"""
GAP (Generalized Assignment Problem) as a primal-dual algorithmic-reasoning
task, in the style of PDNAR (Primal-Dual Neural Algorithmic Reasoning,
He & Vitercik, 2025 -- https://arxiv.org/abs/2505.24067,
code: https://github.com/dransyhe/pdnar).

This module is dependency-light (numpy only) so it can be unit-tested and
used to build supervision data (instances + optimal labels + execution
traces) independently of the GNN model / torch stack.

------------------------------------------------------------------------
Problem (matches the slides "MILP-postановка / Лагранжева релаксация"):

    min   sum_i sum_j c_ij * x_ij
    s.t.  sum_j a_ij * x_ij <= b_i        for all agents i      (capacity)
          sum_i x_ij        == 1          for all tasks j       (coverage)
          x_ij in {0,1},   only for compatible (i, j)

Primal-dual view (Fisher, Jaikumar & Van Wassenhove, 1986):
    - Dual variables  mu_j  (one per task j), relaxing the *coverage*
      equality constraint.
    - For fixed mu, the Lagrangian decomposes into one 0/1-knapsack per
      agent i, with "adjusted" cost  AdjustedCost_ij = c_ij - mu_j.
    - Subgradient update:  mu_j <- mu_j + step * (1 - Count_j)
      where Count_j = number of agents currently selecting task j.

This is *structurally* the same primal-dual loop as PDNAR's Algorithm 1
(General primal-dual approximation algorithm): dual variables grow until
some primal "residual" hits zero (an assignment becomes attractive),
that element is (tentatively) added to the solution, and growth
continues for whatever is still uncovered. The one addition GAP makes
on top of the covering problems in the paper (vertex cover / set cover /
hitting set) is a *packing* side-constraint (agent capacity) -- each
agent runs a local knapsack instead of unconditionally accepting every
"tight" element. See DESIGN.md for the full mapping.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------- #
# Instance representation
# --------------------------------------------------------------------------- #

@dataclass
class GAPInstance:
    n_agents: int
    n_tasks: int
    cost: np.ndarray          # (n_agents, n_tasks) float, c_ij
    resource: np.ndarray      # (n_agents, n_tasks) float, a_ij (capacity used)
    capacity: np.ndarray      # (n_agents,) float, b_i
    compatible: np.ndarray    # (n_agents, n_tasks) bool, A_tv in the slides

    def edges(self):
        """Iterate over compatible (agent, task) candidate pairs."""
        ii, jj = np.nonzero(self.compatible)
        return list(zip(ii.tolist(), jj.tolist()))


def generate_gap_instance(
    n_agents: int,
    n_tasks: int,
    seed: int = 0,
    cost_range=(1.0, 20.0),
    resource_range=(1.0, 10.0),
    capacity_slack: float = 1.35,
    compat_prob: float = 0.85,
) -> GAPInstance:
    """Synthetic GAP instance generator, guaranteed feasible.

    Mirrors the "Barabasi-Albert bipartite graph" style synthetic datasets
    used in the PDNAR appendix (C.1) for other bipartite tasks: a random
    bipartite compatibility graph plus random weights.

    Feasibility is guaranteed by construction (rather than rejection
    sampling, which can be arbitrarily slow for tight instances): a random
    "planted" feasible assignment is drawn first, and each agent's capacity
    is then set to comfortably exceed the resource it would need to serve
    its planted tasks, scaled by `capacity_slack` (>1 loosens the
    instance / makes it more likely several near-optimal solutions exist;
    close to 1 makes the knapsacks tight and the instance harder).
    """
    rng = np.random.default_rng(seed)

    cost = rng.uniform(*cost_range, size=(n_agents, n_tasks))
    resource = rng.uniform(*resource_range, size=(n_agents, n_tasks))
    compatible = rng.random((n_agents, n_tasks)) < compat_prob
    # Every task needs >=1 compatible agent.
    for j in range(n_tasks):
        if not compatible[:, j].any():
            compatible[rng.integers(n_agents), j] = True

    # Plant a feasible assignment: each task -> a uniformly random
    # compatible agent.
    planted_agent = np.array(
        [rng.choice(np.where(compatible[:, j])[0]) for j in range(n_tasks)]
    )
    required = np.zeros(n_agents)
    for j, i in enumerate(planted_agent):
        required[i] += resource[i, j]

    # Capacity comfortably covers what the planted solution needs; agents
    # that were never planted still get a small baseline capacity so they
    # remain useful alternatives during optimization.
    baseline = resource[compatible].mean() if compatible.any() else 1.0
    capacity = np.maximum(required, 0.5 * baseline) * rng.uniform(
        capacity_slack, capacity_slack + 0.3, size=n_agents
    )

    cost = np.where(compatible, cost, np.inf)
    resource = np.where(compatible, resource, np.inf)

    return GAPInstance(
        n_agents=n_agents,
        n_tasks=n_tasks,
        cost=cost,
        resource=resource,
        capacity=capacity,
        compatible=compatible,
    )


# --------------------------------------------------------------------------- #
# Exact solver (small instances only) -- used to label supervision data,
# matching PDNAR Sec. 4.4 "Use of optimal solutions from small instances".
# --------------------------------------------------------------------------- #

def solve_gap_exact(instance: GAPInstance, time_limit_nodes: int = 2_000_000):
    """Branch-and-bound exact solver, task-by-task, agent-capacity pruned.

    Only intended for small instances (tens of tasks at most) -- this is
    meant to generate ground-truth *labels* for training/evaluating the
    GNN, exactly as PDNAR uses exact/near-exact solvers to supervise small
    instances (Gurobi in the paper; here a from-scratch B&B since this
    environment has no MILP solver installed).

    Returns (assignment, objective) or (None, np.inf) if infeasible.
    assignment: dict task_j -> agent_i
    """
    n_agents, n_tasks = instance.n_agents, instance.n_tasks
    order = np.argsort(-instance.compatible.sum(axis=0))  # hardest tasks first (fewest options)
    tasks = order.tolist()

    best = {"obj": np.inf, "assign": None}
    remaining_cap = instance.capacity.copy()
    assign = {}
    nodes = [0]

    def lower_bound(depth):
        # Cheap admissible bound: cheapest compatible (feasible-so-far) cost
        # per remaining task, ignoring future capacity interactions.
        bound = 0.0
        for j in tasks[depth:]:
            feasible_costs = [
                instance.cost[i, j]
                for i in range(n_agents)
                if instance.compatible[i, j] and instance.resource[i, j] <= remaining_cap[i] + 1e-9
            ]
            if not feasible_costs:
                return np.inf
            bound += min(feasible_costs)
        return bound

    def rec(depth, cur_cost):
        nodes[0] += 1
        if nodes[0] > time_limit_nodes:
            return
        if cur_cost >= best["obj"]:
            return
        if depth == len(tasks):
            best["obj"] = cur_cost
            best["assign"] = dict(assign)
            return
        j = tasks[depth]
        bound = cur_cost + lower_bound(depth)
        if bound >= best["obj"]:
            return
        candidates = [
            i for i in range(n_agents)
            if instance.compatible[i, j] and instance.resource[i, j] <= remaining_cap[i] + 1e-9
        ]
        candidates.sort(key=lambda i: instance.cost[i, j])
        for i in candidates:
            remaining_cap[i] -= instance.resource[i, j]
            assign[j] = i
            rec(depth + 1, cur_cost + instance.cost[i, j])
            del assign[j]
            remaining_cap[i] += instance.resource[i, j]

    rec(0, 0.0)
    if best["assign"] is None:
        return None, np.inf
    return best["assign"], best["obj"]


# --------------------------------------------------------------------------- #
# Primal-dual (Lagrangian / subgradient) trace generator -- the "classical
# algorithm" PDNAR trains the GNN to imitate, analogous to Algorithm 1.
# --------------------------------------------------------------------------- #

@dataclass
class GAPStep:
    """One state S^(t) in the execution trace, matching NAR's
    {S^(t)}_{t=0}^{T} formalism (paper, Sec. 3.1)."""
    iteration: int
    mu: np.ndarray                 # (n_tasks,) dual variables
    x: np.ndarray                  # (n_agents, n_tasks) bool, current tentative assignment
    count: np.ndarray              # (n_tasks,) int, agents currently selecting each task
    remaining_capacity: np.ndarray  # (n_agents,) float


def _agent_knapsack(
    mu: np.ndarray,
    instance: GAPInstance,
    agent: int,
    capacity_quantum: float,
) -> np.ndarray:
    """Solve agent `agent`'s 0/1 knapsack over its compatible tasks with
    profit = mu_j - c_ij (only keep strictly-positive-profit items) subject
    to capacity b_i, via DP on a discretized capacity axis.

    Returns a boolean mask over tasks (True = agent selects task).
    """
    n_tasks = instance.n_tasks
    compat = np.where(instance.compatible[agent])[0]
    profit = mu[compat] - instance.cost[agent, compat]
    weight = instance.resource[agent, compat]

    keep = profit > 1e-9
    compat, profit, weight = compat[keep], profit[keep], weight[keep]
    selection = np.zeros(n_tasks, dtype=bool)
    if len(compat) == 0:
        return selection

    cap_units = max(1, int(round(instance.capacity[agent] / capacity_quantum)))
    w_units = np.maximum(1, np.round(weight / capacity_quantum).astype(int))

    # Standard 0/1 knapsack DP: dp[c] = best profit using capacity <= c.
    dp = np.zeros(cap_units + 1)
    choice = np.zeros((len(compat), cap_units + 1), dtype=bool)
    for idx in range(len(compat)):
        w = w_units[idx]
        p = profit[idx]
        if w > cap_units:
            continue
        # iterate capacity downward for 0/1 knapsack
        for c in range(cap_units, w - 1, -1):
            cand = dp[c - w] + p
            if cand > dp[c] + 1e-12:
                dp[c] = cand
                choice[idx, c] = True
    # backtrack
    c = cap_units
    for idx in range(len(compat) - 1, -1, -1):
        w = w_units[idx]
        if choice[idx, c]:
            selection[compat[idx]] = True
            c -= w
    return selection


def solve_gap_lagrangian_pd(
    instance: GAPInstance,
    max_iters: int = 60,
    step0: float = 1.0,
    clip_nonneg: bool = True,
    capacity_quantum: float | None = None,
    record_trace: bool = True,
):
    """Primal-dual (Lagrangian relaxation + subgradient ascent) heuristic.

    Mirrors Algorithm 1 of the PDNAR paper: dual variables (mu) are grown
    (here via subgradient rather than the paper's uniform-increase-until-
    tight rule, since GAP's dual is a genuine Lagrangian multiplier, not a
    packing LP dual restricted to y>=0 -- see DESIGN.md, section
    "Relation to Algorithm 1"), and whenever growth makes an assignment
    profitable, the corresponding agent's local knapsack picks it up
    (matching "once tight, add e to A").

    Returns:
        trace: list[GAPStep]
        assignment: dict task_j -> agent_i (after greedy repair)
        objective: float
    """
    n_agents, n_tasks = instance.n_agents, instance.n_tasks
    mu = np.zeros(n_tasks)
    if capacity_quantum is None:
        finite_res = instance.resource[np.isfinite(instance.resource)]
        capacity_quantum = max(finite_res.min() / 4.0, 1e-3) if len(finite_res) else 1.0

    trace: list[GAPStep] = []
    x = np.zeros((n_agents, n_tasks), dtype=bool)

    for t in range(max_iters):
        for i in range(n_agents):
            x[i] = _agent_knapsack(mu, instance, i, capacity_quantum)
        count = x.sum(axis=0)

        remaining_capacity = instance.capacity - np.where(
            instance.compatible, np.where(x, instance.resource, 0.0), 0.0
        ).sum(axis=1)

        if record_trace:
            trace.append(
                GAPStep(
                    iteration=t,
                    mu=mu.copy(),
                    x=x.copy(),
                    count=count.copy(),
                    remaining_capacity=remaining_capacity.copy(),
                )
            )

        if np.all(count == 1):
            break  # dual-feasible & primal-feasible: exact partition found

        step = step0 / np.sqrt(t + 1)
        mu = mu + step * (1 - count)
        if clip_nonneg:
            mu = np.maximum(mu, 0.0)

    assignment, objective = _greedy_repair(instance, x)
    return trace, assignment, objective


def _greedy_repair(instance: GAPInstance, x: np.ndarray):
    """Turn a (possibly infeasible: Count_j != 1) Lagrangian solution into
    a feasible partition via greedy repair, matching the slides'
    "Iterative Repair (Корректировка)" step:
      - tasks with Count_j > 1: keep cheapest agent, free the rest
      - tasks with Count_j == 0: assign to cheapest agent with remaining
        capacity, else leave unassigned
    """
    n_agents, n_tasks = instance.n_agents, instance.n_tasks
    remaining_capacity = instance.capacity.copy()
    assignment: dict[int, int] = {}

    count = x.sum(axis=0)
    # First pass: uniquely-covered and over-covered tasks.
    for j in range(n_tasks):
        agents_selecting = np.where(x[:, j])[0]
        if len(agents_selecting) == 0:
            continue
        agents_selecting = sorted(agents_selecting, key=lambda i: instance.cost[i, j])
        for i in agents_selecting:
            if instance.resource[i, j] <= remaining_capacity[i] + 1e-9:
                assignment[j] = i
                remaining_capacity[i] -= instance.resource[i, j]
                break

    # Second pass: still-unassigned tasks -> cheapest feasible agent.
    for j in range(n_tasks):
        if j in assignment:
            continue
        candidates = [
            i for i in range(n_agents)
            if instance.compatible[i, j]
            and instance.resource[i, j] <= remaining_capacity[i] + 1e-9
        ]
        if not candidates:
            continue  # left unassigned, matches "unassigned" list in the slides
        i_best = min(candidates, key=lambda i: instance.cost[i, j])
        assignment[j] = i_best
        remaining_capacity[i_best] -= instance.resource[i_best, j]

    objective = sum(instance.cost[i, j] for j, i in assignment.items())
    return assignment, objective


def regret2_repair(instance: GAPInstance, x: np.ndarray):
    """Stronger repair pass using Regret-2 insertion (slides: "Оператор
    восстановления: Regret-2 Insertion"), instead of plain greedy.

    For each unassigned task, look at its two cheapest *currently feasible*
    agents; insert first whichever task has the largest gap (regret)
    between its best and second-best option, since that task has the most
    to lose from being delayed. Falls back to greedy for tasks with only
    one feasible agent.
    """
    n_agents, n_tasks = instance.n_agents, instance.n_tasks
    remaining_capacity = instance.capacity.copy()
    assignment: dict[int, int] = {}

    # Seed with the uniquely-covered tasks from the Lagrangian solution,
    # same as the first pass of the greedy repair.
    for j in range(n_tasks):
        agents_selecting = np.where(x[:, j])[0]
        if len(agents_selecting) == 0:
            continue
        agents_selecting = sorted(agents_selecting, key=lambda i: instance.cost[i, j])
        for i in agents_selecting:
            if instance.resource[i, j] <= remaining_capacity[i] + 1e-9:
                assignment[j] = i
                remaining_capacity[i] -= instance.resource[i, j]
                break

    pending = [j for j in range(n_tasks) if j not in assignment]
    while pending:
        best_regret, best_j, best_i = -np.inf, None, None
        for j in pending:
            candidates = [
                (instance.cost[i, j], i)
                for i in range(n_agents)
                if instance.compatible[i, j]
                and instance.resource[i, j] <= remaining_capacity[i] + 1e-9
            ]
            if not candidates:
                continue
            candidates.sort()
            best_cost, best_agent = candidates[0]
            second_cost = candidates[1][0] if len(candidates) > 1 else best_cost
            regret = second_cost - best_cost
            if regret > best_regret:
                best_regret, best_j, best_i = regret, j, best_agent
        if best_j is None:
            break  # remaining pending tasks have no feasible agent left
        assignment[best_j] = best_i
        remaining_capacity[best_i] -= instance.resource[best_i, best_j]
        pending.remove(best_j)

    objective = sum(instance.cost[i, j] for j, i in assignment.items())
    return assignment, objective


if __name__ == "__main__":
    inst = generate_gap_instance(n_agents=4, n_tasks=8, seed=6)
    trace, assign, obj = solve_gap_lagrangian_pd(inst)
    last_x = trace[-1].x
    r_assign, r_obj = regret2_repair(inst, last_x)
    exact_assign, exact_obj = solve_gap_exact(inst)
    print(f"Exact objective:            {exact_obj:.3f}")
    print(f"PD + greedy repair:         {obj:.3f} (covered {len(assign)}/{inst.n_tasks})")
    print(f"PD + regret-2 repair:       {r_obj:.3f} (covered {len(r_assign)}/{inst.n_tasks})")
    print(f"Trace length: {len(trace)} iterations")
