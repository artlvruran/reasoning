"""
Builds GAP samples in the NAR format PDNAR expects: an input instance, the
algorithm's execution trace {S^(t)}, and the final output y = A(x)
(paper Sec. 3.1). Meant to sit next to whatever `vertex_cover_dataset.py` /
`set_cover_dataset.py` / `hitting_set_dataset.py` already do in the real
`src/data/` package -- see README_INTEGRATION.md for exact wiring, since
this repository's actual dataset base-classes could not be inspected
(GitHub blocked automated directory browsing of the private file tree from
this environment).

Each sample is a plain dict of numpy arrays so it has zero hard dependency
on torch/PyG; `src/models/gap_bipartite.py` shows how to convert one of
these into a PyG `HeteroData` object for the GNN.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional

import numpy as np

from src.dataset.algorithms.gap_pd import (
    GAPInstance,
    generate_gap_instance,
    solve_gap_exact,
    solve_gap_lagrangian_pd,
    regret2_repair,
)

# Small enough that branch-and-bound finishes in well under a second per
# instance; above this, fall back to the (label-free) PD+regret heuristic
# as a "silver" label, exactly as PDNAR uses exact solvers only for small
# instances and the approximation algorithm's own output elsewhere.
EXACT_LABEL_TASK_LIMIT = 14


def build_gap_sample(
    n_agents: int,
    n_tasks: int,
    seed: int,
    max_pd_iters: int = 60,
    use_exact_label: Optional[bool] = None,
) -> dict:
    instance = generate_gap_instance(n_agents=n_agents, n_tasks=n_tasks, seed=seed)

    trace, heuristic_assignment, heuristic_obj = solve_gap_lagrangian_pd(
        instance, max_iters=max_pd_iters
    )
    regret_assignment, regret_obj = regret2_repair(instance, trace[-1].x)
    if regret_obj <= heuristic_obj and len(regret_assignment) >= len(heuristic_assignment):
        heuristic_assignment, heuristic_obj = regret_assignment, regret_obj

    if use_exact_label is None:
        use_exact_label = n_tasks <= EXACT_LABEL_TASK_LIMIT

    if use_exact_label:
        exact_assignment, exact_obj = solve_gap_exact(instance)
    else:
        exact_assignment, exact_obj = None, None

    label_assignment = exact_assignment if exact_assignment is not None else heuristic_assignment
    label_source = "exact" if exact_assignment is not None else "pd_heuristic"

    y = np.zeros((n_agents, n_tasks), dtype=bool)
    for j, i in label_assignment.items():
        y[i, j] = True

    trace_arrays = {
        "mu": np.stack([s.mu for s in trace]),                       # (T, n_tasks)
        "x": np.stack([s.x for s in trace]),                         # (T, n_agents, n_tasks)
        "count": np.stack([s.count for s in trace]),                 # (T, n_tasks)
        "remaining_capacity": np.stack([s.remaining_capacity for s in trace]),  # (T, n_agents)
    }

    return {
        "n_agents": n_agents,
        "n_tasks": n_tasks,
        "seed": seed,
        # ---- input x -----------------------------------------------------
        "cost": instance.cost,
        "resource": instance.resource,
        "capacity": instance.capacity,
        "compatible": instance.compatible,
        # ---- algorithm trace {S^(t)}_{t=0}^{T} ----------------------------
        "trace": trace_arrays,
        "num_steps": len(trace),
        # ---- output y = A(x) ----------------------------------------------
        "y_assignment": y,
        "y_objective": float(sum(
            instance.cost[i, j] for j, i in label_assignment.items()
        )),
        "y_source": label_source,   # "exact" or "pd_heuristic" -- like the
                                     # Gurobi-vs-approximation split in
                                     # PDNAR Sec 4.4 / Appendix C.2
        "pd_heuristic_objective": heuristic_obj,
        "exact_objective": exact_obj,
    }


def build_gap_dataset(
    n_instances: int,
    n_agents_range=(3, 8),
    n_tasks_range=(6, 20),
    seed0: int = 0,
) -> list[dict]:
    """Batch generator, mirroring the size-randomized synthetic datasets in
    PDNAR Sec 5.2 ("Size and OOD generalization")."""
    rng = np.random.default_rng(seed0)
    dataset = []
    for k in range(n_instances):
        n_agents = int(rng.integers(n_agents_range[0], n_agents_range[1] + 1))
        n_tasks = int(rng.integers(n_tasks_range[0], n_tasks_range[1] + 1))
        dataset.append(build_gap_sample(n_agents, n_tasks, seed=seed0 + k))
    return dataset


if __name__ == "__main__":
    samples = build_gap_dataset(n_instances=5, seed0=42)
    for s in samples:
        print(
            f"n_agents={s['n_agents']:2d} n_tasks={s['n_tasks']:2d} "
            f"label={s['y_source']:12s} obj={s['y_objective']:8.3f} "
            f"pd_obj={s['pd_heuristic_objective']:8.3f} steps={s['num_steps']}"
        )
