"""
GAP bipartite primal-dual graph + GNN skeleton, following PDNAR's
architecture (paper Sec 4.1: Bipartite graph construction / Encoder /
Processor / Decoder).

IMPORTANT: torch and torch_geometric are not installed in the environment
this file was authored in, so this module could not be executed/unit
tested here. It is written to standard, current PyG APIs
(`MessagePassing`, `HeteroData`) and kept deliberately close to the
class/function names implied by the PDNAR README (`model.model.eps`,
an `Encoder`/`Processor`/`Decoder` split) -- but the exact base classes,
constructor signatures and hidden-dim conventions of the real
`src/models/` package could not be inspected (see README_INTEGRATION.md).
Treat this as a faithful *reference* implementation to adapt into
whatever the repo's actual `PDNARModel` base class looks like, not as a
drop-in file.

------------------------------------------------------------------------
Design: mapping GAP onto PDNAR's element/set bipartite formalism
------------------------------------------------------------------------
PDNAR's other three tasks (vertex cover, set cover, hitting set) are all
pure *covering* problems: elements e in E carry a primal decision x_e and
a residual r_e; sets T in a family carry a dual variable y_T; e in T bipartite
edges connect them. Algorithm 1 grows y_T uniformly until some r_e hits
zero, then e joins the solution A.

GAP adds a *packing* side-constraint (agent capacity) that plain covering
doesn't have, so the roles are assigned as follows:

  * DUAL nodes  = tasks j.            Carries mu_j (dual var for the
                                       relaxed "assigned exactly once"
                                       constraint) and count_j (how many
                                       agents currently want it) --
                                       directly analogous to y_T.
  * PRIMAL nodes = agents i.          Carries capacity b_i and remaining
                                       capacity -- analogous to an
                                       element's weight w_e / residual r_e,
                                       except the "residual" is a resource
                                       budget consumed by *possibly many*
                                       edges rather than a scalar that
                                       hits zero once.
  * EDGES (i, j) = candidate assignments, present iff compatible[i, j].
    Edge features: cost c_ij, resource a_ij, current reduced cost
    r_ij = c_ij - mu_j. The edge-level hidden state is what the decoder
    reads out to predict x_ij (the actual primal decision -- in PDNAR's
    other tasks this lives on elements directly; here it lives on edges
    because one "element" (agent) can satisfy many "sets" (tasks)).

Each processor message-passing round is aligned (for step-training /
"hint" supervision, PDNAR Sec 4.1 "Training") with one iteration of
`solve_gap_lagrangian_pd`: agents recompute their knapsack picks, tasks
recompute mu via the subgradient step. See gap_pd.GAPStep for the
matching trace format.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import MessagePassing
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - executed in torch-free environments
    _TORCH_AVAILABLE = False
    torch = None
    nn = object  # type: ignore
    HeteroData = object  # type: ignore
    MessagePassing = object  # type: ignore


def _require_torch():
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "torch / torch_geometric are required for src.models.gap_bipartite "
            "but are not installed in this environment. Install them in the "
            "real pdnar conda env (see environment.yml) before using this "
            "module for training/inference."
        )


# --------------------------------------------------------------------------- #
# Graph construction: GAP sample (numpy, from gap_dataset.py) -> HeteroData
# --------------------------------------------------------------------------- #

def gap_sample_to_hetero_data(sample: dict, step: int = -1):
    """Convert one sample from `build_gap_sample` into a PyG `HeteroData`
    bipartite graph, following the node/edge typing described in the module
    docstring. `step` selects which point in the trace to use for input
    dual/primal state features (-1 = final/converged state; 0 = the raw
    instance with mu initialized to zero, i.e. what the model should see at
    inference time before it has produced anything)."""
    _require_torch()

    n_agents, n_tasks = sample["n_agents"], sample["n_tasks"]
    compat = sample["compatible"]
    ii, jj = np.nonzero(compat)

    trace = sample["trace"]
    mu = trace["mu"][step] if step is not None else np.zeros(n_tasks)
    remaining_capacity = (
        trace["remaining_capacity"][step] if step is not None else sample["capacity"]
    )

    data = HeteroData()

    # Primal (agent) nodes: [capacity, remaining_capacity]
    data["agent"].x = torch.tensor(
        np.stack([sample["capacity"], remaining_capacity], axis=1), dtype=torch.float32
    )
    # Dual (task) nodes: [mu, count] -- count only meaningful for step>=0
    count = trace["count"][step] if step is not None and step >= 0 else np.zeros(n_tasks)
    data["task"].x = torch.tensor(np.stack([mu, count], axis=1), dtype=torch.float32)

    reduced_cost = sample["cost"][ii, jj] - mu[jj]
    edge_attr = np.stack(
        [sample["cost"][ii, jj], sample["resource"][ii, jj], reduced_cost], axis=1
    )
    data["agent", "candidate", "task"].edge_index = torch.tensor(
        np.stack([ii, jj]), dtype=torch.long
    )
    data["agent", "candidate", "task"].edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    # Reverse edges for bidirectional message passing (PyG convention).
    data["task", "rev_candidate", "agent"].edge_index = torch.tensor(
        np.stack([jj, ii]), dtype=torch.long
    )
    data["task", "rev_candidate", "agent"].edge_attr = data[
        "agent", "candidate", "task"
    ].edge_attr

    if step is not None:
        data["agent", "candidate", "task"].y = torch.tensor(
            trace["x"][step][ii, jj], dtype=torch.float32
        )
    data["agent", "candidate", "task"].y_final = torch.tensor(
        sample["y_assignment"][ii, jj], dtype=torch.float32
    )
    return data


# --------------------------------------------------------------------------- #
# Encoder / Processor / Decoder, matching PDNAR Sec 4.1 terminology
# --------------------------------------------------------------------------- #

if _TORCH_AVAILABLE:

    class GAPEncoder(nn.Module):
        """Projects raw node/edge features into the shared latent space."""

        def __init__(self, hidden_dim: int = 128):
            super().__init__()
            self.agent_lin = nn.Linear(2, hidden_dim)
            self.task_lin = nn.Linear(2, hidden_dim)
            self.edge_lin = nn.Linear(3, hidden_dim)

        def forward(self, data: "HeteroData"):
            h_agent = self.agent_lin(data["agent"].x)
            h_task = self.task_lin(data["task"].x)
            h_edge = self.edge_lin(data["agent", "candidate", "task"].edge_attr)
            return h_agent, h_task, h_edge

    class _BipartiteConv(MessagePassing):
        """One direction of bipartite message passing (agent -> task or
        task -> agent), edge-conditioned, matching how PDNAR aligns GNN
        message-passing rounds with primal-dual algorithm iterations."""

        def __init__(self, hidden_dim: int):
            super().__init__(aggr="add")
            self.msg_mlp = nn.Sequential(
                nn.Linear(3 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
            )
            self.upd_mlp = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
            )

        def forward(self, h_src, h_dst, edge_index, h_edge):
            out = self.propagate(edge_index, x=(h_src, h_dst), edge_attr=h_edge)
            return self.upd_mlp(torch.cat([h_dst, out], dim=-1))

        def message(self, x_j, x_i, edge_attr):
            return self.msg_mlp(torch.cat([x_j, x_i, edge_attr], dim=-1))

    class GAPProcessor(nn.Module):
        """Bipartite message-passing GNN core. `n_steps` message-passing
        rounds are meant to be run once per outer primal-dual iteration
        (agent knapsack update + dual subgradient step) when doing
        step-aligned / "hint" training against `gap_pd.GAPStep` traces;
        for inference-only rollouts they can also be unrolled for a fixed
        budget and read out only at the end."""

        def __init__(self, hidden_dim: int = 128):
            super().__init__()
            self.task_to_agent = _BipartiteConv(hidden_dim)
            self.agent_to_task = _BipartiteConv(hidden_dim)
            self.edge_update = nn.Sequential(
                nn.Linear(3 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
            )

        def forward(self, h_agent, h_task, h_edge, edge_index):
            agent_idx, task_idx = edge_index
            h_agent = self.task_to_agent(h_task, h_agent, torch.stack([task_idx, agent_idx]), h_edge)
            h_task = self.agent_to_task(h_agent, h_task, torch.stack([agent_idx, task_idx]), h_edge)
            h_edge = self.edge_update(
                torch.cat([h_agent[agent_idx], h_task[task_idx], h_edge], dim=-1)
            )
            return h_agent, h_task, h_edge

    class GAPDecoder(nn.Module):
        """Reads out (a) the predicted dual variable mu_j per task, and
        (b) the predicted assignment probability x_ij per candidate edge
        -- the GAP analogues of y_T and x_e in PDNAR's covering tasks."""

        def __init__(self, hidden_dim: int = 128):
            super().__init__()
            self.mu_head = nn.Linear(hidden_dim, 1)
            self.edge_head = nn.Linear(hidden_dim, 1)

        def forward(self, h_task, h_edge):
            mu_pred = self.mu_head(h_task).squeeze(-1)
            x_logits = self.edge_head(h_edge).squeeze(-1)
            return mu_pred, x_logits

    class GAPPDNARModel(nn.Module):
        """End-to-end model: Encoder -> `n_steps` x Processor -> Decoder,
        exposing the same `model.model.eps`-style toggle used by
        vertex_cover/hitting_set for whether to use the paper's uniform-
        increase dual rule (Sec 4.2) as an inductive bias on the mu head."""

        def __init__(self, hidden_dim: int = 128, n_steps: int = 10, eps: bool = False):
            super().__init__()
            self.encoder = GAPEncoder(hidden_dim)
            self.processor = GAPProcessor(hidden_dim)
            self.decoder = GAPDecoder(hidden_dim)
            self.n_steps = n_steps
            self.eps = eps  # placeholder flag; see gap_pd.solve_gap_lagrangian_pd's
                             # `clip_nonneg` for the analogous classical-algorithm knob

        def forward(self, data: "HeteroData"):
            h_agent, h_task, h_edge = self.encoder(data)
            edge_index = data["agent", "candidate", "task"].edge_index
            mu_trace, x_trace = [], []
            for _ in range(self.n_steps):
                h_agent, h_task, h_edge = self.processor(h_agent, h_task, h_edge, edge_index)
                mu_pred, x_logits = self.decoder(h_task, h_edge)
                mu_trace.append(mu_pred)
                x_trace.append(x_logits)
            return {
                "mu_trace": torch.stack(mu_trace),      # (n_steps, n_tasks)
                "x_logit_trace": torch.stack(x_trace),  # (n_steps, n_edges)
                "x_logits_final": x_trace[-1],
            }

else:  # pragma: no cover
    GAPEncoder = GAPProcessor = GAPDecoder = GAPPDNARModel = None  # type: ignore


if __name__ == "__main__":
    if not _TORCH_AVAILABLE:
        print(
            "torch/torch_geometric not installed here -- this file is a "
            "reference implementation only. See DESIGN.md for the mapping "
            "and README_INTEGRATION.md for wiring instructions."
        )
    else:  # pragma: no cover
        import sys
        sys.path.insert(0, ".")
        from src.dataset.gap_dataset import build_gap_sample

        sample = build_gap_sample(n_agents=4, n_tasks=8, seed=0)
        data = gap_sample_to_hetero_data(sample, step=0)
        model = GAPPDNARModel(hidden_dim=32, n_steps=5)
        out = model(data)
        print({k: v.shape for k, v in out.items()})
