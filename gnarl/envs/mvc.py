"""Minimum Vertex Cover MDP (Definition 2, Table 12, Algorithm 6, Appendix D.7)."""
from __future__ import annotations

from typing import List, Optional

import networkx as nx
import numpy as np

from ..model import FeatureSpec
from .base import GNARLEnv, GraphState

try:
    import pulp
    _HAS_PULP = True
except Exception:
    _HAS_PULP = False


class MVCEnv(GNARLEnv):
    node_specs = (
        FeatureSpec("w", "node", 1),
        FeatureSpec("in_cover", "node", 1),
    )
    edge_specs = (FeatureSpec("adj", "edge", 1),)
    graph_specs = (FeatureSpec("p", "graph", 1),)

    def __init__(self):
        super().__init__()
        self.directed = False

    def reset(self, G: nx.Graph, weights: Optional[np.ndarray] = None) -> GraphState:
        self.G = G
        self.n = G.number_of_nodes()
        self._build_edge_index()
        if weights is None:
            weights = np.ones((self.n,), dtype=np.float32)
        self.weights = np.asarray(weights, dtype=np.float32)
        self.node_feat["w"] = self.weights.reshape(-1, 1).copy()
        self.node_feat["in_cover"] = self._zeros_node(1)
        self.edge_feat["adj"] = np.ones((len(self.edge_list), 1), dtype=np.float32)
        self.graph_feat["p"] = np.ones((1,), dtype=np.float32)
        self.t = 0
        self.h = self.n
        self.done = False
        return self.state()

    def _uncovered_edges(self):
        cover = self.node_feat["in_cover"][:, 0].astype(bool)
        return [(u, v) for (u, v) in self.G.edges() if not (cover[u] or cover[v])]

    def action_mask(self) -> np.ndarray:
        cover = self.node_feat["in_cover"][:, 0].astype(bool)
        return ~cover

    def step(self, action: int):
        v = int(action)
        prev_obj = self.objective()
        self.node_feat["in_cover"][v, 0] = 1.0
        self.t += 1
        self.done = (len(self._uncovered_edges()) == 0) or (self.t >= self.h)
        reward = self.objective() - prev_obj
        return self.state(), reward, self.done, {}

    def is_correct(self) -> bool:
        return len(self._uncovered_edges()) == 0

    def objective(self) -> float:
        cover = self.node_feat["in_cover"][:, 0].astype(bool)
        return -float(self.weights[cover].sum())

    def expert_probs(self) -> Optional[np.ndarray]:
        if not hasattr(self, "_expert_cover"):
            return None
        remaining = [v for v in self._expert_cover
                     if self.node_feat["in_cover"][v, 0] == 0.0]
        if not remaining:
            return None
        probs = np.zeros((self.n,), dtype=np.float64)
        for v in remaining:
            probs[v] = 1.0 / len(remaining)
        return probs

    def set_expert_cover(self, cover: List[int]):
        self._expert_cover = list(cover)


def exact_min_vertex_cover(G: nx.Graph, weights: Optional[np.ndarray] = None) -> List[int]:
    """Exact MVC via ILP (CBC through PuLP), substituting for the exact ILP
    solver used in the paper (He & Vitercik, 2025). Falls back to the
    standard 2-approximation (Khuller et al., 1994) if PuLP/CBC is
    unavailable."""
    n = G.number_of_nodes()
    if weights is None:
        weights = np.ones((n,))
    if _HAS_PULP:
        prob = pulp.LpProblem("mvc", pulp.LpMinimize)
        x = {v: pulp.LpVariable(f"x_{v}", cat="Binary") for v in G.nodes()}
        prob += pulp.lpSum(weights[v] * x[v] for v in G.nodes())
        for u, v in G.edges():
            prob += x[u] + x[v] >= 1
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        return [v for v in G.nodes() if pulp.value(x[v]) > 0.5]
    return two_approx_vertex_cover(G, weights)


def two_approx_vertex_cover(G: nx.Graph, weights=None, epsilon: float = 0.1) -> List[int]:
    """Khuller et al. (1994) 2/(1-epsilon)-approximation, used as the
    baseline heuristic in Section 5.2 / Table 3."""
    H = G.copy()
    cover = set()
    while H.number_of_edges() > 0:
        u, v = next(iter(H.edges()))
        cover.add(u)
        cover.add(v)
        H.remove_node(u)
        H.remove_node(v)
    return list(cover)
