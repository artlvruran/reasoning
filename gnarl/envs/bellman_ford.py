"""Bellman-Ford MDP (Table 7, Algorithm 3, expert Algorithm 11, Appendix C/D.4)."""
from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np

from ..model import FeatureSpec
from .base import GNARLEnv, GraphState

P = 2


class BellmanFordEnv(GNARLEnv):
    node_specs = (
        FeatureSpec("vs", "node", 1),
        FeatureSpec("psi_1", "node", 1),
        FeatureSpec("psi_2", "node", 1),
        FeatureSpec("mask", "node", 1),
        FeatureSpec("d", "node", 1),
    )
    edge_specs = (
        FeatureSpec("A", "edge", 1),
        FeatureSpec("is_pred", "edge", 1),
    )
    graph_specs = (FeatureSpec("p", "graph", P),)

    def __init__(self, horizon_mult: float = 2.0):
        super().__init__()
        self.directed = False
        self.horizon_mult = horizon_mult

    def reset(self, G: nx.Graph, start: int = 0) -> GraphState:
        self.G = G
        self.n = G.number_of_nodes()
        self._build_edge_index()
        self.start = start
        self.pred = np.arange(self.n)
        self.node_feat["vs"] = self._zeros_node(1)
        self.node_feat["vs"][start, 0] = 1.0
        self.node_feat["psi_1"] = self._zeros_node(1)
        self.node_feat["psi_2"] = self._zeros_node(1)
        self.node_feat["mask"] = self._zeros_node(1)
        self.node_feat["mask"][start, 0] = 1.0
        self.node_feat["d"] = self._zeros_node(1)
        self.edge_feat["A"] = np.array(
            [[self.edge_weight(u, v)] for (u, v) in self.edge_list], dtype=np.float32
        ) if self.edge_list else np.zeros((0, 1), dtype=np.float32)
        self.edge_feat["is_pred"] = self._zeros_edge(1)
        self.graph_feat["p"] = np.zeros((P,), dtype=np.float32)
        self.graph_feat["p"][0] = 1.0
        self.p, self.t = 1, 0
        m = len(self.edge_list)
        # generous horizon; true expert trajectory length is typically far shorter
        self.h = int(self.horizon_mult * P * max(1, self.n - 1) * max(1, m))
        self.done = False
        self._ref_dist = None
        return self.state()

    # -- helpers -----------------------------------------------------------
    def _reference_distances(self):
        if self._ref_dist is None:
            self._ref_dist = nx.single_source_dijkstra_path_length(self.G, self.start, weight="weight")
        return self._ref_dist

    def _get_psi(self, m):
        col = self.node_feat[f"psi_{m}"][:, 0]
        if col.sum() == 0:
            return None
        return int(np.argmax(col))

    def _set_psi(self, m, v):
        key = f"psi_{m}"
        self.node_feat[key][:] = 0.0
        self.node_feat[key][v, 0] = 1.0

    def action_mask(self) -> np.ndarray:
        mask = np.zeros((self.n,), dtype=bool)
        if self.p == 1:
            reached = self.node_feat["mask"][:, 0].astype(bool)
            mask[:] = reached
            if not mask.any():
                mask[self.start] = True
        else:
            psi1 = self._get_psi(1)
            for v in self.neighbours(psi1):
                mask[v] = True
            if not mask.any():
                mask[:] = self.node_feat["mask"][:, 0].astype(bool)
        return mask

    def step(self, action: int):
        v = int(action)
        if self.p == 2:
            u = self._get_psi(1)
            if self.G.has_edge(u, v):
                w = self.edge_weight(u, v)
                self.node_feat["d"][v, 0] = self.node_feat["d"][u, 0] + w
                self.pred[v] = u
                self.node_feat["mask"][v, 0] = 1.0
                e = (u, v)
                if e in self.edge_idx:
                    self.edge_feat["is_pred"][:, 0] = 0.0
                    self.edge_feat["is_pred"][self.edge_idx[e], 0] = 1.0
        self._set_psi(self.p, v)
        self.p = (self.p % P) + 1
        self.graph_feat["p"][:] = 0.0
        self.graph_feat["p"][self.p - 1] = 1.0
        self.t += 1
        solved = self.is_correct()
        self.done = solved or self.t >= self.h
        reward = 0.0
        return self.state(), reward, self.done, {}

    # -- expert policy (Algorithm 11) --------------------------------------
    def _phase_two_candidates(self, u):
        d = self.node_feat["d"][:, 0]
        mask = self.node_feat["mask"][:, 0].astype(bool)
        cand = []
        for v in self.neighbours(u):
            if v == u:
                continue
            w = self.edge_weight(u, v)
            if (d[u] + w < d[v]) or (not mask[v]):
                cand.append(v)
        return cand

    def expert_probs(self) -> Optional[np.ndarray]:
        probs = np.zeros((self.n,), dtype=np.float64)
        if self.p == 1:
            mask = self.node_feat["mask"][:, 0].astype(bool)
            possible = [] if mask.any() else [self.start]
            for u in np.where(mask)[0]:
                if self._phase_two_candidates(int(u)):
                    possible.append(int(u))
            if not possible:
                return None
            for u in possible:
                probs[u] = 1.0 / len(possible)
            return probs
        else:
            u = self._get_psi(1)
            cand = self._phase_two_candidates(u)
            if not cand:
                return None
            for v in cand:
                probs[v] = 1.0 / len(cand)
            return probs

    def is_correct(self) -> bool:
        ref = self._reference_distances()
        if len(ref) != self.n:
            return False
        d = self.node_feat["d"][:, 0]
        for v in range(self.n):
            if abs(d[v] - ref[v]) > 1e-6:
                return False
        return True

    def objective(self) -> float:
        return 1.0 if self.is_correct() else 0.0
