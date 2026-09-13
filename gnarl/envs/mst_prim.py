"""MST-Prim MDP (Table 10, Algorithm 4, expert Algorithm 12, Appendix D.5)."""
from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np

from ..model import FeatureSpec
from .base import GNARLEnv, GraphState

P = 2
BIG = 1e6


class MSTPrimEnv(GNARLEnv):
    node_specs = (
        FeatureSpec("vs", "node", 1),
        FeatureSpec("psi_1", "node", 1),
        FeatureSpec("psi_2", "node", 1),
        FeatureSpec("key", "node", 1),
        FeatureSpec("mark", "node", 1),
        FeatureSpec("in_queue", "node", 1),
    )
    edge_specs = (
        FeatureSpec("A", "edge", 1),
        FeatureSpec("is_pred", "edge", 1),
    )
    graph_specs = (FeatureSpec("p", "graph", P),)

    def __init__(self):
        super().__init__()
        self.directed = False

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
        self.node_feat["key"] = self._zeros_node(1)
        self.node_feat["mark"] = self._zeros_node(1)
        self.node_feat["in_queue"] = self._zeros_node(1)
        self.edge_feat["A"] = np.array(
            [[self.edge_weight(u, v)] for (u, v) in self.edge_list], dtype=np.float32
        ) if self.edge_list else np.zeros((0, 1), dtype=np.float32)
        self.edge_feat["is_pred"] = self._zeros_edge(1)
        self.graph_feat["p"] = np.zeros((P,), dtype=np.float32)
        self.graph_feat["p"][0] = 1.0
        self.p, self.t = 1, 0
        self.h = P * self.n * self.n
        self.done = False
        # seed: pretend psi_1 = start so phase-2 can expand from it first round
        self._psi1_seed = start
        return self.state()

    def _get_psi(self, m):
        col = self.node_feat[f"psi_{m}"][:, 0]
        if col.sum() == 0:
            return None
        return int(np.argmax(col))

    def _set_psi(self, m, v):
        key = f"psi_{m}"
        self.node_feat[key][:] = 0.0
        self.node_feat[key][v, 0] = 1.0

    def _current_root(self):
        p1 = self._get_psi(1)
        return p1 if p1 is not None else self._psi1_seed

    def action_mask(self) -> np.ndarray:
        mask = np.zeros((self.n,), dtype=bool)
        if self.p == 1:
            root = self._current_root()
            mask[root] = True
            inq = self.node_feat["in_queue"][:, 0].astype(bool)
            mask |= inq
            if not mask.any():
                mask[self.start] = True
        else:
            u = self._current_root()
            for v in self.neighbours(u):
                mask[v] = True
            if not mask.any():
                mask[:] = True
        return mask

    def step(self, action: int):
        v = int(action)
        if self.p == 1:
            self.node_feat["mark"][v, 0] = 1.0
            self.node_feat["in_queue"][v, 0] = 0.0
        else:
            u = self._current_root()
            if self.G.has_edge(u, v) and self.node_feat["mark"][v, 0] == 0.0:
                w = self.edge_weight(u, v)
                if self.node_feat["in_queue"][v, 0] == 0.0 or w < self.node_feat["key"][v, 0]:
                    self.pred[v] = u
                    self.node_feat["key"][v, 0] = w
                    self.node_feat["in_queue"][v, 0] = 1.0
                    e = (u, v)
                    if e in self.edge_idx:
                        self.edge_feat["is_pred"][:, 0] = 0.0
                        self.edge_feat["is_pred"][self.edge_idx[e], 0] = 1.0
        self._set_psi(self.p, v)
        self.p = (self.p % P) + 1
        self.graph_feat["p"][:] = 0.0
        self.graph_feat["p"][self.p - 1] = 1.0
        self.t += 1
        solved = self.node_feat["mark"][:, 0].all()
        self.done = bool(solved) or self.t >= self.h
        return self.state(), 0.0, self.done, {}

    # -- expert policy (Algorithm 12) --------------------------------------
    def _phase_two_candidates(self, u):
        mark = self.node_feat["mark"][:, 0].astype(bool)
        key = self.node_feat["key"][:, 0]
        inq = self.node_feat["in_queue"][:, 0].astype(bool)
        cand = []
        for v in self.neighbours(u):
            if v == u or mark[v]:
                continue
            w = self.edge_weight(u, v)
            if (not inq[v]) or (w < key[v]):
                cand.append(v)
        return cand

    def expert_probs(self) -> Optional[np.ndarray]:
        probs = np.zeros((self.n,), dtype=np.float64)
        if self.p == 1:
            root = self._current_root()
            if self._phase_two_candidates(root):
                k = root
            else:
                inq = np.where(self.node_feat["in_queue"][:, 0].astype(bool))[0]
                key = self.node_feat["key"][:, 0]
                order = inq[np.argsort(key[inq])]
                k = None
                for u in order:
                    if self._phase_two_candidates(int(u)):
                        k = int(u)
                        break
                if k is None:
                    return None
            probs[k] = 1.0
            return probs
        else:
            u = self._current_root()
            cand = self._phase_two_candidates(u)
            if not cand:
                return None
            for v in cand:
                probs[v] = 1.0 / len(cand)
            return probs

    def is_correct(self) -> bool:
        if not self.node_feat["mark"][:, 0].all():
            return False
        weight = sum(self.edge_weight(int(self.pred[v]), v) for v in range(self.n) if self.pred[v] != v)
        ref = nx.minimum_spanning_tree(self.G, weight="weight")
        ref_weight = sum(d["weight"] for _, _, d in ref.edges(data=True))
        return abs(weight - ref_weight) < 1e-6

    def objective(self) -> float:
        weight = sum(self.edge_weight(int(self.pred[v]), v) for v in range(self.n) if self.pred[v] != v)
        return -weight
