"""BFS and DFS MDPs (paper Section 4.1 example + Appendix D.3).

Transition function: Algorithm 1.
Expert policies: Algorithm 9 (BFS), Algorithm 10 (DFS).
Correctness checks: BFS -- predecessor-tree depths match true shortest-path
distances from the source; DFS -- Algorithm 8, reformulated here via the
standard tin/tout ancestor-interval characterisation of a valid DFS forest
(every non-tree edge must connect an ancestor/descendant pair) -- this is
mathematically equivalent to Algorithm 8's recursive cycle check but is
simpler and more numerically robust to implement directly.

Note on a likely OCR/transcription artefact: Algorithm 9's PhaseTwoPolicy as
printed selects neighbours with `reach_j = 1`; by symmetry with Algorithm 10
(DFS) which selects `reach_j = 0` (unvisited neighbours), and because a BFS
frontier expansion must extend to *unvisited* neighbours, we implement
PhaseTwoPolicy for BFS with `reach_j = 0` as well.
"""
from __future__ import annotations

from typing import List, Optional

import networkx as nx
import numpy as np

from ..model import FeatureSpec
from .base import GNARLEnv, GraphState

P = 2  # number of phases: select source node, then select target node


class _BFSDFSBase(GNARLEnv):
    edge_specs = (
        FeatureSpec("adj", "edge", 1),
        FeatureSpec("is_pred", "edge", 1),
    )
    graph_specs = (FeatureSpec("p", "graph", P),)

    def __init__(self, directed: bool = False):
        super().__init__()
        self.directed = directed
        self.pred: Optional[np.ndarray] = None

    def _common_reset(self, G: nx.Graph):
        self.G = G
        self.n = G.number_of_nodes()
        self._build_edge_index()
        self.pred = np.arange(self.n)  # pred_v = v initially (Table 9)
        self.node_feat["psi_1"] = self._zeros_node(1)
        self.node_feat["psi_2"] = self._zeros_node(1)
        self.node_feat["reach"] = self._zeros_node(1)
        self.edge_feat["adj"] = np.ones((len(self.edge_list), 1), dtype=np.float32)
        self.edge_feat["is_pred"] = self._zeros_edge(1)
        self.graph_feat["p"] = np.zeros((P,), dtype=np.float32)
        self.graph_feat["p"][0] = 1.0  # p = 1
        self.p = 1
        self.t = 0
        self.h = P * (self.n - 1)
        self.done = False

    def _phase(self) -> int:
        return int(np.argmax(self.graph_feat["p"])) + 1

    def action_mask(self) -> np.ndarray:
        mask = np.zeros((self.n,), dtype=bool)
        p = self._phase()
        if p == 1:
            mask[:] = True
        else:
            psi1 = int(np.argmax(self.node_feat["psi_1"][:, 0])) if self.node_feat["psi_1"].sum() > 0 else None
            if psi1 is not None:
                for v in self.neighbours(psi1):
                    mask[v] = True
            if not mask.any():
                mask[:] = True  # degenerate isolated node fallback
        return mask

    def _set_psi(self, m: int, v: int):
        key = f"psi_{m}"
        self.node_feat[key][:] = 0.0
        self.node_feat[key][v, 0] = 1.0

    def _get_psi(self, m: int) -> Optional[int]:
        col = self.node_feat[f"psi_{m}"][:, 0]
        if col.sum() == 0:
            return None
        return int(np.argmax(col))

    def step(self, action: int):
        v = int(action)
        p = self._phase()
        if p == P:
            psi1 = self._get_psi(1)
            self.node_feat["reach"][psi1, 0] = 1.0
            self.node_feat["reach"][v, 0] = 1.0
            self.pred[v] = psi1
            e = (psi1, v)
            if e in self.edge_idx:
                self.edge_feat["is_pred"][:, 0] = 0.0
                self.edge_feat["is_pred"][self.edge_idx[e], 0] = 1.0
        self._set_psi(p, v)
        new_p = (p % P) + 1
        self.graph_feat["p"][:] = 0.0
        self.graph_feat["p"][new_p - 1] = 1.0
        self.t += 1
        self.done = self.t >= self.h
        return self.state(), 0.0, self.done, {}

    # -- shared depth-counter helper (used by BFS/DFS expert policies) ----
    # Depth in the predecessor tree, restricted to nodes that are actually
    # `visited_mask`-True; not-yet-visited nodes get +inf so that they never
    # out-rank a genuine (reached but unexpanded) frontier node -- they only
    # matter, via the equal-min/-max tie, when *no* node has been visited
    # yet (handled by the BFS/DFS-specific fallbacks around this helper).
    def _depths(self, visited_mask: np.ndarray, unvisited_value: float) -> np.ndarray:
        depths = np.full((self.n,), unvisited_value)
        computed = np.zeros((self.n,), dtype=bool)

        def get_depth(v):
            if computed[v]:
                return depths[v]
            computed[v] = True  # guard against pathological cycles
            if not visited_mask[v]:
                depths[v] = unvisited_value
            elif self.pred[v] == v:
                depths[v] = 0.0
            else:
                depths[v] = get_depth(self.pred[v]) + 1
            return depths[v]

        for v in range(self.n):
            get_depth(v)
        return depths


class BFSEnv(_BFSDFSBase):
    node_specs = (
        FeatureSpec("vs", "node", 1),
        FeatureSpec("psi_1", "node", 1),
        FeatureSpec("psi_2", "node", 1),
        FeatureSpec("reach", "node", 1),
    )

    def __init__(self):
        super().__init__(directed=False)

    def reset(self, G: nx.Graph, start: int = 0) -> GraphState:
        self._common_reset(G)
        self.node_feat["vs"] = self._zeros_node(1)
        self.node_feat["vs"][start, 0] = 1.0
        self.start = start
        return self.state()

    def expert_probs(self) -> Optional[np.ndarray]:
        p = self._phase()
        reach = self.node_feat["reach"][:, 0].astype(bool)
        probs = np.zeros((self.n,), dtype=np.float64)
        if p == 1:
            if not reach.any():
                # Boundary condition: BFS is rooted at the designated source
                # node vs (Table 9). Before anything has been reached, the
                # generic depth-based rule below is degenerate (every node
                # has depth 0), so we force the expert to begin at vs.
                probs[self.start] = 1.0
                return probs
            closed = np.array([
                reach[v] and all(reach[u] for u in self.neighbours(v))
                for v in range(self.n)
            ])
            open_ = ~closed
            if not open_.any():
                return None
            depths = self._depths(reach, unvisited_value=np.inf)
            d_min = depths[open_].min()
            eligible = np.where(open_ & (depths == d_min))[0]
            probs[eligible] = 1.0 / len(eligible)
            return probs
        else:
            psi1 = self._get_psi(1)
            cand = [j for j in self.neighbours(psi1) if j != psi1 and not reach[j]]
            if not cand:
                return None
            for j in cand:
                probs[j] = 1.0 / len(cand)
            return probs

    def is_correct(self) -> bool:
        try:
            dist = nx.single_source_shortest_path_length(self.G, self.start)
        except Exception:
            return False
        if len(dist) != self.n:
            return False
        depths = {}

        def get_depth(v):
            if v in depths:
                return depths[v]
            if self.pred[v] == v:
                depths[v] = 0
            else:
                depths[v] = get_depth(int(self.pred[v])) + 1
            return depths[v]

        for v in range(self.n):
            if get_depth(v) != dist[v]:
                return False
            if self.pred[v] != v and not self.G.has_edge(int(self.pred[v]), v):
                return False
        return True

    def objective(self) -> float:
        return 1.0 if self.is_correct() else 0.0


class DFSEnv(_BFSDFSBase):
    node_specs = (
        FeatureSpec("psi_1", "node", 1),
        FeatureSpec("psi_2", "node", 1),
        FeatureSpec("reach", "node", 1),
    )

    def __init__(self, directed: bool = True):
        super().__init__(directed=directed)

    def reset(self, G: nx.Graph) -> GraphState:
        self._common_reset(G)
        return self.state()

    def _colour(self) -> np.ndarray:
        reach = self.node_feat["reach"][:, 0].astype(bool)
        colour = np.zeros((self.n,), dtype=np.int64)
        for v in range(self.n):
            if not reach[v]:
                colour[v] = 0
            elif all(reach[u] for u in self.neighbours(v)):
                colour[v] = 2
            else:
                colour[v] = 1
        return colour

    def expert_probs(self) -> Optional[np.ndarray]:
        p = self._phase()
        probs = np.zeros((self.n,), dtype=np.float64)
        if p == 1:
            colour = self._colour()
            visited = colour != 0
            depths = self._depths(visited, unvisited_value=-np.inf)
            active = colour != 2
            if not active.any():
                return None
            d_max = depths[active].max()
            eligible = np.where((colour == 1) & (depths == d_max))[0]
            if len(eligible) == 0:
                eligible = np.where((colour == 0) & (depths == d_max))[0]
            if len(eligible) == 0:
                return None
            probs[eligible] = 1.0 / len(eligible)
            return probs
        else:
            reach = self.node_feat["reach"][:, 0].astype(bool)
            psi1 = self._get_psi(1)
            cand = [j for j in self.neighbours(psi1) if j != psi1 and not reach[j]]
            if not cand and not reach[psi1]:
                cand = [psi1]
            if not cand:
                return None
            for j in cand:
                probs[j] = 1.0 / len(cand)
            return probs

    def is_correct(self) -> bool:
        return _check_valid_dfs_forest(self.G, self.pred, self.n)

    def objective(self) -> float:
        return 1.0 if self.is_correct() else 0.0


def _check_valid_dfs_forest(G: nx.Graph, pred: np.ndarray, n: int) -> bool:
    """Direct translation of Algorithm 8 ("Check if a Predecessor Forest is
    a Valid DFS Solution"), Appendix H. Recursively verifies that, within
    each set of "active" (still-undecided) nodes, the sub-roots induced by
    `pred` do not have any edges between them that would create a cycle in
    the induced component graph -- and recurses into each sub-root's
    descendants. This also implicitly requires that `pred` spans all nodes
    and that every parent edge (pred[v], v) actually exists in G."""

    for v in range(n):
        if pred[v] != v and not G.has_edge(int(pred[v]), v):
            return False

    def is_descendant(v, root, active_nodes):
        cur = v
        while cur in active_nodes and cur != root:
            nxt = int(pred[cur])
            if nxt == cur:
                return False
            cur = nxt
        return cur == root

    def is_valid_forest_recursive(active_nodes):
        if len(active_nodes) <= 1:
            return True
        subroots = {v for v in active_nodes if int(pred[v]) not in active_nodes or pred[v] == v}
        if not subroots:
            return False
        if len(subroots) > 1:
            subroot_of = {}
            for v in active_nodes:
                cur = v
                while cur not in subroots:
                    cur = int(pred[cur])
                subroot_of[v] = cur
            Gcomp = nx.Graph()
            Gcomp.add_nodes_from(subroots)
            for u, v in G.edges():
                if u in active_nodes and v in active_nodes:
                    su, sv = subroot_of[u], subroot_of[v]
                    if su != sv:
                        Gcomp.add_edge(su, sv)
            if Gcomp.number_of_edges() > 0 and not nx.is_forest(Gcomp):
                return False
        for root in subroots:
            descendants = {v for v in active_nodes if v != root and is_descendant(v, root, active_nodes)}
            if descendants:
                if not is_valid_forest_recursive(descendants):
                    return False
        return True

    return is_valid_forest_recursive(set(range(n)))
