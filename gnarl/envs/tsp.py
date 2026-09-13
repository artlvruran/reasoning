"""TSP MDP (Definition 1, Table 11, Appendix D.6).

Design note / simplification: the paper's Algorithm 5 builds the tour using
an insert-at-head pointer trick (`pred` doubles as a circular successor
pointer that supports insertion anywhere, mirroring Khalil et al. 2017's
insertion heuristic). Per Appendix C.2 ("the choice of representation is not
unique"), we use the simpler, functionally equivalent *append-at-tail*
construction: the tour is built by appending the chosen node to the current
partial tour, `is_next` edge features mark the tour's current tail edges,
and the tour is closed (an edge back to the start node) on the final step.
This preserves the MDP's semantics (a permutation of V built by sequential
node selection, valid by construction) while being simpler to implement
correctly.
"""
from __future__ import annotations

from typing import List, Optional

import networkx as nx
import numpy as np

from ..model import FeatureSpec
from .base import GNARLEnv, GraphState


class TSPEnv(GNARLEnv):
    node_specs = (
        FeatureSpec("vs", "node", 1),
        FeatureSpec("in_tour", "node", 1),
    )
    edge_specs = (
        FeatureSpec("A", "edge", 1),
        FeatureSpec("is_next", "edge", 1),
    )
    graph_specs = (FeatureSpec("p", "graph", 1),)

    def __init__(self):
        super().__init__()
        self.directed = True  # complete graph, but we store both directions

    def reset(self, G: nx.Graph, start: int = 0) -> GraphState:
        # store as complete directed graph (both directions, symmetric weight)
        self.G = nx.DiGraph()
        self.G.add_nodes_from(G.nodes())
        for u, v, d in G.edges(data=True):
            w = d.get("weight", 1.0)
            self.G.add_edge(u, v, weight=w)
            self.G.add_edge(v, u, weight=w)
        self.n = self.G.number_of_nodes()
        self._build_edge_index()
        self.start = start
        self.node_feat["vs"] = self._zeros_node(1)
        self.node_feat["vs"][start, 0] = 1.0
        self.node_feat["in_tour"] = self._zeros_node(1)
        self.edge_feat["A"] = np.array(
            [[self.edge_weight(u, v)] for (u, v) in self.edge_list], dtype=np.float32
        )
        self.edge_feat["is_next"] = self._zeros_edge(1)
        self.graph_feat["p"] = np.ones((1,), dtype=np.float32)
        self.t = 0
        self.h = self.n
        self.done = False
        self.tour: List[int] = []
        return self.state()

    def action_mask(self) -> np.ndarray:
        mask = np.zeros((self.n,), dtype=bool)
        if self.t == 0:
            mask[self.start] = True
        else:
            in_tour = self.node_feat["in_tour"][:, 0].astype(bool)
            mask[:] = ~in_tour
        return mask

    def step(self, action: int):
        v = int(action)
        self.node_feat["in_tour"][v, 0] = 1.0
        reward = 0.0
        if self.tour:
            prev = self.tour[-1]
            w = self.edge_weight(prev, v)
            reward -= w
            e = (prev, v)
            if e in self.edge_idx:
                self.edge_feat["is_next"][self.edge_idx[e], 0] = 1.0
        self.tour.append(v)
        self.t += 1
        self.done = self.t >= self.h
        if self.done:
            # close the tour
            last, first = self.tour[-1], self.tour[0]
            reward -= self.edge_weight(last, first)
            e = (last, first)
            if e in self.edge_idx:
                self.edge_feat["is_next"][self.edge_idx[e], 0] = 1.0
        return self.state(), reward, self.done, {}

    def tour_length(self, tour: Optional[List[int]] = None) -> float:
        tour = tour if tour is not None else self.tour
        total = 0.0
        for i in range(len(tour)):
            total += self.edge_weight(tour[i], tour[(i + 1) % len(tour)])
        return total

    def is_correct(self) -> bool:
        return len(self.tour) == self.n and len(set(self.tour)) == self.n

    def objective(self) -> float:
        return -self.tour_length()

    # -- exact expert (Held-Karp DP), Concorde substitute for small n ------
    def set_expert_tour(self, tour: List[int]):
        self._expert_tour = list(tour)

    def expert_probs(self) -> Optional[np.ndarray]:
        if not hasattr(self, "_expert_tour"):
            return None
        idx = len(self.tour)
        if idx >= len(self._expert_tour):
            return None
        # verify the current partial tour matches the expert prefix
        if self.tour != self._expert_tour[:idx]:
            return None
        probs = np.zeros((self.n,), dtype=np.float64)
        probs[self._expert_tour[idx]] = 1.0
        return probs


def held_karp(G: nx.Graph, start: int = 0) -> List[int]:
    """Exact TSP via Held-Karp DP, O(2^n n^2). Used as an exact-solver
    substitute for the Concorde solver (Applegate et al., 1998) used in the
    paper, for small n (<= ~15)."""
    nodes = list(G.nodes())
    n = len(nodes)
    idx_of = {u: i for i, u in enumerate(nodes)}
    start_i = idx_of[start]
    W = np.zeros((n, n))
    for u in nodes:
        for v in nodes:
            if u != v:
                W[idx_of[u], idx_of[v]] = G[u][v]["weight"] if G.has_edge(u, v) else 1e9

    others = [i for i in range(n) if i != start_i]
    from itertools import combinations

    # C[(subset, j)] = min cost of a path start_i -> ... -> j visiting
    # exactly the nodes in `subset` (subset always includes j).
    C = {}
    parent = {}
    for j in others:
        C[(frozenset([j]), j)] = W[start_i, j]

    for size in range(2, len(others) + 1):
        for subset in combinations(others, size):
            subset_fs = frozenset(subset)
            for j in subset:
                prev_fs = subset_fs - {j}
                best_cost, best_k = float("inf"), None
                for k in subset:
                    if k == j:
                        continue
                    prev_cost = C.get((prev_fs, k))
                    if prev_cost is None:
                        continue
                    cost = prev_cost + W[k, j]
                    if cost < best_cost:
                        best_cost, best_k = cost, k
                C[(subset_fs, j)] = best_cost
                parent[(subset_fs, j)] = best_k

    full = frozenset(others)
    best_cost, best_j = float("inf"), None
    for j in others:
        cost = C[(full, j)] + W[j, start_i]
        if cost < best_cost:
            best_cost, best_j = cost, j

    # reconstruct path start_i -> ... -> best_j by walking `parent` backwards
    tour_idx = []
    subset = full
    j = best_j
    while j is not None:
        tour_idx.append(j)
        k = parent.get((subset, j))
        subset = subset - {j}
        j = k
    tour_idx.append(start_i)
    tour_idx.reverse()  # now start_i -> ... -> best_j
    assert len(tour_idx) == n, (len(tour_idx), n)
    return [nodes[i] for i in tour_idx]
