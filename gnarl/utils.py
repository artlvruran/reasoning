"""Graph generation and small shared helpers."""
from __future__ import annotations

import networkx as nx
import numpy as np


def random_er_graph(n: int, p: float, directed: bool = False, seed=None,
                     weighted: bool = False, connected: bool = True) -> nx.Graph:
    """Erdos-Renyi graph, optionally forced connected (as used for BFS/DFS/
    Bellman-Ford/MST-Prim training data, Appendix D.3/D.4/D.5)."""
    rng = np.random.default_rng(seed)
    cls = nx.DiGraph if directed else nx.Graph
    for attempt in range(200):
        G = nx.gnp_random_graph(n, p, seed=int(rng.integers(0, 1 << 30)), directed=directed)
        if not connected or nx.is_connected(G.to_undirected() if directed else G):
            break
    if weighted:
        for u, v in G.edges():
            G[u][v]["weight"] = float(rng.uniform(0.1, 1.0))
    return G


def random_ba_graph(n: int, m: int, seed=None, weighted: bool = False) -> nx.Graph:
    rng = np.random.default_rng(seed)
    G = nx.barabasi_albert_graph(n, m, seed=int(rng.integers(0, 1 << 30)))
    if weighted:
        for u, v in G.edges():
            G[u][v]["weight"] = float(rng.uniform(0.1, 1.0))
    return G


def random_euclidean_tsp(n: int, seed=None):
    """Random points in unit square + complete weighted graph for TSP."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, 1, size=(n, 2))
    G = nx.complete_graph(n)
    for u, v in G.edges():
        G[u][v]["weight"] = float(np.linalg.norm(pts[u] - pts[v]))
    return G, pts


def one_hot(idx: int, size: int) -> np.ndarray:
    v = np.zeros((size,), dtype=np.float32)
    if idx is not None and 0 <= idx < size:
        v[idx] = 1.0
    return v
