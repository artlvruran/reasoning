"""Common infrastructure for GNARL graph-algorithm MDPs (Section 4.1).

Every concrete environment (BFS, DFS, Bellman-Ford, MST-Prim, TSP, MVC ...)
subclasses `GNARLEnv` and is responsible for:
  * building the initial node/edge/graph feature arrays for a given input
    graph (Table 1, 7, 9-13 in the paper),
  * defining the transition function T (the various "Algorithm N" boxes),
  * defining the legal-action mask A(s) at each step (action-masking makes
    solutions valid by construction, Section 4.4),
  * (optionally) an expert action-distribution policy for imitation learning
    (Algorithms 8-12), and a correctness/objective check for evaluation.

Feature bookkeeping is done in plain NumPy (mutated in place each step);
`GraphState.from_env` produces an immutable JAX snapshot to feed to
`gnarl.model.forward`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import jax.numpy as jnp
import networkx as nx
import numpy as np


@dataclass
class GraphState:
    n_nodes: int
    edge_index: jnp.ndarray  # (2, m) int32
    node_feats: Dict[str, jnp.ndarray]
    edge_feats: Dict[str, jnp.ndarray]
    graph_feats: Dict[str, jnp.ndarray]
    action_mask: jnp.ndarray  # (n,) bool
    edge_valid: Optional[jnp.ndarray] = None  # (m,) float32, 1 for real edges,
    # 0 for padding edges introduced when batching variable-sized graphs
    # together (see `pad_state_to`/`stack_states` below). `None` means "all
    # edges are real", i.e. the common, un-padded case.

    def __post_init__(self):
        if self.edge_valid is None:
            m = self.edge_index.shape[1]
            self.edge_valid = jnp.ones((m,), dtype=jnp.float32)


def _graphstate_flatten(gs: "GraphState"):
    node_keys = tuple(sorted(gs.node_feats.keys()))
    edge_keys = tuple(sorted(gs.edge_feats.keys()))
    graph_keys = tuple(sorted(gs.graph_feats.keys()))
    children = (
        gs.edge_index,
        tuple(gs.node_feats[k] for k in node_keys),
        tuple(gs.edge_feats[k] for k in edge_keys),
        tuple(gs.graph_feats[k] for k in graph_keys),
        gs.action_mask,
        gs.edge_valid,
    )
    aux = (gs.n_nodes, node_keys, edge_keys, graph_keys)
    return children, aux


def _graphstate_unflatten(aux, children):
    n_nodes, node_keys, edge_keys, graph_keys = aux
    edge_index, node_vals, edge_vals, graph_vals, action_mask, edge_valid = children
    return GraphState(
        n_nodes=n_nodes,
        edge_index=edge_index,
        node_feats=dict(zip(node_keys, node_vals)),
        edge_feats=dict(zip(edge_keys, edge_vals)),
        graph_feats=dict(zip(graph_keys, graph_vals)),
        action_mask=action_mask,
        edge_valid=edge_valid,
    )


import jax
jax.tree_util.register_pytree_node(GraphState, _graphstate_flatten, _graphstate_unflatten)


def pad_state_to(state: "GraphState", max_m: int) -> "GraphState":
    """Pads a GraphState's edge arrays up to `max_m` edges, so that states
    from graphs with differing edge counts (but the same node count) can be
    stacked into a single batch and processed with `jax.vmap`. Padding
    edges are routed to node 0 with all-zero features and marked invalid
    via `edge_valid`, so `_segment_agg` (sum/max aggregation) is unaffected
    (see `gnarl.model._process`)."""
    m = state.edge_index.shape[1]
    if m == max_m:
        return state
    pad = max_m - m
    if pad < 0:
        raise ValueError(f"max_m={max_m} smaller than existing m={m}")
    pad_index = jnp.zeros((2, pad), dtype=state.edge_index.dtype)
    new_edge_index = jnp.concatenate([state.edge_index, pad_index], axis=1)
    new_edge_feats = {
        k: jnp.concatenate([v, jnp.zeros((pad,) + v.shape[1:], dtype=v.dtype)], axis=0)
        for k, v in state.edge_feats.items()
    }
    new_edge_valid = jnp.concatenate([
        jnp.asarray(state.edge_valid, dtype=jnp.float32),
        jnp.zeros((pad,), dtype=jnp.float32),
    ])
    return GraphState(
        n_nodes=state.n_nodes,
        edge_index=new_edge_index,
        node_feats=state.node_feats,
        edge_feats=new_edge_feats,
        graph_feats=state.graph_feats,
        action_mask=state.action_mask,
        edge_valid=new_edge_valid,
    )


def stack_states(states: List["GraphState"]) -> "GraphState":
    """Stacks a list of same-n_nodes, same-max-edge-count GraphStates (after
    `pad_state_to`) into a single batched GraphState whose leaves gain a
    leading batch dimension -- suitable for `jax.vmap(forward, in_axes=(None, None, 0))`."""
    n_nodes = states[0].n_nodes
    node_keys = states[0].node_feats.keys()
    edge_keys = states[0].edge_feats.keys()
    graph_keys = states[0].graph_feats.keys()
    return GraphState(
        n_nodes=n_nodes,
        edge_index=jnp.stack([s.edge_index for s in states]),
        node_feats={k: jnp.stack([s.node_feats[k] for s in states]) for k in node_keys},
        edge_feats={k: jnp.stack([s.edge_feats[k] for s in states]) for k in edge_keys},
        graph_feats={k: jnp.stack([s.graph_feats[k] for s in states]) for k in graph_keys},
        action_mask=jnp.stack([s.action_mask for s in states]),
        edge_valid=jnp.stack([jnp.asarray(s.edge_valid, dtype=jnp.float32) for s in states]),
    )


class GNARLEnv:
    """Abstract base class for a graph-algorithm MDP M = <S, A, T, R, h>."""

    #: subclasses set these to lists of gnarl.model.FeatureSpec
    node_specs: Tuple = ()
    edge_specs: Tuple = ()
    graph_specs: Tuple = ()

    def __init__(self):
        self.G: Optional[nx.Graph] = None
        self.n: int = 0
        self.directed: bool = False
        self.edge_list: List[Tuple[int, int]] = []
        self.edge_idx: Dict[Tuple[int, int], int] = {}
        self.node_feat: Dict[str, np.ndarray] = {}
        self.edge_feat: Dict[str, np.ndarray] = {}
        self.graph_feat: Dict[str, np.ndarray] = {}
        self.t = 0
        self.h = 0  # horizon
        self.done = False

    # -- graph bookkeeping ---------------------------------------------
    def _build_edge_index(self):
        """Populate self.edge_list / self.edge_idx from self.G.

        For undirected graphs we materialise both (u, v) and (v, u) so that
        the sparse MPNN processor can pass messages in both directions
        (Appendix D.2), while `has_edge` below still respects the logical
        (undirected) adjacency for action-legality checks.
        """
        self.edge_list = []
        self.edge_idx = {}
        if self.directed:
            edges = list(self.G.edges())
        else:
            edges = []
            for u, v in self.G.edges():
                edges.append((u, v))
                edges.append((v, u))
        for e in edges:
            self.edge_idx[e] = len(self.edge_list)
            self.edge_list.append(e)

    def has_edge(self, u, v) -> bool:
        return self.G.has_edge(u, v)

    def neighbours(self, u) -> List[int]:
        if self.directed:
            return list(self.G.successors(u))
        return list(self.G.neighbors(u))

    def edge_weight(self, u, v) -> float:
        return float(self.G[u][v].get("weight", 1.0))

    # -- feature array helpers -------------------------------------------
    def _zeros_node(self, dim):
        return np.zeros((self.n, dim), dtype=np.float32)

    def _zeros_edge(self, dim):
        return np.zeros((len(self.edge_list), dim), dtype=np.float32)

    def state(self) -> GraphState:
        m = len(self.edge_list)
        if m > 0:
            ei = np.array(self.edge_list, dtype=np.int32).T  # (2, m) as (src,dst)
        else:
            ei = np.zeros((2, 0), dtype=np.int32)
        node_feats = {k: jnp.asarray(v) for k, v in self.node_feat.items()}
        edge_feats = {k: jnp.asarray(v) for k, v in self.edge_feat.items()}
        graph_feats = {k: jnp.asarray(v) for k, v in self.graph_feat.items()}
        mask = jnp.asarray(self.action_mask())
        return GraphState(
            n_nodes=self.n,
            edge_index=jnp.asarray(ei),
            node_feats=node_feats,
            edge_feats=edge_feats,
            graph_feats=graph_feats,
            action_mask=mask,
        )

    # -- interface to be implemented by subclasses ------------------------
    def reset(self, G: nx.Graph, **kwargs) -> GraphState:
        raise NotImplementedError

    def action_mask(self) -> np.ndarray:
        raise NotImplementedError

    def step(self, action: int):
        """Returns (GraphState, reward, done, info)."""
        raise NotImplementedError

    def expert_probs(self) -> Optional[np.ndarray]:
        """Expert action distribution pi*(.|s) for imitation learning, or
        None if unavailable in the current phase/state."""
        return None

    def is_correct(self) -> bool:
        raise NotImplementedError

    def objective(self) -> float:
        raise NotImplementedError
