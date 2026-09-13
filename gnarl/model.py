"""Core GNARL architecture: Encode -> Process -> Act.

Implements (see paper Section 4.2 and Appendix B):
  - Encoder: per-feature linear encoders, aggregated by location (node/edge/graph).
  - Processor: an MPNN that performs L rounds of message passing *within* a
    single MDP step (the cross-timestep recurrent state h^(t-1) is
    deliberately NOT carried over, following Bohde et al. 2024's finding that
    this improves algorithmic (Markov) alignment -- see Sec 4.2).
  - Actor: proto-action mechanism (Darvariu et al. 2021b) -- a learned linear
    map from the graph embedding produces a "proto-action" vector that is
    compared to every node embedding via negative Euclidean distance, then
    passed through a temperature-scaled softmax to obtain the action
    distribution. Invalid actions are masked to -inf before the softmax so
    that GNARL is correct-by-construction (Section 4.4).
  - Critic: an MLP over the graph embedding producing a scalar state-value,
    used only for RL (PPO) training (Section 4.3).

This is implemented in plain JAX (mirroring the paper's own JAX-based
reproducibility stack for the NAR baselines) with a minimal hand-rolled
parameter pytree, so it has no dependency beyond jax/optax and works on
CPU-only, disk constrained environments.
"""
from __future__ import annotations

import functools
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np


# --------------------------------------------------------------------------
# Feature specification
# --------------------------------------------------------------------------

class FeatureSpec(NamedTuple):
    """Describes one named feature used by an environment's MDP.

    location: one of "node", "edge", "graph" (Section 4.1 / Appendix D).
    dim: dimensionality of the raw feature vector fed to its encoder, e.g.
        scalar/mask features are 1-dim, a "phase" categorical feature with
        P values is P-dim (one-hot), etc.
    """

    name: str
    location: str  # "node" | "edge" | "graph"
    dim: int


# --------------------------------------------------------------------------
# Minimal parameter utilities (avoids a flax/haiku dependency)
# --------------------------------------------------------------------------

def _init_linear(key, in_dim: int, out_dim: int, scale: float = None):
    scale = scale if scale is not None else 1.0 / np.sqrt(max(in_dim, 1))
    wkey, bkey = jax.random.split(key)
    W = jax.random.normal(wkey, (in_dim, out_dim)) * scale
    b = jnp.zeros((out_dim,))
    return {"W": W, "b": b}


def _linear(p, x):
    return x @ p["W"] + p["b"]


def _init_mlp(key, dims: Sequence[int]):
    layers = []
    keys = jax.random.split(key, len(dims) - 1)
    for i, k in enumerate(keys):
        layers.append(_init_linear(k, dims[i], dims[i + 1]))
    return layers


def _mlp(layers, x, activation=jax.nn.relu):
    for i, p in enumerate(layers):
        x = _linear(p, x)
        if i < len(layers) - 1:
            x = activation(x)
    return x


# --------------------------------------------------------------------------
# GNARL model
# --------------------------------------------------------------------------

class GNARLConfig(NamedTuple):
    node_specs: Tuple[FeatureSpec, ...]
    edge_specs: Tuple[FeatureSpec, ...]
    graph_specs: Tuple[FeatureSpec, ...]
    embed_dim: int = 64
    num_mp_layers: int = 2
    mlp_layers: int = 2  # number of layers in each MLP (message/update funcs)
    pooling: str = "max"  # "max" | "mean"
    aggregation: str = "max"  # message aggregation over neighbours: "max"|"sum"
    use_critic: bool = True


def init_params(key: jax.Array, cfg: GNARLConfig):
    f = cfg.embed_dim
    keys = jax.random.split(key, 8)
    params = {}

    # --- Encoders: one linear map per named feature -> f dims -------------
    enc_keys = jax.random.split(keys[0], len(cfg.node_specs) + len(cfg.edge_specs) + len(cfg.graph_specs))
    idx = 0
    enc_node, enc_edge, enc_graph = {}, {}, {}
    for spec in cfg.node_specs:
        enc_node[spec.name] = _init_linear(enc_keys[idx], spec.dim, f)
        idx += 1
    for spec in cfg.edge_specs:
        enc_edge[spec.name] = _init_linear(enc_keys[idx], spec.dim, f)
        idx += 1
    for spec in cfg.graph_specs:
        enc_graph[spec.name] = _init_linear(enc_keys[idx], spec.dim, f)
        idx += 1
    params["enc_node"] = enc_node
    params["enc_edge"] = enc_edge
    params["enc_graph"] = enc_graph

    # --- Processor: L message-passing layers -------------------------------
    mp_keys = jax.random.split(keys[1], cfg.num_mp_layers)
    mp_layers = []
    for k in mp_keys:
        mk, uk = jax.random.split(k)
        # message fn M: [h_v, h_u, z_uv, z_v_graph] (3f) -> f
        msg_dims = [3 * f] + [f] * (cfg.mlp_layers - 1) + [f]
        # update fn U: [h_v, z_v, m_v] (3f) -> f
        upd_dims = [3 * f] + [f] * (cfg.mlp_layers - 1) + [f]
        mp_layers.append({
            "M": _init_mlp(mk, msg_dims),
            "U": _init_mlp(uk, upd_dims),
        })
    params["processor"] = mp_layers

    # --- Actor: proto-action linear map + learned temperature -------------
    params["actor_proto"] = _init_linear(keys[2], f, f)
    params["actor_log_temp"] = jnp.array(0.0)  # T = softplus(log_temp) + eps

    # --- Critic: MLP over graph embedding ----------------------------------
    if cfg.use_critic:
        params["critic"] = _init_mlp(keys[3], [f, f, 1])

    return params


def _encode(params, cfg: GNARLConfig, node_feats, edge_feats, graph_feats, n_nodes, n_edges):
    f = cfg.embed_dim
    z_node = jnp.zeros((n_nodes, f))
    for spec in cfg.node_specs:
        x = node_feats[spec.name]
        z_node = z_node + _linear(params["enc_node"][spec.name], x)

    if n_edges > 0:
        z_edge = jnp.zeros((n_edges, f))
        for spec in cfg.edge_specs:
            x = edge_feats[spec.name]
            z_edge = z_edge + _linear(params["enc_edge"][spec.name], x)
    else:
        z_edge = jnp.zeros((0, f))

    z_graph = jnp.zeros((f,))
    for spec in cfg.graph_specs:
        x = graph_feats[spec.name]
        z_graph = z_graph + _linear(params["enc_graph"][spec.name], x)

    return z_node, z_edge, z_graph


def _segment_agg(messages, dst, n_nodes, mode):
    if mode == "sum":
        return jax.ops.segment_sum(messages, dst, num_segments=n_nodes)
    elif mode == "max":
        neg_inf = jnp.full((n_nodes, messages.shape[-1]), -1e9)
        agg = jax.ops.segment_max(messages, dst, num_segments=n_nodes)
        # segment_max returns -inf-like fill for empty segments (isolated nodes);
        # replace with zeros to avoid propagating -inf.
        agg = jnp.where(jnp.isfinite(agg), agg, jnp.zeros_like(agg))
        return agg
    else:
        raise ValueError(mode)


def _process(params, cfg: GNARLConfig, z_node, z_edge, edge_index, n_nodes):
    """L rounds of message passing purely within this MDP step (Sec 4.2)."""
    h = z_node
    if edge_index.shape[1] > 0:
        src, dst = edge_index[0], edge_index[1]
    else:
        src = dst = jnp.zeros((0,), dtype=jnp.int32)

    for layer in params["processor"]:
        if edge_index.shape[1] > 0:
            h_src = h[src]
            h_dst = h[dst]
            msg_in = jnp.concatenate([h_dst, h_src, z_edge], axis=-1)
            messages = _mlp(layer["M"], msg_in)
            agg = _segment_agg(messages, dst, n_nodes, cfg.aggregation)
        else:
            agg = jnp.zeros_like(h)
        upd_in = jnp.concatenate([h, z_node, agg], axis=-1)
        h = _mlp(layer["U"], upd_in)
    return h


def _pool(h, mode):
    if mode == "max":
        return jnp.max(h, axis=0)
    elif mode == "mean":
        return jnp.mean(h, axis=0)
    else:
        raise ValueError(mode)


def forward(params, cfg: GNARLConfig, graph_state) -> Dict[str, jnp.ndarray]:
    """Runs Encode -> Process -> Act for a single graph/state.

    graph_state must expose: n_nodes, edge_index (2,m), node_feats (dict),
    edge_feats (dict), graph_feats (dict), action_mask (n,) bool.
    Returns dict with 'logits', 'probs', 'value' (if critic enabled),
    'node_embed', 'graph_embed'.
    """
    n = graph_state.n_nodes
    m = graph_state.edge_index.shape[1]
    z_node, z_edge, z_graph = _encode(
        params, cfg, graph_state.node_feats, graph_state.edge_feats,
        graph_state.graph_feats, n, m,
    )
    h = _process(params, cfg, z_node, z_edge, graph_state.edge_index, n)
    hbar = _pool(h, cfg.pooling)

    proto = _linear(params["actor_proto"], hbar)  # (f,)
    diff = h - proto[None, :]
    sim = -jnp.sum(diff * diff, axis=-1)  # (n,)
    temp = jax.nn.softplus(params["actor_log_temp"]) + 1e-3
    logits = sim / temp
    mask = graph_state.action_mask
    masked_logits = jnp.where(mask, logits, -1e9)
    probs = jax.nn.softmax(masked_logits)

    out = {
        "logits": masked_logits,
        "probs": probs,
        "node_embed": h,
        "graph_embed": hbar,
    }
    if cfg.use_critic and "critic" in params:
        out["value"] = _mlp(params["critic"], hbar)[0]
    return out


__all__ = [
    "FeatureSpec", "GNARLConfig", "init_params", "forward",
]
