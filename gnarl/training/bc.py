"""Behavioural Cloning trainer (Section 3.2 Eq. 3, Section 4.3).

Trains the actor to match an expert action distribution pi_expert(.|s) by
minimising KL(pi || pi_expert) (equivalently, since pi_expert is a discrete
distribution over a masked action set, this reduces to a cross-entropy-like
objective weighted by the expert probabilities).

Training is done with **real vectorised minibatches**: examples are grouped
by node count (so every example in a group shares an action-space size),
each group's edge lists are zero-padded to a common length within the
group (`gnarl.envs.base.pad_state_to`, with the padding edges excluded from
message aggregation via an `edge_valid` mask), and a whole minibatch is
processed in a single `jax.vmap`'d forward pass (`gnarl.model.forward_batched`).
This is far faster, and yields much less noisy gradients, than dispatching
one small forward/backward pass per example. This matters in practice: with
only per-example SGD and a handful of epochs, GNARL under-fits badly (loss
plateaus well above the expert's own entropy, and greedy-rollout accuracy on
held-out graphs can be close to 0%) simply because too few effective
gradient updates have been taken -- real minibatching lets many more epochs
run in the same wall-clock budget.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..envs.base import GraphState, pad_state_to, stack_states
from ..model import GNARLConfig, forward, forward_batched, init_params


class Trajectory:
    """One (state, expert_probs) pair collected by rolling out an expert
    policy on an environment instance."""

    __slots__ = ("state", "expert_probs")

    def __init__(self, state, expert_probs):
        self.state = state
        self.expert_probs = expert_probs


def collect_bc_dataset(env_factory, graphs, expert_setup: Optional[Callable] = None,
                        reset_kwargs_fn: Optional[Callable] = None) -> List[Trajectory]:
    """Rolls out the *expert* policy on each graph, recording every state
    visited along with the expert's action distribution at that state (skips
    states where the expert has no valid recommendation, e.g. once a BFS/DFS
    tree has already fully formed before the fixed horizon is exhausted)."""
    data = []
    for G in graphs:
        env = env_factory()
        kwargs = reset_kwargs_fn(G) if reset_kwargs_fn is not None else {}
        env.reset(G, **kwargs)
        if expert_setup is not None:
            expert_setup(env, G)
        done = False
        while not done:
            ep = env.expert_probs()
            state = env.state()
            mask = np.asarray(env.action_mask())
            if ep is not None:
                data.append(Trajectory(state, np.asarray(ep)))
                a = int(np.random.choice(len(ep), p=np.asarray(ep) / np.sum(ep)))
            else:
                valid = np.where(mask)[0]
                a = int(np.random.choice(valid))
            _, _, done, _ = env.step(a)
    return data


def _kl_loss_single(params, cfg: GNARLConfig, state, expert_probs):
    out = forward(params, cfg, state)
    log_probs = jax.nn.log_softmax(out["logits"])
    expert_probs = jnp.asarray(expert_probs)
    expert_probs = expert_probs / jnp.clip(expert_probs.sum(), 1e-8)
    return -jnp.sum(expert_probs * log_probs)


def _kl_loss_batched(params, cfg: GNARLConfig, batched_state, expert_probs_batch):
    out = forward_batched(params, cfg, batched_state)
    log_probs = jax.nn.log_softmax(out["logits"], axis=-1)
    ep = expert_probs_batch / jnp.clip(expert_probs_batch.sum(axis=-1, keepdims=True), 1e-8)
    per_example = -jnp.sum(ep * log_probs, axis=-1)
    return jnp.mean(per_example)


def _make_minibatches(dataset: List[Trajectory], batch_size: int, rng: np.random.Generator):
    """Groups trajectories by n_nodes (so each group shares an action-space
    size), shuffles within each group, and yields padded/stacked minibatches
    of (batched_state, expert_probs_batch)."""
    by_n = defaultdict(list)
    for traj in dataset:
        by_n[traj.state.n_nodes].append(traj)

    batches = []
    for n_nodes, trajs in by_n.items():
        order = rng.permutation(len(trajs))
        for start in range(0, len(trajs), batch_size):
            idx = order[start:start + batch_size]
            group = [trajs[i] for i in idx]
            max_m = max(t.state.edge_index.shape[1] for t in group)
            padded = [pad_state_to(t.state, max_m) for t in group]
            batched_state = stack_states(padded)
            expert_probs_batch = jnp.stack([jnp.asarray(t.expert_probs, dtype=jnp.float32) for t in group])
            batches.append((batched_state, expert_probs_batch))
    return batches


def train_bc(cfg: GNARLConfig, dataset: List[Trajectory], key: jax.Array,
             epochs: int = 20, lr: float = 1e-3, batch_size: int = 16,
             params=None, verbose: bool = True, grad_clip: float = 5.0):
    if params is None:
        params = init_params(key, cfg)
    opt = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr))
    opt_state = opt.init(params)

    loss_and_grad = jax.jit(jax.value_and_grad(_kl_loss_batched), static_argnums=(1,))

    rng = np.random.default_rng(0)
    history = []
    n = len(dataset)
    for epoch in range(epochs):
        batches = _make_minibatches(dataset, batch_size, rng)
        random.shuffle(batches)
        epoch_loss = 0.0
        n_batches = 0
        for batched_state, expert_probs_batch in batches:
            loss, grads = loss_and_grad(params, cfg, batched_state, expert_probs_batch)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            epoch_loss += float(loss)
            n_batches += 1
        epoch_loss /= max(n_batches, 1)
        history.append(epoch_loss)
        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(f"  [BC] epoch {epoch+1}/{epochs}  loss={epoch_loss:.4f}  ({n_batches} minibatches, {n} examples)")
    return params, history
