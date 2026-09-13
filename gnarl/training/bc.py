"""Behavioural Cloning trainer (Section 3.2 Eq. 3, Section 4.3).

Trains the actor to match an expert action distribution pi_expert(.|s) by
minimising KL(pi || pi_expert) (equivalently, since pi_expert is a discrete
distribution over a masked action set, this reduces to a cross-entropy-like
objective weighted by the expert probabilities).
"""
from __future__ import annotations

import time
from typing import Callable, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..model import GNARLConfig, forward, init_params


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


def _kl_loss(params, cfg: GNARLConfig, state, expert_probs):
    out = forward(params, cfg, state)
    log_probs = jax.nn.log_softmax(out["logits"])
    expert_probs = jnp.asarray(expert_probs)
    expert_probs = expert_probs / jnp.clip(expert_probs.sum(), 1e-8)
    # cross-entropy between expert distribution and predicted distribution;
    # equivalent (up to the expert's own constant entropy) to Eq. 3's KL term.
    return -jnp.sum(expert_probs * log_probs)


def train_bc(cfg: GNARLConfig, dataset: List[Trajectory], key: jax.Array,
             epochs: int = 20, lr: float = 1e-3, batch_size: int = 16,
             params=None, verbose: bool = True):
    if params is None:
        params = init_params(key, cfg)
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    loss_and_grad = jax.jit(jax.value_and_grad(_kl_loss), static_argnums=(1,))

    n = len(dataset)
    history = []
    idxs = np.arange(n)
    for epoch in range(epochs):
        np.random.shuffle(idxs)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            batch_idx = idxs[start:start + batch_size]
            for i in batch_idx:
                traj = dataset[i]
                loss, grads = loss_and_grad(params, cfg, traj.state, traj.expert_probs)
                updates, opt_state = opt.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                epoch_loss += float(loss)
        epoch_loss /= max(n, 1)
        history.append(epoch_loss)
        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(f"  [BC] epoch {epoch+1}/{epochs}  loss={epoch_loss:.4f}")
    return params, history
