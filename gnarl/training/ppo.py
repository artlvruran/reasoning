"""A minimal PPO trainer (Schulman et al., 2017) with action masking,
supporting the RL training mode of GNARL (Section 3.2 Eq. 2, Section 4.3).

Because GNARL operates on variable-sized graphs (no fixed action-space
dimensionality across episodes), this implementation processes one
state/transition at a time rather than using large vectorised minibatches --
adequate for the small-scale demonstrations in this notebook, at the cost of
some training speed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..model import GNARLConfig, forward, init_params


@dataclass
class Transition:
    state: object
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool


def collect_episode(env, params, cfg: GNARLConfig, key, greedy: bool = False) -> List[Transition]:
    transitions = []
    done = False
    while not done:
        state = env.state()
        out = forward(params, cfg, state)
        probs = np.asarray(out["probs"])
        probs = probs / probs.sum()
        if greedy:
            a = int(np.argmax(probs))
        else:
            a = int(np.random.choice(len(probs), p=probs))
        logp = float(np.log(probs[a] + 1e-12))
        value = float(out["value"]) if "value" in out else 0.0
        _, reward, done, _ = env.step(a)
        transitions.append(Transition(state, a, logp, value, reward, done))
    return transitions


def compute_gae(transitions: List[Transition], gamma: float = 1.0, lam: float = 0.95):
    T = len(transitions)
    advs = np.zeros(T)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = transitions[t + 1].value if t + 1 < T else 0.0
        delta = transitions[t].reward + gamma * next_value - transitions[t].value
        last_gae = delta + gamma * lam * last_gae
        advs[t] = last_gae
    returns = advs + np.array([tr.value for tr in transitions])
    return advs, returns


def _ppo_loss(params, cfg, state, action, old_log_prob, advantage, ret, clip_eps, vf_coef, ent_coef):
    out = forward(params, cfg, state)
    log_probs = jax.nn.log_softmax(out["logits"])
    new_log_prob = log_probs[action]
    ratio = jnp.exp(new_log_prob - old_log_prob)
    clipped = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    policy_loss = -jnp.minimum(ratio * advantage, clipped * advantage)
    value = out.get("value", 0.0)
    value_loss = (ret - value) ** 2
    probs = out["probs"]
    entropy = -jnp.sum(jnp.where(probs > 0, probs * jnp.log(probs + 1e-12), 0.0))
    return policy_loss + vf_coef * value_loss - ent_coef * entropy


def train_ppo(cfg: GNARLConfig, env_factory, graph_sampler: Callable, key: jax.Array,
              n_updates: int = 50, episodes_per_update: int = 8, ppo_epochs: int = 4,
              lr: float = 5e-4, clip_eps: float = 0.2, gamma: float = 1.0, lam: float = 0.95,
              vf_coef: float = 0.5, ent_coef: float = 0.01, params=None, verbose: bool = True,
              reset_kwargs_fn: Optional[Callable] = None):
    if params is None:
        params = init_params(key, cfg)
    opt = optax.adam(lr)
    opt_state = opt.init(params)
    loss_and_grad = jax.jit(
        jax.value_and_grad(_ppo_loss),
        static_argnums=(1,),
    )

    history = {"mean_reward": [], "mean_return": []}
    for update in range(n_updates):
        all_trans: List[Transition] = []
        ep_rewards = []
        for _ in range(episodes_per_update):
            G = graph_sampler()
            env = env_factory()
            kwargs = reset_kwargs_fn(G) if reset_kwargs_fn is not None else {}
            env.reset(G, **kwargs)
            trans = collect_episode(env, params, cfg, key)
            advs, returns = compute_gae(trans, gamma, lam)
            for tr, adv, ret in zip(trans, advs, returns):
                tr.advantage = adv
                tr.ret = ret
            all_trans.extend(trans)
            ep_rewards.append(sum(t.reward for t in trans))

        advs_arr = np.array([t.advantage for t in all_trans])
        advs_arr = (advs_arr - advs_arr.mean()) / (advs_arr.std() + 1e-8)
        for i, t in enumerate(all_trans):
            t.advantage = advs_arr[i]

        for _ in range(ppo_epochs):
            order = np.random.permutation(len(all_trans))
            for i in order:
                t = all_trans[i]
                loss, grads = loss_and_grad(
                    params, cfg, t.state, t.action, t.log_prob, t.advantage, t.ret,
                    clip_eps, vf_coef, ent_coef,
                )
                updates, opt_state = opt.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)

        mean_reward = float(np.mean(ep_rewards))
        history["mean_reward"].append(mean_reward)
        if verbose and (update % max(1, n_updates // 10) == 0 or update == n_updates - 1):
            print(f"  [PPO] update {update+1}/{n_updates}  mean_episode_reward={mean_reward:.4f}")
    return params, history
