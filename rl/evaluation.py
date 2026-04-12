from __future__ import annotations

import numpy as np
import torch

from rl.actor import sample_actor_step
from rl.common import filter_obs


@torch.no_grad()
def evaluate_actor(actor, env, obs_keys, n_episodes: int, max_ep_len: int):
    successes = 0
    horizons = []
    returns = []
    for _ in range(n_episodes):
        obs = filter_obs(env.reset(), obs_keys)
        rnn_state = None
        total_reward = 0.0
        for t in range(max_ep_len):
            action, rnn_state = sample_actor_step(actor, obs, rnn_state=rnn_state, deterministic=True)
            next_obs, _, done, _ = env.step(action)
            success = env.is_success()["task"]
            reward = float(success)
            total_reward += reward
            obs = filter_obs(next_obs, obs_keys)
            if done or success:
                successes += int(success)
                horizons.append(t + 1)
                returns.append(total_reward)
                break
        else:
            horizons.append(max_ep_len)
            returns.append(total_reward)
    return {
        "Success_Rate": successes / n_episodes,
        "Return": float(np.mean(returns)),
        "Horizon": float(np.mean(horizons)),
        "Num_Success": float(successes),
    }

