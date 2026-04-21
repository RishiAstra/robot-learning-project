from __future__ import annotations

import os

import imageio
import numpy as np
import torch
from tqdm import tqdm

from rl.actor import sample_actor_step
from rl.common import filter_obs


@torch.no_grad()
def evaluate_actor(actor, env, obs_keys, n_episodes: int, max_ep_len: int, video_dir: str | None = None):
    if video_dir is not None:
        os.makedirs(video_dir, exist_ok=True)
    successes = 0
    horizons = []
    returns = []
    for ep in tqdm(range(n_episodes), desc="eval", unit="ep"):
        obs = filter_obs(env.reset(), obs_keys)
        rnn_state = None
        total_reward = 0.0
        frames = []
        for t in range(max_ep_len):
            if video_dir is not None:
                frames.append(env.render(mode="rgb_array", height=512, width=512))
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
                if video_dir is not None:
                    label = "success" if success else "fail"
                    imageio.mimwrite(os.path.join(video_dir, f"ep{ep:03d}_{label}.mp4"), frames, fps=20)
                break
        else:
            horizons.append(max_ep_len)
            returns.append(total_reward)
            if video_dir is not None:
                imageio.mimwrite(os.path.join(video_dir, f"ep{ep:03d}_fail.mp4"), frames, fps=20)
    return {
        "Success_Rate": successes / n_episodes,
        "Return": float(np.mean(returns)),
        "Horizon": float(np.mean(horizons)),
        "Num_Success": float(successes),
    }
