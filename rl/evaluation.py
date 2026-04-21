from __future__ import annotations

import numpy as np

from robomimic.scripts.run_trained_agent import rollout


def evaluate_policy(policy, env, n_episodes: int, max_ep_len: int):
    if n_episodes <= 0:
        return {
            "Success_Rate": 0.0,
            "Return": 0.0,
            "Horizon": 0.0,
            "Num_Success": 0.0,
        }

    rollout_stats = []
    for _ in range(n_episodes):
        stats, _ = rollout(
            policy=policy,
            env=env,
            horizon=max_ep_len,
            render=False,
            video_writer=None,
            video_skip=5,
            return_obs=False,
            camera_names=["agentview"],
        )
        rollout_stats.append(stats)

    success_rates = [stats["Success_Rate"] for stats in rollout_stats]
    returns = [stats["Return"] for stats in rollout_stats]
    horizons = [stats["Horizon"] for stats in rollout_stats]
    return {
        "Success_Rate": float(np.mean(success_rates)),
        "Return": float(np.mean(returns)),
        "Horizon": float(np.mean(horizons)),
        "Num_Success": float(np.sum(success_rates)),
    }
