from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import tqdm

# Avoid robosuite / numba cache issues in moved workspaces unless the caller
# explicitly overrides this environment variable beforehand.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.scripts.run_trained_agent import rollout


def resolve_agents(agent_args, agent_globs):
    paths = []
    for agent in agent_args:
        paths.append(Path(agent).expanduser().resolve())
    for pattern in agent_globs:
        matches = sorted(Path().glob(pattern))
        if not matches:
            raise FileNotFoundError(f"pattern matched no checkpoints: {pattern}")
        paths.extend(match.resolve() for match in matches)

    # Preserve order while deduplicating.
    deduped = []
    seen = set()
    for path in paths:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    if not deduped:
        raise ValueError("no checkpoints provided")
    return deduped


def evaluate_checkpoint(agent_path, n_rollouts, horizon, env_name, seed):
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=str(agent_path),
        device=device,
        verbose=False,
    )

    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        env_name=env_name,
        render=False,
        render_offscreen=False,
        verbose=False,
    )

    if horizon is None:
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        rollout_horizon = config.experiment.rollout.horizon
    else:
        rollout_horizon = horizon

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    rollout_stats = []
    for _ in tqdm.tqdm(range(n_rollouts), desc=f"Evaluating {agent_path}"):
        stats, _ = rollout(
            policy=policy,
            env=env,
            horizon=rollout_horizon,
            render=False,
            video_writer=None,
            video_skip=5,
            return_obs=False,
            camera_names=["agentview"],
        )
        rollout_stats.append(stats)

    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    avg_rollout_stats = {k: float(np.mean(rollout_stats[k])) for k in rollout_stats}
    avg_rollout_stats["Num_Success"] = float(np.sum(rollout_stats["Success_Rate"]))
    avg_rollout_stats["n_rollouts"] = n_rollouts
    avg_rollout_stats["agent"] = str(agent_path)
    avg_rollout_stats["env_name"] = env.name
    avg_rollout_stats["horizon_used"] = rollout_horizon
    return avg_rollout_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        action="append",
        default=[],
        help="checkpoint path to evaluate; pass multiple times for multiple agents",
    )
    parser.add_argument(
        "--agent-glob",
        action="append",
        default=[],
        help="glob pattern for checkpoint paths, relative to the current working directory",
    )
    parser.add_argument("--n-rollouts", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="optional path to write the evaluation summary as JSON",
    )
    args = parser.parse_args()

    agents = resolve_agents(args.agent, args.agent_glob)
    results = []
    for agent_path in agents:
        result = evaluate_checkpoint(
            agent_path=agent_path,
            n_rollouts=args.n_rollouts,
            horizon=args.horizon,
            env_name=args.env,
            seed=args.seed,
        )
        results.append(result)
        print(json.dumps(result, indent=4))

    if args.output_json is not None:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=4) + "\n")
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
