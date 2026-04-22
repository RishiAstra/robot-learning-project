#!/usr/bin/env python3
"""
record_videos.py — run rollouts on a .pth checkpoint and save labelled videos.

Output structure (next to the .pth):
    videos/<pth_stem>/success/success_001.mp4
    videos/<pth_stem>/success/success_002.mp4
    videos/<pth_stem>/fail/fail_001.mp4
    ...

Usage:
    python record_videos.py path/to/checkpoint.pth
    python record_videos.py path/to/checkpoint.pth --n-rollouts 20 --horizon 400
    python record_videos.py path/to/checkpoint.pth --camera frontview --fps 20
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import imageio
import numpy as np
import torch
import tqdm

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import robosuite
if not hasattr(robosuite, "__version__"):
    robosuite.__version__ = "1.4.0"  # patch missing attr

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.scripts.run_trained_agent import rollout


def record_videos(
    ckpt_path: Path,
    n_rollouts: int,
    horizon: int | None,
    camera: str,
    fps: int,
    video_skip: int,
    seed: int | None,
):
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=str(ckpt_path),
        device=device,
        verbose=False,
    )

    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        render=False,
        render_offscreen=True,
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

    # output dirs: videos/<stem>/success/ and videos/<stem>/fail/
    stem       = ckpt_path.stem
    video_root = ckpt_path.parent / "videos" / stem
    success_dir = video_root / "success"
    fail_dir    = video_root / "fail"
    success_dir.mkdir(parents=True, exist_ok=True)
    fail_dir.mkdir(parents=True, exist_ok=True)

    n_success = n_fail = 0

    for i in tqdm.tqdm(range(n_rollouts), desc=f"Recording {stem}"):
        # write to a temp writer, collect frames, then save to the right folder
        frames: list[np.ndarray] = []

        class _FrameCollector:
            """Mimics imageio writer interface, just collects frames."""
            def append_data(self, frame):
                frames.append(frame)

        stats, _ = rollout(
            policy=policy,
            env=env,
            horizon=rollout_horizon,
            render=False,
            video_writer=_FrameCollector(),
            video_skip=video_skip,
            return_obs=False,
            camera_names=[camera],
        )

        success = bool(stats.get("Success_Rate", 0))

        if success:
            n_success += 1
            out_path = success_dir / f"success_{n_success:03d}.mp4"
        else:
            n_fail += 1
            out_path = fail_dir / f"fail_{n_fail:03d}.mp4"

        writer = imageio.get_writer(str(out_path), fps=fps)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

    print(f"\nDone. {n_success} success, {n_fail} fail")
    print(f"Videos saved to: {video_root}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ckpt", help="Path to .pth checkpoint.")
    parser.add_argument("--n-rollouts", type=int, default=10)
    parser.add_argument("--horizon",    type=int, default=None,
                        help="Max steps per rollout. Defaults to value in checkpoint.")
    parser.add_argument("--camera",     default="agentview",
                        help="Camera name to render (default: agentview).")
    parser.add_argument("--fps",        type=int, default=20,
                        help="Video FPS (default: 20).")
    parser.add_argument("--video-skip", type=int, default=1,
                        help="Render every N steps (default: 1 = every step).")
    parser.add_argument("--seed",       type=int, default=101)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    record_videos(
        ckpt_path=ckpt_path,
        n_rollouts=args.n_rollouts,
        horizon=args.horizon,
        camera=args.camera,
        fps=args.fps,
        video_skip=args.video_skip,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()