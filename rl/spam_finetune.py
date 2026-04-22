#!/usr/bin/env python3
"""
spam_rl_finetune.py — run RL finetuning for all (task, method) combos.

Finds BC checkpoints automatically from the training results folder,
launches train_rl.py for each combo, and skips combos whose final
eval JSON already exists.

All evals land in:
    <out-dir>/rl_eval/step_{N:06d}/{task}_{method}.json

The "final result" for a run of N total steps is simply step_{N:06d}/.
If you also pass --eval-interval, intermediate steps produce their own
subfolders — each one a complete usable result.

Usage:
    # full grid: 3 tasks × 2 difficulties × 5 methods
    python spam_rl_finetune.py coffee_d1 coffee_d2 square_d1 square_d2 mug_d1 mug_d2

    # specific methods only
    python spam_rl_finetune.py coffee_d1 coffee_d2 --methods sac_fd ppo

    # step-count ablation: run one combo with evals every 2k
    python spam_rl_finetune.py coffee_d1 --methods sac_fd --total-steps 10000 --eval-interval 2000
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT  = Path(os.environ.get("PROJECT_ROOT", SCRIPT_DIR.parent))
PYTHON_BIN    = os.environ.get("PYTHON_BIN",
                    str(Path.home() / "miniconda3/envs/mimicgen/bin/python"))

TRAINING_ROOT = SCRIPT_DIR.parent.parent / "mimicgen/training_results/core"
TRAIN_SCRIPT  = SCRIPT_DIR.parent / "train_rl.py"

ALL_METHODS   = ["sac_bc_init", "sac_dapg", "sac_fd", "ppo", "ppo_dapg"]

# ── hardcoded training hyperparameters ────────────────────────────────────────
# Tune these here rather than passing dozens of flags on the command line.

HPARAMS = {
    "seed":                   101,
    "batch_size":             16,
    "burnin_len":             5,
    "learn_len":              5,
    "online_buffer_capacity": 5000,
    "demo_buffer_capacity":   4000,
    "demo_batch_fraction":    0.5,
    "actor_lr":               3e-5,
    "critic_lr":              3e-4,
    "alpha_lr":               3e-4,
    "value_lr":               3e-4,
    "gamma":                  0.99,
    "tau":                    0.005,
    "update_every":           10,
    "warmup_steps":           1000,
    "max_ep_len":             400,
    "rollout_batch_steps":    1024,
    "ppo_epochs":             4,
    "ppo_clip_coef":          0.2,
    "value_coef":             0.5,
    "entropy_coef":           0.0,
    "gae_lambda":             0.95,
    "actor_bc_weight":        0.5,
    "actor_bc_decay":         0.99995,
    "eval_rollouts":          50,
    "eval_horizon":           400,
}

# ── helpers ───────────────────────────────────────────────────────────────────

def find_bc_checkpoint(task: str) -> Path | None:
    """Find model_epoch_2000*.pth for a given task under TRAINING_ROOT."""
    matches = sorted(TRAINING_ROOT.glob(
        f"{task}/low_dim/trained_models/*/*/models/model_epoch_2000*.pth"
    ))
    return matches[0] if matches else None


def final_json_exists(eval_root: Path, task: str, method: str, total_steps: int) -> bool:
    p = eval_root / f"step_{total_steps:06d}" / f"{task}_{method}.json"
    return p.exists()


def build_train_cmd(
    task: str,
    method: str,
    ckpt_path: Path,
    out_dir: Path,
    eval_root: Path,
    total_steps: int,
    eval_interval: int,
    hparams: dict,
) -> list[str]:
    checkpoint_interval = eval_interval if eval_interval > 0 else total_steps
    cmd = [
        PYTHON_BIN, str(TRAIN_SCRIPT),
        "--method",               method,
        "--ckpt-path",            str(ckpt_path),
        "--total-steps",          str(total_steps),
        "--output-dir",           str(out_dir / f"{task}_{method}"),
        "--eval-output-dir",      str(eval_root),
        "--task-name",            task,
        "--eval-interval",        str(eval_interval),
        "--checkpoint-interval",  str(checkpoint_interval),
    ]
    # append all hardcoded hparams
    flag_map = {k.replace("_", "-"): v for k, v in hparams.items()}
    for flag, val in flag_map.items():
        if isinstance(val, bool):
            if val:
                cmd.append(f"--{flag}")
        else:
            cmd += [f"--{flag}", str(val)]
    return cmd

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tasks", nargs="+", metavar="TASK",
                        help="Tasks to finetune (e.g. coffee_d1 coffee_d2).")
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS,
                        choices=ALL_METHODS, metavar="METHOD",
                        help=f"Methods to run. Default: all ({', '.join(ALL_METHODS)}).")
    parser.add_argument("--total-steps",   type=int, default=20_000,
                        help="Training steps per run.")
    parser.add_argument("--eval-interval", type=int, default=0,
                        help="Eval every N steps. 0 = final step only.")
    parser.add_argument("--out-dir",       type=Path,
                        default=SCRIPT_DIR.parent.parent / "tmp_runs" / "rl_finetune",
                        help="Root output directory.")
    args = parser.parse_args()

    eval_root = args.out_dir / "rl_eval"
    hparams   = dict(HPARAMS)

    # build work list
    combos = [(task, method) for task in args.tasks for method in args.methods]

    print(f"[INFO] {len(combos)} combo(s)  |  steps={args.total_steps}  eval_interval={args.eval_interval}")
    print(f"[INFO] eval root: {eval_root}")
    print("─" * 70)

    done = skipped = failed = 0

    for task, method in combos:
        label = f"{task}/{method}"

        # skip if final eval JSON already exists
        if final_json_exists(eval_root, task, method, args.total_steps):
            print(f"[SKIP] {label}  (final eval JSON exists)")
            skipped += 1
            continue

        # find BC checkpoint
        ckpt = find_bc_checkpoint(task)
        if ckpt is None:
            print(f"[ERR]  {label}  — no BC checkpoint found under {TRAINING_ROOT / task}")
            failed += 1
            continue

        cmd = build_train_cmd(
            task=task,
            method=method,
            ckpt_path=ckpt,
            out_dir=args.out_dir,
            eval_root=eval_root,
            total_steps=args.total_steps,
            eval_interval=args.eval_interval,
            hparams=hparams,
        )

        print(f"[RUN]  {label}  (ckpt: {ckpt.name})")

        result = subprocess.run(cmd, check=False)

        if result.returncode != 0:
            print(f"[ERR]  {label}  exited with code {result.returncode}")
            failed += 1
            continue

        # verify the final JSON was actually written
        if final_json_exists(eval_root, task, method, args.total_steps):
            print(f"[DONE] {label}")
            done += 1
        else:
            print(f"[WARN] {label}  — training finished but final eval JSON missing")
            failed += 1

    print("─" * 70)
    print(f"[INFO] Done: {done}  |  Skipped: {skipped}  |  Failed: {failed}")
    print(f"[INFO] Results in: {eval_root}")


if __name__ == "__main__":
    main()