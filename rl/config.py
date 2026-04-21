from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


SCRIPT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_CKPT = REPO_ROOT / "mimicgen/datasets/core_training_results/bc_rnn_low_dim_ds_stack_D0_seed_101/20260312210424/models/model_epoch_100_demo_success_1.0.pth"


@dataclass
class Args:
    method: str
    ckpt_path: str
    total_steps: int
    seed: int
    batch_size: int
    burnin_len: int
    learn_len: int
    online_buffer_capacity: int
    demo_buffer_capacity: int
    demo_max_demos: Optional[int]
    demo_batch_fraction: float
    actor_lr: float
    critic_lr: float
    alpha_lr: float
    value_lr: float
    gamma: float
    tau: float
    update_every: int
    warmup_steps: int
    eval_interval: int
    eval_episodes: int
    max_ep_len: int
    rollout_batch_steps: int
    ppo_epochs: int
    ppo_clip_coef: float
    value_coef: float
    entropy_coef: float
    gae_lambda: float
    actor_bc_weight: float
    actor_bc_decay: float
    output_dir: str


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["sac_bc_init", "sac_dapg", "sac_fd", "ppo", "ppo_dapg"], required=True)
    parser.add_argument("--ckpt-path", default=str(DEFAULT_CKPT))
    parser.add_argument("--total-steps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--burnin-len", type=int, default=5)
    parser.add_argument("--learn-len", type=int, default=5)
    parser.add_argument("--online-buffer-capacity", type=int, default=5000)
    parser.add_argument("--demo-buffer-capacity", type=int, default=4000)
    parser.add_argument("--demo-max-demos", type=int, default=None)
    parser.add_argument("--demo-batch-fraction", type=float, default=0.5)
    parser.add_argument("--actor-lr", type=float, default=3e-5)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--value-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--update-every", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=2000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--max-ep-len", type=int, default=400)
    parser.add_argument("--rollout-batch-steps", type=int, default=1024)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--actor-bc-weight", type=float, default=5.0)
    parser.add_argument("--actor-bc-decay", type=float, default=0.999)
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "rl_runs"))
    ns = parser.parse_args()
    return Args(
        method=ns.method,
        ckpt_path=str(Path(ns.ckpt_path).expanduser().resolve()),
        total_steps=ns.total_steps,
        seed=ns.seed,
        batch_size=ns.batch_size,
        burnin_len=ns.burnin_len,
        learn_len=ns.learn_len,
        online_buffer_capacity=ns.online_buffer_capacity,
        demo_buffer_capacity=ns.demo_buffer_capacity,
        demo_max_demos=ns.demo_max_demos,
        demo_batch_fraction=ns.demo_batch_fraction,
        actor_lr=ns.actor_lr,
        critic_lr=ns.critic_lr,
        alpha_lr=ns.alpha_lr,
        value_lr=ns.value_lr,
        gamma=ns.gamma,
        tau=ns.tau,
        update_every=ns.update_every,
        warmup_steps=ns.warmup_steps,
        eval_interval=ns.eval_interval,
        eval_episodes=ns.eval_episodes,
        max_ep_len=ns.max_ep_len,
        rollout_batch_steps=ns.rollout_batch_steps,
        ppo_epochs=ns.ppo_epochs,
        ppo_clip_coef=ns.ppo_clip_coef,
        value_coef=ns.value_coef,
        entropy_coef=ns.entropy_coef,
        gae_lambda=ns.gae_lambda,
        actor_bc_weight=ns.actor_bc_weight,
        actor_bc_decay=ns.actor_bc_decay,
        output_dir=ns.output_dir,
    )
