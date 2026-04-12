from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import robosuite as suite


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def flatten_obs(obs: Dict[str, np.ndarray], obs_keys: Optional[Sequence[str]] = None) -> np.ndarray:
    if obs_keys is None:
        preferred = [key for key in ("robot0_proprio-state", "object-state") if key in obs]
        keys = preferred if preferred else [k for k, v in obs.items() if np.asarray(v).ndim == 1]
    else:
        keys = list(obs_keys)
    parts = [np.asarray(obs[key], dtype=np.float32).reshape(-1) for key in keys]
    return np.concatenate(parts, axis=0)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done):
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.obs[idx], device=self.device),
            torch.tensor(self.next_obs[idx], device=self.device),
            torch.tensor(self.actions[idx], device=self.device),
            torch.tensor(self.rewards[idx], device=self.device),
            torch.tensor(self.dones[idx], device=self.device),
        )


class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        return self.net(torch.cat([obs, action], dim=-1))


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, action_low, action_high, hidden_dim: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        action_low = torch.as_tensor(action_low, dtype=torch.float32)
        action_high = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)

    def forward(self, obs: torch.Tensor):
        hidden = self.backbone(obs)
        mean = self.mean(hidden)
        log_std = self.log_std(hidden).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor):
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        correction = torch.log(self.action_scale * (1.0 - y_t.pow(2)) + 1e-6)
        log_prob = (log_prob - correction).sum(dim=-1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action


@dataclass
class Args:
    env: str
    total_steps: int
    seed: int
    batch_size: int
    buffer_size: int
    gamma: float
    tau: float
    actor_lr: float
    critic_lr: float
    alpha_lr: float
    learning_starts: int
    policy_frequency: int
    eval_interval: int
    eval_episodes: int
    max_ep_len: int
    reward_shaping: bool
    output_json: Optional[str]


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Lift")
    parser.add_argument("--total-steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--alpha-lr", type=float, default=1e-3)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--policy-frequency", type=int, default=2)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--max-ep-len", type=int, default=200)
    parser.add_argument("--reward-shaping", action="store_true")
    parser.add_argument("--output-json", default=None)
    ns = parser.parse_args()
    return Args(
        env=ns.env,
        total_steps=ns.total_steps,
        seed=ns.seed,
        batch_size=ns.batch_size,
        buffer_size=ns.buffer_size,
        gamma=ns.gamma,
        tau=ns.tau,
        actor_lr=ns.actor_lr,
        critic_lr=ns.critic_lr,
        alpha_lr=ns.alpha_lr,
        learning_starts=ns.learning_starts,
        policy_frequency=ns.policy_frequency,
        eval_interval=ns.eval_interval,
        eval_episodes=ns.eval_episodes,
        max_ep_len=ns.max_ep_len,
        reward_shaping=ns.reward_shaping,
        output_json=ns.output_json,
    )


def make_env(env_name: str, reward_shaping: bool):
    return suite.make(
        env_name,
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=reward_shaping,
        control_freq=20,
    )


def random_action(low, high):
    return np.random.uniform(low, high).astype(np.float32)


def get_success(env) -> float:
    if hasattr(env, "is_success"):
        try:
            success = env.is_success()
            if isinstance(success, dict):
                if "task" in success:
                    return float(success["task"])
                return float(any(success.values()))
            return float(success)
        except Exception:
            pass
    if hasattr(env, "_check_success"):
        return float(env._check_success())
    return 0.0


@torch.no_grad()
def evaluate(actor, env_name: str, seed: int, n_episodes: int, max_ep_len: int, reward_shaping: bool):
    env = make_env(env_name, reward_shaping)
    returns = []
    successes = 0
    for ep in range(n_episodes):
        np.random.seed(seed + ep)
        obs = flatten_obs(env.reset())
        total_reward = 0.0
        for _ in range(max_ep_len):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=next(actor.parameters()).device).unsqueeze(0)
            _, _, action = actor.sample(obs_tensor)
            next_obs, reward, done, _ = env.step(action.squeeze(0).cpu().numpy())
            total_reward += float(reward)
            obs = flatten_obs(next_obs)
            success = get_success(env)
            if done or success:
                successes += int(success > 0.0)
                break
        returns.append(total_reward)
    env.close()
    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "success_rate": float(successes / n_episodes),
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(args.env, args.reward_shaping)
    action_low, action_high = env.action_spec
    obs_dim = flatten_obs(env.reset()).shape[0]
    action_dim = int(env.action_dim)

    actor = SquashedGaussianActor(obs_dim, action_dim, action_low, action_high).to(device)
    qf1 = SoftQNetwork(obs_dim, action_dim).to(device)
    qf2 = SoftQNetwork(obs_dim, action_dim).to(device)
    qf1_target = SoftQNetwork(obs_dim, action_dim).to(device)
    qf2_target = SoftQNetwork(obs_dim, action_dim).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    for param in list(qf1_target.parameters()) + list(qf2_target.parameters()):
        param.requires_grad_(False)

    q_optimizer = torch.optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.critic_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
    target_entropy = -float(action_dim)

    replay = ReplayBuffer(args.buffer_size, obs_dim, action_dim, device)
    obs = flatten_obs(env.reset())
    ep_return = 0.0
    ep_len = 0
    recent_returns = deque(maxlen=10)
    records: List[dict] = []
    start_time = time.time()

    for step in range(args.total_steps):
        if step < args.learning_starts:
            action = random_action(action_low, action_high)
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action, _, _ = actor.sample(obs_tensor)
                action = action.squeeze(0).cpu().numpy()

        next_obs_dict, reward, done, _ = env.step(action)
        next_obs = flatten_obs(next_obs_dict)
        success = get_success(env)
        ep_return += float(reward)
        ep_len += 1
        terminal = bool(done or success or ep_len >= args.max_ep_len)
        replay.add(obs, next_obs, action, reward, terminal)
        obs = next_obs

        if terminal:
            recent_returns.append(ep_return)
            obs = flatten_obs(env.reset())
            ep_return = 0.0
            ep_len = 0

        if step >= args.learning_starts and replay.size >= args.batch_size:
            obs_batch, next_obs_batch, actions_batch, rewards_batch, dones_batch = replay.sample(args.batch_size)

            with torch.no_grad():
                next_actions, next_log_prob, _ = actor.sample(next_obs_batch)
                min_q_target = torch.min(
                    qf1_target(next_obs_batch, next_actions),
                    qf2_target(next_obs_batch, next_actions),
                )
                alpha = log_alpha.exp()
                next_q_value = rewards_batch + (1.0 - dones_batch) * args.gamma * (min_q_target - alpha * next_log_prob)

            qf1_loss = F.mse_loss(qf1(obs_batch, actions_batch), next_q_value)
            qf2_loss = F.mse_loss(qf2(obs_batch, actions_batch), next_q_value)
            q_optimizer.zero_grad()
            (qf1_loss + qf2_loss).backward()
            q_optimizer.step()

            if step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    actions_pi, log_prob_pi, _ = actor.sample(obs_batch)
                    q_pi = torch.min(qf1(obs_batch, actions_pi), qf2(obs_batch, actions_pi))
                    alpha = log_alpha.exp()
                    actor_loss = (alpha * log_prob_pi - q_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    alpha_loss = -(log_alpha * (log_prob_pi.detach() + target_entropy)).mean()
                    alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    alpha_optimizer.step()

                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1.0 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1.0 - args.tau) * target_param.data)

        if step == 0 or (step + 1) % args.eval_interval == 0:
            eval_stats = evaluate(
                actor=actor,
                env_name=args.env,
                seed=args.seed + 1000 + step,
                n_episodes=args.eval_episodes,
                max_ep_len=args.max_ep_len,
                reward_shaping=args.reward_shaping,
            )
            record = {
                "step": step + 1,
                "recent_train_return": float(np.mean(recent_returns)) if recent_returns else 0.0,
                "alpha": float(log_alpha.exp().item()),
                **eval_stats,
            }
            records.append(record)
            print(json.dumps(record))

    env.close()
    summary = {
        "env": args.env,
        "total_steps": args.total_steps,
        "elapsed_sec": round(time.time() - start_time, 1),
        "final_eval": records[-1] if records else None,
        "best_return_mean": max((r["return_mean"] for r in records), default=None),
        "best_success_rate": max((r["success_rate"] for r in records), default=None),
    }
    print("summary")
    print(json.dumps(summary, indent=4))

    if args.output_json is not None:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"records": records, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
