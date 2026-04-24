from __future__ import annotations

import copy
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

from rl.actor import infer_feature_dim, sample_actor_step, sample_actor_step_with_log_prob
from rl.common import filter_obs, set_seed
from rl.replay import SequenceReplayBuffer, cat_batches, load_demo_sequences
from rl.ppo import RolloutEpisodeBuffer, ValueHead, ppo_update
from rl.sac import TwinQNetwork, sac_update


os.environ.setdefault("NUMBA_DISABLE_JIT", "1")


def save_robomimic_policy_checkpoint(policy, ckpt_dict, save_path: Path, rl_state: dict) -> None:
    updated_ckpt = dict(ckpt_dict)
    updated_ckpt["model"] = policy.policy.serialize()
    updated_ckpt["rl_state"] = rl_state
    torch.save(updated_ckpt, save_path)
    print(f"saved {save_path}")


def _state_step(ckpt_dict: dict) -> int:
    return int(ckpt_dict.get("rl_state", {}).get("step", 0))


def _decoupled_decay_param_groups(module: torch.nn.Module, weight_decay: float):
    decay_params = []
    no_decay_params = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if weight_decay > 0.0 and param.ndim > 1 and not name.endswith(".bias"):
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return param_groups


def _run_label(args) -> str:
    """Canonical name for this run, used in all output filenames."""
    label = args.method
    if getattr(args, "critic_layer_norm", False):
        label += "_ln"
    if getattr(args, "critic_weight_decay", 0.0) > 0.0:
        label += "_l2"
    return label


# ── eval helper ───────────────────────────────────────────────────────────────

def _maybe_run_eval(policy, ckpt_dict: dict, args, step: int) -> None:
    eval_output_dir = getattr(args, "eval_output_dir", "")
    if not eval_output_dir:
        return

    eval_script = Path(__file__).parent.parent / "evaluate_checkpoints.py"

    task = (getattr(args, "task_name", "")
            or ckpt_dict.get("env_metadata", {}).get("env_name", "task")
                        .lower().replace("-", "_"))

    step_dir = Path(eval_output_dir) / f"step_{step:06d}"
    json_path = step_dir / f"{task}_{_run_label(args)}.json"

    if json_path.exists():
        print(f"[eval] step={step} already evaluated — skipping")
        return

    step_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = step_dir / f"{task}_{_run_label(args)}.pth"

    save_robomimic_policy_checkpoint(
        policy, ckpt_dict, ckpt_path,
        rl_state={"step": step},
    )

    n_rollouts = getattr(args, "eval_rollouts", 20)
    horizon    = getattr(args, "eval_horizon",  getattr(args, "max_ep_len", 400))

    result = subprocess.run(
        [sys.executable, str(eval_script),
         "--agent",       str(ckpt_path),
         "--n-rollouts",  str(n_rollouts),
         "--horizon",     str(horizon),
         "--output-json", str(json_path)],
        check=False,
    )
    if result.returncode != 0:
        print(f"[eval] evaluate_checkpoints.py failed at step={step} (exit {result.returncode})")
        json_path.unlink(missing_ok=True)


# ── trainers ──────────────────────────────────────────────────────────────────

class SACTrainer:
    def __init__(self, args):
        self.args = args
        set_seed(args.seed)
        self.device = TorchUtils.get_torch_device(try_to_use_cuda=True)

        self.policy, self.ckpt_dict = FileUtils.policy_from_checkpoint(
            ckpt_path=args.ckpt_path,
            device=self.device,
            verbose=False,
        )
        self.config, _ = FileUtils.config_from_checkpoint(ckpt_dict=self.ckpt_dict)
        self.obs_keys = list(self.config.observation.modalities.obs.low_dim)
        if len(self.config.observation.modalities.obs.rgb) > 0:
            raise ValueError("train_rl.py currently supports low-dim checkpoints only")

        self.actor = self.policy.policy.nets["policy"]
        self.actor.train()
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.actor_target.train()
        for param in self.actor_target.parameters():
            param.requires_grad_(False)
        self.rnn_horizon = int(getattr(self.config.algo.rnn, "horizon", 1)) if hasattr(self.config.algo, "rnn") else 1

        self.env = EnvUtils.create_env_from_metadata(
            self.ckpt_dict["env_metadata"],
            render=False,
            render_offscreen=False,
            use_image_obs=False,
        )
        obs_feat_dim = infer_feature_dim(self.actor, self.obs_keys, self.env, self.device)
        action_dim = int(self.env.action_dimension)
        self.critic = TwinQNetwork(obs_feat_dim, action_dim=action_dim, layer_norm=args.critic_layer_norm).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for param in self.critic_target.parameters():
            param.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = torch.optim.AdamW(
            _decoupled_decay_param_groups(self.critic, args.critic_weight_decay),
            lr=args.critic_lr,
        )
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.alpha_lr)
        self.target_entropy = -float(action_dim)

        self.start_step = _state_step(self.ckpt_dict)

        # adaptive BC — initialise before _load_rl_state so resume can override
        self.bc_weight = args.actor_bc_weight
        self.ema_success_rate = args.bc_baseline_success
        self.baseline_success_rate = args.bc_baseline_success
        self.best_ema_success_rate = args.bc_baseline_success
        self.episode_ema_history: list = []   # [(step, ema, bc_weight), ...]
        self.episode_count = 0

        self._load_rl_state()

        self.seq_len = args.burnin_len + args.learn_len
        self.online_replay = SequenceReplayBuffer(args.online_buffer_capacity, self.seq_len, self.obs_keys)
        self.demo_replay = SequenceReplayBuffer(args.demo_buffer_capacity, self.seq_len, self.obs_keys)

        _data = self.config.train.data
        if isinstance(_data, str):
            dataset_path = _data
        elif isinstance(_data[0], dict):
            dataset_path = _data[0]["path"]
        else:
            dataset_path = _data[0]
        load_demo_sequences(dataset_path, self.demo_replay, self.obs_keys, max_demos=args.demo_max_demos)

        self.output_dir = Path(args.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_rl_state(self) -> None:
        rl_state = self.ckpt_dict.get("rl_state")
        if not rl_state:
            return

        if "critic" in rl_state:
            self.critic.load_state_dict(rl_state["critic"])
        if "critic_target" in rl_state:
            self.critic_target.load_state_dict(rl_state["critic_target"])
        if "actor_target" in rl_state:
            self.actor_target.load_state_dict(rl_state["actor_target"])
        else:
            self.actor_target.load_state_dict(self.actor.state_dict())
        if "log_alpha" in rl_state:
            self.log_alpha.data.copy_(rl_state["log_alpha"].to(self.device))

        # optimizer states — prevents momentum reset on resume
        if "actor_optimizer" in rl_state:
            self.actor_optimizer.load_state_dict(rl_state["actor_optimizer"])
        if "critic_optimizer" in rl_state:
            self.critic_optimizer.load_state_dict(rl_state["critic_optimizer"])
        if "alpha_optimizer" in rl_state:
            self.alpha_optimizer.load_state_dict(rl_state["alpha_optimizer"])

        # adaptive BC state
        if self.args.adaptive_bc:
            self.ema_success_rate      = rl_state.get("ema_success_rate",      self.args.bc_baseline_success)
            self.best_ema_success_rate = rl_state.get("best_ema_success_rate", self.args.bc_baseline_success)
            self.bc_weight             = rl_state.get("bc_weight",             self.args.actor_bc_weight)
            self.baseline_success_rate = rl_state.get("baseline_success_rate", self.args.bc_baseline_success)
            self.episode_ema_history   = rl_state.get("episode_ema_history",   [])
            self.episode_count         = rl_state.get("episode_count",         0)

        print(f"resumed rl_state from step {self.start_step}")

    def select_update_batches(self):
        args = self.args
        if args.method == "sac_bc_init":
            return self.online_replay.sample(args.batch_size, self.device), None, 0.0

        n_demo = max(1, int(args.batch_size * args.demo_batch_fraction))
        n_demo = min(n_demo, len(self.demo_replay))
        n_online = args.batch_size - n_demo
        if n_online <= 0:
            n_online = 1
            n_demo = args.batch_size - 1

        online_batch = self.online_replay.sample(n_online, self.device)
        demo_batch = self.demo_replay.sample(n_demo, self.device)
        train_batch = cat_batches(online_batch, demo_batch) if args.method == "sac_fd" else online_batch
        bc_batch = demo_batch if args.method in ("sac_dapg", "sac_fd") else None
        bc_weight = args.actor_bc_weight if args.method in ("sac_dapg", "sac_fd") else 0.0
        return train_batch, bc_batch, bc_weight

    def maybe_update(self, step: int):
        args = self.args
        if step < args.warmup_steps:
            return None
        if len(self.online_replay) < args.batch_size or step % args.update_every != 0:
            return None

        batch, bc_batch, bc_weight = self.select_update_batches()
        if args.adaptive_bc:
            bc_weight = self.bc_weight          # live adaptive value, no per-step decay
        elif bc_weight > 0.0:
            bc_weight *= args.actor_bc_decay ** step

        return sac_update(
            actor=self.actor,
            actor_target=self.actor_target,
            critic=self.critic,
            critic_target=self.critic_target,
            batch=batch,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.critic_optimizer,
            log_alpha=self.log_alpha,
            alpha_optimizer=self.alpha_optimizer,
            target_entropy=self.target_entropy,
            burnin_len=args.burnin_len,
            gamma=args.gamma,
            tau=args.tau,
            bc_weight=bc_weight,
            bc_batch=bc_batch,
        )

    def save_checkpoint(self, step: int, stats):
        save_path = self.output_dir / f"{_run_label(self.args)}_step_{step:06d}.pth"
        rl_state = {
            "critic":          self.critic.state_dict(),
            "critic_target":   self.critic_target.state_dict(),
            "actor_target":    self.actor_target.state_dict(),
            "log_alpha":       self.log_alpha.detach().cpu(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer":self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "step": step, "stats": stats, "args": vars(self.args),
        }
        if self.args.adaptive_bc:
            rl_state.update({
                "ema_success_rate":      self.ema_success_rate,
                "best_ema_success_rate": self.best_ema_success_rate,
                "bc_weight":             self.bc_weight,
                "baseline_success_rate": self.baseline_success_rate,
                "episode_ema_history":   self.episode_ema_history,
                "episode_count":         self.episode_count,
            })
        save_robomimic_policy_checkpoint(self.policy, self.ckpt_dict, save_path, rl_state=rl_state)

    def train(self):
        args = self.args

        obs = filter_obs(self.env.reset(), self.obs_keys)
        rollout_state = None
        rollout_rnn_steps = 0
        ep_len = 0
        ep_success = False          # tracks success anywhere in the episode
        start_time = time.time()

        for step in range(self.start_step, args.total_steps):
            if rollout_rnn_steps % self.rnn_horizon == 0:
                rollout_state = None
            action, rollout_state = sample_actor_step(self.actor, obs, rnn_state=rollout_state, deterministic=False)
            next_obs_raw, reward, done, _ = self.env.step(action)
            next_obs = filter_obs(next_obs_raw, self.obs_keys)
            success = self.env.is_success()["task"]
            ep_success = ep_success or bool(success)
            reward = float(reward)
            terminal = bool(done or success or (ep_len + 1) >= args.max_ep_len)
            self.online_replay.add_step(obs, action, reward, next_obs, terminal)

            obs = next_obs
            ep_len += 1
            rollout_rnn_steps += 1

            if terminal:
                self.online_replay.flush_episode()

                if args.adaptive_bc:
                    self.ema_success_rate = (
                        args.bc_ema_alpha * float(ep_success)
                        + (1.0 - args.bc_ema_alpha) * self.ema_success_rate
                    )
                    self.best_ema_success_rate = max(self.best_ema_success_rate, self.ema_success_rate)
                    if self.ema_success_rate < self.best_ema_success_rate:
                        pass  # hold bc_weight steady
                    else:
                        self.bc_weight = max(self.bc_weight * 0.97, 0.0)
                    self.episode_count += 1
                    self.episode_ema_history.append((step, self.ema_success_rate, self.bc_weight))
                    print(
                        f"episode={self.episode_count} step={step} "
                        f"success={int(ep_success)} ema={self.ema_success_rate:.3f} "
                        f"best={self.best_ema_success_rate:.3f} bc_weight={self.bc_weight:.4f}"
                    )

                obs = filter_obs(self.env.reset(), self.obs_keys)
                rollout_state = None
                rollout_rnn_steps = 0
                ep_len = 0
                ep_success = False

            logs = self.maybe_update(step)
            if logs is not None and step % 200 == 0:
                print(
                    f"step={step} method={args.method} "
                    f"critic={logs['critic_loss']:.3f} actor={logs['actor_loss']:.3f} "
                    f"td={logs['td_loss']:.3f} "
                    f"rl={logs['rl_loss']:.3f} bc={logs['bc_loss']:.3f} "
                    f"bc_w={logs['bc_weight']:.4f} alpha={logs['alpha']:.4f}"
                )

            if step > self.start_step and step % args.checkpoint_interval == 0:
                self.save_checkpoint(step, stats={})

            if (args.eval_interval > 0
                    and step > self.start_step
                    and step % args.eval_interval == 0):
                _maybe_run_eval(self.policy, self.ckpt_dict, args, step)

        self.save_checkpoint(args.total_steps, stats={})
        _maybe_run_eval(self.policy, self.ckpt_dict, args, args.total_steps)
        print(f"elapsed_sec={time.time() - start_time:.1f}")
        return {}


class PPOTrainer:
    def __init__(self, args):
        self.args = args
        set_seed(args.seed)
        self.device = TorchUtils.get_torch_device(try_to_use_cuda=True)

        self.policy, self.ckpt_dict = FileUtils.policy_from_checkpoint(
            ckpt_path=args.ckpt_path,
            device=self.device,
            verbose=False,
        )
        self.config, _ = FileUtils.config_from_checkpoint(ckpt_dict=self.ckpt_dict)
        self.obs_keys = list(self.config.observation.modalities.obs.low_dim)
        if len(self.config.observation.modalities.obs.rgb) > 0:
            raise ValueError("train_rl.py currently supports low-dim checkpoints only")

        self.actor = self.policy.policy.nets["policy"]
        self.actor.train()
        self.rnn_horizon = int(getattr(self.config.algo.rnn, "horizon", 1)) if hasattr(self.config.algo, "rnn") else 1

        self.env = EnvUtils.create_env_from_metadata(
            self.ckpt_dict["env_metadata"],
            render=False,
            render_offscreen=False,
            use_image_obs=False,
        )
        obs_feat_dim = infer_feature_dim(self.actor, self.obs_keys, self.env, self.device)
        self.value_net = ValueHead(obs_feat_dim, layer_norm=args.critic_layer_norm).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.value_optimizer = torch.optim.AdamW(
            _decoupled_decay_param_groups(self.value_net, args.critic_weight_decay),
            lr=args.value_lr,
        )
        self.start_step = _state_step(self.ckpt_dict)
        self._load_rl_state()

        self.rollout_buffer = RolloutEpisodeBuffer()
        self.demo_replay = SequenceReplayBuffer(
            args.demo_buffer_capacity,
            args.burnin_len + args.learn_len,
            self.obs_keys,
        )
        _data = self.config.train.data
        if isinstance(_data, str):
            dataset_path = _data
        elif isinstance(_data[0], dict):
            dataset_path = _data[0]["path"]
        else:
            dataset_path = _data[0]
        load_demo_sequences(dataset_path, self.demo_replay, self.obs_keys, max_demos=args.demo_max_demos)

        self.output_dir = Path(args.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.env_steps = 0

    def _load_rl_state(self) -> None:
        rl_state = self.ckpt_dict.get("rl_state")
        if not rl_state:
            return

        if "value_net" in rl_state:
            self.value_net.load_state_dict(rl_state["value_net"])

        print(f"resumed PPO rl_state from step {self.start_step}")

    def save_checkpoint(self, step: int, stats):
        save_path = self.output_dir / f"{_run_label(self.args)}_step_{step:06d}.pth"
        save_robomimic_policy_checkpoint(
            self.policy, self.ckpt_dict, save_path,
            rl_state={"value_net": self.value_net.state_dict(), "step": step, "stats": stats, "args": vars(self.args)},
        )

    def maybe_update(self):
        args = self.args
        if self.rollout_buffer.num_steps() < args.rollout_batch_steps:
            return None
        if len(self.rollout_buffer) == 0:
            return None

        demo_batch = None
        if args.method == "ppo_dapg" and len(self.demo_replay) > 0:
            n_demo = max(1, int(args.batch_size * args.demo_batch_fraction))
            n_demo = min(n_demo, len(self.demo_replay))
            demo_batch = self.demo_replay.sample(n_demo, self.device)

        stats = ppo_update(
            actor=self.actor,
            value_net=self.value_net,
            episodes=self.rollout_buffer.episodes,
            obs_keys=self.obs_keys,
            device=self.device,
            actor_optimizer=self.actor_optimizer,
            value_optimizer=self.value_optimizer,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_coef=args.ppo_clip_coef,
            value_clip_coef=args.ppo_clip_coef,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=1.0,
            update_epochs=args.ppo_epochs,
            demo_batch=demo_batch,
            bc_weight=args.actor_bc_weight if args.method == "ppo_dapg" else 0.0,
            bc_decay=args.actor_bc_decay,
            env_step=self.env_steps,
            rnn_horizon=self.rnn_horizon,
        )
        self.rollout_buffer.clear()
        return stats

    def train(self):
        args = self.args

        obs = filter_obs(self.env.reset(), self.obs_keys)
        rollout_state = None
        rollout_rnn_steps = 0
        ep_len = 0
        start_time = time.time()

        for step in range(self.start_step, args.total_steps):
            if rollout_rnn_steps % self.rnn_horizon == 0:
                rollout_state = None
            action, log_prob, rollout_state = sample_actor_step_with_log_prob(
                self.actor,
                obs,
                rnn_state=rollout_state,
                deterministic=False,
            )
            obs_tensor = {
                k: torch.tensor(v, dtype=torch.float32, device=self.device).unsqueeze(0)
                for k, v in obs.items()
            }
            with torch.no_grad():
                obs_feat = self.actor.nets["encoder"](obs=obs_tensor)
                value = self.value_net(obs_feat).item()

            next_obs_raw, reward, done, _ = self.env.step(action)
            next_obs = filter_obs(next_obs_raw, self.obs_keys)
            success = self.env.is_success()["task"]
            reward = float(reward)
            terminal = bool(done or success or (ep_len + 1) >= args.max_ep_len)
            self.rollout_buffer.add_step(obs, action, reward, terminal, log_prob, value)

            obs = next_obs
            ep_len += 1
            self.env_steps += 1
            rollout_rnn_steps += 1

            if terminal:
                self.rollout_buffer.flush_episode()
                obs = filter_obs(self.env.reset(), self.obs_keys)
                rollout_state = None
                rollout_rnn_steps = 0
                ep_len = 0
                logs = self.maybe_update()
                if logs is not None:
                    print(
                        f"update@{step} method={args.method} "
                        f"policy={logs['policy_loss']:.3f} value={logs['value_loss']:.3f} "
                        f"bc={logs['bc_loss']:.3f} kl={logs['approx_kl']:.4f} "
                        f"clip={logs['clip_fraction']:.3f}"
                    )

            if step > self.start_step and step % args.checkpoint_interval == 0:
                self.save_checkpoint(step, stats={})

            if (args.eval_interval > 0
                    and step > self.start_step
                    and step % args.eval_interval == 0):
                _maybe_run_eval(self.policy, self.ckpt_dict, args, step)

        self.rollout_buffer.flush_episode()
        if len(self.rollout_buffer) > 0:
            logs = self.maybe_update()
            if logs is not None:
                print(
                    f"final_update method={args.method} "
                    f"policy={logs['policy_loss']:.3f} value={logs['value_loss']:.3f} "
                    f"bc={logs['bc_loss']:.3f} kl={logs['approx_kl']:.4f} "
                    f"clip={logs['clip_fraction']:.3f}"
                )

        self.save_checkpoint(args.total_steps, stats={})
        _maybe_run_eval(self.policy, self.ckpt_dict, args, args.total_steps)
        print(f"elapsed_sec={time.time() - start_time:.1f}")