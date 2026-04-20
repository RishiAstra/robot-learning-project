from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path

import torch

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

from rl.actor import infer_feature_dim, sample_actor_step
from rl.common import filter_obs, set_seed
from rl.evaluation import evaluate_actor
from rl.replay import SequenceReplayBuffer, cat_batches, load_demo_sequences
from rl.sac import TwinQNetwork, sac_update


os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

FINAL_EVAL_EPISODES = 50


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

        self.env = EnvUtils.create_env_from_metadata(
            self.ckpt_dict["env_metadata"],
            render=False,
            render_offscreen=False,
            use_image_obs=False,
        )
        obs_feat_dim = infer_feature_dim(self.actor, self.obs_keys, self.env, self.device)
        action_dim = int(self.env.action_dimension)
        self.critic = TwinQNetwork(obs_feat_dim, action_dim=action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for param in self.critic_target.parameters():
            param.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        self.log_alpha = torch.tensor(-4.0, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.alpha_lr)
        self.target_entropy = -float(action_dim)

        self.seq_len = args.burnin_len + args.learn_len
        self.online_replay = SequenceReplayBuffer(args.online_buffer_capacity, self.seq_len, self.obs_keys)
        self.demo_replay = SequenceReplayBuffer(args.demo_buffer_capacity, self.seq_len, self.obs_keys)

        # dataset_path = self.config.train.data[0]["path"]
        data_cfg = self.config.train.data
        if isinstance(data_cfg, str):
            dataset_path = data_cfg
        else:
            dataset_path = data_cfg[0]["path"]
        load_demo_sequences(dataset_path, self.demo_replay, self.obs_keys, max_demos=args.demo_max_demos)

        self.output_dir = Path(args.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
        bc_batch = demo_batch if args.method == "sac_dapg" else None
        bc_weight = args.actor_bc_weight if args.method == "sac_dapg" else 0.0
        return train_batch, bc_batch, bc_weight

    def maybe_update(self, step: int):
        args = self.args
        if step < args.warmup_steps:
            return None
        if len(self.online_replay) < args.batch_size or step % args.update_every != 0:
            return None

        batch, bc_batch, bc_weight = self.select_update_batches()
        if bc_weight > 0.0:
            bc_weight *= args.actor_bc_decay ** step
        return sac_update(
            actor=self.actor,
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

    def save_best(self, step: int, stats):
        save_path = self.output_dir / f"{self.args.method}_best.pth"
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "step": step,
                "stats": stats,
                "args": vars(self.args),
            },
            save_path,
        )
        print(f"saved {save_path}")

    def save_final(self, step: int, stats):
        """Save the final finetuned model and write accuracy results to a JSON file."""
        method = self.args.method

        model_path = self.output_dir / f"{method}_final.pth"
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "step": step,
                "stats": stats,
                "args": vars(self.args),
            },
            model_path,
        )
        print(f"saved final model → {model_path}")

        stats_path = self.output_dir / f"{method}_final_stats.json"
        with open(stats_path, "w") as f:
            json.dump({"step": step, "eval_episodes": FINAL_EVAL_EPISODES, **stats}, f, indent=4)
        print(f"saved final stats → {stats_path}")

    def train(self):
        args = self.args
        baseline = evaluate_actor(self.actor, self.env, self.obs_keys, args.eval_episodes, args.max_ep_len)
        print("BC baseline")
        print(json.dumps(baseline, indent=4))

        obs = filter_obs(self.env.reset(), self.obs_keys)
        rollout_state = None
        ep_len = 0
        best_success = baseline["Success_Rate"]
        start_time = time.time()

        for step in range(args.total_steps):
            action, rollout_state = sample_actor_step(self.actor, obs, rnn_state=rollout_state, deterministic=False)
            next_obs_raw, _, done, _ = self.env.step(action)
            next_obs = filter_obs(next_obs_raw, self.obs_keys)
            success = self.env.is_success()["task"]
            reward = float(success)
            terminal = bool(done or success or (ep_len + 1) >= args.max_ep_len)
            self.online_replay.add_step(obs, action, reward, next_obs, terminal)

            obs = next_obs
            ep_len += 1

            if terminal:
                self.online_replay.flush_episode()
                obs = filter_obs(self.env.reset(), self.obs_keys)
                rollout_state = None
                ep_len = 0

            logs = self.maybe_update(step)
            if logs is not None and step % 200 == 0:
                print(
                    f"step={step} method={args.method} "
                    f"critic={logs['critic_loss']:.3f} actor={logs['actor_loss']:.3f} "
                    f"rl={logs['rl_loss']:.3f} bc={logs['bc_loss']:.3f} alpha={logs['alpha']:.4f}"
                )

            if step > 0 and step % args.eval_interval == 0:
                stats = evaluate_actor(self.actor, self.env, self.obs_keys, args.eval_episodes, args.max_ep_len)
                print(f"eval@{step}")
                print(json.dumps(stats, indent=4))
                if stats["Success_Rate"] > best_success:
                    best_success = stats["Success_Rate"]
                    self.save_best(step, stats)

        # High-resolution final evaluation (50 episodes) + save model and stats
        final_stats = evaluate_actor(
            self.actor, self.env, self.obs_keys, FINAL_EVAL_EPISODES, args.max_ep_len
        )
        print("final")
        print(json.dumps(final_stats, indent=4))
        print(f"elapsed_sec={time.time() - start_time:.1f}")

        self.save_final(step=args.total_steps, stats=final_stats)
        return final_stats
