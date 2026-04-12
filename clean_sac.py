# SAC for Robomimic BC-RNN fine-tuning.
# Stays as close as possible to CleanRL's sac_continuous_action.py.
#
# The ONLY changes from vanilla CleanRL are:
#   1. Robomimic env instead of gym
#   2. Dict obs → frozen BC encoder → flat feature vector stored in the buffer
#   3. GMM actor (from BC-RNN) instead of a squashed Gaussian
#      - actor called with T=1 / reset RNN state each call (stateless, like CleanRL)
#
# Everything else — replay buffer, update order, alpha tuning, target updates —
# is copied verbatim from CleanRL.

import copy
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_CKPT = REPO_ROOT / "mimicgen/datasets/core_training_results/bc_rnn_low_dim_ds_stack_D0_seed_101/20260312210424/models/model_epoch_100_demo_success_1.0.pth"


# ─────────────────────────────────────────────────────────────────────────────
# Args  (CleanRL pattern)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Args:
    ckpt_path: str = str(DEFAULT_CKPT)

    total_timesteps: int = 100_000
    buffer_size: int = int(1e5)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = 5_000
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    policy_frequency: int = 2
    target_network_frequency: int = 1
    alpha: float = 0.2
    autotune: bool = True

    # Robomimic-specific (only additions)
    max_ep_len: int = 400
    action_dim: int = 7
    eval_interval: int = 5_000
    eval_episodes: int = 20


OBS_KEYS = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "object",
]


# ─────────────────────────────────────────────────────────────────────────────
# Encode a single-timestep obs dict → flat feature vector  (no grad, frozen)
# Storing features in the buffer means we never re-encode and the buffer stays
# the same flat-array format as CleanRL.
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def encode_obs(actor, obs_dict: dict, device) -> np.ndarray:
    """obs_dict values are raw numpy arrays (no batch/time dim). Returns (F,) numpy."""
    tensor_obs = {
        k: torch.tensor(obs_dict[k], dtype=torch.float32, device=device).unsqueeze(0)
        for k in OBS_KEYS
        if k in obs_dict
    }
    feat = actor.nets["encoder"](obs=tensor_obs)  # (1, F)
    return feat.squeeze(0).cpu().numpy()


def infer_feat_dim(actor, env, device: torch.device) -> int:
    obs = env.reset()
    return encode_obs(actor, obs, device).shape[0]


# ─────────────────────────────────────────────────────────────────────────────
# Replay buffer  (same as CleanRL but stores pre-encoded flat obs features)
# ─────────────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device):
        self.capacity = capacity
        self.device   = device
        self.ptr      = 0
        self.size     = 0
        self.obs        = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.next_obs   = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.actions    = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards    = np.zeros((capacity, 1),          dtype=np.float32)
        self.dones      = np.zeros((capacity, 1),          dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done):
        self.obs[self.ptr]      = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.dones[self.ptr]    = done
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.obs[idx],      device=self.device),
            torch.tensor(self.next_obs[idx], device=self.device),
            torch.tensor(self.actions[idx],  device=self.device),
            torch.tensor(self.rewards[idx],  device=self.device),
            torch.tensor(self.dones[idx],    device=self.device),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Q-network  (identical to CleanRL's SoftQNetwork)
# ─────────────────────────────────────────────────────────────────────────────
class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ─────────────────────────────────────────────────────────────────────────────
# Actor helper: replaces CleanRL's Actor.get_action(obs)
#
# The BC-RNN actor uses a GMM head instead of a squashed Gaussian.
# We call forward_train with T=1 and rnn_init_state=None each time so the RNN
# is stateless during training — exactly matching CleanRL's per-step pattern.
# ─────────────────────────────────────────────────────────────────────────────
def actor_get_action(actor, obs_feat: torch.Tensor):
    """
    obs_feat : (B, F) pre-encoded flat features.
    Returns action (B, A) and log_prob (B, 1) — same interface as CleanRL.
    """
    # The BC-RNN encoder expects a dict of raw obs, but we've already encoded.
    # We call the RNN + GMM head directly by passing a no-op through the net.
    # Robomimic's RNNActorNetwork.forward_train flattens obs through the encoder
    # first; to skip that we wrap our feature as a single-key dict and replace
    # the encoder with identity-compatible input.
    #
    # Simplest correct route: use nets["policy"] (the RNN+head) directly,
    # bypassing nets["encoder"] by constructing the expected input manually.
    # actor.nets["policy"] is a RNNGMMActorNetwork that takes encoded features.

    B = obs_feat.shape[0]
    feat_seq = obs_feat.unsqueeze(1)  # (B, 1, F) — T=1, stateless

    # Forward through RNN + GMM head only (encoder already applied)
    dist, _ = actor.nets["policy"].forward(
        feat_seq, rnn_init_state=None, return_state=True
    )  # dist is a MixtureSameFamily GMM over (B, 1, A)

    # Reparameterised GMM sample via Gumbel-softmax component selection
    gumbel_w    = F.gumbel_softmax(dist.mixture_distribution.logits, tau=1.0, hard=True)
    all_samples = dist.component_distribution.rsample()           # (B, 1, K, A)
    action      = (gumbel_w.unsqueeze(-1) * all_samples).sum(-2)  # (B, 1, A)
    action      = action.squeeze(1)                               # (B, A)
    log_prob    = dist.log_prob(action.unsqueeze(1)).squeeze(1)   # (B,)

    return action, log_prob.unsqueeze(1)  # (B, A), (B, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation  (unchanged from your original)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(policy, env, n_episodes: int, max_ep_len: int) -> float:
    actor = policy.policy.nets["policy"]
    actor.eval()
    successes = 0
    for _ in range(n_episodes):
        obs = env.reset()
        policy.start_episode()
        for _ in range(max_ep_len):
            action = policy(obs)
            obs, _, _, _ = env.step(action)
            if env.is_success()["task"]:
                successes += 1
                break
    actor.train()
    return successes / n_episodes


# ─────────────────────────────────────────────────────────────────────────────
# Main  (mirrors CleanRL's __main__ block as closely as possible)
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args   = tyro.cli(Args)
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    ckpt_path = str(Path(args.ckpt_path).expanduser().resolve())

    # ── Load BC-RNN policy ───────────────────────────────────────────────────
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=ckpt_path, device=device, verbose=False
    )
    actor = policy.policy.nets["policy"]

    # Freeze encoder — features are pre-encoded and stored in the buffer.
    for p in actor.nets["encoder"].parameters():
        p.requires_grad_(False)
    actor.train()

    # ── Environment ──────────────────────────────────────────────────────────
    env = EnvUtils.create_env_from_metadata(
        ckpt_dict["env_metadata"],
        render=False, render_offscreen=False, use_image_obs=False,
    )

    obs_dim = infer_feat_dim(actor, env, device)
    print(f"Obs feature dim: {obs_dim}")

    # ── Networks (identical to CleanRL) ──────────────────────────────────────
    qf1        = SoftQNetwork(obs_dim, args.action_dim).to(device)
    qf2        = SoftQNetwork(obs_dim, args.action_dim).to(device)
    qf1_target = copy.deepcopy(qf1)
    qf2_target = copy.deepcopy(qf2)
    for p in list(qf1_target.parameters()) + list(qf2_target.parameters()):
        p.requires_grad_(False)

    q_optimizer     = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr
    )
    actor_optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, actor.parameters()), lr=args.policy_lr
    )

    # Automatic entropy tuning (identical to CleanRL)
    if args.autotune:
        target_entropy = -float(args.action_dim)
        log_alpha      = torch.zeros(1, requires_grad=True, device=device)
        alpha          = log_alpha.exp().item()
        a_optimizer    = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    rb         = ReplayBuffer(args.buffer_size, obs_dim, args.action_dim, device)
    start_time = time.time()

    # ── BC baseline ──────────────────────────────────────────────────────────
    print("Evaluating BC baseline...")
    bc_sr   = evaluate(policy, env, args.eval_episodes, args.max_ep_len)
    best_sr = bc_sr
    print(f"BC baseline: {bc_sr:.1%}")

    # ── Training loop (mirrors CleanRL's loop structure verbatim) ─────────────
    obs      = env.reset()
    obs_feat = encode_obs(actor, obs, device)
    ep_t     = 0

    for global_step in range(args.total_timesteps):

        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            action = np.random.uniform(-1.0, 1.0, size=(args.action_dim,)).astype(np.float32)
        else:
            obs_feat_t = torch.tensor(obs_feat, device=device).unsqueeze(0)
            with torch.no_grad():
                action_t, _, _ = actor_get_action(actor, obs_feat_t)
            action = action_t.squeeze(0).cpu().numpy()

        next_obs, _, done, _ = env.step(action)
        success              = env.is_success()["task"]
        reward               = float(success)
        next_obs_feat        = encode_obs(actor, next_obs, device)
        ep_t                += 1

        rb.add(obs_feat, next_obs_feat, action, reward, float(done or success))

        obs      = next_obs
        obs_feat = next_obs_feat

        if success or ep_t >= args.max_ep_len:
            obs      = env.reset()
            obs_feat = encode_obs(actor, obs, device)
            ep_t     = 0
            policy.start_episode()

        # ALGO LOGIC: training
        if global_step > args.learning_starts:
            obs_b, next_obs_b, actions_b, rewards_b, dones_b = rb.sample(args.batch_size)

            with torch.no_grad():
                next_state_actions, next_state_log_pi = actor_get_action(actor, next_obs_b)
                qf1_next_target = qf1_target(next_obs_b, next_state_actions)
                qf2_next_target = qf2_target(next_obs_b, next_state_actions)
                min_qf_next_target = (
                    torch.min(qf1_next_target, qf2_next_target)
                    - alpha * next_state_log_pi
                )
                next_q_value = (
                    rewards_b.flatten()
                    + (1 - dones_b.flatten()) * args.gamma * min_qf_next_target.view(-1)
                )

            qf1_a_values = qf1(obs_b, actions_b).view(-1)
            qf2_a_values = qf2(obs_b, actions_b).view(-1)
            qf1_loss     = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss     = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss      = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    pi, log_pi = actor_get_action(actor, obs_b)
                    qf1_pi     = qf1(obs_b, pi)
                    qf2_pi     = qf2(obs_b, pi)
                    min_qf_pi  = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi = actor_get_action(actor, obs_b)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )

            if global_step % 1000 == 0:
                print(
                    f"step={global_step} | "
                    f"qf1={qf1_loss.item():.3f} qf2={qf2_loss.item():.3f} | "
                    f"actor={actor_loss.item():.3f} | "
                    f"alpha={alpha:.4f} | "
                    f"SPS={int(global_step / (time.time() - start_time))}"
                )

        # Evaluation
        if global_step > 0 and global_step % args.eval_interval == 0:
            sr = evaluate(policy, env, args.eval_episodes, args.max_ep_len)
            print(f">>> step={global_step} | SR={sr:.1%}  (BC={bc_sr:.1%}, best={best_sr:.1%})")
            if sr > best_sr:
                best_sr = sr
                os.makedirs("rl_finetune", exist_ok=True)
                torch.save(
                    dict(
                        actor=actor.state_dict(),
                        qf1=qf1.state_dict(),
                        qf2=qf2.state_dict(),
                        step=global_step,
                        sr=sr,
                    ),
                    "rl_finetune/sac_best.pth",
                )
                print(f"  *** New best saved (SR={sr:.1%}) ***")

    print(f"\nFinal SR: {evaluate(policy, env, 50, args.max_ep_len):.1%}")
    print(f"BC baseline was: {bc_sr:.1%}")


if __name__ == "__main__":
    main()
