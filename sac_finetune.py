import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from collections import deque
import random
import logging
import os
from datetime import datetime
from tqdm import tqdm

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.env_utils as EnvUtils

# ─────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────
os.makedirs("rl_finetune/logs", exist_ok=True)
run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = f"rl_finetune/logs/sac_{run_name}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(),  # also print to terminal
    ]
)
log = logging.getLogger()


# ─────────────────────────────────────────────
# 1. GMM reparameterized sample
# ─────────────────────────────────────────────
def gmm_rsample_with_log_prob(dist):
    mixture     = dist.mixture_distribution
    components  = dist.component_distribution
    logits      = mixture.logits
    gumbel_w    = F.gumbel_softmax(logits, tau=1.0, hard=True)
    all_samples = components.rsample()
    sample      = (gumbel_w.unsqueeze(-1) * all_samples).sum(dim=-2)
    log_prob    = dist.log_prob(sample)
    return sample, log_prob


# ─────────────────────────────────────────────
# 2. Twin Q-network
# ─────────────────────────────────────────────
class TwinQNetwork(nn.Module):
    def __init__(self, obs_feat_dim, action_dim, hidden_dim=1024):
        super().__init__()
        in_dim = obs_feat_dim + action_dim
        self.q1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs_feat, action):
        x = torch.cat([obs_feat, action], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


# ─────────────────────────────────────────────
# 3. Replay buffer
# ─────────────────────────────────────────────
OBS_KEYS = ['agentview_image', 'robot0_eye_in_hand_image',
            'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']

class SequenceReplayBuffer:
    def __init__(self, capacity, seq_len, obs_keys):
        self.capacity    = capacity
        self.seq_len     = seq_len
        self.obs_keys    = obs_keys
        self.buffer      = deque(maxlen=capacity)
        self._current_ep = []

    def _filter_obs(self, obs):
        return {k: obs[k] for k in self.obs_keys}

    def add_step(self, obs, action, reward, next_obs, done):
        self._current_ep.append({
            'obs':      self._filter_obs(obs),
            'action':   action,
            'reward':   reward,
            'next_obs': self._filter_obs(next_obs),
            'done':     float(done),
        })

    def flush_episode(self):
        ep = self._current_ep
        for start in range(0, len(ep), self.seq_len // 2):
            seq = ep[start: start + self.seq_len]
            if len(seq) == self.seq_len:
                self.buffer.append(seq)
        self._current_ep = []

    def sample(self, batch_size, device):
        seqs = random.sample(self.buffer, batch_size)

        def stack_key(key, subkey=None):
            if subkey:
                vals = [[s[key][subkey] for s in seq] for seq in seqs]
            else:
                vals = [[s[key] for s in seq] for seq in seqs]
            return torch.tensor(np.array(vals, dtype=np.float32), device=device)

        obs_batch      = {k: stack_key('obs', k)      for k in self.obs_keys}
        next_obs_batch = {k: stack_key('next_obs', k) for k in self.obs_keys}
        actions        = stack_key('action')
        rewards        = stack_key('reward')
        dones          = stack_key('done')
        return obs_batch, actions, rewards, next_obs_batch, dones

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────
# 4. Encode obs
# ─────────────────────────────────────────────

def encode_obs(actor, obs_dict):
    """
    obs_dict values: (B, T, ...) 
    Conv2d can't handle the T dim, so fold it into B, encode, then unfold.
    """
    # Get B and T from any key
    sample_val = next(iter(obs_dict.values()))
    B, T = sample_val.shape[:2]

    # Fold T into B: (B, T, ...) -> (B*T, ...)
    flat_obs = {k: v.reshape(B * T, *v.shape[2:]) for k, v in obs_dict.items()}

    # Encode: (B*T, 137)
    feat = actor.nets["encoder"](obs=flat_obs)

    # Unfold back: (B*T, 137) -> (B, T, 137)
    return feat.reshape(B, T, -1)


# ─────────────────────────────────────────────
# 5. Actor forward with burn-in
# ─────────────────────────────────────────────
BURNIN_LEN = 5
LEARN_LEN  = 5

def actor_forward_with_burnin(actor, obs_dict):
    burnin_obs = {k: v[:, :BURNIN_LEN] for k, v in obs_dict.items()}
    learn_obs  = {k: v[:, BURNIN_LEN:] for k, v in obs_dict.items()}
    with torch.no_grad():
        _, rnn_state = actor.forward_train(
            burnin_obs, rnn_init_state=None, return_state=True
        )
    dist, _ = actor.forward_train(
        learn_obs, rnn_init_state=rnn_state, return_state=True
    )
    return dist


# ─────────────────────────────────────────────
# 6. SAC update
# ─────────────────────────────────────────────
def sac_update(actor, critic, critic_target, log_alpha,
               actor_optim, critic_optim, alpha_optim,
               batch, target_entropy, gamma=0.99, tau=0.005,
               bc_reg_weight=1.0):
    actor.train()  # always enforce — eval() elsewhere can leak into here
    critic.train()

    obs, actions, rewards, next_obs, dones = batch
    actions_learn = actions[:, BURNIN_LEN:]
    rewards_learn = rewards[:, BURNIN_LEN:]
    dones_learn   = dones[:, BURNIN_LEN:]
    alpha         = log_alpha.exp().detach()

    # Critic
    with torch.no_grad():
        next_dist               = actor_forward_with_burnin(actor, next_obs)
        next_sample, next_lp    = gmm_rsample_with_log_prob(next_dist)
        next_lp                 = next_lp.clamp(-20, 2)
        next_obs_learn          = {k: v[:, BURNIN_LEN:] for k, v in next_obs.items()}
        next_feat               = encode_obs(actor, next_obs_learn)
        q1_next, q2_next        = critic_target(next_feat, next_sample)
        q_next                  = torch.min(q1_next, q2_next) - alpha * next_lp
        q_target                = rewards_learn + gamma * (1.0 - dones_learn) * q_next

    obs_learn   = {k: v[:, BURNIN_LEN:] for k, v in obs.items()}
    obs_feat    = encode_obs(actor, obs_learn)
    q1, q2      = critic(obs_feat, actions_learn)
    critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

    critic_optim.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
    critic_optim.step()

    # Actor
    dist                   = actor_forward_with_burnin(actor, obs)
    sample, lp             = gmm_rsample_with_log_prob(dist)
    lp                     = lp.clamp(-20, 2)
    obs_feat_actor         = encode_obs(actor, obs_learn)
    q1_new, q2_new         = critic(obs_feat_actor, sample)
    actor_loss             = (alpha * lp - torch.min(q1_new, q2_new)).mean()

    actor_optim.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    actor_optim.step()

    # Alpha
    alpha_loss = -(log_alpha * (lp.detach() + target_entropy)).mean()
    alpha_optim.zero_grad()
    alpha_loss.backward()
    alpha_optim.step()

    # Soft target update
    for p, p_t in zip(critic.parameters(), critic_target.parameters()):
        p_t.data.copy_(tau * p.data + (1 - tau) * p_t.data)

    return {
        'critic_loss': critic_loss.item(),
        'actor_loss':  actor_loss.item(),
        'alpha':       log_alpha.exp().item(),
        'mean_q':      q1.mean().item(),
    }


# ─────────────────────────────────────────────
# 7. Evaluation
# ─────────────────────────────────────────────
def evaluate(policy, env, n_episodes=10):
    successes = 0
    actor = policy.policy.nets["policy"]
    actor.eval()
    for i in tqdm(range(n_episodes), desc="  Evaluating", leave=False):
        obs = env.reset()
        policy.start_episode()
        t = 0
        while t < 400:
            action = policy(obs)
            obs, _, _, _ = env.step(action)
            if env.is_success()["task"]:
                successes += 1
                break
            t += 1
    actor.train()
    return successes / n_episodes


# ─────────────────────────────────────────────
# 8. Main
# ─────────────────────────────────────────────
def main():
    device    = TorchUtils.get_torch_device(try_to_use_cuda=True)
    ckpt_path = "rl_finetune/model_epoch_220_Coffee_D1_success_0.9.pth"

    log.info(f"Logging to {log_path}")
    log.info("Loading policy...")

    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=ckpt_path, device=device, verbose=False
    )
    bc_algo = policy.policy
    actor   = bc_algo.nets["policy"]
    actor.train()

    env_meta = ckpt_dict["env_metadata"]
    env = EnvUtils.create_env_from_metadata(
        env_meta, render=False, render_offscreen=True, use_image_obs=True
    )

    # Critic
    critic        = TwinQNetwork(137, 7).to(device)
    critic_target = copy.deepcopy(critic).to(device)
    for p in critic_target.parameters():
        p.requires_grad = False

    # Optimizers
    actor_optim = torch.optim.Adam(actor.parameters(),  lr=3e-5)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=3e-4)
    log_alpha   = torch.tensor(-4.0, requires_grad=True, device=device)
    alpha_optim = torch.optim.Adam([log_alpha], lr=3e-4)
    target_entropy = -7.0

    replay = SequenceReplayBuffer(capacity=5000, seq_len=10, obs_keys=OBS_KEYS)

    # ── Seed buffer ──
    log.info("Seeding replay buffer with BC rollouts...")
    n_seed = 20
    with tqdm(total=n_seed, desc="Seeding") as pbar:
        for ep in range(n_seed):
            obs = env.reset()
            policy.start_episode()
            t = 0
            while t < 400:
                action = policy(obs)
                next_obs, _, done, _ = env.step(action)
                success = env.is_success()["task"]
                replay.add_step(obs, action, float(success), next_obs, done)
                obs = next_obs
                t += 1
                if success:
                    break
            replay.flush_episode()
            pbar.set_postfix(ep_len=t, buffer=len(replay))
            pbar.update(1)

    log.info(f"Buffer seeded with {len(replay)} sequences.")

    # ── BC baseline ──
    log.info("Evaluating BC baseline...")
    bc_sr = evaluate(policy, env, n_episodes=10)
    log.info(f"BC baseline success rate: {bc_sr:.1%}")

    # ── SAC loop ──
    TOTAL_STEPS   = 20_000
    UPDATE_EVERY  = 10
    BATCH_SIZE    = 16
    EVAL_INTERVAL = 2_000

    obs = env.reset()
    policy.start_episode()
    ep_reward = 0
    t = 0
    best_sr = bc_sr

    pbar = tqdm(total=TOTAL_STEPS, desc="SAC training")
    for step in range(TOTAL_STEPS):

        with torch.no_grad():
            action = policy(obs)

        next_obs, _, done, _ = env.step(action)
        success = env.is_success()["task"]
        reward  = float(success)
        replay.add_step(obs, action, reward, next_obs, done)
        ep_reward += reward
        obs = next_obs
        t  += 1

        if success or t >= 400:
            replay.flush_episode()
            obs = env.reset()
            policy.start_episode()
            ep_reward = 0
            t = 0

        # Update
        if len(replay) >= BATCH_SIZE and step % UPDATE_EVERY == 0:
            actor.train()
            batch = replay.sample(BATCH_SIZE, device)
            logs  = sac_update(
                actor, critic, critic_target, log_alpha,
                actor_optim, critic_optim, alpha_optim,
                batch, target_entropy
            )
            pbar.set_postfix(
                critic=f"{logs['critic_loss']:.3f}",
                actor=f"{logs['actor_loss']:.3f}",
                alpha=f"{logs['alpha']:.3f}",
                q=f"{logs['mean_q']:.3f}",
            )
            if step % 500 == 0:
                log.info(
                    f"Step {step:5d} | critic={logs['critic_loss']:.3f} "
                    f"actor={logs['actor_loss']:.3f} "
                    f"alpha={logs['alpha']:.3f} "
                    f"mean_q={logs['mean_q']:.3f} "
                    f"buffer={len(replay)}"
                )

        # Eval
        if step > 0 and step % EVAL_INTERVAL == 0:
            sr = evaluate(policy, env, n_episodes=10)
            log.info(f">>> Step {step:5d} | success rate: {sr:.1%} (BC was {bc_sr:.1%})")
            if sr > best_sr:
                best_sr = sr
                torch.save({
                    'actor':        actor.state_dict(),
                    'critic':       critic.state_dict(),
                    'step':         step,
                    'success_rate': sr,
                }, "rl_finetune/sac_best.pth")
                log.info(f"  *** New best model saved (sr={sr:.1%}) ***")

        pbar.update(1)

    pbar.close()

    # Final eval
    final_sr = evaluate(policy, env, n_episodes=20)
    log.info(f"\nFinal SAC success rate:  {final_sr:.1%}")
    log.info(f"BC baseline was:         {bc_sr:.1%}")
    log.info(f"Best during training:    {best_sr:.1%}")
    log.info(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()