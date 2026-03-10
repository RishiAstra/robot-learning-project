import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from collections import deque
import random

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.env_utils as EnvUtils

# ─────────────────────────────────────────────
# 1. GMM reparameterized sample helper
# ─────────────────────────────────────────────
def gmm_rsample_with_log_prob(dist):
    mixture    = dist.mixture_distribution
    components = dist.component_distribution
    logits     = mixture.logits                              # (B, T, 5)
    gumbel_w   = F.gumbel_softmax(logits, tau=1.0, hard=True)  # (B, T, 5)
    all_samples = components.rsample()                       # (B, T, 5, 7)
    sample     = (gumbel_w.unsqueeze(-1) * all_samples).sum(dim=-2)  # (B, T, 7)
    log_prob   = dist.log_prob(sample)                       # (B, T)
    return sample, log_prob


# ─────────────────────────────────────────────
# 2. Twin Q-network (critic)
#    Input: encoded obs features + action
#    The actor's encoder is shared — we pass
#    features directly, not raw obs
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
# 3. Replay buffer — stores sequences of length
#    SEQ_LEN so we can do burn-in
# ─────────────────────────────────────────────
OBS_KEYS = ['agentview_image', 'robot0_eye_in_hand_image',
            'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']

class SequenceReplayBuffer:
    def __init__(self, capacity, seq_len, obs_keys):
        self.capacity = capacity
        self.seq_len  = seq_len
        self.obs_keys = obs_keys
        self.buffer   = deque(maxlen=capacity)
        # Current episode being built
        self._current_ep = []

    def _filter_obs(self, obs):
        return {k: obs[k] for k in self.obs_keys}

    def add_step(self, obs, action, reward, next_obs, done):
        self._current_ep.append({
            'obs':      self._filter_obs(obs),
            'action':   action,
            'reward':   reward,
            'next_obs': self._filter_obs(next_obs),
            'done':     done,
        })
        if done:
            # Slice episode into overlapping sequences
            ep = self._current_ep
            for start in range(0, len(ep), self.seq_len // 2):
                seq = ep[start: start + self.seq_len]
                if len(seq) == self.seq_len:
                    self.buffer.append(seq)
            self._current_ep = []

    def sample(self, batch_size, device):
        seqs = random.sample(self.buffer, batch_size)
        # Stack into tensors: (B, T, ...)
        def stack_key(key, subkey=None):
            if subkey:
                vals = [[s[key][subkey] for s in seq] for seq in seqs]
            else:
                vals = [[s[key] for s in seq] for seq in seqs]
            arr = np.array(vals, dtype=np.float32)
            return torch.tensor(arr, device=device)

        obs_batch      = {k: stack_key('obs', k)      for k in self.obs_keys}
        next_obs_batch = {k: stack_key('next_obs', k) for k in self.obs_keys}
        actions  = stack_key('action')                   # (B, T, 7)
        rewards  = stack_key('reward')                   # (B, T)
        dones    = stack_key('done')                     # (B, T)
        return obs_batch, actions, rewards, next_obs_batch, dones

    def flush_episode(self):
        """Force-flush current episode even if done wasn't set."""
        ep = self._current_ep
        for start in range(0, len(ep), self.seq_len // 2):
            seq = ep[start: start + self.seq_len]
            if len(seq) == self.seq_len:
                self.buffer.append(seq)
        self._current_ep = []

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────
# 4. Helper: encode obs dict → flat feature
#    Uses the actor's trained encoder (shared)
# ─────────────────────────────────────────────
def encode_obs(actor, obs_dict):
    """obs_dict values shape: (B, T, ...) → returns (B, T, 137)"""
    return actor.nets["encoder"](obs=obs_dict)


# ─────────────────────────────────────────────
# 5. Helper: run actor with burn-in
#    Returns dist and rnn_state over the LEARN
#    half of the sequence only
# ─────────────────────────────────────────────
BURNIN_LEN = 5
LEARN_LEN  = 5

def actor_forward_with_burnin(actor, obs_dict):
    """
    obs_dict values: (B, T=10, ...)
    Returns dist over last LEARN_LEN steps and the rnn state.
    """
    # Split into burn-in and learn halves
    burnin_obs = {k: v[:, :BURNIN_LEN] for k, v in obs_dict.items()}
    learn_obs  = {k: v[:, BURNIN_LEN:] for k, v in obs_dict.items()}

    # Burn-in: no gradient
    with torch.no_grad():
        _, rnn_state = actor.forward_train(
            burnin_obs, rnn_init_state=None, return_state=True
        )

    # Learn half: gradient flows
    dist, _ = actor.forward_train(
        learn_obs, rnn_init_state=rnn_state, return_state=True
    )
    return dist   # MixtureSameFamily over (B, LEARN_LEN, 7)


# ─────────────────────────────────────────────
# 6. SAC update step
# ─────────────────────────────────────────────
def sac_update(actor, critic, critic_target, log_alpha,
               actor_optim, critic_optim, alpha_optim,
               batch, target_entropy, gamma=0.99, tau=0.005, device='cuda'):

    obs, actions, rewards, next_obs, dones = batch
    # All shapes: (B, T, ...) but we only learn on last LEARN_LEN steps
    # Slice rewards/dones/actions to learn half
    actions_learn = actions[:, BURNIN_LEN:]   # (B, 5, 7)
    rewards_learn = rewards[:, BURNIN_LEN:]   # (B, 5)
    dones_learn   = dones[:, BURNIN_LEN:]     # (B, 5)

    alpha = log_alpha.exp().detach()

    # ── Critic update ──────────────────────────────
    with torch.no_grad():
        next_dist = actor_forward_with_burnin(actor, next_obs)
        next_sample, next_lp = gmm_rsample_with_log_prob(next_dist)
        next_lp = next_lp.clamp(-20, 2)

        # Need features for critic — encode next_obs learn half
        next_obs_learn = {k: v[:, BURNIN_LEN:] for k, v in next_obs.items()}
        next_feat = encode_obs(actor, next_obs_learn)  # (B, 5, 137)

        q1_next, q2_next = critic_target(next_feat, next_sample)
        q_next   = torch.min(q1_next, q2_next) - alpha * next_lp
        q_target = rewards_learn + gamma * (1.0 - dones_learn) * q_next

    obs_learn = {k: v[:, BURNIN_LEN:] for k, v in obs.items()}
    obs_feat  = encode_obs(actor, obs_learn)   # (B, 5, 137)
    q1, q2    = critic(obs_feat, actions_learn)
    critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

    critic_optim.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
    critic_optim.step()

    # ── Actor update ───────────────────────────────
    dist = actor_forward_with_burnin(actor, obs)
    sample, lp = gmm_rsample_with_log_prob(dist)
    lp = lp.clamp(-20, 2)

    obs_feat_actor = encode_obs(actor, obs_learn)
    q1_new, q2_new = critic(obs_feat_actor, sample)
    actor_loss = (alpha * lp - torch.min(q1_new, q2_new)).mean()

    actor_optim.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    actor_optim.step()

    # ── Alpha update ───────────────────────────────
    alpha_loss = -(log_alpha * (lp.detach() + target_entropy)).mean()
    alpha_optim.zero_grad()
    alpha_loss.backward()
    alpha_optim.step()

    # ── Soft update target critic ──────────────────
    for p, p_t in zip(critic.parameters(), critic_target.parameters()):
        p_t.data.copy_(tau * p.data + (1 - tau) * p_t.data)

    return {
        'critic_loss': critic_loss.item(),
        'actor_loss':  actor_loss.item(),
        'alpha':       alpha.item(),
        'mean_q':      q1.mean().item(),
    }


# ─────────────────────────────────────────────
# 7. Evaluation
# ─────────────────────────────────────────────
def evaluate(policy, env, n_episodes=20):
    successes = 0
    for _ in range(n_episodes):
        obs = env.reset()
        policy.start_episode()
        done = False
        t = 0
        while not done and t < 400:
            action = policy(obs)
            obs, reward, done, info = env.step(action)
            if env.is_success()["task"]:
                successes += 1
                break
            t += 1
    return successes / n_episodes



# ─────────────────────────────────────────────
# 8. Main training loop
# ─────────────────────────────────────────────
def main():
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    ckpt_path = "rl_finetune/model_epoch_220_Coffee_D1_success_0.9.pth"

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

    # Critic setup
    OBS_FEAT_DIM  = 137
    ACTION_DIM    = 7
    critic        = TwinQNetwork(OBS_FEAT_DIM, ACTION_DIM).to(device)
    critic_target = copy.deepcopy(critic).to(device)
    for p in critic_target.parameters():
        p.requires_grad = False

    # Optimizers
    actor_optim  = torch.optim.Adam(actor.parameters(),  lr=3e-5)  # lower LR for fine-tuning
    critic_optim = torch.optim.Adam(critic.parameters(), lr=3e-4)
    log_alpha    = torch.tensor(0.0, requires_grad=True, device=device)
    alpha_optim  = torch.optim.Adam([log_alpha], lr=3e-4)

    target_entropy = -ACTION_DIM  # standard SAC heuristic: -dim(A)

    # Replay buffer
    replay = SequenceReplayBuffer(
        capacity=5000, seq_len=BURNIN_LEN + LEARN_LEN, obs_keys=OBS_KEYS
    )

    # ── Collect initial BC rollouts to seed the buffer ──
    print("Collecting BC rollouts to seed replay buffer...")
    n_seed_episodes = 20
    for ep in range(n_seed_episodes):
        obs = env.reset()
        policy.start_episode()
        done = False
        t = 0
        while not done and t < 400:
            action = policy(obs)
            next_obs, reward, done, info = env.step(action)
            
            success = env.is_success()["task"]  # returns {"task": True/False}
            reward = float(success)
            
            replay.add_step(obs, action, reward, next_obs, done)
            obs = next_obs
            t += 1

            if success:
                break  # stop the episode early on success

        replay.flush_episode()  # <-- force flush regardless of done
        print(f"  Seed episode {ep+1}/{n_seed_episodes}, ep_len={t}, buffer size: {len(replay)}")

    print(f"\nBuffer seeded with {len(replay)} sequences. Starting SAC fine-tuning...\n")

    # ── Evaluate BC baseline before any RL ──
    actor.eval()
    bc_success = evaluate(policy, env, n_episodes=20)
    actor.train()
    print(f"BC baseline success rate: {bc_success:.1%}\n")

    # ── SAC training loop ──
    total_steps   = 0
    update_every  = 10   # env steps between updates
    batch_size    = 16
    eval_interval = 1000

    obs = env.reset()
    policy.start_episode()
    ep_reward = 0

    for step in range(50_000):
        if step < 10 or step < 100 and step % 10 == 0:
            print(f"Starting step {step} / 50,000")

        # Act in environment
        with torch.no_grad():
            action = policy(obs)

        next_obs, reward, done, info = env.step(action)
        success = env.is_success()["task"]
        reward = float(success)
        replay.add_step(obs, action, reward, next_obs, done)
        ep_reward += reward
        obs = next_obs
        total_steps += 1
        t += 1  # actually increment t

        if success or t >= 400:
            replay.flush_episode()
            print(f"  Episode done, ep_reward={ep_reward:.3f}, buffer={len(replay)}")
            obs = env.reset()
            policy.start_episode()
            ep_reward = 0
            t = 0


        # Update networks
        if len(replay) >= batch_size and step % update_every == 0:
            batch = replay.sample(batch_size, device)
            logs  = sac_update(
                actor, critic, critic_target, log_alpha,
                actor_optim, critic_optim, alpha_optim,
                batch, target_entropy, device=device
            )

            if step % 100 == 0:
                print(f"Step {step:5d} | critic={logs['critic_loss']:.3f} "
                      f"actor={logs['actor_loss']:.3f} "
                      f"alpha={logs['alpha']:.3f} "
                      f"mean_q={logs['mean_q']:.3f}")

        # Evaluate
        if step > 0 and step % eval_interval == 0:
            actor.eval()
            sr = evaluate(policy, env, n_episodes=20)
            actor.train()
            print(f"\n>>> Step {step} eval success rate: {sr:.1%}\n")

            # Save checkpoint
            torch.save({
                'actor':  actor.state_dict(),
                'critic': critic.state_dict(),
                'step':   step,
                'success_rate': sr,
            }, f"rl_finetune/sac_checkpoint_step{step}.pth")

    # Final eval
    actor.eval()
    final_sr = evaluate(policy, env, n_episodes=50)
    print(f"\nFinal SAC success rate: {final_sr:.1%}")
    print(f"BC baseline was:        {bc_success:.1%}")


if __name__ == "__main__":
    main()