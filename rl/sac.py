from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.actor import actor_forward_with_burnin, encode_obs_sequence
from rl.common import gmm_rsample_with_log_prob


class TwinQNetwork(nn.Module):
    def __init__(self, obs_feat_dim: int, action_dim: int, hidden_dim: int = 512, layer_norm: bool = False):
        super().__init__()
        in_dim = obs_feat_dim + action_dim
        self.q1 = self._build_q(in_dim, hidden_dim, layer_norm)
        self.q2 = self._build_q(in_dim, hidden_dim, layer_norm)

    @staticmethod
    def _build_q(in_dim: int, hidden_dim: int, layer_norm: bool) -> nn.Sequential:
        # RLPD-style: LayerNorm after each Linear and before activation keeps
        # Q-values from exploding when mixing demo and online data.
        layers: list = [nn.Linear(in_dim, hidden_dim)]
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))
        return nn.Sequential(*layers)

    def forward(self, obs_feat: torch.Tensor, action: torch.Tensor):
        x = torch.cat([obs_feat, action], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


def sac_update(
    actor,
    actor_target,
    critic,
    critic_target,
    batch,
    actor_optimizer,
    critic_optimizer,
    log_alpha,
    alpha_optimizer,
    target_entropy,
    burnin_len: int,
    gamma: float,
    tau: float,
    bc_weight: float = 0.0,
    bc_batch=None,
):
    actor.train()
    critic.train()

    obs, actions, rewards, next_obs, dones = batch
    actions_learn = actions[:, burnin_len:]
    rewards_learn = rewards[:, burnin_len:]
    dones_learn = dones[:, burnin_len:]
    alpha = log_alpha.exp().detach()
    target_actor = actor_target if actor_target is not None else actor

    with torch.no_grad():
        next_dist = actor_forward_with_burnin(target_actor, next_obs, burnin_len)
        next_action, next_logp = gmm_rsample_with_log_prob(next_dist)
        next_logp = next_logp.clamp(-20, 2)
        next_obs_learn = {k: v[:, burnin_len:] for k, v in next_obs.items()}
        next_feat = encode_obs_sequence(target_actor, next_obs_learn, detach=True)
        q1_next, q2_next = critic_target(next_feat, next_action)
        q_target = rewards_learn + gamma * (1.0 - dones_learn) * (torch.min(q1_next, q2_next) - alpha * next_logp)

    obs_learn = {k: v[:, burnin_len:] for k, v in obs.items()}
    # Encode once with gradients; detach a view for the critic to avoid
    # pulling the encoder in two directions simultaneously.
    obs_feat_actor = encode_obs_sequence(actor, obs_learn, detach=False)
    obs_feat = obs_feat_actor.detach()
    q1, q2 = critic(obs_feat, actions_learn)
    critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
    critic_optimizer.step()

    dist = actor_forward_with_burnin(actor, obs, burnin_len)
    action_sample, logp = gmm_rsample_with_log_prob(dist)
    logp = logp.clamp(-20, 2)
    # Reuse the already-computed actor features (with gradients intact)
    q1_pi, q2_pi = critic(obs_feat_actor, action_sample)
    rl_loss = (alpha * logp - torch.min(q1_pi, q2_pi)).mean()

    bc_loss = torch.tensor(0.0, device=rl_loss.device)
    if bc_weight > 0.0 and bc_batch is not None:
        bc_obs, bc_actions, _, _, _ = bc_batch
        bc_dist = actor_forward_with_burnin(actor, bc_obs, burnin_len)
        bc_actions_learn = bc_actions[:, burnin_len:]
        bc_loss = -bc_dist.log_prob(bc_actions_learn).mean()

    actor_loss = rl_loss + bc_weight * bc_loss
    actor_optimizer.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    actor_optimizer.step()

    alpha_loss = -(log_alpha * (logp.detach() + target_entropy)).mean()
    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    alpha_optimizer.step()

    for p, p_t in zip(critic.parameters(), critic_target.parameters()):
        p_t.data.copy_(tau * p.data + (1.0 - tau) * p_t.data)

    if actor_target is not None:
        for p, p_t in zip(actor.parameters(), actor_target.parameters()):
            p_t.data.copy_(tau * p.data + (1.0 - tau) * p_t.data)

    return {
        "critic_loss": float(critic_loss.item()),
        "actor_loss": float(actor_loss.item()),
        "rl_loss": float(rl_loss.item()),
        "bc_loss": float(bc_loss.item()),
        "bc_weight": float(bc_weight),
        "alpha": float(log_alpha.exp().item()),
        "mean_q": float(q1.mean().item()),
    }
