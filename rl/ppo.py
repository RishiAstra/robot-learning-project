from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.actor import encode_obs_sequence


class ValueHead(nn.Module):
    def __init__(self, obs_feat_dim: int, hidden_dim: int = 256, layer_norm: bool = False):
        super().__init__()
        layers: list = [nn.Linear(obs_feat_dim, hidden_dim)]
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs_feat: torch.Tensor):
        return self.net(obs_feat).squeeze(-1)


class RolloutEpisodeBuffer:
    def __init__(self):
        self.episodes: List[List[dict]] = []
        self._current_episode: List[dict] = []

    def add_step(self, obs, action, reward, done, log_prob, value):
        self._current_episode.append(
            {
                "obs": obs,
                "action": np.asarray(action, dtype=np.float32),
                "reward": float(reward),
                "done": float(done),
                "log_prob": float(log_prob),
                "value": float(value),
            }
        )

    def flush_episode(self):
        if self._current_episode:
            self.episodes.append(self._current_episode)
            self._current_episode = []

    def clear(self):
        self.episodes = []
        self._current_episode = []

    def num_steps(self) -> int:
        return sum(len(ep) for ep in self.episodes) + len(self._current_episode)

    def __len__(self):
        return len(self.episodes)


def _pad_episode_field(episodes: Sequence[Sequence[dict]], field: str, device: torch.device):
    lengths = [len(ep) for ep in episodes]
    max_len = max(lengths)
    first_value = episodes[0][0][field]
    value_shape = np.asarray(first_value).shape
    batch = np.zeros((len(episodes), max_len) + value_shape, dtype=np.float32)
    for batch_idx, episode in enumerate(episodes):
        for t, step in enumerate(episode):
            batch[(batch_idx, t)] = np.asarray(step[field], dtype=np.float32)
    return torch.tensor(batch, device=device), torch.tensor(lengths, device=device)


def _pad_episode_obs(episodes: Sequence[Sequence[dict]], obs_keys: Sequence[str], device: torch.device):
    lengths = [len(ep) for ep in episodes]
    max_len = max(lengths)
    sample_obs = episodes[0][0]["obs"]
    obs_batch = {}
    for key in obs_keys:
        obs_shape = np.asarray(sample_obs[key]).shape
        batch = np.zeros((len(episodes), max_len) + obs_shape, dtype=np.float32)
        for batch_idx, episode in enumerate(episodes):
            for t, step in enumerate(episode):
                batch[(batch_idx, t)] = np.asarray(step["obs"][key], dtype=np.float32)
        obs_batch[key] = torch.tensor(batch, device=device)
    return obs_batch, torch.tensor(lengths, device=device)


def _mask_from_lengths(lengths: torch.Tensor, max_len: int):
    return (torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)).float()


def _sequence_log_probs(actor, obs, actions, lengths, rnn_horizon: int):
    batch_size, max_len = actions.shape[:2]
    log_probs = torch.zeros((batch_size, max_len), device=actions.device, dtype=torch.float32)

    for start in range(0, max_len, rnn_horizon):
        end = min(start + rnn_horizon, max_len)
        chunk_obs = {k: v[:, start:end] for k, v in obs.items()}
        chunk_actions = actions[:, start:end]
        chunk_dist = actor.forward_train(chunk_obs, rnn_init_state=None, return_state=False)
        chunk_log_probs = chunk_dist.log_prob(chunk_actions)
        valid = (lengths > start).float().unsqueeze(1)
        log_probs[:, start:end] = chunk_log_probs * valid

    return log_probs


def _sequence_sampled_entropy(actor, obs, lengths, rnn_horizon: int):
    sample_value = next(iter(obs.values()))
    batch_size, max_len = sample_value.shape[:2]
    entropy = torch.zeros((batch_size, max_len), device=sample_value.device, dtype=torch.float32)

    for start in range(0, max_len, rnn_horizon):
        end = min(start + rnn_horizon, max_len)
        chunk_obs = {k: v[:, start:end] for k, v in obs.items()}
        chunk_dist = actor.forward_train(chunk_obs, rnn_init_state=None, return_state=False)
        with torch.no_grad():
            sampled_actions = chunk_dist.sample()
        chunk_entropy = -chunk_dist.log_prob(sampled_actions)
        valid = (lengths > start).float().unsqueeze(1)
        entropy[:, start:end] = chunk_entropy * valid

    return entropy


def _compute_gae(episodes: Sequence[Sequence[dict]], gamma: float, gae_lambda: float):
    advantages = []
    returns = []
    for episode in episodes:
        rewards = np.asarray([step["reward"] for step in episode], dtype=np.float32)
        values = np.asarray([step["value"] for step in episode], dtype=np.float32)
        dones = np.asarray([step["done"] for step in episode], dtype=np.float32)

        adv = np.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(len(episode))):
            next_value = 0.0 if t == len(episode) - 1 else values[t + 1]
            next_nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
            last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
            adv[t] = last_gae
        advantages.append(adv)
        returns.append(adv + values)
    return advantages, returns


def prepare_ppo_batch(
    episodes: Sequence[Sequence[dict]],
    obs_keys: Sequence[str],
    device: torch.device,
    gamma: float,
    gae_lambda: float,
):
    obs_batch, lengths = _pad_episode_obs(episodes, obs_keys, device)
    actions, _ = _pad_episode_field(episodes, "action", device)
    rewards, _ = _pad_episode_field(episodes, "reward", device)
    dones, _ = _pad_episode_field(episodes, "done", device)
    old_log_probs, _ = _pad_episode_field(episodes, "log_prob", device)
    values, _ = _pad_episode_field(episodes, "value", device)

    advantages_np, returns_np = _compute_gae(episodes, gamma, gae_lambda)
    max_len = actions.shape[1]
    advantages = np.zeros((len(episodes), max_len), dtype=np.float32)
    returns = np.zeros((len(episodes), max_len), dtype=np.float32)
    for idx, (adv, ret) in enumerate(zip(advantages_np, returns_np)):
        advantages[idx, : len(adv)] = adv
        returns[idx, : len(ret)] = ret

    advantages = torch.tensor(advantages, device=device)
    returns = torch.tensor(returns, device=device)
    mask = _mask_from_lengths(lengths, max_len)

    valid_adv = advantages[mask.bool()]
    advantages = (advantages - valid_adv.mean()) / (valid_adv.std(unbiased=False) + 1e-8)
    advantages = advantages * mask

    return {
        "obs": obs_batch,
        "actions": actions,
        "old_log_probs": old_log_probs,
        "values": values,
        "advantages": advantages,
        "returns": returns,
        "mask": mask,
        "lengths": lengths,
    }


def ppo_update(
    actor,
    value_net: nn.Module,
    episodes: Sequence[Sequence[dict]],
    obs_keys: Sequence[str],
    device: torch.device,
    actor_optimizer,
    value_optimizer,
    gamma: float,
    gae_lambda: float,
    clip_coef: float,
    value_clip_coef: float,
    value_coef: float,
    entropy_coef: float,
    critic_output_l2_weight: float,
    max_grad_norm: float,
    update_epochs: int,
    demo_batch=None,
    bc_weight: float = 0.0,
    bc_decay: float = 1.0,
    env_step: int = 0,
    rnn_horizon: int = 10,
):
    actor.train()
    value_net.train()

    batch = prepare_ppo_batch(episodes, obs_keys, device, gamma, gae_lambda)
    obs = batch["obs"]
    actions = batch["actions"]
    old_log_probs = batch["old_log_probs"]
    old_values = batch["values"]
    advantages = batch["advantages"]
    returns = batch["returns"]
    mask = batch["mask"]
    valid_count = mask.sum().clamp_min(1.0)
    bc_weight = bc_weight * (bc_decay ** env_step) if bc_weight > 0.0 else 0.0

    stats = {}
    for _ in range(update_epochs):
        new_log_probs = _sequence_log_probs(actor, obs, actions, batch["lengths"], rnn_horizon=rnn_horizon)
        log_ratio = new_log_probs - old_log_probs
        ratio = log_ratio.exp()
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * advantages
        policy_loss = -(torch.minimum(unclipped, clipped) * mask).sum() / valid_count

        obs_feat = encode_obs_sequence(actor, obs, detach=True)
        value_pred = value_net(obs_feat).view_as(old_values)
        value_pred_clipped = old_values + torch.clamp(value_pred - old_values, -value_clip_coef, value_clip_coef)
        value_loss_unclipped = (value_pred - returns).pow(2)
        value_loss_clipped = (value_pred_clipped - returns).pow(2)
        raw_value_loss = 0.5 * (torch.maximum(value_loss_unclipped, value_loss_clipped) * mask).sum() / valid_count
        value_output_l2 = ((value_pred.pow(2) * mask).sum() / valid_count)
        value_loss = value_coef * raw_value_loss + critic_output_l2_weight * value_output_l2

        if entropy_coef != 0.0:
            entropy_values = _sequence_sampled_entropy(actor, obs, batch["lengths"], rnn_horizon=rnn_horizon)
            entropy = (entropy_values * mask).sum() / valid_count
        else:
            entropy = torch.tensor(0.0, device=device)
        total_actor_loss = policy_loss - entropy_coef * entropy

        bc_loss = torch.tensor(0.0, device=device)
        if bc_weight > 0.0 and demo_batch is not None:
            bc_obs, bc_actions, _, _, _ = demo_batch
            demo_lengths = torch.full(
                (bc_actions.shape[0],),
                bc_actions.shape[1],
                dtype=torch.long,
                device=device,
            )
            bc_log_prob = _sequence_log_probs(actor, bc_obs, bc_actions, demo_lengths, rnn_horizon=rnn_horizon)
            bc_mask = torch.ones_like(bc_log_prob)
            bc_loss = -(bc_log_prob * bc_mask).sum() / bc_mask.sum()
            total_actor_loss = total_actor_loss + bc_weight * bc_loss

        actor_optimizer.zero_grad()
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
        actor_optimizer.step()

        value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)
        value_optimizer.step()

        stats = {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(raw_value_loss.item()),
            "value_output_l2": float(value_output_l2.item()),
            "entropy_loss": float(entropy.item()),
            "bc_loss": float(bc_loss.item()),
            "approx_kl": float((old_log_probs - new_log_probs).masked_select(mask.bool()).mean().item()),
            "clip_fraction": float((torch.abs(ratio - 1.0) > clip_coef).masked_select(mask.bool()).float().mean().item()),
            "mean_value": float(value_pred.masked_select(mask.bool()).mean().item()),
        }

    stats["bc_weight"] = float(bc_weight)
    return stats
