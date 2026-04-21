from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from rl.common import filter_obs


def encode_obs_sequence(actor, obs_dict: Dict[str, torch.Tensor], detach: bool = False):
    sample_value = next(iter(obs_dict.values()))
    batch_size, timesteps = sample_value.shape[:2]
    flat_obs = {k: v.reshape(batch_size * timesteps, *v.shape[2:]) for k, v in obs_dict.items()}
    feat = actor.nets["encoder"](obs=flat_obs)
    feat = feat.reshape(batch_size, timesteps, -1)
    if detach:
        feat = feat.detach()
    return feat


def actor_forward_with_burnin(actor, obs_dict, burnin_len: int):
    burnin_obs = {k: v[:, :burnin_len] for k, v in obs_dict.items()}
    learn_obs = {k: v[:, burnin_len:] for k, v in obs_dict.items()}
    with torch.no_grad():
        _, rnn_state = actor.forward_train(
            burnin_obs,
            rnn_init_state=None,
            return_state=True,
        )
    dist, _ = actor.forward_train(
        learn_obs,
        rnn_init_state=rnn_state,
        return_state=True,
    )
    return dist


def sample_actor_step(actor, obs_dict, rnn_state=None, deterministic: bool = False):
    actor_input = {
        k: torch.tensor(v, dtype=torch.float32, device=next(actor.parameters()).device).unsqueeze(0)
        for k, v in obs_dict.items()
    }
    was_training = actor.training
    actor.eval() if deterministic else actor.train()
    dist, rnn_state = actor.forward_train_step(actor_input, rnn_state=rnn_state)
    if deterministic:
        action = dist.component_distribution.base_dist.loc[
            torch.arange(dist.mixture_distribution.logits.shape[0]),
            dist.mixture_distribution.logits.argmax(dim=-1),
        ]
    else:
        action = dist.sample()
    if was_training:
        actor.train()
    else:
        actor.eval()
    return action.squeeze(0).detach().cpu().numpy(), rnn_state


def sample_actor_step_with_log_prob(actor, obs_dict, rnn_state=None, deterministic: bool = False):
    actor_input = {
        k: torch.tensor(v, dtype=torch.float32, device=next(actor.parameters()).device).unsqueeze(0)
        for k, v in obs_dict.items()
    }
    was_training = actor.training
    actor.train()
    dist, rnn_state = actor.forward_train_step(actor_input, rnn_state=rnn_state)
    if deterministic:
        action = dist.component_distribution.base_dist.loc[
            torch.arange(dist.mixture_distribution.logits.shape[0], device=next(actor.parameters()).device),
            dist.mixture_distribution.logits.argmax(dim=-1),
        ]
    else:
        action = dist.sample()
    log_prob = dist.log_prob(action)
    if was_training:
        actor.train()
    else:
        actor.eval()
    return action.squeeze(0).detach().cpu().numpy(), float(log_prob.squeeze(0).item()), rnn_state


def infer_feature_dim(actor, obs_keys, env, device):
    obs = filter_obs(env.reset(), obs_keys)
    tensor_obs = {
        k: torch.tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
        for k, v in obs.items()
    }
    return actor.nets["encoder"](obs=tensor_obs).shape[-1]
