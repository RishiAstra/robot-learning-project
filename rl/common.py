from __future__ import annotations

import random
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def filter_obs(obs: Dict[str, np.ndarray], obs_keys: Sequence[str]) -> Dict[str, np.ndarray]:
    return {k: obs[k] for k in obs_keys}


def gmm_rsample_with_log_prob(dist):
    logits = dist.mixture_distribution.logits
    gumbel_w = F.gumbel_softmax(logits, tau=1.0, hard=True)
    all_samples = dist.component_distribution.rsample()
    sample = (gumbel_w.unsqueeze(-1) * all_samples).sum(dim=-2)
    log_prob = dist.log_prob(sample)
    return sample, log_prob

