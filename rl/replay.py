from __future__ import annotations

import random
from collections import deque
from typing import List, Optional, Sequence

import h5py
import numpy as np
import torch

from rl.common import filter_obs


class SequenceReplayBuffer:
    def __init__(self, capacity: int, seq_len: int, obs_keys: Sequence[str]):
        self.capacity = capacity
        self.seq_len = seq_len
        self.obs_keys = list(obs_keys)
        # deque with maxlen gives O(1) append-and-evict vs O(n) list.pop(0)
        self.storage: deque = deque(maxlen=capacity)
        self._current_episode: List[dict] = []

    def add_step(self, obs, action, reward, next_obs, done):
        self._current_episode.append(
            {
                "obs": filter_obs(obs, self.obs_keys),
                "action": np.asarray(action, dtype=np.float32),
                "reward": float(reward),
                "next_obs": filter_obs(next_obs, self.obs_keys),
                "done": float(done),
            }
        )

    def add_sequence(self, sequence: List[dict]):
        if len(sequence) != self.seq_len:
            return
        self.storage.append(sequence)

    def flush_episode(self, stride: Optional[int] = None):
        # Default stride = seq_len so sequences align with RNN reset boundaries.
        # Using seq_len // 2 would produce sequences that span resets, giving the
        # burnin the wrong RNN context for half the training data.
        stride = stride if stride is not None else self.seq_len
        episode = self._current_episode
        for start in range(0, max(1, len(episode) - self.seq_len + 1), stride):
            seq = episode[start : start + self.seq_len]
            if len(seq) == self.seq_len:
                self.add_sequence(seq)
        self._current_episode = []

    def sample(self, batch_size: int, device: torch.device):
        sequences = random.sample(list(self.storage), batch_size)

        def stack_scalar(key: str):
            values = [[step[key] for step in seq] for seq in sequences]
            return torch.tensor(np.asarray(values, dtype=np.float32), device=device)

        def stack_obs(obs_key: str, prefix: str):
            values = [[step[prefix][obs_key] for step in seq] for seq in sequences]
            return torch.tensor(np.asarray(values, dtype=np.float32), device=device)

        obs = {k: stack_obs(k, "obs") for k in self.obs_keys}
        next_obs = {k: stack_obs(k, "next_obs") for k in self.obs_keys}
        actions = stack_scalar("action")
        rewards = stack_scalar("reward")
        dones = stack_scalar("done")
        return obs, actions, rewards, next_obs, dones

    def __len__(self):
        return len(self.storage)


def load_demo_sequences(
    dataset_path: str,
    replay: SequenceReplayBuffer,
    obs_keys: Sequence[str],
    max_demos: Optional[int] = None,
) -> None:
    with h5py.File(dataset_path, "r") as dataset:
        demo_keys = sorted(dataset["data"].keys())
        if max_demos is not None:
            demo_keys = demo_keys[:max_demos]

        for demo_key in demo_keys:
            demo = dataset["data"][demo_key]
            obs = {k: demo["obs"][k][()] for k in obs_keys}
            next_obs_group = demo.get("next_obs")
            next_obs = {k: next_obs_group[k][()] for k in obs_keys} if next_obs_group is not None else None
            actions = demo["actions"][()]
            rewards = demo["rewards"][()] if "rewards" in demo else None
            dones = demo["dones"][()] if "dones" in demo else None
            length = actions.shape[0]
            steps = []
            for t in range(length):
                next_t = min(t + 1, length - 1)
                terminal = float(dones[t]) if dones is not None else float(t == (length - 1))
                if t == (length - 1):
                    terminal = 1.0
                steps.append(
                    {
                        "obs": {k: obs[k][t] for k in obs_keys},
                        "action": np.asarray(actions[t], dtype=np.float32),
                        "reward": float(rewards[t]) if rewards is not None else float(t == (length - 1)),
                        "next_obs": {
                            k: (next_obs[k][t] if next_obs is not None else obs[k][next_t])
                            for k in obs_keys
                        },
                        "done": terminal,
                    }
                )

            stride = max(1, replay.seq_len // 2)
            for start in range(0, max(1, len(steps) - replay.seq_len + 1), stride):
                seq = steps[start : start + replay.seq_len]
                if len(seq) == replay.seq_len:
                    replay.add_sequence(seq)


def cat_batches(batch_a, batch_b):
    obs_a, actions_a, rewards_a, next_obs_a, dones_a = batch_a
    obs_b, actions_b, rewards_b, next_obs_b, dones_b = batch_b
    obs = {k: torch.cat([obs_a[k], obs_b[k]], dim=0) for k in obs_a}
    next_obs = {k: torch.cat([next_obs_a[k], next_obs_b[k]], dim=0) for k in next_obs_a}
    return (
        obs,
        torch.cat([actions_a, actions_b], dim=0),
        torch.cat([rewards_a, rewards_b], dim=0),
        next_obs,
        torch.cat([dones_a, dones_b], dim=0),
    )
