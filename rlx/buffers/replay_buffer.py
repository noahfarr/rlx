import gymnasium as gym
import numpy as np
import mlx.core as mx
from typing import Any, NamedTuple


class Batch(NamedTuple):
    observations: mx.array
    actions: mx.array
    next_observations: mx.array
    rewards: mx.array
    terminations: mx.array
    truncations: mx.array


class ReplayBuffer:

    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        n_envs: int = 1,
    ):
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.pos = 0
        self.full = False

        action_dtype = (
            np.int32 if isinstance(action_space, gym.spaces.Discrete) else np.float32
        )

        self.observations = np.zeros(
            (buffer_size, n_envs, *observation_space.shape), dtype=np.float32
        )
        self.actions = np.zeros(
            (buffer_size, n_envs, *action_space.shape), dtype=action_dtype
        )
        self.next_observations = np.zeros(
            (buffer_size, n_envs, *observation_space.shape), dtype=np.float32
        )
        self.rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.terminations = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.truncations = np.zeros((buffer_size, n_envs), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
    ):
        self.observations[self.pos] = np.array(obs).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.truncations[self.pos] = np.array(truncated).copy()
        self.terminations[self.pos] = np.array(terminated).copy()

        self.pos += 1

        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> Batch:
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)

        env_indices = np.random.randint(0, self.n_envs, size=batch_size)

        b_obs = self.observations[batch_inds, env_indices]
        b_actions = self.actions[batch_inds, env_indices]
        b_next_obs = self.next_observations[batch_inds, env_indices]
        b_rewards = self.rewards[batch_inds, env_indices]
        b_terminations = self.terminations[batch_inds, env_indices]
        b_truncations = self.truncations[batch_inds, env_indices]

        return Batch(
            observations=mx.array(b_obs),
            actions=mx.array(b_actions),
            next_observations=mx.array(b_next_obs),
            rewards=mx.array(b_rewards.reshape(-1, 1)),
            terminations=mx.array(b_terminations.reshape(-1, 1)),
            truncations=mx.array(b_truncations.reshape(-1, 1)),
        )
