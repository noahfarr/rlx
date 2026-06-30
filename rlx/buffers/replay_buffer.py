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
        num_envs: int = 1,
    ):
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.position = 0
        self.full = False

        action_dtype = (
            np.int32 if isinstance(action_space, gym.spaces.Discrete) else np.float32
        )

        self.observations = np.zeros(
            (buffer_size, num_envs, *observation_space.shape), dtype=np.float32
        )
        self.actions = np.zeros(
            (buffer_size, num_envs, *action_space.shape), dtype=action_dtype
        )
        self.next_observations = np.zeros(
            (buffer_size, num_envs, *observation_space.shape), dtype=np.float32
        )
        self.rewards = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.terminations = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.truncations = np.zeros((buffer_size, num_envs), dtype=np.float32)

    def add(
        self,
        observation: np.ndarray,
        next_observation: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
    ):
        self.observations[self.position] = np.array(observation).copy()
        self.next_observations[self.position] = np.array(next_observation).copy()
        self.actions[self.position] = np.array(action).copy()
        self.rewards[self.position] = np.array(reward).copy()
        self.truncations[self.position] = np.array(truncated).copy()
        self.terminations[self.position] = np.array(terminated).copy()

        self.position += 1

        if self.position == self.buffer_size:
            self.full = True
            self.position = 0

    def sample(self, batch_size: int) -> Batch:
        upper_bound = self.buffer_size if self.full else self.position
        batch_indices = np.random.randint(0, upper_bound, size=batch_size)

        env_indices = np.random.randint(0, self.num_envs, size=batch_size)

        batch_observations = self.observations[batch_indices, env_indices]
        batch_actions = self.actions[batch_indices, env_indices]
        batch_next_observations = self.next_observations[batch_indices, env_indices]
        batch_rewards = self.rewards[batch_indices, env_indices]
        batch_terminations = self.terminations[batch_indices, env_indices]
        batch_truncations = self.truncations[batch_indices, env_indices]

        return Batch(
            observations=mx.array(batch_observations),
            actions=mx.array(batch_actions),
            next_observations=mx.array(batch_next_observations),
            rewards=mx.array(batch_rewards.reshape(-1, 1)),
            terminations=mx.array(batch_terminations.reshape(-1, 1)),
            truncations=mx.array(batch_truncations.reshape(-1, 1)),
        )
