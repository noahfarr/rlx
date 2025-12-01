from typing import NamedTuple, Optional, Generator
import gymnasium as gym
import mlx.core as mx


class Batch(NamedTuple):
    observations: mx.array
    actions: mx.array
    next_observations: mx.array
    rewards: mx.array
    terminations: mx.array
    truncations: mx.array


class RolloutBuffer:
    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.observation_space = observation_space
        self.action_space = action_space
        self.gamma = gamma

        if isinstance(action_space, gym.spaces.Discrete):
            self.action_dim = 1
            self.action_dtype = mx.int32
        else:
            self.action_dim = action_space.shape[0]
            self.action_dtype = mx.float32

        self.reset()

    def reset(self):
        self.observations = mx.zeros(
            (self.buffer_size, self.n_envs, *self.observation_space.shape),
            dtype=mx.float32,
        )
        self.next_observations = mx.zeros(
            (self.buffer_size, self.n_envs, *self.observation_space.shape),
            dtype=mx.float32,
        )
        self.actions = mx.zeros(
            (self.buffer_size, self.n_envs, *self.action_space.shape),
            dtype=self.action_dtype,
        )
        self.rewards = mx.zeros((self.buffer_size, self.n_envs), dtype=mx.float32)
        self.returns = mx.zeros((self.buffer_size, self.n_envs), dtype=mx.float32)
        self.terminations = mx.zeros((self.buffer_size, self.n_envs), dtype=mx.float32)
        self.truncations = mx.zeros((self.buffer_size, self.n_envs), dtype=mx.float32)
        self.values = mx.zeros((self.buffer_size, self.n_envs), dtype=mx.float32)
        self.log_probs = mx.zeros((self.buffer_size, self.n_envs), dtype=mx.float32)
        self.advantages = mx.zeros((self.buffer_size, self.n_envs), dtype=mx.float32)
        self.pos = 0
        self.full = False

    def add(
        self,
        obs: mx.array,
        next_obs: mx.array,
        action: mx.array,
        reward: mx.array,
        terminated: mx.array,
        truncated: mx.array,
        value: Optional[mx.array] = None,
        log_prob: Optional[mx.array] = None,
    ):
        self.observations[self.pos] = mx.array(obs)
        self.next_observations[self.pos] = mx.array(next_obs)
        self.actions[self.pos] = mx.array(action)
        self.rewards[self.pos] = mx.array(reward)
        self.terminations[self.pos] = mx.array(terminated)
        self.truncations[self.pos] = mx.array(truncated)
        if value is not None:
            self.values[self.pos] = value.flatten()
        if log_prob is not None:
            self.log_probs[self.pos] = log_prob.flatten()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
