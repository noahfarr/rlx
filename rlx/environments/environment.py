from abc import ABC, abstractmethod

import gymnasium as gym
import mlx.core as mx
from mlx.utils import tree_map

EnvState = dict[str, mx.array]


class Environment(ABC):
    @abstractmethod
    def reset(self, key: mx.array) -> tuple[mx.array, EnvState, dict]: ...

    @abstractmethod
    def step_env(
        self, key: mx.array, state: EnvState, action: mx.array
    ) -> tuple[mx.array, EnvState, mx.array, mx.array, mx.array, dict]: ...

    def step(
        self, key: mx.array, state: EnvState, action: mx.array
    ) -> tuple[mx.array, EnvState, mx.array, mx.array, mx.array, dict]:
        step_key, reset_key = mx.random.split(key)

        observation, state, reward, terminated, truncated, info = self.step_env(
            step_key, state, action
        )
        done = mx.logical_or(terminated, truncated)

        reset_observation, reset_state, _ = self.reset(reset_key)
        state = tree_map(lambda r, s: mx.where(done, r, s), reset_state, state)
        observation = mx.where(done, reset_observation, observation)

        return observation, state, reward, terminated, truncated, info

    @property
    @abstractmethod
    def observation_space(self) -> gym.Space: ...

    @property
    @abstractmethod
    def action_space(self) -> gym.Space: ...
