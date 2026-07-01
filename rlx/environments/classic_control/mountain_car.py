import numpy as np
import gymnasium as gym
import mlx.core as mx

from rlx.environments.environment import Environment, EnvState


class MountainCar(Environment):
    min_position = -1.2
    max_position = 0.6
    max_speed = 0.07
    goal_position = 0.5
    goal_velocity = 0.0
    force = 0.001
    gravity = 0.0025

    def __init__(self, max_episode_steps: int = 200):
        self.max_episode_steps = max_episode_steps

        low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        high = np.array([self.max_position, self.max_speed], dtype=np.float32)
        self._observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        self._action_space = gym.spaces.Discrete(3)

    def _observation(self, state: EnvState) -> mx.array:
        return mx.stack([state["position"], state["velocity"]])

    def reset(self, key: mx.array) -> tuple[mx.array, EnvState, dict]:
        position = mx.random.uniform(low=-0.6, high=-0.4, key=key)
        state: EnvState = {
            "position": position,
            "velocity": mx.zeros_like(position),
            "time": mx.array(0, dtype=mx.int32),
        }
        return self._observation(state), state, {}

    def step_env(
        self, key: mx.array, state: EnvState, action: mx.array
    ) -> tuple[mx.array, EnvState, mx.array, mx.array, mx.array, dict]:
        position, velocity = state["position"], state["velocity"]

        velocity = velocity + (action - 1) * self.force + mx.cos(3.0 * position) * (
            -self.gravity
        )
        velocity = mx.clip(velocity, -self.max_speed, self.max_speed)
        position = position + velocity
        position = mx.clip(position, self.min_position, self.max_position)
        velocity = mx.where(
            (position <= self.min_position) & (velocity < 0), 0.0, velocity
        )

        next_state: EnvState = {
            "position": position,
            "velocity": velocity,
            "time": state["time"] + 1,
        }
        terminated = (position >= self.goal_position) & (velocity >= self.goal_velocity)
        truncated = next_state["time"] >= self.max_episode_steps
        reward = -mx.ones_like(position)

        return self._observation(next_state), next_state, reward, terminated, truncated, {}

    @property
    def observation_space(self) -> gym.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        return self._action_space
