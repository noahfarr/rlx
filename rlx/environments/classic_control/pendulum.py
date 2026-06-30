import math

import numpy as np
import gymnasium as gym
import mlx.core as mx

from rlx.environments.environment import Environment, EnvState


class Pendulum(Environment):
    max_speed = 8.0
    max_torque = 2.0
    dt = 0.05
    gravity = 10.0
    mass = 1.0
    length = 1.0

    def __init__(self, max_episode_steps: int = 200):
        self.max_episode_steps = max_episode_steps

        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self._observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self._action_space = gym.spaces.Box(
            -self.max_torque, self.max_torque, shape=(1,), dtype=np.float32
        )

    def _observation(self, state: EnvState) -> mx.array:
        theta = state["theta"]
        return mx.stack([mx.cos(theta), mx.sin(theta), state["theta_dot"]])

    def reset(self, key: mx.array) -> tuple[mx.array, EnvState]:
        sample = mx.random.uniform(shape=(2,), key=key)
        state: EnvState = {
            "theta": sample[0] * (2.0 * math.pi) - math.pi,
            "theta_dot": sample[1] * 2.0 - 1.0,
            "time": mx.array(0, dtype=mx.int32),
        }
        return self._observation(state), state

    def step_env(
        self, key: mx.array, state: EnvState, action: mx.array
    ) -> tuple[mx.array, EnvState, mx.array, mx.array, mx.array, dict]:
        theta, theta_dot = state["theta"], state["theta_dot"]
        torque = mx.clip(action, -self.max_torque, self.max_torque).reshape(())

        normalized = mx.arctan2(mx.sin(theta), mx.cos(theta))
        cost = normalized**2 + 0.1 * theta_dot**2 + 0.001 * torque**2

        new_theta_dot = theta_dot + (
            3.0 * self.gravity / (2.0 * self.length) * mx.sin(theta)
            + 3.0 / (self.mass * self.length**2) * torque
        ) * self.dt
        new_theta_dot = mx.clip(new_theta_dot, -self.max_speed, self.max_speed)
        new_theta = theta + new_theta_dot * self.dt

        next_state: EnvState = {
            "theta": new_theta,
            "theta_dot": new_theta_dot,
            "time": state["time"] + 1,
        }
        truncated = next_state["time"] >= self.max_episode_steps
        terminated = mx.zeros_like(truncated)

        return self._observation(next_state), next_state, -cost, terminated, truncated, {}

    @property
    def observation_space(self) -> gym.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        return self._action_space
