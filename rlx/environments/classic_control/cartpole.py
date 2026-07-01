import math

import numpy as np
import gymnasium as gym
import mlx.core as mx

from rlx.environments.environment import Environment, EnvState


class CartPole(Environment):
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masspole + masscart
    length = 0.5
    polemass_length = masspole * length
    force_mag = 10.0
    tau = 0.02

    theta_threshold_radians = 12 * 2 * math.pi / 360
    x_threshold = 2.4

    def __init__(self, max_episode_steps: int = 500):
        self.max_episode_steps = max_episode_steps

        high = np.array(
            [
                self.x_threshold * 2,
                np.inf,
                self.theta_threshold_radians * 2,
                np.inf,
            ],
            dtype=np.float32,
        )
        self._observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self._action_space = gym.spaces.Discrete(2)

    def _observation(self, state: EnvState) -> mx.array:
        return mx.stack(
            [state["x"], state["x_dot"], state["theta"], state["theta_dot"]]
        )

    def reset(self, key: mx.array) -> tuple[mx.array, EnvState, dict]:
        init = mx.random.uniform(low=-0.05, high=0.05, shape=(4,), key=key)
        state: EnvState = {
            "x": init[0],
            "x_dot": init[1],
            "theta": init[2],
            "theta_dot": init[3],
            "time": mx.array(0, dtype=mx.int32),
        }
        return self._observation(state), state, {}

    def step_env(
        self, key: mx.array, state: EnvState, action: mx.array
    ) -> tuple[mx.array, EnvState, mx.array, mx.array, mx.array, dict]:
        x, x_dot = state["x"], state["x_dot"]
        theta, theta_dot = state["theta"], state["theta_dot"]

        force = mx.where(action == 1, self.force_mag, -self.force_mag)
        costheta = mx.cos(theta)
        sintheta = mx.sin(theta)

        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        next_state: EnvState = {
            "x": x + self.tau * x_dot,
            "x_dot": x_dot + self.tau * xacc,
            "theta": theta + self.tau * theta_dot,
            "theta_dot": theta_dot + self.tau * thetaacc,
            "time": state["time"] + 1,
        }

        terminated = (
            (next_state["x"] < -self.x_threshold)
            | (next_state["x"] > self.x_threshold)
            | (next_state["theta"] < -self.theta_threshold_radians)
            | (next_state["theta"] > self.theta_threshold_radians)
        )
        truncated = next_state["time"] >= self.max_episode_steps
        reward = mx.ones_like(next_state["x"])

        return self._observation(next_state), next_state, reward, terminated, truncated, {}

    @property
    def observation_space(self) -> gym.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        return self._action_space
