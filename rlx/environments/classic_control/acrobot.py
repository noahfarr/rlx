import math

import numpy as np
import gymnasium as gym
import mlx.core as mx

from rlx.environments.environment import Environment, EnvState


class Acrobot(Environment):
    dt = 0.2
    link_length_1 = 1.0
    link_mass_1 = 1.0
    link_mass_2 = 1.0
    link_com_pos_1 = 0.5
    link_com_pos_2 = 0.5
    link_moi = 1.0
    max_vel_1 = 4 * math.pi
    max_vel_2 = 9 * math.pi
    gravity = 9.8

    def __init__(self, max_episode_steps: int = 500):
        self.max_episode_steps = max_episode_steps

        high = np.array(
            [1.0, 1.0, 1.0, 1.0, self.max_vel_1, self.max_vel_2], dtype=np.float32
        )
        self._observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self._action_space = gym.spaces.Discrete(3)

    def _observation(self, state: EnvState) -> mx.array:
        theta1, theta2 = state["theta1"], state["theta2"]
        return mx.stack(
            [
                mx.cos(theta1),
                mx.sin(theta1),
                mx.cos(theta2),
                mx.sin(theta2),
                state["dtheta1"],
                state["dtheta2"],
            ]
        )

    def reset(self, key: mx.array) -> tuple[mx.array, EnvState]:
        init = mx.random.uniform(low=-0.1, high=0.1, shape=(4,), key=key)
        state: EnvState = {
            "theta1": init[0],
            "theta2": init[1],
            "dtheta1": init[2],
            "dtheta2": init[3],
            "time": mx.array(0, dtype=mx.int32),
        }
        return self._observation(state), state

    def _dsdt(self, s: mx.array, torque: mx.array) -> mx.array:
        theta1, theta2, dtheta1, dtheta2 = s[0], s[1], s[2], s[3]
        m1, m2 = self.link_mass_1, self.link_mass_2
        l1 = self.link_length_1
        lc1, lc2 = self.link_com_pos_1, self.link_com_pos_2
        i1 = i2 = self.link_moi
        g = self.gravity

        d1 = (
            m1 * lc1**2
            + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * mx.cos(theta2))
            + i1
            + i2
        )
        d2 = m2 * (lc2**2 + l1 * lc2 * mx.cos(theta2)) + i2
        phi2 = m2 * lc2 * g * mx.cos(theta1 + theta2 - math.pi / 2)
        phi1 = (
            -m2 * l1 * lc2 * dtheta2**2 * mx.sin(theta2)
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * mx.sin(theta2)
            + (m1 * lc1 + m2 * l1) * g * mx.cos(theta1 - math.pi / 2)
            + phi2
        )
        ddtheta2 = (
            torque + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1**2 * mx.sin(theta2) - phi2
        ) / (m2 * lc2**2 + i2 - d2**2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return mx.stack([dtheta1, dtheta2, ddtheta1, ddtheta2])

    def step_env(
        self, key: mx.array, state: EnvState, action: mx.array
    ) -> tuple[mx.array, EnvState, mx.array, mx.array, mx.array, dict]:
        torque = (action - 1).astype(mx.float32)
        s = mx.stack(
            [state["theta1"], state["theta2"], state["dtheta1"], state["dtheta2"]]
        )

        k1 = self._dsdt(s, torque)
        k2 = self._dsdt(s + self.dt / 2 * k1, torque)
        k3 = self._dsdt(s + self.dt / 2 * k2, torque)
        k4 = self._dsdt(s + self.dt * k3, torque)
        s = s + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        theta1 = mx.arctan2(mx.sin(s[0]), mx.cos(s[0]))
        theta2 = mx.arctan2(mx.sin(s[1]), mx.cos(s[1]))
        dtheta1 = mx.clip(s[2], -self.max_vel_1, self.max_vel_1)
        dtheta2 = mx.clip(s[3], -self.max_vel_2, self.max_vel_2)

        next_state: EnvState = {
            "theta1": theta1,
            "theta2": theta2,
            "dtheta1": dtheta1,
            "dtheta2": dtheta2,
            "time": state["time"] + 1,
        }
        terminated = (-mx.cos(theta1) - mx.cos(theta2 + theta1)) > 1.0
        truncated = next_state["time"] >= self.max_episode_steps
        reward = mx.where(terminated, 0.0, -1.0)

        return self._observation(next_state), next_state, reward, terminated, truncated, {}

    @property
    def observation_space(self) -> gym.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        return self._action_space
