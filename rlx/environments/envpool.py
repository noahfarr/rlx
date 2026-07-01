import warnings

import numpy as np
import envpool

import mlx.core as mx


class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.var = m2 / total_count
        self.count = total_count


class EnvPool:
    def __init__(
        self,
        env_id,
        num_envs,
        seed=1,
        num_threads=1,
        normalize_observations=False,
        normalize_rewards=False,
        gamma=0.99,
        epsilon=1e-8,
        clip=10.0,
    ):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
            self.envs = envpool.make(
                env_id,
                env_type="gymnasium",
                num_envs=num_envs,
                seed=seed,
                num_threads=num_threads,
            )
            self._observation_space = self.envs.observation_space
            self._action_space = self.envs.action_space
        self.num_envs = num_envs

        self.normalize_observations = normalize_observations
        self.normalize_rewards = normalize_rewards
        self.gamma = gamma
        self.epsilon = epsilon
        self.clip = clip

        self.discounted_returns = np.zeros(num_envs, dtype=np.float64)
        self.episode_returns = np.zeros(num_envs, dtype=np.float64)
        self.episode_lengths = np.zeros(num_envs, dtype=np.int64)
        self.observation_rms = RunningMeanStd(self._observation_space.shape)
        self.return_rms = RunningMeanStd(())

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def normalize_observation(self, observation):
        if not self.normalize_observations:
            return observation
        self.observation_rms.update(observation)
        normalized = (observation - self.observation_rms.mean) / np.sqrt(
            self.observation_rms.var + self.epsilon
        )
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    def reset(self, key=None):
        observation, info = self.envs.reset()
        self.discounted_returns[:] = 0
        self.episode_returns[:] = 0
        self.episode_lengths[:] = 0
        return mx.array(self.normalize_observation(observation)), {}, {}

    def step(self, key, state, action):
        observation, reward, terminated, truncated, info = self.envs.step(
            np.asarray(action)
        )
        observation = self.normalize_observation(observation)
        reward = reward.astype(np.float64)

        done = np.logical_or(terminated, truncated)

        self.episode_returns += reward
        self.episode_lengths += 1
        info = {
            "episode": {
                "r": self.episode_returns.copy(),
                "l": self.episode_lengths.copy(),
            },
            "_episode": done,
        }
        self.episode_returns[done] = 0.0
        self.episode_lengths[done] = 0

        if self.normalize_rewards:
            self.discounted_returns = self.discounted_returns * self.gamma + reward
            self.return_rms.update(self.discounted_returns)
            reward = reward / np.sqrt(self.return_rms.var + self.epsilon)
            reward = np.clip(reward, -self.clip, self.clip)
            self.discounted_returns[done] = 0.0

        return (
            mx.array(observation),
            {},
            mx.array(reward.astype(np.float32)),
            mx.array(terminated),
            mx.array(truncated),
            info,
        )

    def close(self):
        self.envs.close()
