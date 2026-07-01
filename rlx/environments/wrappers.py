import mlx.core as mx

from rlx.environments.environment import Environment, EnvState


class RecordEpisodeStatistics(Environment):
    RETURN_KEY = "episode_return"
    LENGTH_KEY = "episode_length"

    def __init__(self, env: Environment):
        self.env = env

    def reset(self, key: mx.array) -> tuple[mx.array, EnvState, dict]:
        observation, state, info = self.env.reset(key)
        state = {
            **state,
            self.RETURN_KEY: mx.array(0.0),
            self.LENGTH_KEY: mx.array(0, dtype=mx.int32),
        }
        return observation, state, info

    def step_env(
        self, key: mx.array, state: EnvState, action: mx.array
    ) -> tuple[mx.array, EnvState, mx.array, mx.array, mx.array, dict]:
        inner_state = {
            key_: value
            for key_, value in state.items()
            if key_ not in (self.RETURN_KEY, self.LENGTH_KEY)
        }
        observation, inner_state, reward, terminated, truncated, info = (
            self.env.step_env(key, inner_state, action)
        )

        episode_return = state[self.RETURN_KEY] + reward
        episode_length = state[self.LENGTH_KEY] + 1
        done = mx.logical_or(terminated, truncated)

        info = {
            **info,
            "episode": {"r": episode_return, "l": episode_length},
            "_episode": done,
        }
        state = {
            **inner_state,
            self.RETURN_KEY: episode_return,
            self.LENGTH_KEY: episode_length,
        }
        return observation, state, reward, terminated, truncated, info

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space


class Vectorize:
    def __init__(self, env: Environment):
        self.env = env
        self._reset = mx.vmap(env.reset)
        self._step = mx.vmap(env.step)

    def reset(self, key: mx.array) -> tuple[mx.array, EnvState, dict]:
        return self._reset(key)

    def step(
        self, key: mx.array, state: EnvState, action: mx.array
    ) -> tuple[mx.array, EnvState, mx.array, mx.array, mx.array, dict]:
        return self._step(key, state, action)

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space
