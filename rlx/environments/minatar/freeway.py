import numpy as np
import gymnasium as gym
import mlx.core as mx

from rlx.environments.environment import Environment, EnvState

PLAYER_SPEED = 3
TIME_LIMIT = 2500
CAR_Y = mx.arange(1, 9)
GRID = mx.arange(10)


class Freeway(Environment):
    def __init__(self):
        self._observation_space = gym.spaces.Box(0, 1, (10, 10, 7), dtype=np.float32)
        self._action_space = gym.spaces.Discrete(6)

    def _random_cars(self, key: mx.array):
        speed_key, direction_key = mx.random.split(key)
        speeds = mx.random.randint(1, 6, shape=(8,), key=speed_key)
        directions = mx.where(
            mx.random.uniform(shape=(8,), key=direction_key) < 0.5, -1, 1
        )
        car_speed = speeds * directions
        return car_speed, mx.abs(car_speed)

    def _observation(self, state: EnvState) -> mx.array:
        pos, car_x, car_speed = state["pos"], state["car_x"], state["car_speed"]

        chicken = (GRID.reshape(10, 1) == pos) & (GRID.reshape(1, 10) == 4)
        row_oh = GRID.reshape(1, 10, 1) == CAR_Y.reshape(8, 1, 1)
        car_oh = row_oh & (GRID.reshape(1, 1, 10) == car_x.reshape(8, 1, 1))
        car_channel = mx.max(car_oh, axis=0)

        back_x = mx.where(car_speed > 0, car_x - 1, car_x + 1)
        back_x = mx.where(back_x < 0, 9, back_x)
        back_x = mx.where(back_x > 9, 0, back_x)
        trail_oh = row_oh & (GRID.reshape(1, 1, 10) == back_x.reshape(8, 1, 1))
        speed_idx = mx.abs(car_speed)

        channels = [chicken, car_channel]
        for k in range(1, 6):
            channels.append(mx.max(trail_oh & (speed_idx == k).reshape(8, 1, 1), axis=0))
        return mx.stack(channels, axis=-1).astype(mx.float32)

    def reset(self, key: mx.array) -> tuple[mx.array, EnvState, dict]:
        car_speed, car_timer = self._random_cars(key)
        state: EnvState = {
            "car_x": mx.zeros((8,), dtype=mx.int32),
            "car_speed": car_speed,
            "car_timer": car_timer,
            "pos": mx.array(9),
            "move_timer": mx.array(PLAYER_SPEED),
            "terminate_timer": mx.array(TIME_LIMIT, dtype=mx.int32),
            "time": mx.array(0, dtype=mx.int32),
        }
        return self._observation(state), state, {}

    def step_env(
        self, key: mx.array, state: EnvState, action: mx.array
    ) -> tuple[mx.array, EnvState, mx.array, mx.array, mx.array, dict]:
        pos, move_timer = state["pos"], state["move_timer"]
        car_x, car_speed, car_timer = (
            state["car_x"],
            state["car_speed"],
            state["car_timer"],
        )
        terminate_timer = state["terminate_timer"]

        can_move = move_timer == 0
        up = (action == 2) & can_move
        down = (action == 4) & can_move
        pos = mx.where(up, mx.maximum(0, pos - 1), pos)
        pos = mx.where(down, mx.minimum(9, pos + 1), pos)
        move_timer = mx.where(up | down, PLAYER_SPEED, move_timer)

        won = pos == 0
        reward = won.astype(mx.float32)
        new_speed, new_timer = self._random_cars(key)
        car_speed = mx.where(won, new_speed, car_speed)
        car_timer = mx.where(won, new_timer, car_timer)
        pos = mx.where(won, 9, pos)

        move = car_timer == 0
        direction = mx.where(car_speed > 0, 1, -1)
        moved_x = car_x + direction
        moved_x = mx.where(moved_x < 0, 9, moved_x)
        moved_x = mx.where(moved_x > 9, 0, moved_x)
        new_car_x = mx.where(move, moved_x, car_x)
        new_car_timer = mx.where(move, mx.abs(car_speed), car_timer - 1)

        pre_collide = (car_x == 4) & (CAR_Y == pos)
        post_collide = move & (new_car_x == 4) & (CAR_Y == pos)
        collide = mx.any(pre_collide | post_collide)
        pos = mx.where(collide, 9, pos)

        move_timer = move_timer - (move_timer > 0).astype(move_timer.dtype)
        terminate_timer = terminate_timer - 1
        terminated = terminate_timer < 0
        truncated = mx.zeros_like(terminated)

        next_state: EnvState = {
            "car_x": new_car_x,
            "car_speed": car_speed,
            "car_timer": new_car_timer,
            "pos": pos,
            "move_timer": move_timer,
            "terminate_timer": terminate_timer,
            "time": state["time"] + 1,
        }
        return self._observation(next_state), next_state, reward, terminated, truncated, {}

    @property
    def observation_space(self) -> gym.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        return self._action_space
