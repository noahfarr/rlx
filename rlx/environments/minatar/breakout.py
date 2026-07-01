import numpy as np
import gymnasium as gym
import mlx.core as mx

from rlx.environments.environment import Environment, EnvState

HORIZONTAL_FLIP = mx.array([1, 0, 3, 2])
VERTICAL_FLIP = mx.array([3, 2, 1, 0])
PADDLE_EDGE_FLIP = mx.array([2, 3, 0, 1])

ROW = mx.arange(10).reshape(10, 1)
COL = mx.arange(10).reshape(1, 10)

BRICK_REFILL = (mx.arange(10).reshape(10, 1) >= 1) & (
    mx.arange(10).reshape(10, 1) <= 3
)
BRICK_REFILL = mx.broadcast_to(BRICK_REFILL, (10, 10)).astype(mx.float32)


class Breakout(Environment):
    def __init__(self):
        self._observation_space = gym.spaces.Box(0, 1, (10, 10, 4), dtype=np.float32)
        self._action_space = gym.spaces.Discrete(6)

    def _observation(self, state: EnvState) -> mx.array:
        ball = (ROW == state["ball_y"]) & (COL == state["ball_x"])
        paddle = (ROW == 9) & (COL == state["pos"])
        trail = (ROW == state["last_y"]) & (COL == state["last_x"])
        brick = state["brick_map"] != 0
        return mx.stack([paddle, ball, trail, brick], axis=-1).astype(mx.float32)

    def reset(self, key: mx.array) -> tuple[mx.array, EnvState, dict]:
        start = mx.random.randint(0, 2, key=key)
        ball_x = mx.where(start == 0, 0, 9)
        ball_dir = mx.where(start == 0, 2, 3)
        brick_map = BRICK_REFILL
        state: EnvState = {
            "ball_x": ball_x,
            "ball_y": mx.array(3),
            "ball_dir": ball_dir,
            "pos": mx.array(4),
            "brick_map": brick_map,
            "strike": mx.array(False),
            "last_x": ball_x,
            "last_y": mx.array(3),
            "time": mx.array(0, dtype=mx.int32),
        }
        return self._observation(state), state, {}

    def step_env(
        self, key: mx.array, state: EnvState, action: mx.array
    ) -> tuple[mx.array, EnvState, mx.array, mx.array, mx.array, dict]:
        ball_x, ball_y, ball_dir = state["ball_x"], state["ball_y"], state["ball_dir"]
        pos, brick_map, strike = state["pos"], state["brick_map"], state["strike"]

        pos = mx.where(action == 1, mx.maximum(0, pos - 1), pos)
        pos = mx.where(action == 3, mx.minimum(9, pos + 1), pos)

        last_x, last_y = ball_x, ball_y
        dx = mx.where((ball_dir == 1) | (ball_dir == 2), 1, -1)
        dy = mx.where((ball_dir == 0) | (ball_dir == 1), -1, 1)
        new_x = ball_x + dx
        new_y = ball_y + dy

        x_oob = (new_x < 0) | (new_x > 9)
        new_x = mx.clip(new_x, 0, 9)
        ball_dir = mx.where(x_oob, HORIZONTAL_FLIP[ball_dir], ball_dir)

        cond_top = new_y < 0
        brick_y = mx.clip(new_y, 0, 9)
        brick_here = brick_map[brick_y, new_x] == 1
        cond_brick = (~cond_top) & brick_here
        cond_bottom = (~cond_top) & (~brick_here) & (new_y == 9)

        ball_dir = mx.where(cond_top, VERTICAL_FLIP[ball_dir], ball_dir)

        effective_brick = cond_brick & (~strike)
        reward = effective_brick.astype(mx.float32)
        ball_dir = mx.where(effective_brick, VERTICAL_FLIP[ball_dir], ball_dir)
        brick_cell = (ROW == brick_y) & (COL == new_x)
        brick_map = mx.where(effective_brick & brick_cell, 0.0, brick_map)

        empty = mx.sum(brick_map) == 0
        brick_map = mx.where(cond_bottom & empty, BRICK_REFILL, brick_map)
        paddle_center = cond_bottom & (last_x == pos)
        paddle_edge = cond_bottom & (~(last_x == pos)) & (new_x == pos)
        terminal_hit = cond_bottom & (~(last_x == pos)) & (~(new_x == pos))
        ball_dir = mx.where(paddle_center, VERTICAL_FLIP[ball_dir], ball_dir)
        ball_dir = mx.where(paddle_edge, PADDLE_EDGE_FLIP[ball_dir], ball_dir)

        new_y = mx.where(cond_top, 0, new_y)
        new_y = mx.where(effective_brick | paddle_center | paddle_edge, last_y, new_y)

        terminated = state["time"] < 0
        terminated = terminated | terminal_hit
        truncated = mx.zeros_like(terminated)

        next_state: EnvState = {
            "ball_x": new_x,
            "ball_y": new_y,
            "ball_dir": ball_dir,
            "pos": pos,
            "brick_map": brick_map,
            "strike": cond_brick,
            "last_x": last_x,
            "last_y": last_y,
            "time": state["time"] + 1,
        }
        return self._observation(next_state), next_state, reward, terminated, truncated, {}

    @property
    def observation_space(self) -> gym.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        return self._action_space
