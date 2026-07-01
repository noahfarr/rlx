import numpy as np
import gymnasium as gym
import mlx.core as mx

from rlx.environments.environment import Environment, EnvState

SHOT_COOL_DOWN = 5
ENEMY_MOVE_INTERVAL = 12
ENEMY_SHOT_INTERVAL = 10

ROW = mx.arange(10).reshape(10, 1)
COL = mx.arange(10).reshape(1, 10)
GRID = mx.arange(10)

WAVE = (
    (mx.arange(10).reshape(10, 1) < 4) & (mx.arange(10).reshape(1, 10) >= 2)
) & (mx.arange(10).reshape(1, 10) < 8)
WAVE = mx.broadcast_to(WAVE, (10, 10)).astype(mx.float32)


class SpaceInvaders(Environment):
    def __init__(self, ramping: bool = True):
        self.ramping = ramping
        self._observation_space = gym.spaces.Box(0, 1, (10, 10, 6), dtype=np.float32)
        self._action_space = gym.spaces.Discrete(6)

    def _observation(self, state: EnvState) -> mx.array:
        alien = state["alien_map"] != 0
        alien_dir = state["alien_dir"]
        return mx.stack(
            [
                (ROW == 9) & (COL == state["pos"]),
                alien,
                alien & (alien_dir < 0),
                alien & (alien_dir > 0),
                state["f_bullet_map"] != 0,
                state["e_bullet_map"] != 0,
            ],
            axis=-1,
        ).astype(mx.float32)

    def reset(self, key: mx.array) -> tuple[mx.array, EnvState, dict]:
        state: EnvState = {
            "pos": mx.array(5),
            "f_bullet_map": mx.zeros((10, 10)),
            "e_bullet_map": mx.zeros((10, 10)),
            "alien_map": WAVE,
            "alien_dir": mx.array(-1),
            "enemy_move_interval": mx.array(ENEMY_MOVE_INTERVAL),
            "alien_move_timer": mx.array(ENEMY_MOVE_INTERVAL),
            "alien_shot_timer": mx.array(ENEMY_SHOT_INTERVAL),
            "ramp_index": mx.array(0),
            "shot_timer": mx.array(0),
            "time": mx.array(0, dtype=mx.int32),
        }
        return self._observation(state), state, {}

    def step_env(
        self, key: mx.array, state: EnvState, action: mx.array
    ) -> tuple[mx.array, EnvState, mx.array, mx.array, mx.array, dict]:
        pos = state["pos"]
        shot_timer = state["shot_timer"]
        f_bullet_map = state["f_bullet_map"]
        e_bullet_map = state["e_bullet_map"]
        alien_map = state["alien_map"]
        alien_dir = state["alien_dir"]
        enemy_move_interval = state["enemy_move_interval"]
        alien_move_timer = state["alien_move_timer"]
        alien_shot_timer = state["alien_shot_timer"]
        ramp_index = state["ramp_index"]

        fire = (action == 5) & (shot_timer == 0)
        f_bullet_map = mx.where(fire & (ROW == 9) & (COL == pos), 1.0, f_bullet_map)
        shot_timer = mx.where(fire, SHOT_COOL_DOWN, shot_timer)
        pos = mx.where(action == 1, mx.maximum(0, pos - 1), pos)
        pos = mx.where(action == 3, mx.minimum(9, pos + 1), pos)

        f_bullet_map = mx.roll(f_bullet_map, -1, axis=0)
        f_bullet_map = mx.where(ROW == 9, 0.0, f_bullet_map)

        e_bullet_map = mx.roll(e_bullet_map, 1, axis=0)
        e_bullet_map = mx.where(ROW == 0, 0.0, e_bullet_map)
        terminated = e_bullet_map[9, pos] != 0
        terminated = terminated | (alien_map[9, pos] != 0)

        move = alien_move_timer == 0
        count = mx.sum(alien_map)
        at_left = mx.sum(alien_map[:, 0]) > 0
        at_right = mx.sum(alien_map[:, 9]) > 0
        edge = (at_left & (alien_dir < 0)) | (at_right & (alien_dir > 0))
        rolled_down = mx.roll(alien_map, 1, axis=0)
        rolled_horizontal = mx.roll(alien_map, alien_dir, axis=1)
        moved_map = mx.where(edge, rolled_down, rolled_horizontal)
        new_alien_map = mx.where(move, moved_map, alien_map)
        alien_dir = mx.where(move & edge, -alien_dir, alien_dir)
        terminated = terminated | (move & edge & (mx.sum(alien_map[9, :]) > 0))
        terminated = terminated | (move & (new_alien_map[9, pos] != 0))
        alien_map = new_alien_map
        alien_move_timer = mx.where(
            move, mx.minimum(count, enemy_move_interval), alien_move_timer
        )

        do_shot = alien_shot_timer == 0
        has_alien_col = mx.sum(alien_map, axis=0) > 0
        score = mx.where(has_alien_col, mx.abs(GRID - pos) * 10 + GRID, 1000)
        target_col = mx.argmin(score)
        column = alien_map[:, target_col]
        target_row = mx.max(mx.where(column > 0, GRID, -1))
        e_bullet_map = mx.where(
            do_shot & (ROW == target_row) & (COL == target_col), 1.0, e_bullet_map
        )
        alien_shot_timer = mx.where(do_shot, ENEMY_SHOT_INTERVAL, alien_shot_timer)

        kill = (alien_map != 0) & (alien_map == f_bullet_map)
        reward = mx.sum(kill).astype(mx.float32)
        alien_map = mx.where(kill, 0.0, alien_map)
        f_bullet_map = mx.where(kill, 0.0, f_bullet_map)

        shot_timer = shot_timer - (shot_timer > 0).astype(shot_timer.dtype)
        alien_move_timer = alien_move_timer - 1
        alien_shot_timer = alien_shot_timer - 1

        cleared = mx.sum(alien_map) == 0
        ramp = cleared & (enemy_move_interval > 6) & self.ramping
        enemy_move_interval = mx.where(ramp, enemy_move_interval - 1, enemy_move_interval)
        ramp_index = mx.where(ramp, ramp_index + 1, ramp_index)
        alien_map = mx.where(cleared, WAVE, alien_map)

        truncated = mx.zeros_like(terminated)
        next_state: EnvState = {
            "pos": pos,
            "f_bullet_map": f_bullet_map,
            "e_bullet_map": e_bullet_map,
            "alien_map": alien_map,
            "alien_dir": alien_dir,
            "enemy_move_interval": enemy_move_interval,
            "alien_move_timer": alien_move_timer,
            "alien_shot_timer": alien_shot_timer,
            "ramp_index": ramp_index,
            "shot_timer": shot_timer,
            "time": state["time"] + 1,
        }
        return self._observation(next_state), next_state, reward, terminated, truncated, {}

    @property
    def observation_space(self) -> gym.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        return self._action_space
