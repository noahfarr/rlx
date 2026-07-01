import numpy as np
import gymnasium as gym
import mlx.core as mx

from rlx.environments.environment import Environment, EnvState

RAMP_INTERVAL = 100
INIT_SPAWN_SPEED = 10
INIT_MOVE_INTERVAL = 5

ROW = mx.arange(10).reshape(10, 1)
COL = mx.arange(10).reshape(1, 10)
GRID = mx.arange(10)
ENTITY_Y = mx.arange(1, 9)
SLOTS = mx.arange(8)


class Asterix(Environment):
    def __init__(self, ramping: bool = True):
        self.ramping = ramping
        self._observation_space = gym.spaces.Box(0, 1, (10, 10, 4), dtype=np.float32)
        self._action_space = gym.spaces.Discrete(6)

    def _observation(self, state: EnvState) -> mx.array:
        active = state["active"]
        x = state["entity_x"]
        gold = state["entity_gold"]
        lr = state["entity_lr"]

        player = (ROW == state["player_y"]) * (COL == state["player_x"])
        row_oh = GRID.reshape(1, 10, 1) == ENTITY_Y.reshape(8, 1, 1)
        entity_oh = (
            row_oh
            * (GRID.reshape(1, 1, 10) == x.reshape(8, 1, 1))
            * active.reshape(8, 1, 1)
        )
        enemy = mx.max(entity_oh * (~gold).reshape(8, 1, 1), axis=0)
        gold_channel = mx.max(entity_oh * gold.reshape(8, 1, 1), axis=0)

        back_x = mx.where(lr, x - 1, x + 1)
        in_bounds = (back_x >= 0) * (back_x <= 9)
        trail_oh = (
            row_oh
            * (GRID.reshape(1, 1, 10) == mx.clip(back_x, 0, 9).reshape(8, 1, 1))
            * (active * in_bounds).reshape(8, 1, 1)
        )
        trail = mx.max(trail_oh, axis=0)
        return mx.stack([player, enemy, trail, gold_channel], axis=-1).astype(mx.float32)

    def reset(self, key: mx.array) -> tuple[mx.array, EnvState, dict]:
        state: EnvState = {
            "player_x": mx.array(5),
            "player_y": mx.array(5),
            "active": mx.zeros((8,), dtype=mx.bool_),
            "entity_x": mx.zeros((8,), dtype=mx.int32),
            "entity_lr": mx.zeros((8,), dtype=mx.bool_),
            "entity_gold": mx.zeros((8,), dtype=mx.bool_),
            "spawn_speed": mx.array(INIT_SPAWN_SPEED),
            "spawn_timer": mx.array(INIT_SPAWN_SPEED),
            "move_speed": mx.array(INIT_MOVE_INTERVAL),
            "move_timer": mx.array(INIT_MOVE_INTERVAL),
            "ramp_timer": mx.array(RAMP_INTERVAL),
            "ramp_index": mx.array(0),
            "time": mx.array(0, dtype=mx.int32),
        }
        return self._observation(state), state, {}

    def step_env(
        self, key: mx.array, state: EnvState, action: mx.array
    ) -> tuple[mx.array, EnvState, mx.array, mx.array, mx.array, dict]:
        player_x, player_y = state["player_x"], state["player_y"]
        active, entity_x = state["active"], state["entity_x"]
        entity_lr, entity_gold = state["entity_lr"], state["entity_gold"]
        spawn_speed, spawn_timer = state["spawn_speed"], state["spawn_timer"]
        move_speed, move_timer = state["move_speed"], state["move_timer"]
        ramp_timer, ramp_index = state["ramp_timer"], state["ramp_index"]

        spawn = spawn_timer == 0
        lr_key, gold_key, slot_key = mx.random.split(key, 3)
        lr_new = mx.random.uniform(key=lr_key) < 0.5
        gold_new = mx.random.uniform(key=gold_key) < (1.0 / 3.0)
        empty = ~active
        priority = mx.where(empty, mx.random.uniform(shape=(8,), key=slot_key), -1.0)
        slot = mx.argmax(priority)
        do_spawn = spawn & mx.any(empty) & (SLOTS == slot)
        active = mx.where(do_spawn, True, active)
        entity_x = mx.where(do_spawn, mx.where(lr_new, 0, 9), entity_x)
        entity_lr = mx.where(do_spawn, lr_new, entity_lr)
        entity_gold = mx.where(do_spawn, gold_new, entity_gold)
        spawn_timer = mx.where(spawn, spawn_speed, spawn_timer)

        player_x = mx.where(action == 1, mx.maximum(0, player_x - 1), player_x)
        player_x = mx.where(action == 3, mx.minimum(9, player_x + 1), player_x)
        player_y = mx.where(action == 2, mx.maximum(1, player_y - 1), player_y)
        player_y = mx.where(action == 4, mx.minimum(8, player_y + 1), player_y)

        hit_pre = active & (entity_x == player_x) & (ENTITY_Y == player_y)
        reward = mx.sum(hit_pre & entity_gold).astype(mx.float32)
        terminated = mx.any(hit_pre & (~entity_gold))
        active = active & (~(hit_pre & entity_gold))

        move = move_timer == 0
        direction = mx.where(entity_lr, 1, -1)
        moved_x = entity_x + direction
        off = (moved_x < 0) | (moved_x > 9)
        entity_x = mx.where(move, moved_x, entity_x)
        active = mx.where(move, active & (~off), active)
        hit_post = move & active & (entity_x == player_x) & (ENTITY_Y == player_y)
        reward = reward + mx.sum(hit_post & entity_gold).astype(mx.float32)
        terminated = terminated | mx.any(hit_post & (~entity_gold))
        active = active & (~(hit_post & entity_gold))
        move_timer = mx.where(move, move_speed, move_timer)

        spawn_timer = spawn_timer - 1
        move_timer = move_timer - 1

        ramp_active = ((spawn_speed > 1) | (move_speed > 1)) & mx.array(self.ramping)
        tick = ramp_active & (ramp_timer >= 0)
        do_ramp = ramp_active & (ramp_timer < 0)
        ramp_timer = mx.where(tick, ramp_timer - 1, ramp_timer)
        move_speed = mx.where(
            do_ramp & (move_speed > 1) & ((ramp_index % 2) == 1), move_speed - 1, move_speed
        )
        spawn_speed = mx.where(do_ramp & (spawn_speed > 1), spawn_speed - 1, spawn_speed)
        ramp_index = mx.where(do_ramp, ramp_index + 1, ramp_index)
        ramp_timer = mx.where(do_ramp, RAMP_INTERVAL, ramp_timer)

        truncated = mx.zeros_like(terminated)
        next_state: EnvState = {
            "player_x": player_x,
            "player_y": player_y,
            "active": active,
            "entity_x": entity_x,
            "entity_lr": entity_lr,
            "entity_gold": entity_gold,
            "spawn_speed": spawn_speed,
            "spawn_timer": spawn_timer,
            "move_speed": move_speed,
            "move_timer": move_timer,
            "ramp_timer": ramp_timer,
            "ramp_index": ramp_index,
            "time": state["time"] + 1,
        }
        return self._observation(next_state), next_state, reward, terminated, truncated, {}

    @property
    def observation_space(self) -> gym.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        return self._action_space
