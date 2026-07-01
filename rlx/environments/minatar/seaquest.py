import numpy as np
import gymnasium as gym
import mlx.core as mx

from rlx.environments.environment import Environment, EnvState

RAMP_INTERVAL = 100
MAX_OXYGEN = 200
INIT_SPAWN_SPEED = 20
DIVER_SPAWN_SPEED = 30
INIT_MOVE_INTERVAL = 5
SHOT_COOL_DOWN = 5
ENEMY_SHOT_INTERVAL = 10
DIVER_MOVE_INTERVAL = 5

CAP = 20
ROW = mx.arange(10).reshape(10, 1)
COL = mx.arange(10).reshape(1, 10)
GRID = mx.arange(10)
SLOTS = mx.arange(CAP)


def _match(a_active, a_x, a_y, b_active, b_x, b_y):
    return (
        a_active.reshape(-1, 1)
        & b_active.reshape(1, -1)
        & (a_x.reshape(-1, 1) == b_x.reshape(1, -1))
        & (a_y.reshape(-1, 1) == b_y.reshape(1, -1))
    )


def _insert(active, fields, add_mask, add_fields):
    empty = ~active
    empty_rank = mx.cumsum(empty.astype(mx.int32)) - 1
    add_rank = mx.cumsum(add_mask.astype(mx.int32)) - 1
    match = (
        empty.reshape(-1, 1)
        & add_mask.reshape(1, -1)
        & (empty_rank.reshape(-1, 1) == add_rank.reshape(1, -1))
    )
    receives = mx.any(match, axis=1)
    idx = mx.argmax(match.astype(mx.int32), axis=1)
    new_fields = tuple(mx.where(receives, af[idx], f) for f, af in zip(fields, add_fields))
    return active | receives, new_fields


def _channel(active, x, y):
    oh = (ROW.reshape(1, 10, 1) == y.reshape(-1, 1, 1)) & (
        COL.reshape(1, 1, 10) == x.reshape(-1, 1, 1)
    )
    return mx.max(oh & active.reshape(-1, 1, 1), axis=0)


def _trail(active, x, y, orientation):
    back_x = mx.where(orientation, x - 1, x + 1)
    in_bounds = (back_x >= 0) & (back_x <= 9)
    return _channel(active & in_bounds, mx.clip(back_x, 0, 9), y)


class Seaquest(Environment):
    def __init__(self, ramping: bool = True):
        self.ramping = ramping
        self._observation_space = gym.spaces.Box(0, 1, (10, 10, 10), dtype=np.float32)
        self._action_space = gym.spaces.Discrete(6)

    def _observation(self, s: EnvState) -> mx.array:
        sub_front = (ROW == s["sub_y"]) & (COL == s["sub_x"])
        back_x = mx.where(s["sub_or"], s["sub_x"] - 1, s["sub_x"] + 1)
        sub_back = (ROW == s["sub_y"]) & (COL == back_x)
        oxygen_lit = mx.floor(mx.maximum(0, s["oxygen"]) * 10 / MAX_OXYGEN).astype(
            mx.int32
        )
        oxygen_gauge = (ROW == 9) * (COL < oxygen_lit)
        diver_gauge = (ROW == 9) * (COL >= 9 - s["diver_count"]) * (COL < 9)

        trail = (
            _trail(s["ef_active"], s["ef_x"], s["ef_y"], s["ef_or"])
            | _trail(s["es_active"], s["es_x"], s["es_y"], s["es_or"])
            | _trail(s["dv_active"], s["dv_x"], s["dv_y"], s["dv_or"])
        )
        return mx.stack(
            [
                sub_front,
                sub_back,
                _channel(s["fb_active"], s["fb_x"], s["fb_y"]),
                trail,
                _channel(s["eb_active"], s["eb_x"], s["eb_y"]),
                _channel(s["ef_active"], s["ef_x"], s["ef_y"]),
                _channel(s["es_active"], s["es_x"], s["es_y"]),
                oxygen_gauge,
                diver_gauge,
                _channel(s["dv_active"], s["dv_x"], s["dv_y"]),
            ],
            axis=-1,
        ).astype(mx.float32)

    def reset(self, key: mx.array) -> tuple[mx.array, EnvState, dict]:
        zeros = mx.zeros((CAP,), dtype=mx.int32)
        falses = mx.zeros((CAP,), dtype=mx.bool_)
        s: EnvState = {
            "sub_x": mx.array(5),
            "sub_y": mx.array(0),
            "sub_or": mx.array(False),
            "oxygen": mx.array(MAX_OXYGEN),
            "diver_count": mx.array(0),
            "fb_active": falses, "fb_x": zeros, "fb_y": zeros, "fb_or": falses,
            "eb_active": falses, "eb_x": zeros, "eb_y": zeros, "eb_or": falses,
            "ef_active": falses, "ef_x": zeros, "ef_y": zeros, "ef_or": falses, "ef_timer": zeros,
            "es_active": falses, "es_x": zeros, "es_y": zeros, "es_or": falses, "es_timer": zeros, "es_shot": zeros,
            "dv_active": falses, "dv_x": zeros, "dv_y": zeros, "dv_or": falses, "dv_timer": zeros,
            "e_spawn_speed": mx.array(INIT_SPAWN_SPEED),
            "e_spawn_timer": mx.array(INIT_SPAWN_SPEED),
            "d_spawn_timer": mx.array(DIVER_SPAWN_SPEED),
            "move_speed": mx.array(INIT_MOVE_INTERVAL),
            "ramp_index": mx.array(0),
            "shot_timer": mx.array(0),
            "surface": mx.array(True),
            "time": mx.array(0, dtype=mx.int32),
        }
        return self._observation(s), s, {}

    def step_env(self, key, state, action):
        s = dict(state)
        reward = mx.array(0.0)
        terminated = state["time"] < 0
        sub_x, sub_y, sub_or = s["sub_x"], s["sub_y"], s["sub_or"]
        move_speed = s["move_speed"]

        # player action
        fire = (action == 5) & (s["shot_timer"] == 0)
        empty_fb = ~s["fb_active"]
        slot = mx.argmax(empty_fb)
        place = fire & mx.any(empty_fb) & (SLOTS == slot)
        fb_active = mx.where(place, True, s["fb_active"])
        fb_x = mx.where(place, sub_x, s["fb_x"])
        fb_y = mx.where(place, sub_y, s["fb_y"])
        fb_or = mx.where(place, sub_or, s["fb_or"])
        shot_timer = mx.where(fire, SHOT_COOL_DOWN, s["shot_timer"])
        sub_x = mx.where(action == 1, mx.maximum(0, sub_x - 1), sub_x)
        sub_or = mx.where(action == 1, False, sub_or)
        sub_x = mx.where(action == 3, mx.minimum(9, sub_x + 1), sub_x)
        sub_or = mx.where(action == 3, True, sub_or)
        sub_y = mx.where(action == 2, mx.maximum(0, sub_y - 1), sub_y)
        sub_y = mx.where(action == 4, mx.minimum(8, sub_y + 1), sub_y)

        ef_active, ef_x, ef_y, ef_or, ef_timer = (
            s["ef_active"], s["ef_x"], s["ef_y"], s["ef_or"], s["ef_timer"])
        es_active, es_x, es_y, es_or, es_timer, es_shot = (
            s["es_active"], s["es_x"], s["es_y"], s["es_or"], s["es_timer"], s["es_shot"])

        # friendly bullets move + collide with current fish (priority) then subs
        fb_x = fb_x + mx.where(fb_or, 1, -1)
        fb_active = fb_active & ~((fb_x < 0) | (fb_x > 9))
        bf = _match(fb_active, fb_x, fb_y, ef_active, ef_x, ef_y)
        bullet_hit_fish = mx.any(bf, axis=1)
        fish_hit = mx.any(bf, axis=0)
        ef_active = ef_active & ~fish_hit
        bs = _match(fb_active & ~bullet_hit_fish, fb_x, fb_y, es_active, es_x, es_y)
        bullet_hit_sub = mx.any(bs, axis=1)
        sub_hit = mx.any(bs, axis=0)
        es_active = es_active & ~sub_hit
        reward = reward + mx.sum(fish_hit) + mx.sum(sub_hit)
        fb_active = fb_active & ~(bullet_hit_fish | bullet_hit_sub)

        # divers
        dv_active, dv_x, dv_y, dv_or, dv_timer = (
            s["dv_active"], s["dv_x"], s["dv_y"], s["dv_or"], s["dv_timer"])
        pickup = dv_active & (dv_x == sub_x) & (dv_y == sub_y) & (s["diver_count"] < 6)
        diver_count = s["diver_count"] + mx.sum(pickup).astype(mx.int32)
        dv_active = dv_active & ~pickup
        move_d = dv_active & (dv_timer == 0)
        dv_x_new = dv_x + mx.where(dv_or, 1, -1)
        dv_off = (dv_x_new < 0) | (dv_x_new > 9)
        dv_x = mx.where(move_d, dv_x_new, dv_x)
        dv_active = mx.where(move_d, dv_active & ~dv_off, dv_active)
        pickup2 = move_d & dv_active & (dv_x == sub_x) & (dv_y == sub_y) & (diver_count < 6)
        diver_count = diver_count + mx.sum(pickup2).astype(mx.int32)
        dv_active = dv_active & ~pickup2
        dv_timer = mx.where(move_d, DIVER_MOVE_INTERVAL, dv_timer - 1)

        # enemy subs: player collision, move, collide with remaining bullets, shoot
        es_active_loop = es_active
        terminated = terminated | mx.any(es_active & (es_x == sub_x) & (es_y == sub_y))
        move_s = es_active & (es_timer == 0)
        es_x_new = es_x + mx.where(es_or, 1, -1)
        es_off = (es_x_new < 0) | (es_x_new > 9)
        es_x = mx.where(move_s, es_x_new, es_x)
        es_active = mx.where(move_s, es_active & ~es_off, es_active)
        terminated = terminated | mx.any(move_s & es_active & (es_x == sub_x) & (es_y == sub_y))
        sb = _match(es_active & move_s, es_x, es_y, fb_active, fb_x, fb_y)
        sub_hit2 = mx.any(sb, axis=1)
        fb_active = fb_active & ~mx.any(sb, axis=0)
        es_active = es_active & ~sub_hit2
        reward = reward + mx.sum(sub_hit2)
        es_timer = mx.where(move_s, move_speed, es_timer - 1)
        shoot = es_active_loop & (es_shot == 0)
        eb_active, (eb_x, eb_y, eb_or) = _insert(
            s["eb_active"], (s["eb_x"], s["eb_y"], s["eb_or"]),
            shoot, (es_x, es_y, es_or))
        es_shot = mx.where(es_shot == 0, ENEMY_SHOT_INTERVAL, es_shot - 1)

        # enemy bullets: player collision, move, player collision
        terminated = terminated | mx.any(eb_active & (eb_x == sub_x) & (eb_y == sub_y))
        eb_x = eb_x + mx.where(eb_or, 1, -1)
        eb_active = eb_active & ~((eb_x < 0) | (eb_x > 9))
        terminated = terminated | mx.any(eb_active & (eb_x == sub_x) & (eb_y == sub_y))

        # enemy fish: player collision, move, collide with remaining bullets
        terminated = terminated | mx.any(ef_active & (ef_x == sub_x) & (ef_y == sub_y))
        move_f = ef_active & (ef_timer == 0)
        ef_x_new = ef_x + mx.where(ef_or, 1, -1)
        ef_off = (ef_x_new < 0) | (ef_x_new > 9)
        ef_x = mx.where(move_f, ef_x_new, ef_x)
        ef_active = mx.where(move_f, ef_active & ~ef_off, ef_active)
        terminated = terminated | mx.any(move_f & ef_active & (ef_x == sub_x) & (ef_y == sub_y))
        ff = _match(ef_active & move_f, ef_x, ef_y, fb_active, fb_x, fb_y)
        fish_hit2 = mx.any(ff, axis=1)
        fb_active = fb_active & ~mx.any(ff, axis=0)
        ef_active = ef_active & ~fish_hit2
        reward = reward + mx.sum(fish_hit2)
        ef_timer = mx.where(move_f, move_speed, ef_timer - 1)

        # timers
        e_spawn_timer = s["e_spawn_timer"] - (s["e_spawn_timer"] > 0).astype(mx.int32)
        d_spawn_timer = s["d_spawn_timer"] - (s["d_spawn_timer"] > 0).astype(mx.int32)
        shot_timer = shot_timer - (shot_timer > 0).astype(mx.int32)

        # oxygen / surfacing
        oxygen = s["oxygen"]
        e_spawn_speed = s["e_spawn_speed"]
        ramp_index = s["ramp_index"]
        surface = s["surface"]
        terminated = terminated | (oxygen <= 0)
        underwater = sub_y > 0
        oxygen = mx.where(underwater, oxygen - 1, oxygen)
        surface = mx.where(underwater, False, surface)

        do_surface = (~underwater) & (~surface)
        terminated = terminated | (do_surface & (diver_count == 0))
        surfaced = do_surface & (diver_count != 0)
        full = surfaced & (diver_count == 6)
        reward = reward + mx.where(full, (oxygen * 10 // MAX_OXYGEN).astype(mx.float32), 0.0)
        diver_count = mx.where(full, 0, diver_count)
        oxygen = mx.where(surfaced, MAX_OXYGEN, oxygen)
        diver_count = mx.where(surfaced, diver_count - 1, diver_count)
        ramp = (
            surfaced
            & mx.array(self.ramping)
            & ((e_spawn_speed > 1) | (move_speed > 2))
        )
        move_speed = mx.where(
            ramp & (move_speed > 2) & ((ramp_index % 2) == 1), move_speed - 1, move_speed)
        e_spawn_speed = mx.where(ramp & (e_spawn_speed > 1), e_spawn_speed - 1, e_spawn_speed)
        ramp_index = mx.where(ramp, ramp_index + 1, ramp_index)
        surface = mx.where(surfaced, True, surface)

        truncated = mx.zeros_like(terminated)
        next_state: EnvState = {
            "sub_x": sub_x, "sub_y": sub_y, "sub_or": sub_or,
            "oxygen": oxygen, "diver_count": diver_count,
            "fb_active": fb_active, "fb_x": fb_x, "fb_y": fb_y, "fb_or": fb_or,
            "eb_active": eb_active, "eb_x": eb_x, "eb_y": eb_y, "eb_or": eb_or,
            "ef_active": ef_active, "ef_x": ef_x, "ef_y": ef_y, "ef_or": ef_or, "ef_timer": ef_timer,
            "es_active": es_active, "es_x": es_x, "es_y": es_y, "es_or": es_or, "es_timer": es_timer, "es_shot": es_shot,
            "dv_active": dv_active, "dv_x": dv_x, "dv_y": dv_y, "dv_or": dv_or, "dv_timer": dv_timer,
            "e_spawn_speed": e_spawn_speed, "e_spawn_timer": e_spawn_timer,
            "d_spawn_timer": d_spawn_timer, "move_speed": move_speed,
            "ramp_index": ramp_index, "shot_timer": shot_timer, "surface": surface,
            "time": s["time"] + 1,
        }
        return self._observation(next_state), next_state, reward, terminated, truncated, {}

    @property
    def observation_space(self) -> gym.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        return self._action_space
