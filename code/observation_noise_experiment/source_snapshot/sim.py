from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .config import SimConfig
from .scenarios import Obstacle, Scenario, make_scenario


ACTION_SET = np.array(
    [
        [-1.0, -1.0],
        [-1.0, 0.0],
        [-1.0, 1.0],
        [0.0, -1.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, -1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ],
    dtype=np.float32,
)

RVO_SAFETY_LAYER_VERSION = "rvo_multistep_density_v1"


def limit_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    scale = np.minimum(1.0, max_norm / np.maximum(norm, 1.0e-8))
    return vec * scale


def rect_distance(point: np.ndarray, obs: Obstacle) -> float:
    return obs.distance(point)


@dataclass
class WorldState:
    pos: np.ndarray
    vel: np.ndarray
    goals: np.ndarray
    obstacles: List[Obstacle]
    bounds: Tuple[float, float, float, float]

    def copy(self) -> "WorldState":
        return WorldState(self.pos.copy(), self.vel.copy(), self.goals.copy(), self.obstacles, self.bounds)


def initial_state(scenario: Scenario) -> WorldState:
    pos = scenario.starts.copy()
    vel = np.zeros_like(pos)
    return WorldState(pos=pos, vel=vel, goals=scenario.goals.copy(), obstacles=scenario.obstacles, bounds=scenario.bounds)


def nearest_neighbors(pos: np.ndarray, i: int, limit: int) -> np.ndarray:
    n = len(pos)
    if n <= 1:
        return np.array([], dtype=np.int64)
    d = np.linalg.norm(pos - pos[i], axis=1)
    order = np.argsort(d)
    return np.array([j for j in order if j != i][:limit], dtype=np.int64)


def base_policy_accel(
    pos: np.ndarray,
    vel: np.ndarray,
    goals: np.ndarray,
    obstacles: List[Obstacle],
    cfg: SimConfig,
    reactive: bool = True,
) -> np.ndarray:
    desired = goals - pos
    desired_vel = limit_norm(desired, cfg.max_speed)
    acc = 1.15 * (desired_vel - vel)
    if reactive:
        acc += pairwise_repulsion(pos, cfg, strength=0.85)
        acc += obstacle_repulsion(pos, obstacles, cfg, strength=0.95)
    return limit_norm(acc, cfg.max_accel)


def pairwise_repulsion(pos: np.ndarray, cfg: SimConfig, strength: float = 1.0) -> np.ndarray:
    n = len(pos)
    acc = np.zeros_like(pos)
    trigger = cfg.safety_distance * 2.4
    for i in range(n):
        for j in range(i + 1, n):
            diff = pos[i] - pos[j]
            d = float(np.linalg.norm(diff)) + 1.0e-6
            if d < trigger:
                direction = diff / d
                mag = strength * (trigger - d) / trigger
                acc[i] += direction * mag
                acc[j] -= direction * mag
    return acc


def obstacle_repulsion(pos: np.ndarray, obstacles: List[Obstacle], cfg: SimConfig, strength: float = 1.0) -> np.ndarray:
    acc = np.zeros_like(pos)
    trigger = cfg.safety_distance * 2.0
    for k, p in enumerate(pos):
        for obs in obstacles:
            nearest = obs.nearest_point(p)
            diff = p - nearest
            d = float(np.linalg.norm(diff))
            signed = obs.distance(p)
            if signed < trigger:
                if d < 1.0e-6:
                    diff = p - obs.center
                    d = float(np.linalg.norm(diff)) + 1.0e-6
                direction = diff / d
                mag = strength * (trigger - signed) / trigger
                acc[k] += direction * mag
    return acc


def _trajectory_conflict_response(
    pos: np.ndarray,
    vel: np.ndarray,
    goals: np.ndarray,
    ego: int,
    ego_plan: np.ndarray,
    cfg: SimConfig,
) -> np.ndarray:
    acc = np.zeros_like(pos)
    if len(ego_plan) == 0:
        return acc
    times = cfg.dt * (np.arange(len(ego_plan), dtype=np.float32) + 1.0)
    for j in range(len(pos)):
        if j == ego:
            continue
        goal_vec = goals[j] - pos[j]
        goal_dir = goal_vec / (float(np.linalg.norm(goal_vec)) + 1.0e-6)
        nominal = pos[j][None, :] + vel[j][None, :] * times[:, None]
        d = np.linalg.norm(nominal - ego_plan, axis=1)
        k = int(np.argmin(d))
        min_d = float(d[k])
        trigger = cfg.safety_distance * 3.4
        if min_d >= trigger:
            continue
        urgency = (trigger - min_d) / trigger
        rel = nominal[k] - ego_plan[k]
        rel_norm = float(np.linalg.norm(rel)) + 1.0e-6
        away = rel / rel_norm
        lateral = np.array([-goal_dir[1], goal_dir[0]], dtype=np.float32)
        cross = float(np.cross(np.append(goal_dir, 0.0), np.append(ego_plan[k] - pos[j], 0.0))[2])
        side = 1.0 if cross >= 0.0 else -1.0
        closing_speed = -float(np.dot(vel[j] - (ego_plan[k] - pos[ego]) / max(times[k], cfg.dt), away))
        yield_bias = 0.65 if j > ego else 0.30
        acc[j] += 1.25 * urgency * away
        acc[j] += 1.10 * urgency * side * lateral
        if closing_speed > 0.0 or min_d < cfg.safety_distance * 1.8:
            acc[j] -= yield_bias * urgency * goal_dir
    return limit_norm(acc, cfg.max_accel)


def integrate(pos: np.ndarray, vel: np.ndarray, acc: np.ndarray, cfg: SimConfig, bounds: Tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    acc = limit_norm(acc, cfg.max_accel)
    vel_next = limit_norm(vel + acc * cfg.dt, cfg.max_speed)
    pos_next = pos + vel_next * cfg.dt
    xmin, xmax, ymin, ymax = bounds
    pos_next[:, 0] = np.clip(pos_next[:, 0], xmin + cfg.robot_radius, xmax - cfg.robot_radius)
    pos_next[:, 1] = np.clip(pos_next[:, 1], ymin + cfg.robot_radius, ymax - cfg.robot_radius)
    return pos_next, vel_next


def rollout_response(
    state: WorldState,
    ego: int,
    ego_accel: np.ndarray,
    cfg: SimConfig,
    horizon: Optional[int] = None,
    reactive_neighbors: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    h = horizon or cfg.horizon
    pos = state.pos.copy()
    vel = state.vel.copy()
    traj = np.zeros((h, len(pos), 2), dtype=np.float32)
    acc0 = np.zeros_like(pos)
    ego_plan_pos = pos[ego].copy()
    ego_plan_vel = vel[ego].copy()
    ego_plan_cache_pos = pos[[ego]].copy()
    ego_plan_cache_vel = vel[[ego]].copy()
    ego_plan = np.zeros((h, 2), dtype=np.float32)
    for tau in range(h):
        planned_acc = ego_accel.reshape(1, 2) if tau < 3 else base_policy_accel(
            ego_plan_cache_pos,
            ego_plan_cache_vel,
            state.goals[[ego]],
            state.obstacles,
            cfg,
            reactive=False,
        )
        ego_plan_cache_pos, ego_plan_cache_vel = integrate(
            ego_plan_cache_pos,
            ego_plan_cache_vel,
            planned_acc,
            cfg,
            state.bounds,
        )
        ego_plan[tau] = ego_plan_cache_pos[0]
    for t in range(h):
        acc = base_policy_accel(pos, vel, state.goals, state.obstacles, cfg, reactive=reactive_neighbors)
        if t < 3:
            acc[ego] = ego_accel
        else:
            acc[ego] = base_policy_accel(pos[[ego]], vel[[ego]], state.goals[[ego]], state.obstacles, cfg, reactive=False)[0]
        # Counterfactual reciprocity: neighbors react to the ego query during
        # data generation. Without this term, different ego actions produce
        # almost identical neighbor futures and intent conditioning has no
        # measurable signal to learn.
        if reactive_neighbors:
            remaining_plan = ego_plan[t:] if t < len(ego_plan) else ego_plan[-1:]
            acc += _trajectory_conflict_response(pos, vel, state.goals, ego, remaining_plan, cfg)
            ego_future_vel = limit_norm((ego_plan_vel + acc[ego] * cfg.dt).reshape(1, 2), cfg.max_speed)[0]
            ego_future_pos = ego_plan_pos + ego_future_vel * cfg.dt
            for j in range(len(pos)):
                if j == ego:
                    continue
                diff = pos[j] - ego_future_pos
                d = float(np.linalg.norm(diff)) + 1.0e-6
                trigger = cfg.safety_distance * 3.8
                if d < trigger:
                    direction = diff / d
                    rel_vel = vel[j] - ego_future_vel
                    closing = float(np.dot(rel_vel, direction))
                    urgency = (trigger - d) / trigger
                    lateral = np.array([-direction[1], direction[0]], dtype=np.float32)
                    side = 1.0 if j < ego else -1.0
                    acc[j] += 1.35 * urgency * direction
                    if closing < 0.25:
                        acc[j] += 0.95 * urgency * side * lateral
            ego_plan_pos, ego_plan_vel = ego_future_pos, ego_future_vel
        if t == 0:
            acc0 = acc.copy()
        pos, vel = integrate(pos, vel, acc, cfg, state.bounds)
        traj[t] = pos
    return traj, acc0


def constant_velocity_prediction(state: WorldState, horizon: int, cfg: SimConfig) -> np.ndarray:
    traj = np.zeros((horizon, len(state.pos), 2), dtype=np.float32)
    for t in range(horizon):
        traj[t] = state.pos + state.vel * cfg.dt * (t + 1)
    return traj


def safety_filter(i: int, acc: np.ndarray, state: WorldState, cfg: SimConfig) -> np.ndarray:
    pos_i = state.pos[i]
    desired_vel = limit_norm((state.vel[i] + acc * cfg.dt).reshape(1, 2), cfg.max_speed)[0]
    for j in range(len(state.pos)):
        if i == j:
            continue
        diff = pos_i - state.pos[j]
        d = float(np.linalg.norm(diff)) + 1.0e-6
        direction = diff / d
        rel_vel = desired_vel - state.vel[j]
        closing = float(np.dot(rel_vel, direction))
        trigger = cfg.safety_distance * 3.2
        if d < trigger:
            lateral = np.array([-direction[1], direction[0]], dtype=np.float32)
            side = 1.0 if i < j else -1.0
            urgency = (trigger - d) / trigger
            desired_vel += 0.55 * cfg.max_speed * urgency * direction
            if closing < 0.25:
                desired_vel += (-closing + 0.18) * direction
                desired_vel += 0.75 * cfg.max_speed * urgency * side * lateral
            if d < cfg.safety_distance * 1.25:
                toward_goal = state.goals[i] - state.pos[i]
                toward_goal = toward_goal / (float(np.linalg.norm(toward_goal)) + 1.0e-6)
                desired_vel -= 0.65 * cfg.max_speed * toward_goal
            if d < cfg.safety_distance * 2.4:
                goal_vec = state.goals[i] - state.pos[i]
                goal_norm = float(np.linalg.norm(goal_vec)) + 1.0e-6
                lane_normal = np.array([-goal_vec[1], goal_vec[0]], dtype=np.float32) / goal_norm
                lane_side = 1.0 if (i % 2 == 0) else -1.0
                desired_vel += 0.35 * cfg.max_speed * urgency * lane_side * lane_normal
    for obs in state.obstacles:
        signed = obs.distance(pos_i)
        if signed < cfg.safety_distance * 2.8:
            nearest = obs.nearest_point(pos_i)
            diff = pos_i - nearest
            d = float(np.linalg.norm(diff)) + 1.0e-6
            urgency = (cfg.safety_distance * 2.8 - signed) / (cfg.safety_distance * 2.8)
            desired_vel += 1.35 * cfg.max_speed * urgency * diff / d
            if signed < cfg.safety_distance * 1.08:
                desired_vel *= 0.72
    desired_vel = limit_norm(desired_vel.reshape(1, 2), cfg.max_speed)[0]
    for _ in range(4):
        projected = pos_i + desired_vel * cfg.dt
        adjusted = False
        for j in range(len(state.pos)):
            if i == j:
                continue
            other_projected = state.pos[j] + state.vel[j] * cfg.dt
            diff = projected - other_projected
            d = float(np.linalg.norm(diff)) + 1.0e-6
            if d < cfg.safety_distance * 1.08:
                direction = diff / d
                desired_vel += (cfg.safety_distance * 1.08 - d) / cfg.dt * direction
                adjusted = True
        for obs in state.obstacles:
            signed = obs.distance(projected)
            if signed < cfg.safety_distance * 1.25:
                nearest = obs.nearest_point(projected)
                diff = projected - nearest
                d = float(np.linalg.norm(diff)) + 1.0e-6
                desired_vel += (cfg.safety_distance * 1.08 - signed) / cfg.dt * diff / d
                adjusted = True
        desired_vel = limit_norm(desired_vel.reshape(1, 2), cfg.max_speed)[0]
        if not adjusted:
            break
    filtered = (desired_vel - state.vel[i]) / max(cfg.dt, 1.0e-6)
    return limit_norm(filtered.reshape(1, 2), cfg.max_accel)[0]


def rvo_safety_filter(
    i: int,
    acc: np.ndarray,
    state: WorldState,
    cfg: SimConfig,
    time_horizon: Optional[float] = None,
) -> np.ndarray:
    """Lightweight RVO-style velocity projection for runtime safety support.

    This is a deterministic local safety layer, not a full ORCA linear-program
    solver. It projects the desired velocity away from short-horizon velocity
    obstacles and keeps the result within the configured actuation limits.
    """
    pos_i = state.pos[i]
    horizon = float(time_horizon if time_horizon is not None else max(5.0 * cfg.dt, 0.90))
    safe_distance = cfg.safety_distance + 0.07
    projection_distance = cfg.safety_distance + 0.10
    hard_distance = cfg.safety_distance + 0.01
    desired_vel = limit_norm((state.vel[i] + acc * cfg.dt).reshape(1, 2), cfg.max_speed)[0]
    sample_times = cfg.dt * (np.arange(1, int(max(2, math.ceil(horizon / cfg.dt))) + 1, dtype=np.float32))
    near_count = int(np.sum(np.linalg.norm(state.pos - pos_i, axis=1) < projection_distance * 2.4)) - 1

    order = np.argsort(np.linalg.norm(state.pos - pos_i, axis=1))
    for j in order:
        if i == int(j):
            continue
        diff = pos_i - state.pos[j]
        dist = float(np.linalg.norm(diff)) + 1.0e-6
        rel_vel = desired_vel - state.vel[j]
        denom = float(np.dot(rel_vel, rel_vel))
        t_star = 0.0 if denom < 1.0e-8 else float(np.clip(-np.dot(diff, rel_vel) / denom, 0.0, horizon))
        closest = diff + rel_vel * t_star
        closest_dist = float(np.linalg.norm(closest)) + 1.0e-6
        closing = float(np.dot(diff / dist, rel_vel))
        trigger = projection_distance * 3.0
        risky = closest_dist < safe_distance or (dist < trigger and closing < 0.0)
        if not risky:
            continue

        away = closest / closest_dist if closest_dist > 1.0e-5 else diff / dist
        deficit = max(0.0, safe_distance - closest_dist)
        if dist < safe_distance:
            deficit += 0.5 * (safe_distance - dist)
        if deficit <= 0.0:
            deficit = 0.18 * (trigger - dist) / max(trigger, 1.0e-6)
        correction_mag = min(cfg.max_speed * 0.82, deficit / max(horizon - t_star + cfg.dt, cfg.dt))
        desired_vel += 0.76 * correction_mag * away

        lateral = np.array([-away[1], away[0]], dtype=np.float32)
        side = 1.0 if i < int(j) else -1.0
        if dist < trigger and closing < 0.15:
            desired_vel += 0.22 * cfg.max_speed * side * lateral * (trigger - dist) / trigger
        desired_vel = limit_norm(desired_vel.reshape(1, 2), cfg.max_speed)[0]

    short_times = sample_times[: min(5, len(sample_times))]
    for _ in range(3):
        adjusted = False
        for j in order:
            if i == int(j):
                continue
            for tau in short_times:
                ego_projected = pos_i + desired_vel * float(tau)
                other_projected = state.pos[j] + state.vel[j] * float(tau)
                diff = ego_projected - other_projected
                dist = float(np.linalg.norm(diff)) + 1.0e-6
                required = projection_distance if tau <= 3.0 * cfg.dt else safe_distance
                if dist >= required:
                    continue
                current_diff = pos_i - state.pos[j]
                current_dist = float(np.linalg.norm(current_diff)) + 1.0e-6
                direction = diff / dist if dist > 1.0e-5 else current_diff / current_dist
                rel_vel = desired_vel - state.vel[j]
                closing = float(np.dot(rel_vel, direction))
                correction = 0.58 * (required - dist) / max(float(tau), cfg.dt) * direction
                if closing < 0.0:
                    correction += 0.42 * (-closing) * direction
                desired_vel += correction
                adjusted = True
        desired_vel = limit_norm(desired_vel.reshape(1, 2), cfg.max_speed)[0]
        if not adjusted:
            break

    for obs in state.obstacles:
        projected = pos_i[None, :] + desired_vel[None, :] * sample_times[:, None]
        distances = np.array([obs.distance(p) for p in projected], dtype=np.float32)
        k = int(np.argmin(distances))
        signed = float(distances[k])
        if signed >= safe_distance:
            continue
        nearest = obs.nearest_point(projected[k])
        diff = projected[k] - nearest
        dist = float(np.linalg.norm(diff)) + 1.0e-6
        if dist < 1.0e-5:
            diff = projected[k] - obs.center
            dist = float(np.linalg.norm(diff)) + 1.0e-6
        away = diff / dist
        deficit = safe_distance - signed
        desired_vel += min(cfg.max_speed * 0.75, deficit / max(sample_times[k], cfg.dt)) * away
        if obs.distance(pos_i) < safe_distance:
            desired_vel *= 0.78
        desired_vel = limit_norm(desired_vel.reshape(1, 2), cfg.max_speed)[0]

    future_min_distance = float("inf")
    for j in order:
        if i == int(j):
            continue
        for tau in short_times:
            ego_projected = pos_i + desired_vel * float(tau)
            other_projected = state.pos[j] + state.vel[j] * float(tau)
            future_min_distance = min(future_min_distance, float(np.linalg.norm(ego_projected - other_projected)))
    if near_count >= 2 and future_min_distance < projection_distance * 1.25:
        speed_cap = cfg.max_speed * (0.62 if future_min_distance < cfg.safety_distance else 0.76)
        desired_vel = limit_norm(desired_vel.reshape(1, 2), speed_cap)[0]

    for _ in range(5):
        projected = pos_i + desired_vel * cfg.dt
        adjusted = False
        for j in range(len(state.pos)):
            if i == j:
                continue
            other_projected = state.pos[j] + state.vel[j] * cfg.dt
            diff = projected - other_projected
            dist = float(np.linalg.norm(diff)) + 1.0e-6
            if dist < projection_distance:
                direction = diff / dist
                desired_vel += 1.10 * (projection_distance - dist) / cfg.dt * direction
                toward_other = float(np.dot(desired_vel - state.vel[j], -direction))
                if toward_other > 0.0:
                    desired_vel -= 0.45 * toward_other * (-direction)
                adjusted = True
        for obs in state.obstacles:
            signed = obs.distance(projected)
            if signed < projection_distance:
                nearest = obs.nearest_point(projected)
                diff = projected - nearest
                dist = float(np.linalg.norm(diff)) + 1.0e-6
                desired_vel += 1.10 * (projection_distance - signed) / cfg.dt * diff / dist
                adjusted = True
        desired_vel = limit_norm(desired_vel.reshape(1, 2), cfg.max_speed)[0]
        if not adjusted:
            break

    for _ in range(4):
        projected = pos_i + desired_vel * cfg.dt
        adjusted = False
        for j in range(len(state.pos)):
            if i == j:
                continue
            other_projected = state.pos[j] + state.vel[j] * cfg.dt
            diff = projected - other_projected
            dist = float(np.linalg.norm(diff)) + 1.0e-6
            if dist < hard_distance:
                desired_vel += 1.25 * (hard_distance - dist) / cfg.dt * diff / dist
                desired_vel *= 0.88
                adjusted = True
        desired_vel = limit_norm(desired_vel.reshape(1, 2), cfg.max_speed)[0]
        if not adjusted:
            break

    filtered = (desired_vel - state.vel[i]) / max(cfg.dt, 1.0e-6)
    return limit_norm(filtered.reshape(1, 2), cfg.max_accel)[0]


def action_to_accel(action_index: int, state: WorldState, i: int, cfg: SimConfig) -> np.ndarray:
    base = base_policy_accel(state.pos[[i]], state.vel[[i]], state.goals[[i]], state.obstacles, cfg, reactive=False)[0]
    goal_vec = state.goals[i] - state.pos[i]
    forward = goal_vec / (float(np.linalg.norm(goal_vec)) + 1.0e-6)
    lateral = np.array([-forward[1], forward[0]], dtype=np.float32)
    local_action = ACTION_SET[action_index]
    action = (local_action[0] * lateral + local_action[1] * forward) * (0.65 * cfg.max_accel)
    return limit_norm((base + action).reshape(1, 2), cfg.max_accel)[0]


def collision_stats(pos: np.ndarray, obstacles: List[Obstacle], cfg: SimConfig) -> tuple[bool, float, bool, bool]:
    min_dist = float("inf")
    collision = False
    safety_violation = False
    obstacle_collision = False
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            d = float(np.linalg.norm(pos[i] - pos[j]))
            min_dist = min(min_dist, d)
            if d < 2.0 * cfg.robot_radius:
                collision = True
            if d < cfg.safety_distance:
                safety_violation = True
        for obs in obstacles:
            signed = obs.distance(pos[i])
            min_dist = min(min_dist, signed + cfg.robot_radius)
            if signed < cfg.robot_radius:
                obstacle_collision = True
            if signed < cfg.safety_distance:
                safety_violation = True
    return collision or obstacle_collision, min_dist, obstacle_collision, safety_violation


def reached_goals(state: WorldState, cfg: SimConfig) -> bool:
    return bool(np.all(np.linalg.norm(state.pos - state.goals, axis=1) < cfg.goal_tolerance))


def planner_cost(
    i: int,
    state: WorldState,
    ego_traj: np.ndarray,
    neighbor_traj: np.ndarray,
    acc: np.ndarray,
    cfg: SimConfig,
) -> float:
    goal_cost = 0.0
    risk_cost = 0.0
    progress_reward = 0.0
    start_goal_dist = float(np.linalg.norm(state.pos[i] - state.goals[i]))
    for t in range(len(ego_traj)):
        p = ego_traj[t]
        dist_to_goal = float(np.linalg.norm(p - state.goals[i]))
        goal_cost += dist_to_goal**2
        progress_reward += max(0.0, start_goal_dist - dist_to_goal)
        for q in neighbor_traj[t]:
            d = float(np.linalg.norm(p - q))
            risk_cost += max(0.0, cfg.safety_distance * 1.25 - d) ** 2
        for obs in state.obstacles:
            signed = obs.distance(p)
            risk_cost += max(0.0, cfg.safety_distance * 1.35 - signed) ** 2
    control_cost = cfg.control_weight * float(np.dot(acc, acc))
    progress_cost = -0.45 * progress_reward
    return goal_cost + cfg.collision_weight * risk_cost + control_cost + progress_cost


def ego_rollout(state: WorldState, i: int, acc: np.ndarray, cfg: SimConfig, horizon: int) -> np.ndarray:
    pos = state.pos[[i]].copy()
    vel = state.vel[[i]].copy()
    goals = state.goals[[i]]
    traj = np.zeros((horizon, 2), dtype=np.float32)
    for t in range(horizon):
        a = acc.reshape(1, 2) if t < 3 else base_policy_accel(pos, vel, goals, state.obstacles, cfg, reactive=False)
        pos, vel = integrate(pos, vel, a, cfg, state.bounds)
        traj[t] = pos[0]
    return traj


def oracle_neighbor_prediction(state: WorldState, i: int, acc: np.ndarray, cfg: SimConfig, horizon: int, neighbor_limit: int) -> tuple[np.ndarray, np.ndarray]:
    traj, _ = rollout_response(state, i, acc, cfg, horizon=horizon, reactive_neighbors=True)
    neighbors = nearest_neighbors(state.pos, i, neighbor_limit)
    return traj[:, neighbors, :], neighbors
