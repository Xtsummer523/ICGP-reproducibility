from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import SimConfig
from .scenarios import make_scenario
from .sim import (
    ACTION_SET,
    WorldState,
    action_to_accel,
    initial_state,
    nearest_neighbors,
    rollout_response,
)


FEATURE_DIM = 8 + 4 * 6 + 2
TARGET_DIM = 8 * 4 * 2


def _pad_neighbors(state: WorldState, ego: int, neighbor_limit: int) -> np.ndarray:
    neighbors = nearest_neighbors(state.pos, ego, neighbor_limit)
    rows = []
    for j in neighbors:
        rel_p = state.pos[j] - state.pos[ego]
        rel_v = state.vel[j] - state.vel[ego]
        rel_g = state.goals[j] - state.pos[j]
        rows.append(np.concatenate([rel_p, rel_v, rel_g]).astype(np.float32))
    while len(rows) < neighbor_limit:
        rows.append(np.zeros(6, dtype=np.float32))
    return np.concatenate(rows, axis=0)


def make_feature(state: WorldState, ego: int, action_index: int, cfg: SimConfig, include_intent: bool) -> np.ndarray:
    goal_vec = state.goals[ego] - state.pos[ego]
    goal_dist = np.linalg.norm(goal_vec) + 1.0e-6
    goal_dir = goal_vec / goal_dist
    base = np.concatenate(
        [
            state.vel[ego] / cfg.max_speed,
            goal_dir.astype(np.float32),
            np.array([goal_dist / 10.0, float(len(state.pos)) / 12.0], dtype=np.float32),
            np.array([np.sin(action_index), np.cos(action_index)], dtype=np.float32) * 0.0,
        ]
    )
    neigh = _pad_neighbors(state, ego, cfg.neighbor_limit) / 6.0
    action = ACTION_SET[action_index].astype(np.float32) if include_intent else np.zeros(2, dtype=np.float32)
    return np.concatenate([base.astype(np.float32), neigh.astype(np.float32), action], axis=0)


def make_target(state: WorldState, ego: int, action_index: int, cfg: SimConfig) -> np.ndarray:
    accel = action_to_accel(action_index, state, ego, cfg)
    traj, _ = rollout_response(state, ego, accel, cfg, horizon=cfg.horizon, reactive_neighbors=True)
    neighbors = nearest_neighbors(state.pos, ego, cfg.neighbor_limit)
    target = np.zeros((cfg.horizon, cfg.neighbor_limit, 2), dtype=np.float32)
    for slot, j in enumerate(neighbors):
        target[:, slot, :] = traj[:, j, :] - state.pos[ego]
    return (target / 6.0).reshape(-1)


def random_state(rng: np.random.Generator, cfg: SimConfig, scenario_name: str, n: int, seed: int) -> WorldState:
    scenario = make_scenario(scenario_name, n, seed)
    state = initial_state(scenario)
    steps = int(rng.integers(0, 36))
    for _ in range(steps):
        acc = rng.normal(0.0, 0.35, size=state.pos.shape).astype(np.float32)
        from .sim import base_policy_accel, integrate

        acc += base_policy_accel(state.pos, state.vel, state.goals, state.obstacles, cfg, reactive=True)
        state.pos, state.vel = integrate(state.pos, state.vel, acc, cfg, state.bounds)
    return state


def interaction_state(rng: np.random.Generator, cfg: SimConfig, n: int, seed: int) -> WorldState:
    from .scenarios import Scenario
    from .sim import integrate, base_policy_accel

    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radius = float(rng.uniform(1.15, 2.25))
    pos = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1).astype(np.float32)
    pos += rng.normal(0.0, 0.10, size=pos.shape).astype(np.float32)
    goals = (-pos + rng.normal(0.0, 0.18, size=pos.shape)).astype(np.float32)
    vel = (0.45 * (goals - pos) / np.maximum(np.linalg.norm(goals - pos, axis=1, keepdims=True), 1.0e-6)).astype(np.float32)
    state = WorldState(pos=pos, vel=vel, goals=goals, obstacles=[], bounds=(-5.0, 5.0, -5.0, 5.0))
    steps = int(rng.integers(0, 8))
    for _ in range(steps):
        acc = base_policy_accel(state.pos, state.vel, state.goals, state.obstacles, cfg, reactive=True)
        state.pos, state.vel = integrate(state.pos, state.vel, acc, cfg, state.bounds)
    return state


def conflict_state(rng: np.random.Generator, cfg: SimConfig, n: int, seed: int) -> WorldState:
    _ = seed
    pattern = str(rng.choice(["head_on", "cross", "merge"]))
    bounds = (-5.0, 5.0, -5.0, 5.0)
    if pattern == "head_on":
        left = (n + 1) // 2
        right = n - left
        y_left = np.linspace(-0.55, 0.55, left) if left else np.array([])
        y_right = np.linspace(0.55, -0.55, right) if right else np.array([])
        pos = np.vstack(
            [
                np.column_stack([np.full(left, -1.75), y_left]),
                np.column_stack([np.full(right, 1.75), y_right]),
            ]
        )
        goals = np.vstack(
            [
                np.column_stack([np.full(left, 4.4), -y_left]),
                np.column_stack([np.full(right, -4.4), -y_right]),
            ]
        )
    elif pattern == "cross":
        base_pos = np.array([[-1.9, -0.15], [1.9, 0.15], [-0.15, 1.9], [0.15, -1.9]], dtype=np.float32)
        base_goals = -base_pos * 2.2
        pos = np.vstack([base_pos[k % 4] + np.array([0.0, 0.32 * (k // 4)]) for k in range(n)])
        goals = np.vstack([base_goals[k % 4] - np.array([0.0, 0.32 * (k // 4)]) for k in range(n)])
    else:
        y = np.linspace(-1.15, 1.15, n)
        pos = np.column_stack([np.full(n, -1.9), y])
        goals = np.column_stack([np.full(n, 4.2), np.zeros(n)])
    pos = pos.astype(np.float32) + rng.normal(0.0, 0.05, size=(n, 2)).astype(np.float32)
    goals = goals.astype(np.float32) + rng.normal(0.0, 0.06, size=(n, 2)).astype(np.float32)
    desired = goals - pos
    vel = (0.72 * desired / np.maximum(np.linalg.norm(desired, axis=1, keepdims=True), 1.0e-6)).astype(np.float32)
    return WorldState(pos=pos, vel=vel, goals=goals, obstacles=[], bounds=bounds)


def generate_prediction_arrays(
    cfg: SimConfig,
    total_samples: int,
    seed: int,
    scenarios: Tuple[str, ...] = (
        "crossing",
        "corridor",
        "warehouse",
        "bottleneck",
        "narrow_gate",
        "merge_bottleneck",
        "four_way_stop",
        "swap_lanes",
    ),
    robot_counts: Tuple[int, ...] = (4, 8, 12),
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_intent = np.zeros((total_samples, FEATURE_DIM), dtype=np.float32)
    x_passive = np.zeros((total_samples, FEATURE_DIM), dtype=np.float32)
    y = np.zeros((total_samples, TARGET_DIM), dtype=np.float32)
    meta = np.zeros((total_samples, 4), dtype=np.int32)
    idx = 0
    while idx < total_samples:
        n_id = int(rng.integers(0, len(robot_counts)))
        n = robot_counts[n_id]
        state_seed = seed * 100000 + idx
        draw = rng.random()
        if draw < 0.45:
            scenario_id = len(scenarios)
            state = interaction_state(rng, cfg, n, state_seed)
        elif draw < 0.75:
            scenario_id = len(scenarios) + 1
            state = conflict_state(rng, cfg, n, state_seed)
        else:
            scenario_id = int(rng.integers(0, len(scenarios)))
            scenario_name = scenarios[scenario_id]
            state = random_state(rng, cfg, scenario_name, n, state_seed)
        ego = int(rng.integers(0, n))
        # Counterfactual grouping is important: the same local state is paired
        # with several ego action queries, so a passive model cannot explain all
        # outcomes by memorizing state alone.
        action_indices = rng.choice(len(ACTION_SET), size=min(5, len(ACTION_SET)), replace=False)
        for action_index in action_indices:
            if idx >= total_samples:
                break
            action_index = int(action_index)
            x_intent[idx] = make_feature(state, ego, action_index, cfg, include_intent=True)
            x_passive[idx] = make_feature(state, ego, action_index, cfg, include_intent=False)
            y[idx] = make_target(state, ego, action_index, cfg)
            meta[idx] = [scenario_id, n, ego, action_index]
            idx += 1
    return {"x_intent": x_intent, "x_passive": x_passive, "y": y, "meta": meta}


class PredictionDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]
