from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from .config import EvalConfig, RESULTS, SimConfig
from .dataset import make_feature
from .observation_noise import (
    OBSERVATION_NOISE_PROTOCOL_VERSION,
    make_noise_schedule,
    observed_state_for_ego,
)
from .scenarios import SCENARIO_GENERATION_VERSION, make_scenario
from .sim import (
    WorldState,
    action_to_accel,
    base_policy_accel,
    collision_stats,
    constant_velocity_prediction,
    ego_rollout,
    initial_state,
    integrate,
    nearest_neighbors,
    planner_cost,
    reached_goals,
    rvo_safety_filter,
    RVO_SAFETY_LAYER_VERSION,
    safety_filter,
)

PLANNER_ACTION_INDICES = [1, 3, 4, 5, 7]
SCREEN_HORIZON_STEPS = 5
SCREEN_MARGIN_THRESHOLD = 0.02
RVO_METHODS = {"icgp_rvo_mpc", "passive_rvo_mpc", "residual_passive_rvo_mpc", "rvo_reactive"}


def _model_prediction(model: torch.nn.Module, state: WorldState, ego: int, action_index: int, cfg: SimConfig, include_intent: bool) -> tuple[np.ndarray, np.ndarray, float]:
    neighbors = nearest_neighbors(state.pos, ego, cfg.neighbor_limit)
    x = make_feature(state, ego, action_index, cfg, include_intent=include_intent).reshape(1, -1)
    start = time.perf_counter()
    with torch.no_grad():
        y = model(torch.from_numpy(x.astype(np.float32))).cpu().numpy().reshape(cfg.horizon, cfg.neighbor_limit, 2) * 6.0
    elapsed = (time.perf_counter() - start) * 1000.0
    pred = np.zeros((cfg.horizon, len(neighbors), 2), dtype=np.float32)
    for slot, _j in enumerate(neighbors):
        pred[:, slot, :] = y[:, slot, :] + state.pos[ego]
    return pred, neighbors, elapsed


def candidate_safety_margin(
    state: WorldState,
    ego: int,
    ego_traj: np.ndarray,
    neighbor_traj: np.ndarray,
    cfg: SimConfig,
    screen_horizon: int = SCREEN_HORIZON_STEPS,
) -> float:
    margin = float("inf")
    horizon = min(screen_horizon, len(ego_traj), len(neighbor_traj))
    for t in range(horizon):
        p = ego_traj[t]
        for q in neighbor_traj[t]:
            margin = min(margin, float(np.linalg.norm(p - q)) - cfg.safety_distance)
        for obs in state.obstacles:
            margin = min(margin, obs.distance(p) - cfg.safety_distance)
    return margin


def runtime_safety_margin(
    state: WorldState,
    ego: int,
    acc: np.ndarray,
    cfg: SimConfig,
    screen_horizon: int = SCREEN_HORIZON_STEPS,
) -> float:
    margin = float("inf")
    pos = state.pos[[ego]].copy()
    vel = state.vel[[ego]].copy()
    goals = state.goals[[ego]]
    horizon = min(screen_horizon, cfg.horizon)
    for t in range(horizon):
        step_acc = acc.reshape(1, 2) if t < 3 else base_policy_accel(pos, vel, goals, state.obstacles, cfg, reactive=False)
        pos, vel = integrate(pos, vel, step_acc, cfg, state.bounds)
        p = pos[0]
        tau = cfg.dt * (t + 1)
        for j in range(len(state.pos)):
            if j == ego:
                continue
            q = state.pos[j] + state.vel[j] * tau
            margin = min(margin, float(np.linalg.norm(p - q)) - cfg.safety_distance)
        for obs in state.obstacles:
            margin = min(margin, obs.distance(p) - cfg.safety_distance)
    return margin


def empty_screen_stats() -> Dict[str, float]:
    return {
        "screen_total_candidates": 0.0,
        "screen_rejected_candidates": 0.0,
        "screen_all_rejected": 0.0,
        "screen_min_margin": 0.0,
        "screen_selected_margin": 0.0,
        "screen_predicted_margin": 0.0,
        "screen_runtime_margin": 0.0,
        "rvo_intervened": 0.0,
    }


def choose_accel(
    method: str,
    state: WorldState,
    ego: int,
    cfg: SimConfig,
    intent_model: torch.nn.Module,
    passive_model: torch.nn.Module,
    residual_passive_model: Optional[torch.nn.Module] = None,
    trace_predictions: bool = False,
) -> tuple[np.ndarray, float, Dict[str, float]]:
    stats = empty_screen_stats()
    if method == "orca_reactive":
        acc = base_policy_accel(state.pos[[ego]], state.vel[[ego]], state.goals[[ego]], state.obstacles, cfg, reactive=False)[0]
        acc = safety_filter(ego, acc, state, cfg)
        return acc, 0.0, stats
    if method == "rvo_reactive":
        acc = base_policy_accel(state.pos[[ego]], state.vel[[ego]], state.goals[[ego]], state.obstacles, cfg, reactive=False)[0]
        filtered = rvo_safety_filter(ego, acc, state, cfg)
        stats["rvo_intervened"] = float(np.linalg.norm(filtered - acc) > 1.0e-5)
        return filtered, 0.0, stats

    best_cost = float("inf")
    best_acc = None
    total_latency = 0.0
    horizon = cfg.horizon
    neighbor_limit = cfg.neighbor_limit
    filter_kind = "rvo" if method in RVO_METHODS else "heuristic"
    predictor = "intent"
    if method == "constant_velocity_mpc":
        predictor = "constant"
    elif method == "passive_predictor_mpc":
        predictor = "passive"
    elif method == "passive_rvo_mpc":
        predictor = "passive"
    elif method == "residual_passive_rvo_mpc":
        predictor = "residual_passive"
    elif method == "icgp_no_filter":
        filter_kind = "none"
    elif method == "icgp_h4":
        horizon = min(4, cfg.horizon)
    elif method == "icgp_k2":
        neighbor_limit = min(2, cfg.neighbor_limit)

    action_indices = PLANNER_ACTION_INDICES
    batched_predictions = {}
    batch_latency = 0.0
    if predictor in {"intent", "passive", "residual_passive"}:
        if predictor == "intent":
            model = intent_model
        elif predictor == "residual_passive":
            model = residual_passive_model if residual_passive_model is not None else intent_model
        else:
            model = passive_model
        include_intent = predictor == "intent"
        features = np.stack(
            [make_feature(state, ego, action_index, cfg, include_intent=include_intent) for action_index in action_indices],
            axis=0,
        ).astype(np.float32)
        start = time.perf_counter()
        with torch.no_grad():
            raw = model(torch.from_numpy(features)).cpu().numpy().reshape(len(action_indices), cfg.horizon, cfg.neighbor_limit, 2) * 6.0
        batch_latency = (time.perf_counter() - start) * 1000.0 / max(1, len(action_indices))
        neighbors_all = nearest_neighbors(state.pos, ego, cfg.neighbor_limit)
        for row, action_index in enumerate(action_indices):
            pred = np.zeros((cfg.horizon, len(neighbors_all), 2), dtype=np.float32)
            for slot, _j in enumerate(neighbors_all):
                pred[:, slot, :] = raw[row, :, slot, :] + state.pos[ego]
            batched_predictions[action_index] = pred

    candidate_rows = []
    prediction_neighbors = np.array([], dtype=np.int64)
    candidate_predictions = {}
    for action_index in action_indices:
        acc = action_to_accel(action_index, state, ego, cfg)
        ego_traj = ego_rollout(state, ego, acc, cfg, horizon)
        if predictor == "constant":
            full_pred = constant_velocity_prediction(state, horizon, cfg)
            neigh = nearest_neighbors(state.pos, ego, neighbor_limit)
            neighbor_traj = full_pred[:, neigh, :]
        elif predictor == "passive":
            total_latency += batch_latency
            neighbor_traj = batched_predictions[action_index][:horizon, :neighbor_limit, :]
        else:
            total_latency += batch_latency
            neighbor_traj = batched_predictions[action_index][:horizon, :neighbor_limit, :]
        if predictor in {"intent", "passive", "residual_passive"}:
            prediction_neighbors = nearest_neighbors(state.pos, ego, cfg.neighbor_limit)
            if trace_predictions:
                candidate_predictions[int(action_index)] = neighbor_traj.copy()
        predicted_margin = candidate_safety_margin(state, ego, ego_traj, neighbor_traj, cfg)
        runtime_margin = runtime_safety_margin(state, ego, acc, cfg)
        margin = min(predicted_margin, runtime_margin)
        cost = planner_cost(ego, state, ego_traj, neighbor_traj, acc, cfg)
        candidate_rows.append((action_index, acc, cost, margin, predicted_margin, runtime_margin))

    if method == "icgp_no_filter":
        selection_rows = candidate_rows
    else:
        safe_rows = [row for row in candidate_rows if row[3] >= SCREEN_MARGIN_THRESHOLD]
        stats["screen_total_candidates"] = float(len(candidate_rows))
        stats["screen_rejected_candidates"] = float(len(candidate_rows) - len(safe_rows))
        stats["screen_min_margin"] = float(min(row[3] for row in candidate_rows))
        if safe_rows:
            selection_rows = safe_rows
        else:
            best_margin = max(row[3] for row in candidate_rows)
            selection_rows = [row for row in candidate_rows if row[3] >= best_margin - 1.0e-6]
            stats["screen_all_rejected"] = 1.0

    selected_prediction = None
    selected_action_index = None
    for action_index, acc, cost, margin, predicted_margin, runtime_margin in selection_rows:
        if cost < best_cost:
            best_cost = cost
            best_acc = acc
            selected_action_index = int(action_index)
            if trace_predictions and int(action_index) in candidate_predictions:
                selected_prediction = candidate_predictions[int(action_index)].copy()
            stats["screen_selected_margin"] = float(margin)
            stats["screen_predicted_margin"] = float(predicted_margin)
            stats["screen_runtime_margin"] = float(runtime_margin)
    assert best_acc is not None
    if filter_kind == "heuristic":
        best_acc = safety_filter(ego, best_acc, state, cfg)
    elif filter_kind == "rvo":
        filtered = rvo_safety_filter(ego, best_acc, state, cfg)
        stats["rvo_intervened"] = float(np.linalg.norm(filtered - best_acc) > 1.0e-5)
        best_acc = filtered
    if trace_predictions and candidate_predictions:
        stats["_candidate_predictions"] = candidate_predictions
        stats["_prediction_neighbors"] = prediction_neighbors.copy()
        stats["_selected_prediction"] = selected_prediction
        stats["_selected_action_index"] = selected_action_index
    return best_acc, total_latency, stats


def deadlock_recovery_accel(state: WorldState, i: int, cfg: SimConfig, filter_kind: str = "heuristic") -> np.ndarray:
    goal_vec = state.goals[i] - state.pos[i]
    goal_norm = float(np.linalg.norm(goal_vec)) + 1.0e-6
    forward = goal_vec / goal_norm
    lateral = np.array([-forward[1], forward[0]], dtype=np.float32)
    side = 1.0 if (i % 2 == 0) else -1.0
    priority = 1.0 if i == int(np.argmin(np.linalg.norm(state.pos - state.goals, axis=1))) else 0.55
    desired_vel = cfg.max_speed * (0.55 * priority * forward + 0.35 * side * lateral)
    acc = (desired_vel - state.vel[i]) / max(cfg.dt, 1.0e-6)
    if filter_kind == "rvo":
        return rvo_safety_filter(i, acc, state, cfg)
    if filter_kind == "none":
        return acc
    return safety_filter(i, acc, state, cfg)


def run_episode(
    scenario_name: str,
    n: int,
    seed: int,
    method: str,
    cfg: SimConfig,
    intent_model: torch.nn.Module,
    passive_model: torch.nn.Module,
    tier: str = "main",
    area_policy: str = "fixed",
    noise_level: str = "clean",
    residual_passive_model: Optional[torch.nn.Module] = None,
    observation_noise_protocol: str = OBSERVATION_NOISE_PROTOCOL_VERSION,
    ar_rho: float = 0.85,
    occlusion_probability: float = 0.10,
    occlusion_max_duration: int = 3,
    trace_predictions: bool = False,
    record_trajectory: bool = True,
) -> tuple[dict, pd.DataFrame]:
    scenario = make_scenario(scenario_name, n, seed, tier=tier, area_policy=area_policy)
    state = initial_state(scenario)
    noise_schedule = make_noise_schedule(
        scenario_name,
        n,
        seed,
        noise_level,
        cfg.max_steps,
        protocol=observation_noise_protocol,
        ar_rho=ar_rho,
        occlusion_probability=occlusion_probability,
        occlusion_max_duration=occlusion_max_duration,
    )
    initial_collision, initial_min_distance, initial_obstacle_collision, initial_safety_violation = collision_stats(
        scenario.starts,
        scenario.obstacles,
        cfg,
    )
    collision = False
    obstacle_collision = False
    safety_violation = False
    min_distance = float("inf")
    path_length = np.zeros(n, dtype=np.float64)
    control_effort = 0.0
    smoothness = 0.0
    last_acc = np.zeros((n, 2), dtype=np.float32)
    latency_values: List[float] = []
    screen_total_candidates = 0.0
    screen_rejected_candidates = 0.0
    screen_all_rejected = 0.0
    screen_min_margins: List[float] = []
    screen_selected_margins: List[float] = []
    screen_predicted_margins: List[float] = []
    screen_runtime_margins: List[float] = []
    rvo_interventions = 0.0
    decisions = 0.0
    no_progress = np.zeros(n, dtype=np.int32)
    last_goal_dist = np.linalg.norm(state.pos - state.goals, axis=1)
    rows = []
    position_history = [state.pos.copy()]
    prediction_events = []
    recovery_filter = "rvo" if method in RVO_METHODS else ("none" if method == "icgp_no_filter" else "heuristic")
    for step in range(cfg.max_steps):
        old_pos = state.pos.copy()
        acc = np.zeros_like(state.pos)
        current_goal_dist = np.linalg.norm(state.pos - state.goals, axis=1)
        no_progress = np.where(current_goal_dist < last_goal_dist - 0.012, 0, no_progress + 1)
        last_goal_dist = current_goal_dist.copy()
        for i in range(n):
            observed_state = observed_state_for_ego(
                state,
                ego=i,
                pos_noise=noise_schedule.pos[step],
                vel_noise=noise_schedule.vel[step],
                occluded=noise_schedule.occluded[step],
            )
            acc[i], latency, decision_stats = choose_accel(
                method,
                observed_state,
                i,
                cfg,
                intent_model,
                passive_model,
                residual_passive_model,
                trace_predictions=trace_predictions,
            )
            if trace_predictions and decision_stats.get("_candidate_predictions"):
                prediction_events.append(
                    {
                        "step": step,
                        "ego": i,
                        "neighbors": decision_stats["_prediction_neighbors"].copy(),
                        "candidate_predictions": decision_stats["_candidate_predictions"],
                        "selected_action_index": decision_stats["_selected_action_index"],
                    }
                )
            if method != "icgp_no_filter" and no_progress[i] > 28:
                acc[i] = deadlock_recovery_accel(observed_state, i, cfg, recovery_filter)
                no_progress[i] = 12
            latency_values.append(latency)
            decisions += 1.0
            screen_total_candidates += decision_stats["screen_total_candidates"]
            screen_rejected_candidates += decision_stats["screen_rejected_candidates"]
            screen_all_rejected += decision_stats["screen_all_rejected"]
            rvo_interventions += decision_stats["rvo_intervened"]
            if decision_stats["screen_total_candidates"] > 0.0:
                screen_min_margins.append(decision_stats["screen_min_margin"])
                screen_selected_margins.append(decision_stats["screen_selected_margin"])
                screen_predicted_margins.append(decision_stats["screen_predicted_margin"])
                screen_runtime_margins.append(decision_stats["screen_runtime_margin"])
        control_effort += float(np.mean(np.sum(acc * acc, axis=1)))
        smoothness += float(np.mean(np.linalg.norm(acc - last_acc, axis=1)))
        last_acc = acc.copy()
        state.pos, state.vel = integrate(state.pos, state.vel, acc, cfg, state.bounds)
        if trace_predictions:
            position_history.append(state.pos.copy())
        path_length += np.linalg.norm(state.pos - old_pos, axis=1)
        hit, md, obs_hit, safe_hit = collision_stats(state.pos, state.obstacles, cfg)
        collision = collision or hit
        obstacle_collision = obstacle_collision or obs_hit
        safety_violation = safety_violation or safe_hit
        min_distance = min(min_distance, md)
        for i in range(n):
            if record_trajectory:
                rows.append(
                    {
                        "scenario": scenario_name,
                        "tier": scenario.tier,
                        "area_policy": scenario.area_policy,
                        "robots": n,
                        "seed": seed,
                        "method": method,
                        "noise_level": noise_schedule.level.name,
                        "sigma_p": float(noise_schedule.level.sigma_p),
                        "sigma_v": float(noise_schedule.level.sigma_v),
                        "dominant_flow_axis": scenario.dominant_flow_axis,
                        "lateral_axis": scenario.lateral_axis,
                        "passage_width": float(scenario.passage_width),
                        "start_bounds_margin": float(scenario.start_bounds_margin),
                        "goal_bounds_margin": float(scenario.goal_bounds_margin),
                        "step": step,
                        "robot": i,
                        "x": float(state.pos[i, 0]),
                        "y": float(state.pos[i, 1]),
                        "goal_x": float(state.goals[i, 0]),
                        "goal_y": float(state.goals[i, 1]),
                    }
                )
        if reached_goals(state, cfg):
            break
    steps = step + 1
    final_goal_dist = np.linalg.norm(state.pos - state.goals, axis=1)
    initial_goal_dist = np.linalg.norm(scenario.starts - scenario.goals, axis=1)
    progress_ratio = 1.0 - float(np.mean(final_goal_dist) / max(float(np.mean(initial_goal_dist)), 1.0e-6))
    success = reached_goals(state, cfg) and not collision
    result = {
        "scenario": scenario_name,
        "scenario_generation_version": SCENARIO_GENERATION_VERSION,
        "rvo_safety_layer_version": RVO_SAFETY_LAYER_VERSION,
        "tier": scenario.tier,
        "area_policy": scenario.area_policy,
        "robots": n,
        "seed": seed,
        "method": method,
        "noise_level": noise_schedule.level.name,
        "sigma_p": float(noise_schedule.level.sigma_p),
        "sigma_v": float(noise_schedule.level.sigma_v),
        "observation_noise_protocol": noise_schedule.protocol,
        "occlusion_probability": float(occlusion_probability),
        "occlusion_max_duration": int(occlusion_max_duration),
        "observation_noise_seed": int(noise_schedule.seed),
        "workspace_width": float(scenario.workspace_width),
        "workspace_height": float(scenario.workspace_height),
        "workspace_area": float(scenario.workspace_area),
        "agent_density": float(scenario.agent_density),
        "area_per_agent": float(scenario.area_per_agent),
        "dominant_flow_axis": scenario.dominant_flow_axis,
        "lateral_axis": scenario.lateral_axis,
        "passage_width": float(scenario.passage_width),
        "agents_per_passage_width": float(n / max(scenario.passage_width, 1.0e-6)),
        "start_bounds_margin": float(scenario.start_bounds_margin),
        "goal_bounds_margin": float(scenario.goal_bounds_margin),
        "starts_out_of_bounds": int(scenario.starts_out_of_bounds),
        "goals_out_of_bounds": int(scenario.goals_out_of_bounds),
        "initial_collision": int(initial_collision),
        "initial_safety_violation": int(initial_safety_violation),
        "initial_obstacle_collision": int(initial_obstacle_collision),
        "initial_min_distance": float(initial_min_distance),
        "success": int(success),
        "collision": int(collision),
        "safety_violation": int(safety_violation),
        "obstacle_collision": int(obstacle_collision),
        "min_distance": float(min_distance),
        "completion_time": float(steps * cfg.dt),
        "steps": int(steps),
        "final_goal_distance": float(np.mean(final_goal_dist)),
        "max_final_goal_distance": float(np.max(final_goal_dist)),
        "progress_ratio": float(progress_ratio),
        "path_length": float(np.mean(path_length)),
        "control_effort": float(control_effort / steps),
        "smoothness": float(smoothness / steps),
        "latency_ms": float(np.mean(latency_values)) if latency_values else 0.0,
        "screen_reject_rate": float(screen_rejected_candidates / screen_total_candidates) if screen_total_candidates > 0.0 else 0.0,
        "screen_all_reject_rate": float(screen_all_rejected / decisions) if decisions > 0.0 else 0.0,
        "screen_min_margin": float(np.mean(screen_min_margins)) if screen_min_margins else 0.0,
        "screen_selected_margin": float(np.mean(screen_selected_margins)) if screen_selected_margins else 0.0,
        "screen_predicted_margin": float(np.mean(screen_predicted_margins)) if screen_predicted_margins else 0.0,
        "screen_runtime_margin": float(np.mean(screen_runtime_margins)) if screen_runtime_margins else 0.0,
        "rvo_intervention_rate": float(rvo_interventions / decisions) if decisions > 0.0 else 0.0,
    }
    if trace_predictions:
        prediction_rows = []
        for event in prediction_events:
            step = int(event["step"])
            neighbors = event["neighbors"]
            selected_action_index = event["selected_action_index"]
            for action_index, prediction in event["candidate_predictions"].items():
                available = min(prediction.shape[0], len(position_history) - step - 1)
                for horizon_step in range(available):
                    actual = position_history[step + horizon_step + 1]
                    for slot, neighbor in enumerate(neighbors):
                        prediction_rows.append(
                            {
                                "scenario": scenario_name,
                                "robots": n,
                                "seed": seed,
                                "method": method,
                                "noise_level": noise_schedule.level.name,
                                "step": step,
                                "ego": int(event["ego"]),
                                "action_index": int(action_index),
                                "selected": int(action_index == selected_action_index),
                                "neighbor": int(neighbor),
                                "horizon_step": horizon_step + 1,
                                "pred_x": float(prediction[horizon_step, slot, 0]),
                                "pred_y": float(prediction[horizon_step, slot, 1]),
                                "actual_x": float(actual[neighbor, 0]),
                                "actual_y": float(actual[neighbor, 1]),
                            }
                        )
        result["_prediction_trace"] = prediction_rows
    return result, pd.DataFrame(rows) if record_trajectory else pd.DataFrame()


def evaluate_methods(
    cfg: SimConfig,
    eval_cfg: EvalConfig,
    intent_model: torch.nn.Module,
    passive_model: torch.nn.Module,
    noise_level: str = "clean",
    residual_passive_model: Optional[torch.nn.Module] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows = []
    trajectory_frames = []
    detailed_methods = {"constant_velocity_mpc", "passive_predictor_mpc", "rvo_reactive", "icgp_mpc", "icgp_rvo_mpc", "passive_rvo_mpc"}
    for scenario in eval_cfg.scenarios:
        for n in eval_cfg.robot_counts:
            for method in eval_cfg.methods:
                for seed_idx in range(eval_cfg.seeds_per_case):
                    seed = 1000 + seed_idx + n * 100 + abs(hash(scenario)) % 1000
                    result, traj = run_episode(
                        scenario,
                        n,
                        seed,
                        method,
                        cfg,
                        intent_model,
                        passive_model,
                        noise_level=noise_level,
                        residual_passive_model=residual_passive_model,
                    )
                    result_rows.append(result)
                    if seed_idx == 0 and n == 8 and method in detailed_methods:
                        trajectory_frames.append(traj)
    results = pd.DataFrame(result_rows)
    trajectories = pd.concat(trajectory_frames, ignore_index=True) if trajectory_frames else pd.DataFrame()
    results.to_csv(RESULTS / "closed_loop_results.csv", index=False)
    trajectories.to_csv(RESULTS / "representative_trajectories.csv", index=False)
    return results, trajectories
