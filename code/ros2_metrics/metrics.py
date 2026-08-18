from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricConfig:
    robot_radius_m: float = 0.22
    safety_margin_m: float = 0.08
    goal_tolerance_m: float = 0.38
    deadlock_speed_mps: float = 0.035
    deadlock_window_s: float = 2.5
    oscillation_heading_flip_rad: float = 1.75
    resample_dt_s: float = 0.05
    max_interpolation_gap_s: float = 0.30

    @property
    def safety_distance_m(self) -> float:
        return 2.0 * self.robot_radius_m + self.safety_margin_m


def pairwise_distances(pos: np.ndarray) -> list[tuple[int, int, float]]:
    rows: list[tuple[int, int, float]] = []
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            rows.append((i, j, float(np.linalg.norm(pos[i] - pos[j]))))
    return rows


def compute_episode_metrics(states: pd.DataFrame, goals: dict[str, tuple[float, float]], cfg: MetricConfig) -> dict[str, float | int | str]:
    required = {"time_s", "robot_id", "x", "y", "vx", "vy"}
    missing = required - set(states.columns)
    if missing:
        raise ValueError(f"state CSV missing columns: {sorted(missing)}")
    if states.empty:
        raise ValueError("state CSV is empty")

    states = states.sort_values(["time_s", "robot_id"]).copy()
    robot_ids = sorted(states["robot_id"].unique())
    if set(robot_ids) != set(goals):
        raise ValueError(f"goals keys {sorted(goals)} do not match state robot ids {robot_ids}")

    raw_duration = float(states["time_s"].max() - states["time_s"].min())
    raw_sample_counts = states.groupby("robot_id").size()
    if "sample_age_s" in states.columns:
        sample_age = pd.to_numeric(states["sample_age_s"], errors="coerce").dropna()
        max_sample_age_s = float(sample_age.max()) if not sample_age.empty else 0.0
        min_sample_age_s = float(sample_age.min()) if not sample_age.empty else 0.0
        max_abs_sample_age_s = float(sample_age.abs().max()) if not sample_age.empty else 0.0
    else:
        max_sample_age_s = 0.0
        min_sample_age_s = 0.0
        max_abs_sample_age_s = 0.0
    resampled = resample_states(states, robot_ids, cfg)

    first = resampled.groupby("robot_id").first()
    last = resampled.groupby("robot_id").last()
    initial_dists = []
    final_dists = []
    for rid in robot_ids:
        gx, gy = goals[rid]
        initial_dists.append(float(math.hypot(first.loc[rid, "x"] - gx, first.loc[rid, "y"] - gy)))
        final_dists.append(float(math.hypot(last.loc[rid, "x"] - gx, last.loc[rid, "y"] - gy)))
    initial_mean = float(np.mean(initial_dists))
    final_mean = float(np.mean(final_dists))
    progress_ratio = 0.0 if initial_mean <= 1.0e-9 else float(np.clip((initial_mean - final_mean) / initial_mean, -1.0, 1.0))

    min_distance = float("inf")
    collision = 0
    safety_violation = 0
    pair_rows = []
    for t, group in resampled.groupby("time_s"):
        group = group.set_index("robot_id").loc[robot_ids]
        pos = group[["x", "y"]].to_numpy(dtype=float)
        for i, j, d in pairwise_distances(pos):
            min_distance = min(min_distance, d)
            if d < 2.0 * cfg.robot_radius_m:
                collision = 1
            if d < cfg.safety_distance_m:
                safety_violation = 1
            pair_rows.append((float(t), robot_ids[i], robot_ids[j], d))

    reached = int(all(d <= cfg.goal_tolerance_m for d in final_dists))
    speeds = np.hypot(resampled["vx"].to_numpy(dtype=float), resampled["vy"].to_numpy(dtype=float))
    mean_speed = float(np.mean(speeds))
    max_speed = float(np.max(speeds))

    duration = float(resampled["time_s"].max() - resampled["time_s"].min())
    deadlock = detect_deadlock(resampled, goals, cfg)
    oscillation_count = count_oscillations(resampled, cfg)
    dropped_fraction = 1.0 - float(len(resampled["time_s"].unique()) / max(int(round(raw_duration / cfg.resample_dt_s)) + 1, 1))

    return {
        "robots": int(len(robot_ids)),
        "duration_s": duration,
        "raw_duration_s": raw_duration,
        "progress_ratio": progress_ratio,
        "final_goal_distance_m": final_mean,
        "collision": collision,
        "safety_violation": safety_violation,
        "min_distance_m": min_distance if math.isfinite(min_distance) else -1.0,
        "reached_all_goals": reached,
        "deadlock": int(deadlock),
        "oscillation_count": int(oscillation_count),
        "mean_speed_mps": mean_speed,
        "max_speed_mps": max_speed,
        "resample_dt_s": cfg.resample_dt_s,
        "max_interpolation_gap_s": cfg.max_interpolation_gap_s,
        "resampled_steps": int(len(resampled["time_s"].unique())),
        "min_raw_samples_per_robot": int(raw_sample_counts.min()),
        "max_raw_samples_per_robot": int(raw_sample_counts.max()),
        "max_sample_age_s": max_sample_age_s,
        "min_sample_age_s": min_sample_age_s,
        "max_abs_sample_age_s": max_abs_sample_age_s,
        "dropped_time_fraction": max(0.0, dropped_fraction),
    }


def pairwise_distance_table(resampled_states: pd.DataFrame, robot_ids: list[str] | None = None) -> pd.DataFrame:
    if resampled_states.empty:
        raise ValueError("resampled state table is empty")
    required = {"time_s", "robot_id", "x", "y"}
    missing = required - set(resampled_states.columns)
    if missing:
        raise ValueError(f"resampled state table missing columns: {sorted(missing)}")
    ids = robot_ids or sorted(resampled_states["robot_id"].unique())
    rows: list[dict[str, float | str]] = []
    for t, group in resampled_states.groupby("time_s"):
        group = group.set_index("robot_id")
        if not set(ids).issubset(set(group.index)):
            continue
        pos = group.loc[ids, ["x", "y"]].to_numpy(dtype=float)
        for i, j, d in pairwise_distances(pos):
            rows.append(
                {
                    "time_s": float(t),
                    "robot_i": ids[i],
                    "robot_j": ids[j],
                    "distance_m": d,
                }
            )
    return pd.DataFrame(rows)


def resample_states(states: pd.DataFrame, robot_ids: list[str], cfg: MetricConfig) -> pd.DataFrame:
    """Synchronize asynchronous robot pose streams onto one common timeline.

    This avoids comparing robot A at t1 with robot B at t2 when odom/mocap
    callbacks arrive at different moments. Samples are linearly interpolated;
    times with a gap larger than cfg.max_interpolation_gap_s for any robot are
    dropped from metric computation.
    """
    t_min = max(float(states[states["robot_id"] == rid]["time_s"].min()) for rid in robot_ids)
    t_max = min(float(states[states["robot_id"] == rid]["time_s"].max()) for rid in robot_ids)
    if t_max <= t_min:
        raise ValueError("state streams do not overlap in time")
    timeline = np.arange(t_min, t_max + 0.5 * cfg.resample_dt_s, cfg.resample_dt_s)
    rows: list[dict[str, float | str]] = []
    for rid in robot_ids:
        group = states[states["robot_id"] == rid].sort_values("time_s").drop_duplicates("time_s")
        times = group["time_s"].to_numpy(dtype=float)
        if len(times) < 2:
            raise ValueError(f"not enough samples for {rid}")
        x = np.interp(timeline, times, group["x"].to_numpy(dtype=float))
        y = np.interp(timeline, times, group["y"].to_numpy(dtype=float))
        vx = np.interp(timeline, times, group["vx"].to_numpy(dtype=float))
        vy = np.interp(timeline, times, group["vy"].to_numpy(dtype=float))
        if "yaw" in group.columns:
            yaw_values = np.unwrap(group["yaw"].to_numpy(dtype=float))
            yaw = np.interp(timeline, times, yaw_values)
        else:
            yaw = np.zeros_like(timeline)
        left_idx = np.searchsorted(times, timeline, side="right") - 1
        right_idx = left_idx + 1
        valid = (left_idx >= 0) & (right_idx < len(times))
        gap = np.full_like(timeline, np.inf, dtype=float)
        gap[valid] = times[right_idx[valid]] - times[left_idx[valid]]
        valid &= gap <= cfg.max_interpolation_gap_s
        for k, t in enumerate(timeline):
            if not bool(valid[k]):
                continue
            rows.append(
                {
                    "time_s": float(t),
                    "robot_id": rid,
                    "x": float(x[k]),
                    "y": float(y[k]),
                    "yaw": float(yaw[k]),
                    "vx": float(vx[k]),
                    "vy": float(vy[k]),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("all resampled states were dropped due to interpolation gaps")
    counts = out.groupby("time_s")["robot_id"].nunique()
    complete_times = counts[counts == len(robot_ids)].index
    out = out[out["time_s"].isin(complete_times)].copy()
    if out.empty:
        raise ValueError("no complete synchronized time steps after resampling")
    return out.sort_values(["time_s", "robot_id"]).reset_index(drop=True)


def detect_deadlock(states: pd.DataFrame, goals: dict[str, tuple[float, float]], cfg: MetricConfig) -> bool:
    robot_ids = sorted(states["robot_id"].unique())
    times = np.array(sorted(states["time_s"].unique()), dtype=float)
    if len(times) < 2:
        return False
    dt = float(np.median(np.diff(times)))
    if dt <= 0:
        return False
    window_steps = max(2, int(round(cfg.deadlock_window_s / dt)))
    pivot = times[-window_steps:]
    tail = states[states["time_s"].isin(pivot)]
    if tail.empty:
        return False
    for rid in robot_ids:
        rtail = tail[tail["robot_id"] == rid]
        if rtail.empty:
            continue
        speed = np.hypot(rtail["vx"].to_numpy(dtype=float), rtail["vy"].to_numpy(dtype=float))
        last = rtail.iloc[-1]
        gx, gy = goals[rid]
        goal_dist = float(math.hypot(last["x"] - gx, last["y"] - gy))
        if float(np.mean(speed)) < cfg.deadlock_speed_mps and goal_dist > cfg.goal_tolerance_m:
            return True
    return False


def count_oscillations(states: pd.DataFrame, cfg: MetricConfig) -> int:
    count = 0
    for _rid, group in states.groupby("robot_id"):
        group = group.sort_values("time_s")
        vx = group["vx"].to_numpy(dtype=float)
        vy = group["vy"].to_numpy(dtype=float)
        speed = np.hypot(vx, vy)
        valid = speed > max(cfg.deadlock_speed_mps, 1.0e-4)
        heading = np.arctan2(vy[valid], vx[valid])
        if len(heading) < 3:
            continue
        delta = np.diff(np.unwrap(heading))
        count += int(np.sum(np.abs(delta) > cfg.oscillation_heading_flip_rad))
    return count


def save_metrics(output_dir: Path, summary: dict[str, float | int | str], prefix: str = "episode") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(output_dir / f"{prefix}_summary.csv", index=False)
