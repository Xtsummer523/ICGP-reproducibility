from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from icgp_experiments.config import load_experiment_spec
from icgp_experiments.metrics import MetricConfig, compute_episode_metrics, pairwise_distance_table, resample_states


def goals_from_spec(config_path: Path) -> dict[str, tuple[float, float]]:
    spec = load_experiment_spec(config_path)
    return {robot.robot_id: robot.goal_xy for robot in spec.robots}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ICGP ROS2 experiment metrics from states.csv.")
    parser.add_argument("--config", required=True, help="Experiment YAML config.")
    parser.add_argument("--run-dir", required=True, help="Directory containing states.csv and optional logs.")
    parser.add_argument("--output-prefix", default="episode", help="Output prefix for summary files.")
    args = parser.parse_args()

    config_path = Path(args.config)
    run_dir = Path(args.run_dir)
    states_path = run_dir / "states.csv"
    if not states_path.exists():
        raise FileNotFoundError(states_path)
    spec = load_experiment_spec(config_path)
    states = pd.read_csv(states_path)
    metric_cfg = MetricConfig(
        robot_radius_m=spec.robot_radius_m,
        safety_margin_m=spec.safety_margin_m,
        goal_tolerance_m=spec.goal_tolerance_m,
    )
    summary = compute_episode_metrics(states, goals_from_spec(config_path), metric_cfg)
    robot_ids = [robot.robot_id for robot in spec.robots]
    resampled = resample_states(states, robot_ids, metric_cfg)
    pairwise = pairwise_distance_table(resampled, robot_ids)
    resampled.to_csv(run_dir / f"{args.output_prefix}_resampled_states.csv", index=False)
    pairwise.to_csv(run_dir / f"{args.output_prefix}_pairwise_distances.csv", index=False)
    time_basis_values = sorted(str(x) for x in states.get("time_basis", pd.Series(dtype=str)).dropna().unique())
    frame_ids = sorted(str(x) for x in states.get("frame_id", pd.Series(dtype=str)).dropna().unique())
    pose_sources = sorted(str(x) for x in states.get("pose_source", pd.Series(dtype=str)).dropna().unique())
    summary.update(
        {
            "experiment_id": spec.experiment_id,
            "scenario": spec.scenario,
            "method": spec.method,
            "enable_motion": spec.enable_motion,
            "pose_sources": ";".join(pose_sources),
            "frame_ids": ";".join(frame_ids),
            "time_basis": ";".join(time_basis_values),
            "run_dir": str(run_dir),
            "resampled_states_csv": str(run_dir / f"{args.output_prefix}_resampled_states.csv"),
            "pairwise_distances_csv": str(run_dir / f"{args.output_prefix}_pairwise_distances.csv"),
        }
    )
    pd.DataFrame([summary]).to_csv(run_dir / f"{args.output_prefix}_summary.csv", index=False)
    (run_dir / f"{args.output_prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
