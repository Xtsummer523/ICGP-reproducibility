from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from icgp_experiments.config import load_experiment_spec, resolve_host_path


def write_trajectory_plot(
    config_path: Path | str,
    run_dir: Path | str,
    output: Path | str | None = None,
    metadata_output: Path | str | None = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = resolve_host_path(config_path)
    run_path = resolve_host_path(run_dir)
    output_path = resolve_host_path(output) if output else run_path / "experiment_screenshots" / "trajectory_overview.png"
    metadata_path = resolve_host_path(metadata_output) if metadata_output else output_path.with_suffix(".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    spec = load_experiment_spec(config)
    states_path = run_path / "states.csv"
    if not states_path.exists():
        raise FileNotFoundError(states_path)
    states = pd.read_csv(states_path)
    required = {"robot_id", "x", "y", "time_s"}
    missing = sorted(required - set(states.columns))
    if missing:
        raise ValueError("states.csv missing columns: " + ", ".join(missing))

    fig, ax = plt.subplots(figsize=(8.2, 3.1), dpi=180)
    colors = plt.get_cmap("tab10")
    for idx, robot in enumerate(spec.robots):
        group = states[states["robot_id"] == robot.robot_id].sort_values("time_s")
        if group.empty:
            continue
        color = colors(idx % 10)
        ax.plot(group["x"], group["y"], color=color, linewidth=2.0, label=robot.robot_id)
        start_xy = robot.start_xy if robot.start_xy is not None else (float(group.iloc[0]["x"]), float(group.iloc[0]["y"]))
        goal_xy = robot.goal_xy
        ax.scatter([start_xy[0]], [start_xy[1]], marker="o", s=42, color=color, edgecolors="black", linewidths=0.6, zorder=3)
        ax.scatter([goal_xy[0]], [goal_xy[1]], marker="X", s=64, color=color, edgecolors="black", linewidths=0.6, zorder=3)
        if len(group) >= 2:
            tail = group.iloc[-2]
            head = group.iloc[-1]
            ax.annotate(
                "",
                xy=(float(head["x"]), float(head["y"])),
                xytext=(float(tail["x"]), float(tail["y"])),
                arrowprops={"arrowstyle": "->", "color": color, "lw": 1.3},
            )

    ax.set_title(f"{spec.method} in {spec.scenario}: audited Gazebo trajectory", fontsize=11, pad=10)
    ax.set_xlabel("x position (m)", labelpad=8)
    ax.set_ylabel("y position (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#d7dce2", linewidth=0.6, alpha=0.9)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        frameon=False,
        borderaxespad=0.0,
    )
    note = f"Markers: circle=start, X=goal. Safety distance={spec.safety_distance_m:.2f} m."
    fig.text(0.5, 0.02, note, ha="center", va="bottom", fontsize=8.5, color="#4b5563")
    fig.tight_layout(rect=(0.03, 0.17, 0.84, 0.95))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    metadata = {
        "schema": "icgp_trajectory_overview_v1",
        "claim_boundary": (
            "Rendered from audited trajectory CSV logs for visual explanation. "
            "It is not a raw GUI screenshot and must not replace metric tables."
        ),
        "config_path": str(config),
        "run_dir": str(run_path),
        "states_path": str(states_path),
        "output_png": str(output_path),
        "robots": len(spec.robots),
        "safety_distance_m": spec.safety_distance_m,
        "metric_source": "CSV/JSON logs, not image processing",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a trajectory overview PNG from an audited ICGP states.csv.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--metadata-output", default="")
    args = parser.parse_args()
    output = write_trajectory_plot(
        Path(args.config),
        Path(args.run_dir),
        Path(args.output) if args.output else None,
        Path(args.metadata_output) if args.metadata_output else None,
    )
    print(str(output), flush=True)


if __name__ == "__main__":
    main()
