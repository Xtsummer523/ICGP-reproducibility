from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source_snapshot.config import (  # noqa: E402
    MODELS,
    SimConfig,
    TrainConfig,
    observation_noise_level,
)
from source_snapshot.dataset import FEATURE_DIM, TARGET_DIM  # noqa: E402
from source_snapshot.evaluate import run_episode  # noqa: E402
from source_snapshot.models import ResidualIntentTrajectoryMLP, TrajectoryMLP  # noqa: E402
from source_snapshot.noise_stats import paired_delta_rows, summarize_method_metrics, summarize_paired_deltas  # noqa: E402
from source_snapshot.observation_noise import (  # noqa: E402
    OBSERVATION_NOISE_PROTOCOL_VERSION,
    SUPPORTED_OBSERVATION_NOISE_PROTOCOLS,
)


CONSTRAINED_SCENARIOS = ("corridor", "narrow_gate", "merge_bottleneck", "four_way_stop")
DEFAULT_ROBOTS = (4, 8)
DEFAULT_SEEDS = 30
DEFAULT_NOISE_LEVELS = ("clean", "low", "medium", "high")
DEFAULT_OUTPUT_DIR = ROOT / "output" / "observation_noise"
DEFAULT_METHODS = (
    "icgp_rvo_mpc",
    "passive_rvo_mpc",
    "rvo_reactive",
    "residual_passive_rvo_mpc",
)
PRIMARY_METHOD = "icgp_rvo_mpc"
REPORT_METRICS = (
    "collision",
    "safety_violation",
    "success",
    "progress_ratio",
    "final_goal_distance",
    "completion_time",
    "min_distance",
)


def parse_csv(value: str, cast=str) -> tuple:
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def stable_episode_seed(scenario: str, robots: int, seed_index: int) -> int:
    return 9000 + robots * 1000 + seed_index * 17 + sum(ord(char) for char in scenario)


def load_models(
    train_cfg: TrainConfig,
    model_dir: Path = MODELS,
) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module | None, str]:
    intent_path = model_dir / "intent_predictor.pt"
    passive_path = model_dir / "passive_predictor.pt"
    missing = [str(path) for path in (intent_path, passive_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Observation-noise evaluation requires the trained predictor artifacts. Missing: "
            f"{', '.join(missing)}. Train them first with scripts/run_pipeline.py or the project training workflow."
        )
    intent = ResidualIntentTrajectoryMLP(FEATURE_DIM, TARGET_DIM, train_cfg.hidden_dim, train_cfg.dropout)
    passive = TrajectoryMLP(FEATURE_DIM, TARGET_DIM, train_cfg.hidden_dim, train_cfg.dropout)
    intent.load_state_dict(torch.load(intent_path, map_location="cpu", weights_only=True))
    passive.load_state_dict(torch.load(passive_path, map_location="cpu", weights_only=True))
    intent.eval()
    passive.eval()
    residual_path = model_dir / "residual_passive_predictor.pt"
    if residual_path.exists():
        residual_passive = ResidualIntentTrajectoryMLP(FEATURE_DIM, TARGET_DIM, train_cfg.hidden_dim, train_cfg.dropout)
        residual_passive.load_state_dict(torch.load(residual_path, map_location="cpu", weights_only=True))
        residual_passive.eval()
        residual_mode = "trained_zero_intent_residual_model"
    else:
        residual_passive = None
        residual_mode = "diagnostic_zero_intent_on_intent_model"
    return intent, passive, residual_passive, residual_mode


def build_cases(
    scenarios: Sequence[str],
    robots: Sequence[int],
    seed_indices: Sequence[int],
    methods: Sequence[str],
    noise_levels: Sequence[str],
) -> list[tuple[str, int, int, int, str, str]]:
    """Yield balanced cases; methods share their scenario-agent-seed noise case."""

    return [
        (scenario, robot_count, seed_index, stable_episode_seed(scenario, robot_count, seed_index), noise_level, method)
        for seed_index in seed_indices
        for scenario in scenarios
        for robot_count in robots
        for noise_level in noise_levels
        for method in methods
    ]


def pairing_status(
    results: pd.DataFrame,
    scenarios: Sequence[str],
    robots: Sequence[int],
    seed_indices: Sequence[int],
    noise_levels: Sequence[str],
    methods: Sequence[str],
) -> dict:
    expected = {
        (noise_level, scenario, robot_count, stable_episode_seed(scenario, robot_count, seed_index), method)
        for seed_index in seed_indices
        for scenario in scenarios
        for robot_count in robots
        for noise_level in noise_levels
        for method in methods
    }
    observed = {
        (str(row.noise_level), str(row.scenario), int(row.robots), int(row.seed), str(row.method))
        for row in results.itertuples(index=False)
    }
    missing = expected - observed
    return {
        "expected_method_seed_rows": len(expected),
        "observed_method_seed_rows": len(observed),
        "missing_method_seed_rows": len(missing),
        "is_complete": not missing,
    }


def plot_robustness(summary: pd.DataFrame, path: Path) -> None:
    """Plot bootstrap 95% CIs for the primary progress and safety outcomes."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "icgp_rvo_mpc": "ICGP+RVO",
        "passive_rvo_mpc": "Passive+RVO",
        "rvo_reactive": "Reactive RVO",
        "residual_passive_rvo_mpc": "Residual-passive+RVO",
    }
    colors = {
        "icgp_rvo_mpc": "#1565c0",
        "passive_rvo_mpc": "#ef6c00",
        "rvo_reactive": "#6a1b9a",
        "residual_passive_rvo_mpc": "#2e7d32",
    }
    ordered_levels = [level for level in DEFAULT_NOISE_LEVELS if level in set(summary["noise_level"])]
    metric_specs = (("progress_ratio", "Progress ratio [higher is better]"), ("safety_violation", "Safety-violation rate [lower is better]"))
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), sharex=True)
    x = np.arange(len(ordered_levels))
    for axis, (metric, ylabel) in zip(axes, metric_specs):
        metric_rows = summary[summary["metric"] == metric]
        for method in DEFAULT_METHODS:
            method_rows = metric_rows[metric_rows["method"] == method].set_index("noise_level").reindex(ordered_levels)
            if method_rows.empty or method_rows["mean"].isna().all():
                continue
            mean = method_rows["mean"].to_numpy(dtype=float)
            low = method_rows["bootstrap_ci_low"].to_numpy(dtype=float)
            high = method_rows["bootstrap_ci_high"].to_numpy(dtype=float)
            errors = np.vstack((mean - low, high - mean))
            axis.errorbar(
                x,
                mean,
                yerr=errors,
                marker="o",
                linewidth=1.6,
                capsize=3,
                label=labels.get(method, method),
                color=colors.get(method),
            )
        axis.set_ylabel(ylabel)
        axis.set_xticks(x)
        axis.set_xticklabels(ordered_levels)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Observation-noise sensitivity (bootstrap 95% CIs)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired neighbor-observation noise sensitivity experiment.")
    parser.add_argument("--tag", default="observation_noise")
    parser.add_argument("--scenarios", default=",".join(CONSTRAINED_SCENARIOS))
    parser.add_argument("--robots", default=",".join(map(str, DEFAULT_ROBOTS)))
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--noise-levels", default=",".join(DEFAULT_NOISE_LEVELS))
    parser.add_argument("--protocol", choices=SUPPORTED_OBSERVATION_NOISE_PROTOCOLS, default=OBSERVATION_NOISE_PROTOCOL_VERSION)
    parser.add_argument("--ar-rho", type=float, default=0.85, help="AR(1) coefficient for the temporal-noise protocol.")
    parser.add_argument("--occlusion-probability", type=float, default=0.10, help="Per-step occlusion-start probability.")
    parser.add_argument("--occlusion-max-duration", type=int, default=3, help="Maximum consecutive occluded steps.")
    parser.add_argument("--model-dir", type=Path, default=None, help="Directory containing intent/passive predictor weights.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for raw results, statistics, metadata, and figures.")
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--limit-cases", type=int, default=0, help="Run only the first N missing cases; safe for smoke tests.")
    parser.add_argument("--torch-threads", type=int, default=0, help="Set PyTorch intra/inter-op CPU threads; 0 preserves the runtime default.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seeds < 1:
        raise ValueError("--seeds must be positive")
    if args.bootstrap_resamples < 1:
        raise ValueError("--bootstrap-resamples must be positive")
    if args.torch_threads < 0:
        raise ValueError("--torch-threads must be non-negative")
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
        try:
            torch.set_num_interop_threads(args.torch_threads)
        except RuntimeError:
            # PyTorch may already have initialized inter-op workers in an
            # embedding process; the intra-op setting remains effective.
            pass
    scenarios = parse_csv(args.scenarios)
    robot_counts = parse_csv(args.robots, int)
    methods = parse_csv(args.methods)
    noise_levels = parse_csv(args.noise_levels)
    if not scenarios or not robot_counts or not methods or not noise_levels:
        raise ValueError("scenarios, robots, methods, and noise levels must not be empty")
    for level in noise_levels:
        observation_noise_level(level)
    if PRIMARY_METHOD not in methods:
        raise ValueError(f"--methods must include the paired primary method '{PRIMARY_METHOD}'")

    seed_indices = tuple(range(args.seed_start, args.seed_start + args.seeds))
    cfg = replace(SimConfig(), max_steps=args.max_steps)
    output_dir = args.output_dir.resolve()
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"observation_noise_{args.tag}_results.csv"
    method_summary_path = results_dir / f"observation_noise_{args.tag}_method_summary.csv"
    paired_detail_path = results_dir / f"observation_noise_{args.tag}_paired_deltas.csv"
    paired_summary_path = results_dir / f"observation_noise_{args.tag}_paired_summary.csv"
    metadata_path = results_dir / f"observation_noise_{args.tag}_metadata.json"
    figure_path = figures_dir / f"observation_noise_{args.tag}.png"

    if result_path.exists():
        results = pd.read_csv(result_path)
        rows = results.to_dict(orient="records")
    else:
        rows = []
        results = pd.DataFrame()
    done_keys = {
        (str(row["noise_level"]), str(row["scenario"]), int(row["robots"]), int(row["seed"]), str(row["method"]))
        for row in rows
    }
    all_cases = build_cases(scenarios, robot_counts, seed_indices, methods, noise_levels)
    pending = [
        case
        for case in all_cases
        if (case[4], case[0], case[1], case[3], case[5]) not in done_keys
    ]
    if args.limit_cases > 0:
        pending = pending[: args.limit_cases]

    print(f"Existing rows: {len(rows)}; pending this run: {len(pending)}; planned total: {len(all_cases)}", flush=True)
    model_dir = (args.model_dir or MODELS).resolve()
    intent_model, passive_model, residual_passive_model, residual_passive_mode = load_models(TrainConfig(), model_dir)
    for index, (scenario, robot_count, seed_index, episode_seed, noise_level, method) in enumerate(pending, start=1):
        result, _ = run_episode(
            scenario,
            robot_count,
            episode_seed,
            method,
            cfg,
            intent_model,
            passive_model,
            noise_level=noise_level,
            residual_passive_model=residual_passive_model,
            observation_noise_protocol=args.protocol,
            ar_rho=args.ar_rho,
            occlusion_probability=args.occlusion_probability,
            occlusion_max_duration=args.occlusion_max_duration,
            record_trajectory=False,
        )
        result["seed_index"] = seed_index
        rows.append(result)
        pd.DataFrame(rows).to_csv(result_path, index=False)
        print(
            f"[{index}/{len(pending)}] noise={noise_level} {scenario} n={robot_count} {method} "
            f"seed_index={seed_index} collision={result['collision']} success={result['success']} "
            f"progress={result['progress_ratio']:.3f}",
            flush=True,
        )

    results = pd.DataFrame(rows)
    if not results.empty:
        method_summary = summarize_method_metrics(results, REPORT_METRICS, n_resamples=args.bootstrap_resamples)
        baselines = tuple(method for method in methods if method != PRIMARY_METHOD)
        paired_detail = paired_delta_rows(results, PRIMARY_METHOD, baselines, REPORT_METRICS)
        paired_summary = summarize_paired_deltas(paired_detail, n_resamples=args.bootstrap_resamples) if not paired_detail.empty else pd.DataFrame()
        method_summary.to_csv(method_summary_path, index=False)
        paired_detail.to_csv(paired_detail_path, index=False)
        paired_summary.to_csv(paired_summary_path, index=False)
        plot_robustness(method_summary, figure_path)
    else:
        method_summary = pd.DataFrame()
        paired_detail = pd.DataFrame()
        paired_summary = pd.DataFrame()

    completion = pairing_status(results, scenarios, robot_counts, seed_indices, noise_levels, methods)
    metadata = {
        "tag": args.tag,
        "purpose": "Controlled simulation stress test, not a real-sensor validation.",
        "observation_scope": "Only neighbor positions and velocities are altered; ego state, goals, obstacle geometry, and simulator dynamics remain true-state quantities.",
        "protocol": args.protocol,
        "protocol_parameters": {
            "ar_rho": args.ar_rho,
            "occlusion_probability": args.occlusion_probability,
            "occlusion_max_duration": args.occlusion_max_duration,
        },
        "common_random_numbers": "A full deterministic schedule is pre-generated from scenario, robot count, episode seed, noise level, and protocol. Every method in a paired case reads the same schedule.",
        "noise_levels": [asdict(observation_noise_level(level)) for level in noise_levels],
        "pairing_keys": ["noise_level", "scenario", "robots", "seed"],
        "methods": methods,
        "primary_method": PRIMARY_METHOD,
        "baseline_methods": [method for method in methods if method != PRIMARY_METHOD],
        "model_dir": str(model_dir),
        "residual_passive_control_mode": residual_passive_mode,
        "scenarios": scenarios,
        "robot_counts": robot_counts,
        "seed_indices": seed_indices,
        "sim_config": asdict(cfg),
        "runtime": {
            "torch_threads_requested": args.torch_threads,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
        },
        "metrics": REPORT_METRICS,
        "bootstrap": {"type": "percentile bootstrap", "confidence": 0.95, "resamples": args.bootstrap_resamples},
        "paired_test": {
            "type": "two-sided paired sign-flip randomization test",
            "small_sample_mode": "exact enumeration for 16 or fewer paired cases",
            "large_sample_mode": "seeded Monte-Carlo sign flips with plus-one correction",
        },
        "completion": completion,
        "outputs": {
            "raw_results": str(result_path),
            "method_summary": str(method_summary_path),
            "paired_deltas": str(paired_detail_path),
            "paired_summary": str(paired_summary_path),
            "figure": str(figure_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {result_path}")
    print(f"Saved {metadata_path}")
    if not paired_summary.empty:
        selected = paired_summary[paired_summary["metric"].isin(("collision", "safety_violation", "progress_ratio", "final_goal_distance"))]
        print(selected.round(4).to_string(index=False))
    if not completion["is_complete"]:
        print(f"Run is incomplete: {completion['missing_method_seed_rows']} paired method-seed rows remain.")


if __name__ == "__main__":
    main()

