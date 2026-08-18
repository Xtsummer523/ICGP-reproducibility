from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_PARENT = ROOT / "observation_noise_experiment"
SOURCE_PARENT = Path(os.environ.get("ICGP_PREDICTION_SOURCE_PARENT", str(DEFAULT_SOURCE_PARENT))).resolve()
SOURCE_ROOT = SOURCE_PARENT / "source_snapshot"
MODEL_ROOT = SOURCE_ROOT.parent / "models"
if str(SOURCE_PARENT) not in sys.path:
    sys.path.insert(0, str(SOURCE_PARENT))

from source_snapshot.config import SimConfig, TrainConfig  # noqa: E402
from source_snapshot.dataset import FEATURE_DIM, TARGET_DIM  # noqa: E402
from source_snapshot.evaluate import run_episode  # noqa: E402
from source_snapshot.models import ResidualIntentTrajectoryMLP, TrajectoryMLP  # noqa: E402
from source_snapshot.observation_noise import (  # noqa: E402
    OBSERVATION_NOISE_PROTOCOL_VERSION,
    SUPPORTED_OBSERVATION_NOISE_PROTOCOLS,
)


METHODS = ("icgp_rvo_mpc", "passive_rvo_mpc", "residual_passive_rvo_mpc")
SOURCE_FILES = (
    "config.py",
    "dataset.py",
    "evaluate.py",
    "models.py",
    "noise_stats.py",
    "observation_noise.py",
    "scenarios.py",
    "sim.py",
)


def stable_episode_seed(scenario: str, robots: int, seed_index: int) -> int:
    return 9000 + robots * 1000 + seed_index * 17 + sum(ord(char) for char in scenario)


def load_models(model_dir: Path, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module | None]:
    train_cfg = TrainConfig()
    intent = ResidualIntentTrajectoryMLP(FEATURE_DIM, TARGET_DIM, train_cfg.hidden_dim, train_cfg.dropout)
    passive = TrajectoryMLP(FEATURE_DIM, TARGET_DIM, train_cfg.hidden_dim, train_cfg.dropout)
    intent.load_state_dict(torch.load(model_dir / "intent_predictor.pt", map_location=device, weights_only=True))
    passive.load_state_dict(torch.load(model_dir / "passive_predictor.pt", map_location=device, weights_only=True))
    residual = None
    residual_path = model_dir / "residual_passive_predictor.pt"
    if residual_path.exists():
        residual = ResidualIntentTrajectoryMLP(FEATURE_DIM, TARGET_DIM, train_cfg.hidden_dim, train_cfg.dropout)
        residual.load_state_dict(torch.load(residual_path, map_location=device, weights_only=True))
    for model in (intent, passive, residual):
        if model is not None:
            model.to(device)
            model.eval()
    return intent, passive, residual


def parse_csv(value: str, cast=str) -> tuple:
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_trace_cases(trace: pd.DataFrame) -> pd.DataFrame:
    if trace.empty:
        return pd.DataFrame()
    trace = trace.copy()
    trace["error"] = np.hypot(trace["pred_x"] - trace["actual_x"], trace["pred_y"] - trace["actual_y"])
    trace = trace.sort_values(
        ["scenario", "robots", "seed", "method", "noise_level", "step", "ego", "action_index", "neighbor", "horizon_step"]
    )
    event_keys = ["scenario", "robots", "seed", "method", "noise_level", "step", "ego", "action_index", "selected", "neighbor"]
    selected = trace[trace["selected"] == 1]
    selected_event = selected.groupby(event_keys, as_index=False).agg(
        ade=("error", "mean"),
        fde=("error", "last"),
        horizons=("horizon_step", "count"),
    )
    diversity_keys = ["scenario", "robots", "seed", "method", "noise_level", "step", "ego", "neighbor", "horizon_step"]
    diversity = trace.groupby(diversity_keys, as_index=False).agg(
        pred_x_std=("pred_x", "std"),
        pred_y_std=("pred_y", "std"),
    )
    diversity["candidate_spread"] = np.hypot(diversity["pred_x_std"].fillna(0.0), diversity["pred_y_std"].fillna(0.0))
    summary_keys = ["scenario", "robots", "seed", "method", "noise_level"]
    summary = selected_event.groupby(summary_keys, as_index=False).agg(
        ADE=("ade", "mean"),
        FDE=("fde", "mean"),
        selected_prediction_events=("ade", "size"),
        selected_horizon_points=("horizons", "sum"),
    )
    summary["effective_horizon"] = summary["selected_horizon_points"] / summary["selected_prediction_events"].clip(lower=1)
    diversity_summary = diversity.groupby(summary_keys, as_index=False).agg(candidate_spread=("candidate_spread", "mean"))
    return summary.merge(diversity_summary, on=summary_keys, how="left")


def summarize_cases(case_summary: pd.DataFrame) -> pd.DataFrame:
    if case_summary.empty:
        return pd.DataFrame()
    return case_summary.groupby(["method", "noise_level"], as_index=False).agg(
        ADE=("ADE", "mean"),
        FDE=("FDE", "mean"),
        candidate_spread=("candidate_spread", "mean"),
        effective_horizon=("effective_horizon", "mean"),
        cases=("ADE", "size"),
        selected_prediction_events=("selected_prediction_events", "sum"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute on-policy ADE/FDE and candidate-response diversity diagnostics.")
    parser.add_argument("--scenarios", default="corridor,narrow_gate,merge_bottleneck,four_way_stop")
    parser.add_argument("--robots", default="4,8")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--noise-levels", default="clean,high")
    parser.add_argument("--protocol", choices=SUPPORTED_OBSERVATION_NOISE_PROTOCOLS, default=OBSERVATION_NOISE_PROTOCOL_VERSION)
    parser.add_argument("--ar-rho", type=float, default=0.85)
    parser.add_argument("--occlusion-probability", type=float, default=0.10)
    parser.add_argument("--occlusion-max-duration", type=int, default=3)
    parser.add_argument("--model-dir", type=Path, default=MODEL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evidence" / "prediction_quality")
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--torch-threads", type=int, default=0, help="Set PyTorch intra/inter-op CPU threads; 0 preserves the runtime default.")
    parser.add_argument("--retain-raw-trace", action="store_true", help="Retain the large per-candidate prediction trace CSV.")
    args = parser.parse_args()

    if args.torch_threads < 0:
        raise ValueError("--torch-threads must be non-negative")
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
        try:
            torch.set_num_interop_threads(args.torch_threads)
        except RuntimeError:
            pass

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in the active PyTorch environment")

    scenarios = parse_csv(args.scenarios)
    robots = parse_csv(args.robots, int)
    noise_levels = parse_csv(args.noise_levels)
    intent, passive, residual = load_models(args.model_dir.resolve(), device)
    cfg = SimConfig(max_steps=args.max_steps)
    records = []
    case_summaries = []
    for seed_index in range(args.seed_start, args.seed_start + args.seeds):
        for scenario in scenarios:
            for robot_count in robots:
                episode_seed = stable_episode_seed(scenario, robot_count, seed_index)
                for noise_level in noise_levels:
                    for method in METHODS:
                        result, _ = run_episode(
                            scenario,
                            robot_count,
                            episode_seed,
                            method,
                            cfg,
                            intent,
                            passive,
                            noise_level=noise_level,
                            residual_passive_model=residual,
                            observation_noise_protocol=args.protocol,
                            ar_rho=args.ar_rho,
                            occlusion_probability=args.occlusion_probability,
                            occlusion_max_duration=args.occlusion_max_duration,
                            trace_predictions=True,
                            record_trajectory=False,
                        )
                        trace = pd.DataFrame(result.pop("_prediction_trace", []))
                        if not trace.empty:
                            case_summary = summarize_trace_cases(trace)
                            case_summary.insert(2, "seed_index", seed_index)
                            case_summaries.append(case_summary)
                            if args.retain_raw_trace:
                                records.extend(trace.to_dict(orient="records"))
    raw = pd.DataFrame(records)
    case_summary = pd.concat(case_summaries, ignore_index=True) if case_summaries else pd.DataFrame()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "prediction_quality_raw.csv"
    case_summary_path = output_dir / "prediction_quality_case_summary.csv"
    summary_path = output_dir / "prediction_quality_summary.csv"
    metadata_path = output_dir / "prediction_quality_metadata.json"
    manifest_path = output_dir / "prediction_quality_manifest.json"
    summary = summarize_cases(case_summary)
    if args.retain_raw_trace:
        raw.to_csv(raw_path, index=False)
    elif raw_path.exists():
        raw_path.unlink()
    case_summary.to_csv(case_summary_path, index=False)
    summary.to_csv(summary_path, index=False)
    metadata = {
        "purpose": "On-policy prediction mechanism diagnostic; not a primary outcome estimand.",
        "scenarios": scenarios,
        "robots": robots,
        "seed_indices": tuple(range(args.seed_start, args.seed_start + args.seeds)),
        "noise_levels": noise_levels,
        "methods": METHODS,
        "protocol": args.protocol,
        "protocol_parameters": {
            "ar_rho": args.ar_rho,
            "occlusion_probability": args.occlusion_probability,
            "occlusion_max_duration": args.occlusion_max_duration,
        },
        "max_steps": args.max_steps,
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "metrics": ["ADE", "FDE", "candidate_spread", "effective_horizon"],
        "aggregation": "case-level mean: one row per scenario, robot count, seed index, method, and noise level; each case is internally averaged over selected prediction events.",
        "raw_trace_retained": bool(args.retain_raw_trace),
        "model_hashes": {str(path): sha256(path) for path in sorted(args.model_dir.resolve().glob("*.pt"))},
        "source_hashes": {str(SOURCE_ROOT / name): sha256(SOURCE_ROOT / name) for name in SOURCE_FILES},
        "runner_hash": sha256(Path(__file__).resolve()),
        "raw_rows": int(len(raw)),
        "case_rows": int(len(case_summary)),
        "summary_rows": int(len(summary)),
        "outputs": {"raw": str(raw_path) if args.retain_raw_trace else None, "case_summary": str(case_summary_path), "summary": str(summary_path)},
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "metadata": str(metadata_path),
        "sha256": {
            str(path): sha256(path)
            for path in (raw_path, case_summary_path, summary_path, metadata_path)
            if path.exists()
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.round(5).to_string(index=False))


if __name__ == "__main__":
    main()

