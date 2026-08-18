"""Run a reproducible architecture-by-candidate-conditioning ablation.

The four cells use the same synthetic states, optimizer schedule, training seed,
candidate set, RVO support layer, and paired Gaussian observation cases. They
only differ in predictor parameterization and whether candidate acceleration is
present in the predictor input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import SimConfig, TrainConfig  # noqa: E402
from src.dataset import FEATURE_DIM, TARGET_DIM, PredictionDataset, generate_prediction_arrays  # noqa: E402
from src.evaluate import run_episode  # noqa: E402
from src.models import ResidualIntentTrajectoryMLP, TrajectoryMLP  # noqa: E402
from src.noise_stats import (  # noqa: E402
    bootstrap_mean_ci,
    paired_delta_rows,
    paired_sign_flip_pvalue,
    summarize_method_metrics,
    summarize_paired_deltas,
)


CONSTRAINED_SCENARIOS = ("corridor", "narrow_gate", "merge_bottleneck", "four_way_stop")
DEFAULT_ROBOTS = (4, 8)
DEFAULT_NOISE_LEVELS = ("clean", "low", "medium", "high")
DEFAULT_ENVIRONMENT_SEEDS = 30
PRIMARY_METRICS = (
    "collision",
    "safety_violation",
    "success",
    "progress_ratio",
    "final_goal_distance",
    "completion_time",
    "min_distance",
)
DIAGNOSTIC_METRICS = (
    "screen_reject_rate",
    "screen_all_reject_rate",
    "rvo_intervention_rate",
    "latency_ms",
)
REPORT_METRICS = PRIMARY_METRICS + DIAGNOSTIC_METRICS


@dataclass(frozen=True)
class ConditionSpec:
    label: str
    architecture: str
    candidate_conditioned: bool
    feature_key: str
    planner_method: str


CONDITION_SPECS = {
    "plain_candidate": ConditionSpec(
        label="plain_candidate",
        architecture="plain",
        candidate_conditioned=True,
        feature_key="x_intent",
        planner_method="icgp_rvo_mpc",
    ),
    "plain_zero": ConditionSpec(
        label="plain_zero",
        architecture="plain",
        candidate_conditioned=False,
        feature_key="x_passive",
        planner_method="passive_rvo_mpc",
    ),
    "residual_candidate": ConditionSpec(
        label="residual_candidate",
        architecture="residual",
        candidate_conditioned=True,
        feature_key="x_intent",
        planner_method="icgp_rvo_mpc",
    ),
    "residual_zero": ConditionSpec(
        label="residual_zero",
        architecture="residual",
        candidate_conditioned=False,
        feature_key="x_passive",
        planner_method="residual_passive_rvo_mpc",
    ),
}

FACTORIAL_CONTRASTS = (
    ("plain_conditioning", "plain_candidate", "plain_zero"),
    ("residual_conditioning", "residual_candidate", "residual_zero"),
    ("candidate_architecture", "residual_candidate", "plain_candidate"),
    ("zero_architecture", "residual_zero", "plain_zero"),
)


def condition_by_label(label: str) -> ConditionSpec:
    try:
        return CONDITION_SPECS[label]
    except KeyError as error:
        raise ValueError(f"Unknown ablation condition: {label}") from error


def expected_case_count(
    scenarios: Sequence[str],
    robots: Sequence[int],
    seed_indices: Sequence[int],
    noise_levels: Sequence[str],
) -> int:
    return len(CONDITION_SPECS) * len(scenarios) * len(robots) * len(seed_indices) * len(noise_levels)


def parse_csv(value: str, cast=str) -> tuple:
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def stable_episode_seed(scenario: str, robots: int, seed_index: int) -> int:
    return 9000 + robots * 1000 + seed_index * 17 + sum(ord(char) for char in scenario)


def formal_train_config(seed: int) -> TrainConfig:
    return TrainConfig(
        seed=seed,
        train_samples=2600,
        val_samples=600,
        test_samples=600,
        batch_size=256,
        epochs=12,
        lr=1.0e-3,
        hidden_dim=96,
        dropout=0.05,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def configure_torch_threads(threads: int) -> None:
    if threads < 1:
        raise ValueError("torch thread count must be positive")
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        # An embedding process may already have initialized inter-op workers.
        pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_metrics(pred: torch.Tensor, target: torch.Tensor, cfg: SimConfig) -> tuple[float, float]:
    prediction = pred.detach().cpu().numpy().reshape(-1, cfg.horizon, cfg.neighbor_limit, 2) * 6.0
    truth = target.detach().cpu().numpy().reshape(-1, cfg.horizon, cfg.neighbor_limit, 2) * 6.0
    valid = np.linalg.norm(truth, axis=-1) > 1.0e-6
    error = np.linalg.norm(prediction - truth, axis=-1)
    ade = float(error[valid].mean()) if np.any(valid) else 0.0
    final_valid = valid[:, -1, :]
    fde = float(error[:, -1, :][final_valid].mean()) if np.any(final_valid) else 0.0
    return ade, fde


def build_model(spec: ConditionSpec, train_cfg: TrainConfig) -> torch.nn.Module:
    if spec.architecture == "plain":
        return TrajectoryMLP(FEATURE_DIM, TARGET_DIM, train_cfg.hidden_dim, train_cfg.dropout)
    if spec.architecture == "residual":
        return ResidualIntentTrajectoryMLP(FEATURE_DIM, TARGET_DIM, train_cfg.hidden_dim, train_cfg.dropout)
    raise ValueError(f"Unsupported architecture: {spec.architecture}")


def model_path(output_dir: Path, label: str) -> Path:
    return output_dir / "models" / f"{label}.pt"


def train_factorial_models(output_dir: Path, training_seed: int, force: bool) -> None:
    train_cfg = formal_train_config(training_seed)
    train_sim_cfg = replace(SimConfig(), max_steps=120)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    for label in CONDITION_SPECS:
        path = model_path(output_dir, label)
        if path.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite existing model {path}; pass --force to retrain.")

    total_samples = train_cfg.train_samples + train_cfg.val_samples + train_cfg.test_samples
    arrays = generate_prediction_arrays(train_sim_cfg, total_samples, seed=training_seed)
    dataset_path = output_dir / "training_dataset.npz"
    np.savez_compressed(dataset_path, **arrays)
    train_slice = slice(0, train_cfg.train_samples)
    val_slice = slice(train_cfg.train_samples, train_cfg.train_samples + train_cfg.val_samples)
    test_slice = slice(train_cfg.train_samples + train_cfg.val_samples, total_samples)
    loss_fn = nn.SmoothL1Loss()
    histories: list[dict] = []
    test_rows: list[dict] = []

    for spec in CONDITION_SPECS.values():
        # Resetting both initialization and sampling seeds means a pair differs
        # only in the final candidate-action feature values and architecture.
        set_seed(training_seed)
        model = build_model(spec, train_cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=1.0e-4)
        train_ds = PredictionDataset(arrays[spec.feature_key][train_slice], arrays["y"][train_slice])
        loader_generator = torch.Generator().manual_seed(training_seed)
        loader = DataLoader(
            train_ds,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            drop_last=False,
            generator=loader_generator,
        )
        val_x = torch.from_numpy(arrays[spec.feature_key][val_slice])
        val_y = torch.from_numpy(arrays["y"][val_slice])
        for epoch in range(1, train_cfg.epochs + 1):
            model.train()
            losses = []
            for xb, yb in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.item()))
            model.eval()
            with torch.no_grad():
                val_pred = model(val_x)
                val_loss = float(loss_fn(val_pred, val_y).item())
                val_ade, val_fde = prediction_metrics(val_pred, val_y, train_sim_cfg)
            histories.append(
                {
                    "condition": spec.label,
                    "architecture": spec.architecture,
                    "candidate_conditioned": int(spec.candidate_conditioned),
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    "val_loss": val_loss,
                    "val_ADE": val_ade,
                    "val_FDE": val_fde,
                }
            )
        test_x = torch.from_numpy(arrays[spec.feature_key][test_slice])
        test_y = torch.from_numpy(arrays["y"][test_slice])
        with torch.no_grad():
            test_pred = model(test_x)
            test_loss = float(loss_fn(test_pred, test_y).item())
            test_ade, test_fde = prediction_metrics(test_pred, test_y, train_sim_cfg)
        model.eval()
        path = model_path(output_dir, spec.label)
        torch.save(model.state_dict(), path)
        test_rows.append(
            {
                "condition": spec.label,
                "architecture": spec.architecture,
                "candidate_conditioned": int(spec.candidate_conditioned),
                "test_loss": test_loss,
                "ADE": test_ade,
                "FDE": test_fde,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "model_sha256": sha256(path),
            }
        )
        print(f"Trained {spec.label}: ADE={test_ade:.4f}, FDE={test_fde:.4f}", flush=True)

    pd.DataFrame(histories).to_csv(output_dir / "training_history.csv", index=False)
    pd.DataFrame(test_rows).to_csv(output_dir / "prediction_test_metrics.csv", index=False)
    manifest = {
        "purpose": "Fully crossed plain/residual architecture by candidate-conditioning experiment.",
        "training_seed": training_seed,
        "training_sim_config": asdict(train_sim_cfg),
        "training_config": asdict(train_cfg),
        "conditions": [asdict(spec) for spec in CONDITION_SPECS.values()],
        "dataset": {"path": str(dataset_path), "sha256": sha256(dataset_path), "samples": total_samples},
        "invariant": "Every condition uses identical generated states, targets, initialization seed, DataLoader permutation seed, optimizer, loss, clipping rule, batch size, and epoch count.",
    }
    (output_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_factorial_models(output_dir: Path, training_seed: int) -> dict[str, torch.nn.Module]:
    train_cfg = formal_train_config(training_seed)
    models: dict[str, torch.nn.Module] = {}
    for label, spec in CONDITION_SPECS.items():
        path = model_path(output_dir, label)
        if not path.exists():
            raise FileNotFoundError(f"Missing trained model: {path}")
        model = build_model(spec, train_cfg)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        models[label] = model
    return models


def shard_path(output_dir: Path, seed_start: int, seeds: int) -> Path:
    return output_dir / "shards" / f"results_seed_{seed_start:03d}_{seed_start + seeds - 1:03d}.csv"


def factorial_interaction_rows(results: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    """Return the residual-minus-plain difference of conditioning effects.

    For a higher-is-better metric, a positive value means candidate conditioning
    improves that metric more under the residual architecture than under the
    plain MLP. For lower-is-better metrics, the raw sign is retained so readers
    can inspect the conventional outcome scale without imposing an artificial
    desirability direction on an interaction term.
    """

    key_columns = ["noise_level", "scenario", "robots", "seed_index"]
    required = set(key_columns) | {"method"} | set(metrics)
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Cannot calculate factorial interaction; missing columns: {sorted(missing)}")
    duplicates = results.duplicated(key_columns + ["method"], keep=False)
    if duplicates.any():
        raise ValueError("Each factorial case must contain one row per condition")

    indexed = results.set_index(key_columns + ["method"])
    rows: list[dict] = []
    for keys, group in indexed.groupby(level=key_columns, sort=False):
        methods = group.reset_index(level=key_columns, drop=True)
        if not set(CONDITION_SPECS).issubset(methods.index):
            continue
        key_dict = dict(zip(key_columns, keys if isinstance(keys, tuple) else (keys,)))
        for metric in metrics:
            residual_effect = float(methods.loc["residual_candidate", metric]) - float(
                methods.loc["residual_zero", metric]
            )
            plain_effect = float(methods.loc["plain_candidate", metric]) - float(methods.loc["plain_zero", metric])
            rows.append(
                {
                    **key_dict,
                    "contrast": "residual_conditioning_minus_plain_conditioning",
                    "metric": metric,
                    "direction": "interaction",
                    "residual_conditioning_effect": residual_effect,
                    "plain_conditioning_effect": plain_effect,
                    "interaction_delta": residual_effect - plain_effect,
                }
            )
    return pd.DataFrame(rows)


def summarize_factorial_interactions(detail: pd.DataFrame, n_resamples: int) -> pd.DataFrame:
    """Summarize each noise-stratified 2x2 interaction with paired resampling."""

    required = {"noise_level", "contrast", "metric", "interaction_delta"}
    missing = required - set(detail.columns)
    if missing:
        raise ValueError(f"Cannot summarize factorial interaction; missing columns: {sorted(missing)}")
    rows: list[dict] = []
    for (noise_level, contrast, metric), group in detail.groupby(["noise_level", "contrast", "metric"], sort=True):
        values = group["interaction_delta"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        low, high = bootstrap_mean_ci(
            values,
            n_resamples=n_resamples,
            seed=int.from_bytes(
                hashlib.blake2b(
                    f"interaction|{noise_level}|{contrast}|{metric}".encode("utf-8"), digest_size=8
                ).digest(),
                "little",
            ),
        )
        rows.append(
            {
                "noise_level": noise_level,
                "contrast": contrast,
                "metric": metric,
                "paired_cases": int(len(values)),
                "mean_interaction_delta": float(values.mean()) if len(values) else float("nan"),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "sign_flip_p": paired_sign_flip_pvalue(
                    values,
                    n_resamples=n_resamples,
                    seed=int.from_bytes(
                        hashlib.blake2b(
                            f"interaction-sign-flip|{noise_level}|{contrast}|{metric}".encode("utf-8"), digest_size=8
                        ).digest(),
                        "little",
                    ),
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["noise_level", "contrast", "metric"]).reset_index(drop=True)


def shard_cases(
    scenarios: Sequence[str],
    robots: Sequence[int],
    seed_indices: Sequence[int],
    noise_levels: Sequence[str],
) -> Iterable[tuple[str, int, int, str, ConditionSpec]]:
    for seed_index in seed_indices:
        for scenario in scenarios:
            for robot_count in robots:
                for noise_level in noise_levels:
                    for spec in CONDITION_SPECS.values():
                        yield scenario, robot_count, seed_index, noise_level, spec


def run_evaluation_shard(
    output_dir: Path,
    training_seed: int,
    scenarios: Sequence[str],
    robots: Sequence[int],
    seed_start: int,
    seeds: int,
    noise_levels: Sequence[str],
    max_steps: int,
) -> Path:
    if seeds < 1:
        raise ValueError("seeds must be positive")
    models = load_factorial_models(output_dir, training_seed)
    path = shard_path(output_dir, seed_start, seeds)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        rows = pd.read_csv(path).to_dict(orient="records")
    else:
        rows = []
    done = {
        (str(row["noise_level"]), str(row["scenario"]), int(row["robots"]), int(row["seed_index"]), str(row["method"]))
        for row in rows
    }
    cfg = replace(SimConfig(), max_steps=max_steps)
    seed_indices = tuple(range(seed_start, seed_start + seeds))
    cases = list(shard_cases(scenarios, robots, seed_indices, noise_levels))
    pending = [
        case
        for case in cases
        if (case[3], case[0], case[1], case[2], case[4].label) not in done
    ]
    print(f"Shard {path.name}: existing={len(rows)}, pending={len(pending)}, planned={len(cases)}", flush=True)
    for index, (scenario, robot_count, seed_index, noise_level, spec) in enumerate(pending, start=1):
        result, _ = run_episode(
            scenario,
            robot_count,
            stable_episode_seed(scenario, robot_count, seed_index),
            spec.planner_method,
            cfg,
            intent_model=models[spec.label],
            passive_model=models[spec.label],
            residual_passive_model=models[spec.label],
            noise_level=noise_level,
            record_trajectory=False,
            batch_across_robots=True,
        )
        result.update(
            {
                "method": spec.label,
                "architecture": spec.architecture,
                "candidate_conditioned": int(spec.candidate_conditioned),
                "feature_key": spec.feature_key,
                "planner_method": spec.planner_method,
                "training_seed": training_seed,
                "seed_index": seed_index,
            }
        )
        rows.append(result)
        pd.DataFrame(rows).to_csv(path, index=False)
        print(
            f"[{index}/{len(pending)}] {spec.label} {noise_level} {scenario} n={robot_count} "
            f"seed={seed_index} progress={result['progress_ratio']:.3f}",
            flush=True,
        )
    return path


def combine_shards(
    output_dir: Path,
    scenarios: Sequence[str],
    robots: Sequence[int],
    seeds: int,
    noise_levels: Sequence[str],
    bootstrap_resamples: int,
    max_steps: int,
) -> None:
    shard_files = sorted((output_dir / "shards").glob("results_seed_*.csv"))
    if not shard_files:
        raise FileNotFoundError(f"No shard result files found in {output_dir / 'shards'}")
    results = pd.concat((pd.read_csv(path) for path in shard_files), ignore_index=True)
    keys = ["noise_level", "scenario", "robots", "seed_index", "method"]
    duplicates = results.duplicated(keys, keep=False)
    if duplicates.any():
        raise ValueError("Duplicate factorial case rows found while merging shards")
    expected = expected_case_count(scenarios, robots, tuple(range(seeds)), noise_levels)
    completion = {
        "expected_rows": expected,
        "observed_rows": int(len(results)),
        "missing_rows": int(max(0, expected - len(results))),
        "is_complete": len(results) == expected,
    }
    combined_path = output_dir / "factorial_results.csv"
    results.sort_values(keys).to_csv(combined_path, index=False)
    method_summary = summarize_method_metrics(results, REPORT_METRICS, n_resamples=bootstrap_resamples)
    detail_frames = []
    for _name, primary, baseline in FACTORIAL_CONTRASTS:
        detail_frames.append(paired_delta_rows(results, primary, (baseline,), PRIMARY_METRICS, key_columns=keys[:-1]))
    paired_detail = pd.concat(detail_frames, ignore_index=True)
    paired_summary = summarize_paired_deltas(paired_detail, n_resamples=bootstrap_resamples)
    interaction_detail = factorial_interaction_rows(results, PRIMARY_METRICS)
    interaction_summary = summarize_factorial_interactions(
        interaction_detail,
        n_resamples=bootstrap_resamples,
    )
    method_summary.to_csv(output_dir / "factorial_method_summary.csv", index=False)
    paired_detail.to_csv(output_dir / "factorial_paired_deltas.csv", index=False)
    paired_summary.to_csv(output_dir / "factorial_paired_summary.csv", index=False)
    interaction_detail.to_csv(output_dir / "factorial_interaction_deltas.csv", index=False)
    interaction_summary.to_csv(output_dir / "factorial_interaction_summary.csv", index=False)
    manifest = {
        "purpose": "Gaussian observation-noise evaluation of a fully crossed architecture-by-candidate-conditioning model set.",
        "scenarios": list(scenarios),
        "robots": list(robots),
        "environment_seed_indices": list(range(seeds)),
        "noise_levels": list(noise_levels),
        "conditions": [asdict(spec) for spec in CONDITION_SPECS.values()],
        "factorial_contrasts": [list(item) for item in FACTORIAL_CONTRASTS],
        "evaluation_sim_config": asdict(replace(SimConfig(), max_steps=max_steps)),
        "completion": completion,
        "outputs": {
            "raw_results": str(combined_path),
            "method_summary": str(output_dir / "factorial_method_summary.csv"),
            "paired_deltas": str(output_dir / "factorial_paired_deltas.csv"),
            "paired_summary": str(output_dir / "factorial_paired_summary.csv"),
            "interaction_deltas": str(output_dir / "factorial_interaction_deltas.csv"),
            "interaction_summary": str(output_dir / "factorial_interaction_summary.csv"),
        },
    }
    (output_dir / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(completion), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "evaluate", "merge"), required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "ICGP_JINT_Submission" / "experiments" / "architecture_conditioning_ablation",
    )
    parser.add_argument("--training-seed", type=int, default=7)
    parser.add_argument("--scenarios", default=",".join(CONSTRAINED_SCENARIOS))
    parser.add_argument("--robots", default=",".join(map(str, DEFAULT_ROBOTS)))
    parser.add_argument("--noise-levels", default=",".join(DEFAULT_NOISE_LEVELS))
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=DEFAULT_ENVIRONMENT_SEEDS)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="Permit replacing existing training artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_torch_threads(args.torch_threads)
    scenarios = parse_csv(args.scenarios)
    robots = parse_csv(args.robots, int)
    noise_levels = parse_csv(args.noise_levels)
    if not scenarios or not robots or not noise_levels:
        raise ValueError("scenarios, robots, and noise levels must not be empty")
    if args.mode == "train":
        train_factorial_models(args.output_dir.resolve(), args.training_seed, args.force)
    elif args.mode == "evaluate":
        run_evaluation_shard(
            args.output_dir.resolve(),
            args.training_seed,
            scenarios,
            robots,
            args.seed_start,
            args.seeds,
            noise_levels,
            args.max_steps,
        )
    else:
        combine_shards(
            args.output_dir.resolve(),
            scenarios,
            robots,
            args.seeds,
            noise_levels,
            args.bootstrap_resamples,
            args.max_steps,
        )


if __name__ == "__main__":
    main()
