from __future__ import annotations

import hashlib
from itertools import product
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


METRIC_DIRECTIONS = {
    "collision": "lower",
    "safety_violation": "lower",
    "success": "higher",
    "min_distance": "higher",
    "completion_time": "lower",
    "final_goal_distance": "lower",
    "max_final_goal_distance": "lower",
    "progress_ratio": "higher",
    "path_length": "lower",
    "control_effort": "lower",
    "smoothness": "lower",
    "latency_ms": "lower",
}


def _finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def bootstrap_mean_ci(values: Iterable[float], n_resamples: int = 10000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for a mean with deterministic resampling."""

    array = _finite(values)
    if len(array) == 0:
        return float("nan"), float("nan")
    if len(array) == 1 or np.all(array == array[0]):
        value = float(array[0])
        return value, value
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(array), size=(n_resamples, len(array)))
    means = array[sample_indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def paired_sign_flip_pvalue(values: Iterable[float], n_resamples: int = 20000, seed: int = 0) -> float:
    """Two-sided paired randomization test for a non-zero mean paired delta.

    Enumerates all sign assignments up to 16 pairs and otherwise uses a seeded
    Monte-Carlo sign-flip approximation with the standard plus-one correction.
    """

    array = _finite(values)
    if len(array) == 0:
        return float("nan")
    observed = abs(float(array.mean()))
    if observed == 0.0:
        return 1.0
    if len(array) <= 16:
        sign_sets = np.asarray(list(product((-1.0, 1.0), repeat=len(array))), dtype=float)
        null_statistics = np.abs((sign_sets * array).mean(axis=1))
        return float(np.mean(null_statistics >= observed - 1.0e-12))
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_resamples, len(array)))
    null_statistics = np.abs((signs * array).mean(axis=1))
    exceedances = int(np.count_nonzero(null_statistics >= observed - 1.0e-12))
    return float((exceedances + 1) / (n_resamples + 1))


def _stable_seed(*parts: object) -> int:
    text = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(text, digest_size=8).digest(), "little")


def paired_delta_rows(
    results: pd.DataFrame,
    primary: str,
    baselines: Sequence[str],
    metrics: Sequence[str],
    key_columns: Sequence[str] = ("noise_level", "scenario", "robots", "seed"),
) -> pd.DataFrame:
    """Create long paired deltas from exact scenario-agent-seed noise cases."""

    required = set(key_columns) | {"method"} | set(metrics)
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Cannot create paired deltas; missing columns: {sorted(missing)}")
    subset = results[results["method"].isin((primary, *baselines))].copy()
    if subset.empty:
        return pd.DataFrame()
    duplicates = subset.duplicated(list(key_columns) + ["method"], keep=False)
    if duplicates.any():
        raise ValueError("Each scenario-agent-seed noise case must contain at most one row per method")

    rows: list[dict] = []
    for baseline in baselines:
        pair = subset[subset["method"].isin((primary, baseline))]
        for keys, group in pair.groupby(list(key_columns), dropna=False):
            keys = keys if isinstance(keys, tuple) else (keys,)
            methods = group.set_index("method")
            if primary not in methods.index or baseline not in methods.index:
                continue
            primary_row = methods.loc[primary]
            baseline_row = methods.loc[baseline]
            for metric in metrics:
                primary_value = float(primary_row[metric])
                baseline_value = float(baseline_row[metric])
                if not np.isfinite(primary_value) or not np.isfinite(baseline_value):
                    continue
                direction = METRIC_DIRECTIONS.get(metric, "higher")
                delta = primary_value - baseline_value
                rows.append(
                    {
                        **dict(zip(key_columns, keys)),
                        "primary": primary,
                        "baseline": baseline,
                        "comparison": f"{primary}_vs_{baseline}",
                        "metric": metric,
                        "direction": direction,
                        "primary_value": primary_value,
                        "baseline_value": baseline_value,
                        "delta": delta,
                        "signed_improvement": -delta if direction == "lower" else delta,
                    }
                )
    return pd.DataFrame(rows)


def summarize_paired_deltas(detail: pd.DataFrame, n_resamples: int = 10000) -> pd.DataFrame:
    """Summarize paired deltas by noise level with bootstrap CIs and sign flips."""

    required = {"noise_level", "comparison", "metric", "direction", "delta", "signed_improvement"}
    missing = required - set(detail.columns)
    if missing:
        raise ValueError(f"Cannot summarize paired deltas; missing columns: {sorted(missing)}")
    rows: list[dict] = []
    group_columns = ["noise_level", "comparison", "metric", "direction"]
    for keys, group in detail.groupby(group_columns, dropna=False):
        values = _finite(group["delta"])
        improvement = _finite(group["signed_improvement"])
        low, high = bootstrap_mean_ci(values, n_resamples=n_resamples, seed=_stable_seed("bootstrap", *keys))
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "paired_cases": int(len(values)),
                "mean_delta": float(values.mean()) if len(values) else float("nan"),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "sign_flip_p": paired_sign_flip_pvalue(values, n_resamples=n_resamples, seed=_stable_seed("sign-flip", *keys)),
                "mean_signed_improvement": float(improvement.mean()) if len(improvement) else float("nan"),
                "primary_better_rate": float(np.mean(improvement > 0.0)) if len(improvement) else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True) if rows else pd.DataFrame()


def summarize_method_metrics(
    results: pd.DataFrame,
    metrics: Sequence[str],
    n_resamples: int = 10000,
) -> pd.DataFrame:
    """Bootstrap method summaries at each observation-noise level."""

    required = {"noise_level", "method"} | set(metrics)
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Cannot summarize methods; missing columns: {sorted(missing)}")
    rows: list[dict] = []
    for (noise_level, method), group in results.groupby(["noise_level", "method"], dropna=False):
        for metric in metrics:
            values = _finite(group[metric])
            low, high = bootstrap_mean_ci(values, n_resamples=n_resamples, seed=_stable_seed("method", noise_level, method, metric))
            rows.append(
                {
                    "noise_level": noise_level,
                    "method": method,
                    "metric": metric,
                    "direction": METRIC_DIRECTIONS.get(metric, "higher"),
                    "trials": int(len(values)),
                    "mean": float(values.mean()) if len(values) else float("nan"),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                }
            )
    return pd.DataFrame(rows).sort_values(["noise_level", "method", "metric"]).reset_index(drop=True)
