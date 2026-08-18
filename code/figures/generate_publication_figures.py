"""Regenerate publication-style data figures for the ICGP JIROS revision.

The script only reads retained CSV evidence files and writes visual summaries.
No plotted value is synthesized.  The visual style follows the downloaded
figures4papers scientific-figure-making guidance: clean white background,
semantic colors, print-safe edges/hatches, compact legends, and PDF+PNG export.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SUPP = ROOT / "data"

LEGACY = ROOT / "data" / "legacy_main" / "selected_episode_results.csv"
FORMAL_METHOD = ROOT / "data" / "gaussian_observation_study" / "method_summary.csv"
FORMAL_PAIR = ROOT / "data" / "gaussian_observation_study" / "paired_summary.csv"
FORMAL_RAW = ROOT / "data" / "gaussian_observation_study" / "episode_results.csv"
PRED_SUMMARY = ROOT / "data" / "prediction_quality" / "summary.csv"

STRESS_FILES = {
    "AR(1), rho=0.85": ROOT / "data" / "autoregressive_observation_study" / "paired_summary.csv",
    "Dropout, p=0.05": ROOT / "data" / "short_dropout_probability_005" / "paired_summary.csv",
    "Dropout, p=0.10": ROOT / "data" / "short_dropout_probability_010" / "paired_summary.csv",
}

PALETTE = {
    "blue_main": "#1B4E8A",
    "blue_secondary": "#4E89C7",
    "green": "#2F8F71",
    "green_soft": "#BBD9C7",
    "red": "#B75A5A",
    "red_soft": "#E8AAA4",
    "amber": "#C99332",
    "teal": "#3E9AA2",
    "violet": "#7F6AA8",
    "neutral": "#CFCECE",
    "neutral_dark": "#4D4D4D",
    "grid": "#E7EAEE",
}

METHODS_MAIN = ["icgp_rvo_mpc", "passive_rvo_mpc", "rvo_reactive", "constant_velocity_mpc"]
METHODS_FORMAL = ["icgp_rvo_mpc", "passive_rvo_mpc", "residual_passive_rvo_mpc", "rvo_reactive"]
METHOD_LABELS = {
    "icgp_rvo_mpc": "ICGP+RVO",
    "passive_rvo_mpc": "Passive+RVO",
    "residual_passive_rvo_mpc": "Residual+RVO",
    "rvo_reactive": "Reactive RVO",
    "constant_velocity_mpc": "CV-MPC",
}
METHOD_COLORS = {
    "icgp_rvo_mpc": PALETTE["blue_main"],
    "passive_rvo_mpc": PALETTE["red"],
    "residual_passive_rvo_mpc": PALETTE["teal"],
    "rvo_reactive": PALETTE["neutral_dark"],
    "constant_velocity_mpc": PALETTE["neutral"],
}
METHOD_HATCHES = {
    "icgp_rvo_mpc": "",
    "passive_rvo_mpc": "//",
    "residual_passive_rvo_mpc": "..",
    "rvo_reactive": "\\\\",
    "constant_velocity_mpc": "xx",
}
METHOD_MARKERS = {
    "icgp_rvo_mpc": "o",
    "passive_rvo_mpc": "s",
    "residual_passive_rvo_mpc": "D",
    "rvo_reactive": "^",
    "constant_velocity_mpc": "P",
}

NOISE_LEVELS = ["clean", "low", "medium", "high"]
NOISE_LABELS = ["Clean", "Low", "Medium", "High"]


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 10.5,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.6,
            "legend.title_fontsize": 8.2,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 1.25,
            "axes.edgecolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "text.color": "#222222",
            "axes.labelcolor": "#222222",
            "axes.titlecolor": "#222222",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


OUTPUT = ROOT / "reproduced_figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

def save(fig: mpl.figure.Figure, number: int, *, pad: float = 0.08) -> None:
    for suffix, kwargs in {
        "pdf": {},
        "png": {"dpi": 450},
    }.items():
        fig.savefig(OUTPUT / f"Fig{number}.{suffix}", bbox_inches="tight", pad_inches=pad, **kwargs)
    plt.close(fig)


def bootstrap_mean_ci(values: np.ndarray, *, seed: int, n_boot: int = 3000) -> tuple[float, float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    mean = float(np.mean(clean))
    if len(clean) <= 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    draws = rng.choice(clean, size=(n_boot, len(clean)), replace=True).mean(axis=1)
    return mean, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def soften_axes(ax: mpl.axes.Axes, *, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color=PALETTE["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color("#30363D")
    ax.spines["bottom"].set_color("#30363D")


def add_panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.07,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color="#222222",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
    )


def figure_2_main_outcomes() -> None:
    raw = pd.read_csv(LEGACY)
    metrics = [
        ("progress_ratio", "Progress ratio", "higher"),
        ("success", "Success", "higher"),
        ("completion_time", "Completion time (s)", "lower"),
        ("safety_violation", "Safety violation", "lower"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.9, 5.25), constrained_layout=False)
    fig.subplots_adjust(left=0.18, right=0.985, top=0.94, bottom=0.10, wspace=0.30, hspace=0.40)
    letters = ["a", "b", "c", "d"]
    for idx, (ax, (metric, title, direction)) in enumerate(zip(axes.flat, metrics)):
        means, lo, hi = [], [], []
        for m_idx, method in enumerate(METHODS_MAIN):
            mean, low, high = bootstrap_mean_ci(raw.loc[raw["method"] == method, metric].to_numpy(), seed=1000 + idx * 10 + m_idx)
            means.append(mean)
            lo.append(mean - low)
            hi.append(high - mean)
        y = np.arange(len(METHODS_MAIN))
        bars = ax.barh(
            y,
            means,
            xerr=np.vstack([lo, hi]),
            height=0.58,
            capsize=2.5,
            color=[METHOD_COLORS[m] for m in METHODS_MAIN],
            edgecolor="#202020",
            linewidth=0.8,
            alpha=0.94,
        )
        for bar, method in zip(bars, METHODS_MAIN):
            bar.set_hatch(METHOD_HATCHES[method])
        ax.set_yticks(y, [METHOD_LABELS[m] for m in METHODS_MAIN])
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontweight="bold", color="#222222")
        ax.set_xlabel("Mean (95% CI)")
        soften_axes(ax, grid_axis="x")
        add_panel_label(ax, letters[idx])
        lows = np.asarray(means) - np.asarray(lo)
        highs = np.asarray(means) + np.asarray(hi)
        vals = np.concatenate([lows, highs])
        span = float(np.nanmax(vals) - np.nanmin(vals))
        if metric in {"progress_ratio", "success", "safety_violation"}:
            margin = max(0.025, span * 0.25)
            ax.set_xlim(max(0.0, float(np.nanmin(vals)) - margin), min(1.0, float(np.nanmax(vals)) + margin))
        else:
            margin = max(0.4, span * 0.30)
            ax.set_xlim(float(np.nanmin(vals)) - margin, float(np.nanmax(vals)) + margin)
    save(fig, 2)


def figure_3_noise_sensitivity() -> None:
    frame = pd.read_csv(FORMAL_METHOD)
    metrics = [
        ("progress_ratio", "Progress ratio", "higher"),
        ("success", "Success", "higher"),
        ("completion_time", "Completion time (s)", "lower"),
        ("safety_violation", "Safety violation", "lower"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.3), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.92, bottom=0.18, wspace=0.28, hspace=0.45)
    x = np.arange(len(NOISE_LEVELS))
    for idx, (ax, (metric, title, direction)) in enumerate(zip(axes.flat, metrics)):
        rows = frame[frame["metric"] == metric]
        for method in METHODS_FORMAL:
            sub = rows[rows["method"] == method].set_index("noise_level").reindex(NOISE_LEVELS)
            mean = sub["mean"].to_numpy(float)
            low = sub["bootstrap_ci_low"].to_numpy(float)
            high = sub["bootstrap_ci_high"].to_numpy(float)
            ax.plot(
                x,
                mean,
                marker="o",
                markersize=4.0,
                linewidth=1.8,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
            ax.fill_between(x, low, high, color=METHOD_COLORS[method], alpha=0.10)
        ax.set_xticks(x, NOISE_LABELS)
        ax.set_title(title, loc="left", fontweight="bold", color="#222222")
        ax.set_xlabel("Neighbor-observation noise" if idx >= 2 else "")
        ax.set_ylabel("Mean")
        soften_axes(ax)
        add_panel_label(ax, "abcd"[idx])
        ax.text(
            0.98,
            0.94,
            "higher is better" if direction == "higher" else "lower is better",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            color="#222222",
        )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.52, 0.025))
    save(fig, 3)


def figure_4_paired_forest() -> None:
    frame = pd.read_csv(FORMAL_PAIR)
    frame = frame[
        (frame["comparison"] == "icgp_rvo_mpc_vs_passive_rvo_mpc")
        & frame["metric"].isin(["progress_ratio", "completion_time", "safety_violation", "final_goal_distance"])
    ].copy()
    metrics = [
        ("progress_ratio", "Progress ratio", "higher"),
        ("completion_time", "Completion time (s)", "lower"),
        ("safety_violation", "Safety violation", "lower"),
        ("final_goal_distance", "Final goal distance (m)", "lower"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.3), constrained_layout=False)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.92, bottom=0.17, wspace=0.30, hspace=0.45)
    for idx, (ax, (metric, title, direction)) in enumerate(zip(axes.flat, metrics)):
        sub = frame[frame["metric"] == metric].set_index("noise_level").reindex(NOISE_LEVELS)
        y = np.arange(len(NOISE_LEVELS))
        mean = sub["mean_delta"].to_numpy(float)
        low = sub["bootstrap_ci_low"].to_numpy(float)
        high = sub["bootstrap_ci_high"].to_numpy(float)
        beneficial = mean > 0 if direction == "higher" else mean < 0
        adverse = mean < 0 if direction == "higher" else mean > 0
        colors = np.where((low > 0) | (high < 0), np.where(beneficial, PALETTE["green"], PALETTE["red"]), PALETTE["neutral_dark"])
        ax.errorbar(
            mean,
            y,
            xerr=np.vstack([mean - low, high - mean]),
            fmt="none",
            ecolor="#202020",
            elinewidth=1.35,
            capsize=3,
            zorder=1,
        )
        ax.scatter(mean, y, s=42, c=colors, edgecolors="#202020", linewidths=0.7, zorder=2)
        ax.axvline(0, color="#8A9099", linewidth=1.0, linestyle=(0, (4, 3)))
        ax.set_yticks(y, NOISE_LABELS)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontweight="bold", color="#222222")
        ax.set_xlabel("Delta" if idx >= 2 else "")
        soften_axes(ax, grid_axis="x")
        add_panel_label(ax, "abcd"[idx])
    legend_handles = [
        mpl.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["green"], markeredgecolor="#202020", label="ICGP better"),
        mpl.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["red"], markeredgecolor="#202020", label="Passive better"),
        mpl.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["neutral_dark"], markeredgecolor="#202020", label="CI crosses 0"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.52, 0.02))
    save(fig, 4)


def figure_5_heterogeneity() -> None:
    raw = pd.read_csv(FORMAL_RAW)
    raw = raw[(raw["noise_level"] == "high") & raw["method"].isin(["icgp_rvo_mpc", "passive_rvo_mpc"])].copy()
    grouped = raw.groupby(["scenario", "robots", "method"], as_index=False)[["completion_time", "safety_violation"]].mean()
    scenarios = ["corridor", "narrow_gate", "merge_bottleneck", "four_way_stop"]
    scenario_labels = ["Corridor", "Narrow gate", "Merge\nbottleneck", "Four-way\nstop"]
    robots = [4, 8]
    metrics = [
        ("completion_time", "Completion-time delta (s)", "lower"),
        ("safety_violation", "Safety-violation delta", "lower"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4), constrained_layout=True)
    for idx, (ax, (metric, title, _direction)) in enumerate(zip(axes, metrics)):
        matrix = np.full((len(scenarios), len(robots)), np.nan)
        for i, scenario in enumerate(scenarios):
            for j, r in enumerate(robots):
                cell = grouped[(grouped["scenario"] == scenario) & (grouped["robots"] == r)].set_index("method")
                if {"icgp_rvo_mpc", "passive_rvo_mpc"}.issubset(cell.index):
                    matrix[i, j] = cell.loc["icgp_rvo_mpc", metric] - cell.loc["passive_rvo_mpc", metric]
        vmax = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(len(robots)), [f"{r} robots" for r in robots])
        ax.set_yticks(np.arange(len(scenarios)), scenario_labels)
        ax.set_title(title, loc="left", fontweight="bold", color="#222222")
        ax.set_xlabel("High Gaussian condition")
        add_panel_label(ax, "ab"[idx])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                label = "NA" if not np.isfinite(val) else f"{val:+.3f}"
                ax.text(j, i, label, ha="center", va="center", fontsize=8.5, color="#111111")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(robots), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(scenarios), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2.0)
        ax.tick_params(which="minor", bottom=False, left=False)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.ax.set_ylabel("Delta", rotation=90)
    save(fig, 5)


def load_stress_pairs() -> pd.DataFrame:
    frames = []
    for protocol, path in STRESS_FILES.items():
        frame = pd.read_csv(path)
        frame = frame[
            (frame["comparison"] == "icgp_rvo_mpc_vs_passive_rvo_mpc")
            & frame["metric"].isin(["completion_time", "safety_violation"])
        ].copy()
        frame["protocol"] = protocol
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def figure_6_stress() -> None:
    frame = load_stress_pairs()
    protocols = list(STRESS_FILES.keys())
    protocol_labels = ["AR(1)\nrho=0.85", "Dropout\n5%", "Dropout\n10%"]
    metrics = [
        ("completion_time", "Completion-time delta (s)", "lower"),
        ("safety_violation", "Safety-violation delta", "lower"),
    ]
    delta_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "icgp_delta",
        [PALETTE["blue_main"], "#F7F7F7", PALETTE["red"]],
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.55), constrained_layout=False)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.83, bottom=0.16, wspace=0.28)
    for idx, (ax, (metric, title, _direction)) in enumerate(zip(axes, metrics)):
        vals = frame[frame["metric"] == metric]
        noise_levels = ["low", "high"]
        noise_labels = ["Low noise", "High noise"]
        matrix = np.zeros((len(noise_levels), len(protocols)))
        low_ci = np.zeros_like(matrix)
        high_ci = np.zeros_like(matrix)
        for i, noise in enumerate(noise_levels):
            sub = vals[vals["noise_level"] == noise].set_index("protocol").reindex(protocols)
            matrix[i, :] = sub["mean_delta"].to_numpy(float)
            low_ci[i, :] = sub["bootstrap_ci_low"].to_numpy(float)
            high_ci[i, :] = sub["bootstrap_ci_high"].to_numpy(float)
        vmax = max(float(np.nanmax(np.abs(matrix))), 0.001)
        norm = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                ci_excludes_zero = (low_ci[i, j] > 0) or (high_ci[i, j] < 0)
                size = 620 + 740 * (abs(val) / vmax)
                ax.scatter(
                    j,
                    i,
                    s=size,
                    color=delta_cmap(norm(val)),
                    edgecolors="#202020" if ci_excludes_zero else "#A4AAB2",
                    linewidths=1.25 if ci_excludes_zero else 0.75,
                    zorder=2,
                )
                label = f"{val:+.3f}"
                ax.text(j, i, label, ha="center", va="center", fontsize=4.8, color="#111111", zorder=3)
        ax.set_xticks(np.arange(len(protocols)), protocol_labels)
        ax.set_yticks(np.arange(len(noise_levels)), noise_labels)
        ax.set_xlim(-0.55, len(protocols) - 0.45)
        ax.set_ylim(len(noise_levels) - 0.55, -0.55)
        ax.grid(color=PALETTE["grid"], linewidth=0.9)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("#30363D")
        ax.set_title(title, loc="left", fontweight="bold", color="#222222")
        ax.set_xlabel("Protocol")
        add_panel_label(ax, "ab"[idx])
    legend_handles = [
        mpl.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor="#F7F7F7", markeredgecolor="#202020", markeredgewidth=1.25, label="CI excludes 0"),
        mpl.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor="#F7F7F7", markeredgecolor="#A4AAB2", markeredgewidth=0.75, label="CI crosses 0"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.52, 0.99))
    save(fig, 6)


def figure_7_prediction_quality() -> None:
    frame = pd.read_csv(PRED_SUMMARY)
    methods = ["icgp_rvo_mpc", "passive_rvo_mpc", "residual_passive_rvo_mpc"]
    metrics = [
        ("ADE", "ADE", "lower"),
        ("FDE", "FDE", "lower"),
        ("candidate_spread", "Candidate spread", "diagnostic"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8.1, 3.55), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.82, bottom=0.18, wspace=0.34)
    x = np.array([0, 1])
    for idx, (ax, (metric, title, _direction)) in enumerate(zip(axes, metrics)):
        clean = frame[frame["noise_level"] == "clean"].set_index("method").reindex(methods)
        high = frame[frame["noise_level"] == "high"].set_index("method").reindex(methods)
        all_vals = []
        for method in methods:
            cval = float(clean.loc[method, metric])
            hval = float(high.loc[method, metric])
            all_vals.extend([cval, hval])
            ax.plot(x, [cval, hval], color=METHOD_COLORS[method], linewidth=2.0, marker="o", markersize=4.8, zorder=2)
        ax.set_title(title, loc="left", fontweight="bold", color="#222222")
        ax.set_xticks(x, ["Clean", "High"])
        ax.set_xlim(-0.12, 1.12)
        ax.set_ylabel("Mean value" if idx == 0 else "")
        soften_axes(ax, grid_axis="y")
        vals = np.asarray(all_vals, dtype=float)
        span = float(np.nanmax(vals) - np.nanmin(vals))
        margin = max(0.02 if metric == "candidate_spread" else 0.05, span * 0.22)
        ax.set_ylim(max(0.0, float(np.nanmin(vals)) - margin), float(np.nanmax(vals)) + margin)
        add_panel_label(ax, "abc"[idx])
    handles = [
        mpl.lines.Line2D([0], [0], marker="o", color=METHOD_COLORS[m], markerfacecolor=METHOD_COLORS[m], label=METHOD_LABELS[m].replace("+RVO", ""))
        for m in methods
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.52, 0.99))
    save(fig, 7)


def main() -> None:
    apply_style()
    figure_2_main_outcomes()
    figure_3_noise_sensitivity()
    figure_4_paired_forest()
    figure_5_heterogeneity()
    figure_6_stress()
    figure_7_prediction_quality()


if __name__ == "__main__":
    main()

