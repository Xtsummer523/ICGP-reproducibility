from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from .config import FIGURES, SimConfig
from .scenarios import make_scenario


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 140,
        "savefig.dpi": 220,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


METHOD_LABELS = {
    "constant_velocity_mpc": "CV-MPC",
    "passive_predictor_mpc": "Passive-MPC",
    "orca_reactive": "Reactive-ORCA",
    "rvo_reactive": "Reactive-RVO",
    "passive_rvo_mpc": "Passive-RVO-MPC",
    "icgp_mpc": "ICGP-MPC",
    "icgp_rvo_mpc": "ICGP-RVO-MPC",
    "icgp_no_filter": "ICGP w/o filter",
    "icgp_h4": "ICGP H=4",
    "icgp_k2": "ICGP K=2",
}


def plot_system_framework() -> Path:
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.axis("off")
    boxes = [
        (0.02, 0.55, 0.18, 0.25, "Local observation\npositions, velocities,\nobstacle layout"),
        (0.28, 0.55, 0.18, 0.25, "Candidate intent\ncontrol query,\ngoal direction"),
        (0.54, 0.55, 0.18, 0.25, "ICGP predictor\nconditional neighbor\ntrajectories"),
        (0.80, 0.55, 0.18, 0.25, "Receding-horizon\ncontrol action"),
        (0.54, 0.12, 0.18, 0.22, "Safety filter\nshort-range\nvelocity correction"),
        (0.80, 0.12, 0.18, 0.22, "Robot dynamics\nnext local state"),
    ]
    for x, y, w, h, text in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, ec="#263238", fc="#eef4f7", lw=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)
    arrows = [((0.20, 0.675), (0.28, 0.675)), ((0.46, 0.675), (0.54, 0.675)), ((0.72, 0.675), (0.80, 0.675)), ((0.89, 0.55), (0.89, 0.34)), ((0.72, 0.23), (0.80, 0.23)), ((0.54, 0.23), (0.46, 0.55))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.3, color="#263238"))
    path = FIGURES / "fig1_system_framework.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_network_diagram() -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.axis("off")
    layers = [
        ("History\nand neighbors", 0.08, 0.55),
        ("Intent query\ncandidate action", 0.08, 0.20),
        ("Interaction\nencoding", 0.34, 0.38),
        ("Temporal MLP\nrollout head", 0.60, 0.38),
        ("Conditional\nfuture trajectories", 0.84, 0.38),
    ]
    for text, x, y in layers:
        ax.add_patch(FancyBboxPatch((x - 0.09, y - 0.10), 0.18, 0.20, boxstyle="round,pad=0.02,rounding_size=0.02", ec="#1b5e20", fc="#eff7ed", lw=1.3))
        ax.text(x, y, text, ha="center", va="center", fontsize=9)
    for start, end in [((0.17, 0.55), (0.25, 0.42)), ((0.17, 0.20), (0.25, 0.34)), ((0.43, 0.38), (0.51, 0.38)), ((0.69, 0.38), (0.75, 0.38))]:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.3, color="#1b5e20"))
    ax.text(0.34, 0.12, "The predictor is queried once for each candidate ego action.", ha="center", fontsize=8.5, color="#455a64")
    path = FIGURES / "fig2_predictor_network.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_control_flow() -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    ax.axis("off")
    steps = ["observe", "enumerate\nintents", "predict\nresponses", "score\nrisk/cost", "filter\nsafety", "execute"]
    x = np.linspace(0.08, 0.92, len(steps))
    for i, (label, xi) in enumerate(zip(steps, x)):
        ax.add_patch(plt.Circle((xi, 0.58), 0.07, ec="#5d4037", fc="#fff8e1", lw=1.3))
        ax.text(xi, 0.58, label, ha="center", va="center", fontsize=8.5)
        if i > 0:
            ax.annotate("", xy=(xi - 0.075, 0.58), xytext=(x[i - 1] + 0.075, 0.58), arrowprops=dict(arrowstyle="->", lw=1.2, color="#5d4037"))
    ax.annotate("", xy=(x[0], 0.66), xytext=(x[-1], 0.66), arrowprops=dict(arrowstyle="->", lw=1.0, color="#6d4c41", connectionstyle="arc3,rad=0.35"))
    ax.text(0.50, 0.86, "closed-loop receding-horizon cycle", ha="center", fontsize=9, color="#5d4037")
    path = FIGURES / "fig3_control_flow.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_training(history: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
    for model, group in history.groupby("model"):
        axes[0].plot(group["epoch"], group["val_ADE"], marker="o", label=model)
        axes[1].plot(group["epoch"], group["val_FDE"], marker="o", label=model)
    axes[0].set_title("Validation ADE")
    axes[1].set_title("Validation FDE")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("m")
        ax.legend(frameon=False)
    path = FIGURES / "fig4_training_curves.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_scenario_layouts(cfg: SimConfig) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 7.0))
    for ax, name in zip(axes.ravel(), ["crossing", "corridor", "warehouse", "bottleneck"]):
        sc = make_scenario(name, 8, seed=42)
        ax.scatter(sc.starts[:, 0], sc.starts[:, 1], c="#1565c0", s=28, label="start")
        ax.scatter(sc.goals[:, 0], sc.goals[:, 1], c="#c62828", marker="x", s=36, label="goal")
        for obs in sc.obstacles:
            ax.add_patch(plt.Rectangle(obs.center - obs.half_size, *(2 * obs.half_size), fc="#9e9e9e", ec="#424242", alpha=0.8))
        ax.set_title(name)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(sc.bounds[0], sc.bounds[1])
        ax.set_ylim(sc.bounds[2], sc.bounds[3])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    path = FIGURES / "fig5_scenario_layouts.png"
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_trajectories(traj: pd.DataFrame, cfg: SimConfig) -> Path:
    methods = ["constant_velocity_mpc", "passive_predictor_mpc", "rvo_reactive", "icgp_rvo_mpc"]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.2))
    for ax, method in zip(axes.ravel(), methods):
        g = traj[(traj["method"] == method) & (traj["scenario"] == "corridor")]
        if g.empty:
            g = traj[traj["method"] == method]
        if g.empty:
            ax.set_title(METHOD_LABELS.get(method, method))
            ax.text(0.5, 0.5, "not recorded", ha="center", va="center", transform=ax.transAxes)
            continue
        for rid, rg in g.groupby("robot"):
            ax.plot(rg["x"], rg["y"], lw=1.2)
            ax.scatter(rg["x"].iloc[0], rg["y"].iloc[0], s=12, c="black")
        ax.set_title(METHOD_LABELS.get(method, method))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    path = FIGURES / "fig6_trajectory_comparison.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_metrics(results: pd.DataFrame) -> tuple[Path, Path]:
    preferred = ["constant_velocity_mpc", "passive_predictor_mpc", "rvo_reactive", "passive_rvo_mpc", "icgp_rvo_mpc"]
    if not results[results["method"].isin(preferred)].empty:
        core = results[results["method"].isin(preferred)].copy()
    else:
        core = results[results["method"].isin(["constant_velocity_mpc", "passive_predictor_mpc", "orca_reactive", "icgp_mpc"])].copy()
    summary = core.groupby("method").agg(collision_rate=("collision", "mean"), safety_violation_rate=("safety_violation", "mean"), success_rate=("success", "mean"), completion_time=("completion_time", "mean"), min_distance=("min_distance", "mean")).reset_index()
    labels = [METHOD_LABELS.get(m, m) for m in summary["method"]]
    x = np.arange(len(summary))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    axes[0].bar(x - 0.17, summary["collision_rate"] * 100, width=0.34, color="#8d6e63", label="physical")
    axes[0].bar(x + 0.17, summary["safety_violation_rate"] * 100, width=0.34, color="#d7ccc8", label="safety margin")
    axes[0].set_ylabel("Event rate [%]")
    axes[0].legend(frameon=False)
    axes[1].bar(x, summary["success_rate"] * 100, color="#2e7d32")
    axes[1].set_ylabel("Success rate [%]")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
    path1 = FIGURES / "fig7_safety_success_bars.png"
    fig.tight_layout()
    fig.savefig(path1)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    axes[0].boxplot([core[core["method"] == m]["completion_time"] for m in summary["method"]], labels=labels, showfliers=False)
    axes[0].set_ylabel("Completion time [s]")
    axes[1].boxplot([core[core["method"] == m]["min_distance"] for m in summary["method"]], labels=labels, showfliers=False)
    axes[1].set_ylabel("Minimum distance [m]")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
    path2 = FIGURES / "fig8_efficiency_distance_boxplots.png"
    fig.tight_layout()
    fig.savefig(path2)
    plt.close(fig)
    return path1, path2


def plot_ablation(results: pd.DataFrame) -> Path:
    ab = results[results["method"].isin(["icgp_mpc", "icgp_no_filter", "icgp_h4", "icgp_k2"])].copy()
    summary = ab.groupby("method").agg(collision_rate=("collision", "mean"), safety_violation_rate=("safety_violation", "mean"), success_rate=("success", "mean"), latency_ms=("latency_ms", "mean")).reset_index()
    labels = [METHOD_LABELS.get(m, m) for m in summary["method"]]
    x = np.arange(len(summary))
    fig, ax1 = plt.subplots(figsize=(7.8, 3.5))
    ax1.bar(x - 0.25, summary["collision_rate"] * 100, width=0.25, label="physical collision", color="#bf360c")
    ax1.bar(x, summary["safety_violation_rate"] * 100, width=0.25, label="safety-margin violation", color="#ef8a62")
    ax1.bar(x + 0.25, summary["success_rate"] * 100, width=0.25, label="success rate", color="#33691e")
    ax1.set_ylabel("%")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.legend(frameon=False)
    path = FIGURES / "fig9_ablation.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_all_figures(history: pd.DataFrame, results: pd.DataFrame, trajectories: pd.DataFrame, cfg: SimConfig) -> Dict[str, Path]:
    paths = {
        "framework": plot_system_framework(),
        "network": plot_network_diagram(),
        "control": plot_control_flow(),
        "training": plot_training(history),
        "scenarios": plot_scenario_layouts(cfg),
    }
    if not trajectories.empty:
        paths["trajectories"] = plot_trajectories(trajectories, cfg)
    metric1, metric2 = plot_metrics(results)
    paths["metrics"] = metric1
    paths["boxplots"] = metric2
    paths["ablation"] = plot_ablation(results)
    return paths
