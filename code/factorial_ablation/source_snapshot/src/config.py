from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
FIGURES = OUTPUT / "figures"
RESULTS = OUTPUT / "results"
MODELS = OUTPUT / "models"
MANUSCRIPTS = OUTPUT / "manuscripts"


@dataclass(frozen=True)
class ObservationNoiseLevel:
    """A synthetic local-observation perturbation for sensitivity analysis."""

    name: str
    sigma_p: float
    sigma_v: float
    description: str


OBSERVATION_NOISE_LEVELS: tuple[ObservationNoiseLevel, ...] = (
    ObservationNoiseLevel("clean", 0.00, 0.00, "Clean simulated neighbor observations."),
    ObservationNoiseLevel("low", 0.02, 0.04, "Mild localization error sensitivity condition."),
    ObservationNoiseLevel("medium", 0.05, 0.10, "Common simulation/localization error sensitivity condition."),
    ObservationNoiseLevel("high", 0.10, 0.20, "Clearly degraded observation sensitivity condition."),
)


def observation_noise_level(name: str) -> ObservationNoiseLevel:
    for level in OBSERVATION_NOISE_LEVELS:
        if level.name == name:
            return level
    available = ", ".join(level.name for level in OBSERVATION_NOISE_LEVELS)
    raise ValueError(f"Unknown observation-noise level '{name}'. Expected one of: {available}")


@dataclass(frozen=True)
class SimConfig:
    dt: float = 0.12
    horizon: int = 8
    max_steps: int = 140
    robot_radius: float = 0.22
    safety_margin: float = 0.08
    max_speed: float = 1.15
    max_accel: float = 1.8
    goal_tolerance: float = 0.38
    neighbor_limit: int = 4
    collision_weight: float = 7.5
    control_weight: float = 0.05
    smooth_weight: float = 0.025
    safety_filter_gain: float = 2.0

    @property
    def safety_distance(self) -> float:
        return 2.0 * self.robot_radius + self.safety_margin


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 7
    train_samples: int = 8000
    val_samples: int = 1600
    test_samples: int = 1600
    batch_size: int = 256
    epochs: int = 18
    lr: float = 1.0e-3
    hidden_dim: int = 96
    dropout: float = 0.05


@dataclass(frozen=True)
class EvalConfig:
    seeds_per_case: int = 30
    robot_counts: tuple[int, ...] = (4, 8, 12)
    scenarios: tuple[str, ...] = (
        "crossing",
        "corridor",
        "warehouse",
        "bottleneck",
        "narrow_gate",
        "merge_bottleneck",
        "four_way_stop",
        "swap_lanes",
    )
    methods: tuple[str, ...] = (
        "constant_velocity_mpc",
        "passive_predictor_mpc",
        "rvo_reactive",
        "passive_rvo_mpc",
        "icgp_rvo_mpc",
        "icgp_mpc",
        "icgp_no_filter",
        "icgp_h4",
        "icgp_k2",
    )


PROFILES: Dict[str, Dict[str, int]] = {
    "quick": {
        "train_samples": 2600,
        "val_samples": 600,
        "test_samples": 600,
        "epochs": 12,
        "seeds_per_case": 2,
    },
    "standard": {
        "train_samples": 8000,
        "val_samples": 1600,
        "test_samples": 1600,
        "epochs": 18,
        "seeds_per_case": 30,
    },
    "cloud": {
        "train_samples": 60000,
        "val_samples": 10000,
        "test_samples": 10000,
        "epochs": 45,
        "seeds_per_case": 80,
    },
}


def apply_profile(train: TrainConfig, eval_cfg: EvalConfig, profile: str) -> tuple[TrainConfig, EvalConfig]:
    overrides = PROFILES[profile]
    train_kwargs = asdict(train)
    eval_kwargs = asdict(eval_cfg)
    for key, value in overrides.items():
        if key in train_kwargs:
            train_kwargs[key] = value
        if key in eval_kwargs:
            eval_kwargs[key] = value
    return TrainConfig(**train_kwargs), EvalConfig(**eval_kwargs)


def ensure_dirs() -> None:
    for path in (OUTPUT, FIGURES, RESULTS, MODELS, MANUSCRIPTS):
        path.mkdir(parents=True, exist_ok=True)
