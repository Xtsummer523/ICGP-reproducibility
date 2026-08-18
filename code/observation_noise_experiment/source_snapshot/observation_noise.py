from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .config import OBSERVATION_NOISE_LEVELS, ObservationNoiseLevel, observation_noise_level
from .sim import WorldState


OBSERVATION_NOISE_PROTOCOL_VERSION = "neighbor_gaussian_v1"
AR1_OBSERVATION_NOISE_PROTOCOL_VERSION = "neighbor_ar1_v1"
OCCLUSION_OBSERVATION_NOISE_PROTOCOL_VERSION = "neighbor_dropout_zero_v1"
SUPPORTED_OBSERVATION_NOISE_PROTOCOLS = (
    OBSERVATION_NOISE_PROTOCOL_VERSION,
    AR1_OBSERVATION_NOISE_PROTOCOL_VERSION,
    OCCLUSION_OBSERVATION_NOISE_PROTOCOL_VERSION,
)


@dataclass(frozen=True)
class NoiseSchedule:
    """Pre-generated perturbations so paired methods share common random noise."""

    level: ObservationNoiseLevel
    seed: int
    protocol: str
    pos: np.ndarray
    vel: np.ndarray
    occluded: np.ndarray


def observation_noise_seed(
    scenario: str,
    robots: int,
    episode_seed: int,
    level_name: str,
    protocol: str = OBSERVATION_NOISE_PROTOCOL_VERSION,
) -> int:
    """Create a stable seed independent of Python's randomized ``hash`` implementation."""

    # Keep the original Gaussian seed byte-for-byte stable so the archived
    # formal corpus remains reproducible after adding new stress protocols.
    protocol_suffix = "" if protocol == OBSERVATION_NOISE_PROTOCOL_VERSION else f"|{protocol}"
    payload = f"{OBSERVATION_NOISE_PROTOCOL_VERSION}|{scenario}|{robots}|{episode_seed}|{level_name}{protocol_suffix}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _ar1_noise(rng: np.random.Generator, shape: tuple[int, ...], sigma: float, rho: float) -> np.ndarray:
    if sigma == 0.0:
        return np.zeros(shape, dtype=np.float32)
    innovation_scale = sigma * np.sqrt(max(0.0, 1.0 - rho * rho))
    values = np.empty(shape, dtype=np.float32)
    values[0] = rng.normal(0.0, sigma, size=shape[1:]).astype(np.float32)
    for step in range(1, shape[0]):
        innovation = rng.normal(0.0, innovation_scale, size=shape[1:]).astype(np.float32)
        values[step] = rho * values[step - 1] + innovation
    return values


def _occlusion_mask(
    rng: np.random.Generator,
    max_steps: int,
    robots: int,
    probability: float,
    max_duration: int,
) -> np.ndarray:
    mask = np.zeros((max_steps, robots), dtype=bool)
    for robot in range(robots):
        step = 0
        while step < max_steps:
            if rng.random() < probability:
                duration = int(rng.integers(1, max_duration + 1))
                mask[step : min(max_steps, step + duration), robot] = True
                # Keep adjacent runs separated so the observable contiguous
                # dropout duration is bounded by ``max_duration``.
                step += duration + 1
            else:
                step += 1
    return mask


def make_noise_schedule(
    scenario: str,
    robots: int,
    episode_seed: int,
    level_name: str,
    max_steps: int,
    protocol: str = OBSERVATION_NOISE_PROTOCOL_VERSION,
    ar_rho: float = 0.85,
    occlusion_probability: float = 0.10,
    occlusion_max_duration: int = 3,
) -> NoiseSchedule:
    """Generate a full episode noise table before a controller is evaluated.

    This common-random-number design gives every method at a
    ``scenario x robots x seed x noise level`` case the same perturbation at
    each simulation step. The true simulator state is never changed here.
    """

    if robots < 1:
        raise ValueError("robots must be positive")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if protocol not in SUPPORTED_OBSERVATION_NOISE_PROTOCOLS:
        available = ", ".join(SUPPORTED_OBSERVATION_NOISE_PROTOCOLS)
        raise ValueError(f"Unknown observation-noise protocol '{protocol}'. Expected one of: {available}")
    if not 0.0 <= ar_rho < 1.0:
        raise ValueError("ar_rho must satisfy 0 <= ar_rho < 1")
    if not 0.0 <= occlusion_probability <= 1.0:
        raise ValueError("occlusion_probability must be between 0 and 1")
    if occlusion_max_duration < 1:
        raise ValueError("occlusion_max_duration must be positive")
    level = observation_noise_level(level_name)
    seed = observation_noise_seed(scenario, robots, episode_seed, level.name, protocol)
    shape = (max_steps, robots, 2)
    rng = np.random.default_rng(seed)
    if protocol == AR1_OBSERVATION_NOISE_PROTOCOL_VERSION:
        pos = _ar1_noise(rng, shape, level.sigma_p, ar_rho)
        vel = _ar1_noise(rng, shape, level.sigma_v, ar_rho)
    else:
        pos = rng.normal(0.0, level.sigma_p, size=shape).astype(np.float32)
        vel = rng.normal(0.0, level.sigma_v, size=shape).astype(np.float32)
    occluded = (
        _occlusion_mask(rng, max_steps, robots, occlusion_probability, occlusion_max_duration)
        if protocol == OCCLUSION_OBSERVATION_NOISE_PROTOCOL_VERSION
        else np.zeros((max_steps, robots), dtype=bool)
    )
    return NoiseSchedule(
        level=level,
        seed=seed,
        protocol=protocol,
        pos=pos,
        vel=vel,
        occluded=occluded,
    )


def observed_state_for_ego(
    state: WorldState,
    ego: int,
    pos_noise: np.ndarray,
    vel_noise: np.ndarray,
    occluded: np.ndarray | None = None,
) -> WorldState:
    """Return one robot's local observation while retaining its own exact state.

    Each decentralized controller receives perturbed neighbor position and
    velocity measurements. Goals and obstacle geometry remain exact by design,
    so this is a targeted neighbor-state sensitivity check rather than a full
    sensor-model validation.
    """

    if not 0 <= ego < len(state.pos):
        raise IndexError(f"ego index {ego} is outside the state with {len(state.pos)} robots")
    pos_noise = np.asarray(pos_noise, dtype=np.float32)
    vel_noise = np.asarray(vel_noise, dtype=np.float32)
    if pos_noise.shape != state.pos.shape or vel_noise.shape != state.vel.shape:
        raise ValueError("position and velocity noise must match the corresponding state arrays")

    observed_pos = state.pos + pos_noise
    observed_vel = state.vel + vel_noise
    if occluded is not None:
        occluded = np.asarray(occluded, dtype=bool)
        if occluded.shape != (len(state.pos),):
            raise ValueError("occlusion mask must contain one flag per robot")
        observed_pos[occluded] = 0.0
        observed_vel[occluded] = 0.0
    observed_pos[ego] = state.pos[ego]
    observed_vel[ego] = state.vel[ego]
    return WorldState(
        pos=observed_pos.astype(np.float32, copy=False),
        vel=observed_vel.astype(np.float32, copy=False),
        goals=state.goals.copy(),
        obstacles=state.obstacles,
        bounds=state.bounds,
    )


__all__ = [
    "OBSERVATION_NOISE_LEVELS",
    "OBSERVATION_NOISE_PROTOCOL_VERSION",
    "AR1_OBSERVATION_NOISE_PROTOCOL_VERSION",
    "OCCLUSION_OBSERVATION_NOISE_PROTOCOL_VERSION",
    "SUPPORTED_OBSERVATION_NOISE_PROTOCOLS",
    "NoiseSchedule",
    "make_noise_schedule",
    "observation_noise_seed",
    "observed_state_for_ego",
]
