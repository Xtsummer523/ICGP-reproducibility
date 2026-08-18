from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .config import OBSERVATION_NOISE_LEVELS, ObservationNoiseLevel, observation_noise_level
from .sim import WorldState


OBSERVATION_NOISE_PROTOCOL_VERSION = "neighbor_gaussian_v1"


@dataclass(frozen=True)
class NoiseSchedule:
    """Pre-generated perturbations so paired methods share common random noise."""

    level: ObservationNoiseLevel
    seed: int
    pos: np.ndarray
    vel: np.ndarray


def observation_noise_seed(scenario: str, robots: int, episode_seed: int, level_name: str) -> int:
    """Create a stable seed independent of Python's randomized ``hash`` implementation."""

    payload = f"{OBSERVATION_NOISE_PROTOCOL_VERSION}|{scenario}|{robots}|{episode_seed}|{level_name}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def make_noise_schedule(
    scenario: str,
    robots: int,
    episode_seed: int,
    level_name: str,
    max_steps: int,
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
    level = observation_noise_level(level_name)
    seed = observation_noise_seed(scenario, robots, episode_seed, level.name)
    shape = (max_steps, robots, 2)
    if level.sigma_p == 0.0 and level.sigma_v == 0.0:
        return NoiseSchedule(level, seed, np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32))
    rng = np.random.default_rng(seed)
    return NoiseSchedule(
        level=level,
        seed=seed,
        pos=rng.normal(0.0, level.sigma_p, size=shape).astype(np.float32),
        vel=rng.normal(0.0, level.sigma_v, size=shape).astype(np.float32),
    )


def observed_state_for_ego(
    state: WorldState,
    ego: int,
    pos_noise: np.ndarray,
    vel_noise: np.ndarray,
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
    "NoiseSchedule",
    "make_noise_schedule",
    "observation_noise_seed",
    "observed_state_for_ego",
]
