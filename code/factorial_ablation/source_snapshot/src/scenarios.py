from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np


SCENARIO_GENERATION_VERSION = "density_v4_interaction_tiers"


@dataclass(frozen=True)
class Obstacle:
    center: np.ndarray
    half_size: np.ndarray

    def distance(self, point: np.ndarray) -> float:
        q = np.abs(point - self.center) - self.half_size
        outside = np.maximum(q, 0.0)
        outside_norm = float(np.linalg.norm(outside))
        inside = min(float(np.max(q)), 0.0)
        return outside_norm + inside

    def nearest_point(self, point: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(point, self.center - self.half_size), self.center + self.half_size)


@dataclass
class Scenario:
    name: str
    starts: np.ndarray
    goals: np.ndarray
    bounds: Tuple[float, float, float, float]
    obstacles: List[Obstacle]
    tier: str = "main"
    area_policy: str = "fixed"

    @property
    def workspace_width(self) -> float:
        return float(self.bounds[1] - self.bounds[0])

    @property
    def workspace_height(self) -> float:
        return float(self.bounds[3] - self.bounds[2])

    @property
    def workspace_area(self) -> float:
        return self.workspace_width * self.workspace_height

    @property
    def agent_density(self) -> float:
        return float(len(self.starts) / max(self.workspace_area, 1.0e-6))

    @property
    def area_per_agent(self) -> float:
        return float(self.workspace_area / max(len(self.starts), 1))

    @property
    def dominant_flow_axis(self) -> str:
        displacement = np.mean(np.abs(self.goals - self.starts), axis=0)
        return "x" if float(displacement[0]) >= float(displacement[1]) else "y"

    @property
    def lateral_axis(self) -> str:
        return "y" if self.dominant_flow_axis == "x" else "x"

    @property
    def passage_width(self) -> float:
        """Largest free lateral gap for the dominant motion direction."""
        axis = 1 if self.lateral_axis == "y" else 0
        lower = self.bounds[2] if axis == 1 else self.bounds[0]
        upper = self.bounds[3] if axis == 1 else self.bounds[1]
        intervals = []
        for obs in self.obstacles:
            intervals.append((float(obs.center[axis] - obs.half_size[axis]), float(obs.center[axis] + obs.half_size[axis])))
        gaps = _free_axis_gaps(float(lower), float(upper), intervals)
        return float(max(gaps) if gaps else upper - lower)

    @property
    def start_bounds_margin(self) -> float:
        return _min_bounds_margin(self.starts, self.bounds)

    @property
    def goal_bounds_margin(self) -> float:
        return _min_bounds_margin(self.goals, self.bounds)

    @property
    def starts_out_of_bounds(self) -> bool:
        return self.start_bounds_margin < 0.0

    @property
    def goals_out_of_bounds(self) -> bool:
        return self.goal_bounds_margin < 0.0


def _min_bounds_margin(points: np.ndarray, bounds: Tuple[float, float, float, float]) -> float:
    if len(points) == 0:
        return float("inf")
    xmin, xmax, ymin, ymax = bounds
    margins = np.column_stack(
        [
            points[:, 0] - xmin,
            xmax - points[:, 0],
            points[:, 1] - ymin,
            ymax - points[:, 1],
        ]
    )
    return float(np.min(margins))


def _jitter(rng: np.random.Generator, arr: np.ndarray, scale: float) -> np.ndarray:
    return arr + rng.normal(0.0, scale, size=arr.shape)


def _min_pairwise_distance(arr: np.ndarray) -> float:
    if len(arr) < 2:
        return float("inf")
    best = float("inf")
    for i in range(len(arr)):
        distances = np.linalg.norm(arr[i + 1 :] - arr[i], axis=1)
        if len(distances):
            best = min(best, float(np.min(distances)))
    return best


def _bounded_jitter(
    rng: np.random.Generator,
    arr: np.ndarray,
    requested_scale: float,
    target_min_distance: float = 0.66,
) -> np.ndarray:
    """Apply deterministic-bounded noise without invalidating dense starts."""
    base_min = _min_pairwise_distance(arr)
    if not np.isfinite(base_min):
        return arr.copy()
    max_axis_jitter = max(0.0, (base_min - target_min_distance) / (2.0 * math.sqrt(2.0)))
    scale = min(requested_scale, max_axis_jitter)
    if scale <= 1.0e-9:
        return arr.copy()
    return arr + rng.uniform(-scale, scale, size=arr.shape)


def _stress_pack_positions(
    starts: np.ndarray,
    goals: np.ndarray,
    bounds: Tuple[float, float, float, float],
    obstacles: List[Obstacle],
    min_spacing: float = 0.62,
) -> np.ndarray:
    """Queue crowded stress starts near their original side without overlap."""
    min_obstacle_clearance = float("inf")
    for point in starts:
        for obstacle in obstacles:
            min_obstacle_clearance = min(min_obstacle_clearance, float(obstacle.distance(point)))
    if len(starts) < 2 or (_min_pairwise_distance(starts) >= min_spacing and min_obstacle_clearance >= 0.34):
        return starts

    packed = starts.copy()
    xmin, xmax, ymin, ymax = bounds
    margin = 0.70
    usable_x = (xmin + margin, xmax - margin)
    usable_y = (ymin + margin, ymax - margin)
    center = np.array([(xmin + xmax) * 0.5, (ymin + ymax) * 0.5], dtype=np.float32)
    span = min(4.2, max(xmax - xmin, ymax - ymin) * 0.38)
    groups: dict[tuple[int, int], list[int]] = {}

    for idx, (start, goal) in enumerate(zip(starts, goals)):
        delta = goal - start
        if abs(float(delta[0])) >= abs(float(delta[1])):
            axis = 0
            side = -1 if start[0] <= center[0] else 1
        else:
            axis = 1
            side = -1 if start[1] <= center[1] else 1
        groups.setdefault((axis, side), []).append(idx)

    for (axis, side), indices in groups.items():
        lateral_axis = 1 - axis
        if axis == 0:
            long_lo, long_hi = usable_x
            lat_lo, lat_hi = usable_y
        else:
            long_lo, long_hi = usable_y
            lat_lo, lat_hi = usable_x
        obstacle_buffer = 0.38
        lateral_blocked = [
            (
                float(obs.center[lateral_axis] - obs.half_size[lateral_axis]),
                float(obs.center[lateral_axis] + obs.half_size[lateral_axis]),
            )
            for obs in obstacles
        ]
        lat_lo, lat_hi = _safe_axis_limits(lat_lo, lat_hi, lateral_blocked, clearance=obstacle_buffer)

        if side < 0:
            zone_lo = long_lo
            zone_hi = min(long_hi, long_lo + span)
        else:
            zone_lo = max(long_lo, long_hi - span)
            zone_hi = long_hi

        original_lateral = starts[indices, lateral_axis]
        original_span = float(np.max(original_lateral) - np.min(original_lateral)) if len(original_lateral) else 0.0
        original_center = float(np.mean(original_lateral)) if len(original_lateral) else 0.5 * (lat_lo + lat_hi)
        target_lanes = min(
            len(indices),
            max(1, int(math.floor((original_span + 2.0 * min_spacing) / min_spacing)) + 1),
        )
        target_width = max(original_span + 2.0 * min_spacing, (target_lanes - 1) * min_spacing)
        target_width = min(target_width, lat_hi - lat_lo)
        band_lo = max(lat_lo, original_center - 0.5 * target_width)
        band_hi = min(lat_hi, original_center + 0.5 * target_width)
        if band_hi - band_lo < target_width:
            if band_lo <= lat_lo:
                band_hi = min(lat_hi, band_lo + target_width)
            elif band_hi >= lat_hi:
                band_lo = max(lat_lo, band_hi - target_width)

        lateral_blocked_with_buffer = [
            (
                float(obs.center[lateral_axis] - obs.half_size[lateral_axis] - obstacle_buffer),
                float(obs.center[lateral_axis] + obs.half_size[lateral_axis] + obstacle_buffer),
            )
            for obs in obstacles
        ]
        lateral_values = _axis_lanes(band_lo, band_hi, min_spacing, lateral_blocked_with_buffer)
        lanes = max(1, len(lateral_values))
        rows = int(math.ceil(len(indices) / lanes))
        available_depth = max(zone_hi - zone_lo, min_spacing)
        row_spacing = min_spacing if rows <= 1 else min(min_spacing, available_depth / max(rows - 1, 1))
        if rows == 1:
            longitudinal_values = np.array([(zone_lo + zone_hi) * 0.5])
        elif side < 0:
            longitudinal_values = zone_lo + np.arange(rows) * row_spacing
        else:
            longitudinal_values = zone_hi - np.arange(rows) * row_spacing

        ordered = sorted(indices, key=lambda idx: float(starts[idx, lateral_axis]))
        for slot, idx in enumerate(ordered):
            row = slot // lanes
            col = slot % lanes
            candidate = packed[idx].copy()
            candidate[axis] = longitudinal_values[min(row, len(longitudinal_values) - 1)]
            candidate[lateral_axis] = lateral_values[col]
            packed[idx] = candidate

    return packed


def _compact_line_positions(count: int, half_span: float, min_spacing: float) -> np.ndarray:
    if count <= 0:
        return np.array([])
    required_half_span = 0.5 * min_spacing * max(count - 1, 0)
    span = max(half_span, required_half_span)
    return np.linspace(-span, span, count)


def _bounded_line_positions(
    count: int,
    half_span: float,
    min_spacing: float,
    lower: float,
    upper: float,
    margin: float,
) -> np.ndarray:
    if count <= 0:
        return np.array([])
    safe_lower = lower + margin
    safe_upper = upper - margin
    if safe_upper <= safe_lower:
        return np.full(count, 0.5 * (lower + upper))
    center = 0.5 * (safe_lower + safe_upper)
    required_half_span = 0.5 * min_spacing * max(count - 1, 0)
    requested_span = max(half_span, required_half_span)
    bounded_span = min(requested_span, 0.5 * (safe_upper - safe_lower))
    if count == 1:
        return np.array([center])
    return np.linspace(center - bounded_span, center + bounded_span, count)


def _free_axis_intervals(lower: float, upper: float, intervals: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    clipped = []
    for a, b in intervals:
        lo = max(lower, min(a, b))
        hi = min(upper, max(a, b))
        if hi > lo:
            clipped.append((lo, hi))
    if not clipped:
        return [(lower, upper)]
    clipped.sort()
    merged = []
    cur_lo, cur_hi = clipped[0]
    for lo, hi in clipped[1:]:
        if lo <= cur_hi:
            cur_hi = max(cur_hi, hi)
        else:
            merged.append((cur_lo, cur_hi))
            cur_lo, cur_hi = lo, hi
    merged.append((cur_lo, cur_hi))
    free = []
    cursor = lower
    for lo, hi in merged:
        if lo > cursor:
            free.append((cursor, lo))
        cursor = max(cursor, hi)
    if cursor < upper:
        free.append((cursor, upper))
    return free


def _axis_lanes(
    lower: float,
    upper: float,
    min_spacing: float,
    blocked: Iterable[Tuple[float, float]],
) -> np.ndarray:
    lanes = []
    for lo, hi in _free_axis_intervals(lower, upper, blocked):
        width = hi - lo
        if width < 0.5 * min_spacing:
            continue
        count = max(1, int(math.floor(width / min_spacing)) + 1)
        lanes.extend(np.linspace(lo, hi, count))
    if not lanes:
        lanes = list(np.linspace(lower, upper, max(1, int(math.floor((upper - lower) / min_spacing)) + 1)))
    return np.array(sorted(set(round(float(v), 6) for v in lanes)), dtype=np.float32)


def _safe_axis_limits(
    lower: float,
    upper: float,
    blocked: Iterable[Tuple[float, float]],
    clearance: float,
) -> Tuple[float, float]:
    free = _free_axis_intervals(lower, upper, blocked)
    if not free:
        return lower, upper
    lo, hi = max(free, key=lambda interval: interval[1] - interval[0])
    safe_lo = lo + clearance
    safe_hi = hi - clearance
    if safe_hi <= safe_lo:
        center = 0.5 * (lo + hi)
        return center, center
    return safe_lo, safe_hi


def _free_axis_gaps(lower: float, upper: float, intervals: Iterable[Tuple[float, float]]) -> List[float]:
    return [hi - lo for lo, hi in _free_axis_intervals(lower, upper, intervals)]


def scaled_workspace_size(n: int) -> float:
    if n <= 8:
        return 10.0
    if n <= 20:
        return 20.0
    if n <= 50:
        return 40.0
    return 40.0 * math.sqrt(float(n) / 50.0)


def _scaled_bounds(size: float) -> Tuple[float, float, float, float]:
    half = 0.5 * size
    return (-half, half, -half, half)


def _scaled_obstacle(size: float, x: float, y: float, hx: float, hy: float) -> Obstacle:
    return Obstacle(np.array([x * size, y * size], dtype=np.float32), np.array([hx * size, hy * size], dtype=np.float32))


def _make_scaled_scenario(name: str, n: int, seed: int, tier: str, area_policy: str) -> Scenario:
    rng = np.random.default_rng(seed)
    size = scaled_workspace_size(n)
    bounds = _scaled_bounds(size)
    edge = 0.43 * size
    jitter = 0.008 * size
    obstacles: List[Obstacle] = []

    if name == "open":
        y = np.linspace(-0.32 * size, 0.32 * size, n)
        starts = np.column_stack([np.full(n, -edge), y])
        goals = np.column_stack([np.full(n, edge), y])
    elif name == "sparse":
        y = np.linspace(-0.36 * size, 0.36 * size, n)
        x_shift = np.linspace(-0.04 * size, 0.04 * size, n)
        starts = np.column_stack([np.full(n, -edge) + x_shift, y])
        goals = np.column_stack([np.full(n, edge) - x_shift, y + 0.04 * size])
    elif name == "warehouse_large":
        rows = int(math.ceil(n / 2))
        y = np.linspace(-0.40 * size, 0.40 * size, rows)
        starts_left = np.column_stack([np.full(rows, -edge), y])[: (n + 1) // 2]
        starts_right = np.column_stack([np.full(rows, edge), y[::-1]])[: n // 2]
        starts = np.vstack([starts_left, starts_right])
        goals = np.vstack(
            [
                np.column_stack([np.full(len(starts_left), edge), starts_left[:, 1]]),
                np.column_stack([np.full(len(starts_right), -edge), starts_right[:, 1]]),
            ]
        )
        obstacles = [
            _scaled_obstacle(size, -0.20, 0.24, 0.035, 0.075),
            _scaled_obstacle(size, 0.20, 0.24, 0.035, 0.075),
            _scaled_obstacle(size, -0.20, -0.24, 0.035, 0.075),
            _scaled_obstacle(size, 0.20, -0.24, 0.035, 0.075),
        ]
    elif name == "crossing":
        angles = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
        starts = np.stack([edge * np.cos(angles), edge * np.sin(angles)], axis=1)
        goals = -starts
    elif name == "corridor":
        left = n // 2
        right = n - left
        y_left = np.linspace(-0.18 * size, 0.18 * size, left) if left else np.array([])
        y_right = np.linspace(-0.18 * size, 0.18 * size, right) if right else np.array([])
        starts = np.vstack(
            [
                np.column_stack([np.full(left, -edge), y_left]),
                np.column_stack([np.full(right, edge), y_right]),
            ]
        )
        goals = np.vstack(
            [
                np.column_stack([np.full(left, edge), y_left[::-1] if left else y_left]),
                np.column_stack([np.full(right, -edge), y_right[::-1] if right else y_right]),
            ]
        )
        obstacles = [
            _scaled_obstacle(size, 0.0, 0.435, 0.48, 0.052),
            _scaled_obstacle(size, 0.0, -0.435, 0.48, 0.052),
        ]
    elif name == "warehouse":
        rows = int(math.ceil(n / 2))
        y = np.linspace(-0.38 * size, 0.38 * size, rows)
        starts_left = np.column_stack([np.full(rows, -edge), y])[: (n + 1) // 2]
        starts_right = np.column_stack([np.full(rows, edge), y[::-1]])[: n // 2]
        starts = np.vstack([starts_left, starts_right])
        goals = np.vstack(
            [
                np.column_stack([np.full(len(starts_left), edge), starts_left[:, 1]]),
                np.column_stack([np.full(len(starts_right), -edge), starts_right[:, 1]]),
            ]
        )
        obstacles = [
            _scaled_obstacle(size, -0.16, 0.19, 0.04, 0.09),
            _scaled_obstacle(size, 0.16, 0.19, 0.04, 0.09),
            _scaled_obstacle(size, -0.16, -0.19, 0.04, 0.09),
            _scaled_obstacle(size, 0.16, -0.19, 0.04, 0.09),
        ]
    elif name == "bottleneck":
        y = np.linspace(-0.34 * size, 0.34 * size, n)
        starts = np.column_stack([np.full(n, -edge), y])
        goals = np.column_stack([np.full(n, edge), -y])
        obstacles = [
            _scaled_obstacle(size, 0.0, 0.30, 0.055, 0.18),
            _scaled_obstacle(size, 0.0, -0.30, 0.055, 0.18),
        ]
    elif name == "narrow_gate":
        left = (n + 1) // 2
        right = n - left
        y_left = np.linspace(-0.24 * size, 0.24 * size, left) if left else np.array([])
        y_right = np.linspace(0.24 * size, -0.24 * size, right) if right else np.array([])
        starts = np.vstack(
            [
                np.column_stack([np.full(left, -edge), y_left]),
                np.column_stack([np.full(right, edge), y_right]),
            ]
        )
        goals = np.vstack(
            [
                np.column_stack([np.full(left, edge), -y_left]),
                np.column_stack([np.full(right, -edge), -y_right]),
            ]
        )
        obstacles = [
            _scaled_obstacle(size, 0.0, 0.31, 0.09, 0.17),
            _scaled_obstacle(size, 0.0, -0.31, 0.09, 0.17),
        ]
    elif name == "merge_bottleneck":
        lanes = np.linspace(-0.36 * size, 0.36 * size, n)
        exit_lanes = np.linspace(-0.16 * size, 0.16 * size, n)
        starts = np.column_stack([np.full(n, -edge), lanes])
        goals = np.column_stack([np.full(n, edge), exit_lanes])
        obstacles = [
            _scaled_obstacle(size, -0.02, 0.31, 0.05, 0.17),
            _scaled_obstacle(size, -0.02, -0.31, 0.05, 0.17),
        ]
    elif name == "merge_bottleneck_asym":
        lanes = np.linspace(-0.34 * size, 0.38 * size, n)
        x_offsets = np.linspace(-0.05 * size, 0.07 * size, n)
        exit_lanes = np.linspace(-0.13 * size, 0.18 * size, n)
        exit_lanes = exit_lanes + 0.015 * size * np.sin(np.linspace(0.0, math.pi, n))
        starts = np.column_stack([np.full(n, -edge) + x_offsets, lanes])
        goals = np.column_stack([np.full(n, edge), exit_lanes])
        obstacles = [
            _scaled_obstacle(size, -0.01, 0.32, 0.05, 0.16),
            _scaled_obstacle(size, -0.04, -0.30, 0.05, 0.18),
        ]
    elif name in {"four_way_stop", "intersection"}:
        starts_rows = []
        goals_rows = []
        counts = [n // 4 + (1 if k < n % 4 else 0) for k in range(4)]
        for direction, count in enumerate(counts):
            offsets = np.linspace(-0.20 * size, 0.20 * size, count) if count else np.array([])
            for offset in offsets:
                if direction == 0:
                    starts_rows.append([-edge, offset])
                    goals_rows.append([edge, offset])
                elif direction == 1:
                    starts_rows.append([edge, offset])
                    goals_rows.append([-edge, offset])
                elif direction == 2:
                    starts_rows.append([offset, edge])
                    goals_rows.append([offset, -edge])
                else:
                    starts_rows.append([offset, -edge])
                    goals_rows.append([offset, edge])
        starts = np.array(starts_rows, dtype=np.float32)
        goals = np.array(goals_rows, dtype=np.float32)
    elif name == "swap_lanes":
        left = (n + 1) // 2
        right = n - left
        y_left = np.linspace(-0.20 * size, 0.20 * size, left) if left else np.array([])
        y_right = np.linspace(0.20 * size, -0.20 * size, right) if right else np.array([])
        starts = np.vstack(
            [
                np.column_stack([np.full(left, -edge), y_left]),
                np.column_stack([np.full(right, edge), y_right]),
            ]
        )
        goals = np.vstack(
            [
                np.column_stack([np.full(left, edge), y_left + 0.08 * size]),
                np.column_stack([np.full(right, -edge), y_right - 0.08 * size]),
            ]
        )
    else:
        raise ValueError(f"unknown scenario: {name}")

    starts = _bounded_jitter(rng, starts, jitter)
    goals = _bounded_jitter(rng, goals, jitter)
    if tier == "stress":
        starts = _stress_pack_positions(starts, goals, bounds, obstacles)

    return Scenario(
        name=name,
        starts=starts.astype(np.float32),
        goals=goals.astype(np.float32),
        bounds=bounds,
        obstacles=obstacles,
        tier=tier,
        area_policy=area_policy,
    )


def make_scenario(name: str, n: int, seed: int, tier: str = "main", area_policy: str = "fixed") -> Scenario:
    if tier not in {"main", "stress", "scaled"}:
        raise ValueError(f"unknown scenario tier: {tier}")
    if area_policy not in {"fixed", "scaled"}:
        raise ValueError(f"unknown area policy: {area_policy}")
    effective_area_policy = "scaled" if tier == "scaled" else area_policy
    if tier == "scaled" or effective_area_policy == "scaled":
        return _make_scaled_scenario(name, n, seed, tier=tier, area_policy=effective_area_policy)

    rng = np.random.default_rng(seed)
    bounds = (-6.0, 6.0, -6.0, 6.0)
    obstacles: List[Obstacle] = []

    if name == "open":
        bounds = (-6.0, 6.0, -6.0, 6.0)
        y = _bounded_line_positions(n, 4.0, 0.95, bounds[2], bounds[3], margin=0.95)
        starts = np.column_stack([np.full(n, -5.0), y])
        goals = np.column_stack([np.full(n, 5.0), y])
        starts = _bounded_jitter(rng, starts, 0.04)
        goals = _bounded_jitter(rng, goals, 0.04)
    elif name == "sparse":
        bounds = (-8.0, 8.0, -8.0, 8.0)
        y = _bounded_line_positions(n, 5.6, 1.35, bounds[2], bounds[3], margin=1.1)
        x_shift = np.linspace(-0.45, 0.45, n)
        starts = np.column_stack([np.full(n, -6.6) + x_shift, y])
        goals = np.column_stack([np.full(n, 6.6) - x_shift, y + 0.45])
        starts = _bounded_jitter(rng, starts, 0.08)
        goals = _bounded_jitter(rng, goals, 0.08)
    elif name == "warehouse_large":
        bounds = (-8.0, 8.0, -8.0, 8.0)
        rows = int(math.ceil(n / 2))
        y = np.linspace(-6.0, 6.0, rows)
        starts_left = np.column_stack([np.full(rows, -6.7), y])[: (n + 1) // 2]
        starts_right = np.column_stack([np.full(rows, 6.7), y[::-1]])[: n // 2]
        starts = np.vstack([starts_left, starts_right])
        goals = np.vstack(
            [
                np.column_stack([np.full(len(starts_left), 6.7), starts_left[:, 1]]),
                np.column_stack([np.full(len(starts_right), -6.7), starts_right[:, 1]]),
            ]
        )
        starts = _jitter(rng, starts, 0.08)
        goals = _jitter(rng, goals, 0.08)
        obstacles = [
            Obstacle(np.array([-2.8, 3.0]), np.array([0.35, 0.85])),
            Obstacle(np.array([2.8, 3.0]), np.array([0.35, 0.85])),
            Obstacle(np.array([-2.8, -3.0]), np.array([0.35, 0.85])),
            Obstacle(np.array([2.8, -3.0]), np.array([0.35, 0.85])),
        ]
    elif name == "crossing":
        angles = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
        starts = np.stack([5.15 * np.cos(angles), 5.15 * np.sin(angles)], axis=1)
        goals = -starts
        starts = _jitter(rng, starts, 0.12)
        goals = _jitter(rng, goals, 0.12)
    elif name == "corridor":
        bounds = (-6.0, 6.0, -2.7, 2.7)
        left = n // 2
        right = n - left
        y_left = _bounded_line_positions(left, 1.05, 0.68, bounds[2], bounds[3], margin=0.78)
        y_right = _bounded_line_positions(right, 1.05, 0.68, bounds[2], bounds[3], margin=0.78)
        starts = np.vstack(
            [
                np.column_stack([np.full(left, -5.0), y_left]),
                np.column_stack([np.full(right, 5.0), y_right]),
            ]
        )
        goals = np.vstack(
            [
                np.column_stack([np.full(left, 5.0), y_left[::-1] if left else y_left]),
                np.column_stack([np.full(right, -5.0), y_right[::-1] if right else y_right]),
            ]
        )
        starts = _bounded_jitter(rng, starts, 0.04)
        goals = _bounded_jitter(rng, goals, 0.04)
        obstacles = [
            Obstacle(np.array([0.0, 2.35]), np.array([5.8, 0.28])),
            Obstacle(np.array([0.0, -2.35]), np.array([5.8, 0.28])),
        ]
    elif name == "warehouse":
        rows = int(math.ceil(n / 2))
        y = np.linspace(-4.5, 4.5, rows)
        starts_left = np.column_stack([np.full(rows, -5.0), y])[: (n + 1) // 2]
        starts_right = np.column_stack([np.full(rows, 5.0), y[::-1]])[: n // 2]
        starts = np.vstack([starts_left, starts_right])
        goals = np.vstack(
            [
                np.column_stack([np.full(len(starts_left), 5.0), starts_left[:, 1]]),
                np.column_stack([np.full(len(starts_right), -5.0), starts_right[:, 1]]),
            ]
        )
        starts = _jitter(rng, starts, 0.10)
        goals = _jitter(rng, goals, 0.10)
        obstacles = [
            Obstacle(np.array([-1.8, 2.1]), np.array([0.45, 1.0])),
            Obstacle(np.array([1.8, 2.1]), np.array([0.45, 1.0])),
            Obstacle(np.array([-1.8, -2.1]), np.array([0.45, 1.0])),
            Obstacle(np.array([1.8, -2.1]), np.array([0.45, 1.0])),
        ]
    elif name == "bottleneck":
        bounds = (-6.0, 6.0, -4.0, 4.0)
        y = _bounded_line_positions(n, 2.7, 0.66, bounds[2], bounds[3], margin=0.70)
        starts = np.column_stack([np.full(n, -5.2), y])
        goals = np.column_stack([np.full(n, 5.2), -y])
        starts = _bounded_jitter(rng, starts, 0.07)
        goals = _bounded_jitter(rng, goals, 0.07)
        obstacles = [
            Obstacle(np.array([0.0, 2.35]), np.array([0.65, 1.45])),
            Obstacle(np.array([0.0, -2.35]), np.array([0.65, 1.45])),
        ]
    elif name == "narrow_gate":
        bounds = (-6.0, 6.0, -4.0, 4.0)
        left = (n + 1) // 2
        right = n - left
        y_left = np.linspace(-1.4, 1.4, left) if left else np.array([])
        y_right = np.linspace(1.4, -1.4, right) if right else np.array([])
        starts = np.vstack(
            [
                np.column_stack([np.full(left, -5.1), y_left]),
                np.column_stack([np.full(right, 5.1), y_right]),
            ]
        )
        goals = np.vstack(
            [
                np.column_stack([np.full(left, 5.1), -y_left]),
                np.column_stack([np.full(right, -5.1), -y_right]),
            ]
        )
        starts = _jitter(rng, starts, 0.07)
        goals = _jitter(rng, goals, 0.07)
        obstacles = [
            Obstacle(np.array([0.0, 2.45]), np.array([1.05, 1.35])),
            Obstacle(np.array([0.0, -2.45]), np.array([1.05, 1.35])),
        ]
    elif name == "merge_bottleneck":
        bounds = (-6.0, 6.0, -4.0, 4.0)
        lanes = _bounded_line_positions(n, 2.55, 0.68, bounds[2], bounds[3], margin=0.70)
        starts = np.column_stack([np.full(n, -5.2), lanes])
        exit_lanes = np.linspace(-0.85, 0.85, n)
        goals = np.column_stack([np.full(n, 5.1), exit_lanes])
        starts = _bounded_jitter(rng, starts, 0.05)
        goals = _bounded_jitter(rng, goals, 0.07)
        obstacles = [
            Obstacle(np.array([-0.2, 2.45]), np.array([0.55, 1.35])),
            Obstacle(np.array([-0.2, -2.45]), np.array([0.55, 1.35])),
        ]
    elif name == "merge_bottleneck_asym":
        bounds = (-6.0, 6.0, -4.0, 4.0)
        lanes = _bounded_line_positions(n, 2.50, 0.68, bounds[2], bounds[3], margin=0.70)
        x_offsets = np.linspace(-0.42, 0.54, n)
        starts = np.column_stack([np.full(n, -5.15) + x_offsets, lanes + 0.10])
        exit_lanes = np.linspace(-0.70, 0.95, n)
        exit_lanes = exit_lanes + 0.10 * np.sin(np.linspace(0.0, math.pi, n))
        goals = np.column_stack([np.full(n, 5.1), exit_lanes])
        starts = _bounded_jitter(rng, starts, 0.04)
        goals = _bounded_jitter(rng, goals, 0.06)
        obstacles = [
            Obstacle(np.array([-0.12, 2.48]), np.array([0.55, 1.32])),
            Obstacle(np.array([-0.32, -2.38]), np.array([0.55, 1.42])),
        ]
    elif name in {"four_way_stop", "intersection"}:
        bounds = (-5.5, 5.5, -5.5, 5.5)
        edge = 4.8
        counts = [n // 4 + (1 if k < n % 4 else 0) for k in range(4)]
        starts_rows = []
        goals_rows = []
        for direction, count in enumerate(counts):
            offsets = _bounded_line_positions(count, 0.70, 0.70, bounds[2], bounds[3], margin=0.70)
            for offset in offsets:
                if direction == 0:
                    starts_rows.append([-edge, offset])
                    goals_rows.append([edge, offset])
                elif direction == 1:
                    starts_rows.append([edge, offset])
                    goals_rows.append([-edge, offset])
                elif direction == 2:
                    starts_rows.append([offset, edge])
                    goals_rows.append([offset, -edge])
                else:
                    starts_rows.append([offset, -edge])
                    goals_rows.append([offset, edge])
        starts = np.array(starts_rows, dtype=np.float32)
        goals = np.array(goals_rows, dtype=np.float32)
        starts = _bounded_jitter(rng, starts, 0.05)
        goals = _bounded_jitter(rng, goals, 0.05)
    elif name == "swap_lanes":
        bounds = (-6.0, 6.0, -3.0, 3.0)
        left = (n + 1) // 2
        right = n - left
        y_left = np.linspace(-1.0, 1.0, left) if left else np.array([])
        y_right = np.linspace(1.0, -1.0, right) if right else np.array([])
        starts = np.vstack(
            [
                np.column_stack([np.full(left, -5.0), y_left]),
                np.column_stack([np.full(right, 5.0), y_right]),
            ]
        )
        goals = np.vstack(
            [
                np.column_stack([np.full(left, 5.0), y_left + 0.75]),
                np.column_stack([np.full(right, -5.0), y_right - 0.75]),
            ]
        )
        starts = _jitter(rng, starts, 0.05)
        goals = _jitter(rng, goals, 0.05)
    else:
        raise ValueError(f"unknown scenario: {name}")

    if tier == "stress":
        starts = _stress_pack_positions(starts, goals, bounds, obstacles)

    return Scenario(
        name=name,
        starts=starts.astype(np.float32),
        goals=goals.astype(np.float32),
        bounds=bounds,
        obstacles=obstacles,
        tier=tier,
        area_policy=effective_area_policy,
    )


def obstacle_array(obstacles: Iterable[Obstacle]) -> np.ndarray:
    rows = []
    for obs in obstacles:
        rows.append([obs.center[0], obs.center[1], obs.half_size[0], obs.half_size[1]])
    if not rows:
        return np.zeros((0, 4), dtype=np.float32)
    return np.array(rows, dtype=np.float32)
