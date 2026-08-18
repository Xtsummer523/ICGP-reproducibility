from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HASH_LIST = ROOT / "SHA256SUMS.csv"
REPORT_FILE = ROOT / "verification_report.json"
VERSION_NAME_PATTERN = re.compile(r"(?:^|[_-])v\d{2,}(?:[_\-.]|$)", re.IGNORECASE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_csv(
    relative_path: str,
    expected_rows: int,
    *,
    unique_fields: tuple[str, ...] = (),
    expected_values: dict[str, set[str]] | None = None,
) -> dict:
    path = ROOT / relative_path
    result = {
        "path": relative_path,
        "exists": path.is_file(),
        "expected_rows": expected_rows,
        "observed_rows": None,
        "duplicate_keys": None,
        "value_checks": {},
        "passed": False,
    }
    if not path.is_file():
        return result

    count = 0
    keys: set[tuple[str, ...]] = set()
    duplicate_keys = 0
    observed_values = {field: set() for field in (expected_values or {})}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        missing_key_fields = set(unique_fields) - fields
        missing_value_fields = set(observed_values) - fields
        if missing_key_fields or missing_value_fields:
            result["missing_fields"] = sorted(missing_key_fields | missing_value_fields)
            return result
        for row in reader:
            count += 1
            if unique_fields:
                key = tuple(row[field] for field in unique_fields)
                if key in keys:
                    duplicate_keys += 1
                keys.add(key)
            for field in observed_values:
                observed_values[field].add(row[field])

    result["observed_rows"] = count
    result["duplicate_keys"] = duplicate_keys
    value_pass = True
    for field, expected in (expected_values or {}).items():
        observed = observed_values[field]
        field_pass = observed == expected
        value_pass = value_pass and field_pass
        result["value_checks"][field] = {
            "expected": sorted(expected),
            "observed": sorted(observed),
            "passed": field_pass,
        }
    result["passed"] = count == expected_rows and duplicate_keys == 0 and value_pass
    return result


def check_nonempty_csv(relative_path: str) -> dict:
    path = ROOT / relative_path
    result = {"path": relative_path, "exists": path.is_file(), "rows": 0, "passed": False}
    if not path.is_file():
        return result
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        next(reader, None)
        result["rows"] = sum(1 for _ in reader)
    result["passed"] = result["rows"] > 0
    return result


def check_hashes() -> dict:
    result = {
        "manifest": "SHA256SUMS.csv",
        "exists": HASH_LIST.is_file(),
        "entries": 0,
        "missing_files": [],
        "size_mismatches": [],
        "hash_mismatches": [],
        "unlisted_files": [],
        "stale_entries": [],
        "passed": False,
    }
    if not HASH_LIST.is_file():
        return result

    declared: set[str] = set()
    with HASH_LIST.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            relative = row["relative_path"].replace("\\", "/")
            declared.add(relative)
            result["entries"] += 1
            path = ROOT / Path(relative)
            if not path.is_file():
                result["missing_files"].append(relative)
                continue
            expected_size = int(row["size_bytes"])
            actual_size = path.stat().st_size
            if actual_size != expected_size:
                result["size_mismatches"].append(
                    {"path": relative, "expected": expected_size, "actual": actual_size}
                )
            expected_hash = row["sha256"].lower()
            actual_hash = sha256(path)
            if actual_hash != expected_hash:
                result["hash_mismatches"].append(
                    {"path": relative, "expected": expected_hash, "actual": actual_hash}
                )

    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and path not in {HASH_LIST, REPORT_FILE}
        and "__pycache__" not in path.parts
    }
    result["unlisted_files"] = sorted(actual - declared)
    result["stale_entries"] = sorted(declared - actual)
    result["passed"] = not any(
        (
            result["missing_files"],
            result["size_mismatches"],
            result["hash_mismatches"],
            result["unlisted_files"],
            result["stale_entries"],
        )
    )
    return result


def check_names() -> dict:
    bad = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if VERSION_NAME_PATTERN.search(relative):
            bad.append(relative)
    return {"version_like_names": sorted(bad), "passed": not bad}


def build_report() -> dict:
    methods_main = {
        "icgp_rvo_mpc",
        "passive_rvo_mpc",
        "rvo_reactive",
        "constant_velocity_mpc",
    }
    methods_formal = {
        "icgp_rvo_mpc",
        "passive_rvo_mpc",
        "rvo_reactive",
        "residual_passive_rvo_mpc",
    }
    factorial_methods = {
        "plain_candidate",
        "plain_zero",
        "residual_candidate",
        "residual_zero",
    }
    scenarios = {"corridor", "narrow_gate", "merge_bottleneck", "four_way_stop"}

    csv_checks = [
        check_csv(
            "data/legacy_main/selected_episode_results.csv",
            960,
            unique_fields=("scenario", "robots", "seed", "method"),
            expected_values={"method": methods_main},
        ),
        check_csv(
            "data/gaussian_observation_study/episode_results.csv",
            3840,
            unique_fields=("noise_level", "scenario", "robots", "seed_index", "method"),
            expected_values={
                "method": methods_formal,
                "scenario": scenarios,
                "noise_level": {"clean", "low", "medium", "high"},
                "robots": {"4", "8"},
            },
        ),
        check_csv(
            "data/autoregressive_observation_study/episode_results.csv",
            1920,
            unique_fields=("noise_level", "scenario", "robots", "seed_index", "method"),
            expected_values={
                "method": methods_formal,
                "scenario": scenarios,
                "noise_level": {"low", "high"},
                "robots": {"4", "8"},
            },
        ),
        check_csv(
            "data/short_dropout_probability_005/episode_results.csv",
            1920,
            unique_fields=("noise_level", "scenario", "robots", "seed_index", "method"),
            expected_values={
                "method": methods_formal,
                "scenario": scenarios,
                "noise_level": {"low", "high"},
                "robots": {"4", "8"},
            },
        ),
        check_csv(
            "data/short_dropout_probability_010/episode_results.csv",
            1920,
            unique_fields=("noise_level", "scenario", "robots", "seed_index", "method"),
            expected_values={
                "method": methods_formal,
                "scenario": scenarios,
                "noise_level": {"low", "high"},
                "robots": {"4", "8"},
            },
        ),
        check_csv(
            "data/prediction_quality/case_summary.csv",
            1440,
            unique_fields=("scenario", "robots", "seed_index", "method", "noise_level"),
            expected_values={
                "method": {
                    "icgp_rvo_mpc",
                    "passive_rvo_mpc",
                    "residual_passive_rvo_mpc",
                },
                "scenario": scenarios,
                "noise_level": {"clean", "high"},
                "robots": {"4", "8"},
            },
        ),
        check_csv(
            "data/gazebo_boundary/eight_robot_seed_results.csv",
            30,
            unique_fields=("seed", "method_key"),
            expected_values={"method_key": {"icgp", "passive", "residual_passive"}},
        ),
        check_csv(
            "data/gazebo_boundary/eight_robot_paired_differences.csv",
            20,
            unique_fields=("seed", "baseline"),
            expected_values={"baseline": {"Passive", "Residual-passive"}},
        ),
        check_csv(
            "data/ros2_gazebo_rviz/single_seed_method_summary.csv",
            3,
            unique_fields=("method",),
            expected_values={
                "method": {
                    "icgp_rvo_mpc",
                    "passive_rvo_mpc",
                    "residual_passive_rvo_mpc",
                }
            },
        ),
        check_csv(
            "data/factorial_ablation/factorial_results.csv",
            3840,
            unique_fields=("noise_level", "scenario", "robots", "seed_index", "method"),
            expected_values={
                "method": factorial_methods,
                "scenario": scenarios,
                "noise_level": {"clean", "low", "medium", "high"},
                "robots": {"4", "8"},
            },
        ),
    ]

    nonempty_checks = [
        check_nonempty_csv("data/ros2_gazebo_rviz/icgp_states.csv"),
        check_nonempty_csv("data/ros2_gazebo_rviz/icgp_pairwise_distances.csv"),
        check_nonempty_csv("data/ros2_gazebo_rviz/passive_states.csv"),
        check_nonempty_csv("data/ros2_gazebo_rviz/passive_pairwise_distances.csv"),
        check_nonempty_csv("data/ros2_gazebo_rviz/residual_passive_states.csv"),
        check_nonempty_csv("data/ros2_gazebo_rviz/residual_passive_pairwise_distances.csv"),
        check_nonempty_csv("data/ros2_gazebo_rviz/progress_timeseries.csv"),
        check_nonempty_csv("data/ros2_gazebo_rviz/minimum_distance_timeseries.csv"),
    ]

    hashes = check_hashes()
    names = check_names()
    passed = all(item["passed"] for item in csv_checks + nonempty_checks) and hashes["passed"] and names["passed"]
    return {
        "archive": "ICGP JIROS Online Resource 1",
        "csv_design_checks": csv_checks,
        "nonempty_log_checks": nonempty_checks,
        "hash_check": hashes,
        "filename_check": names,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the ICGP supplementary archive.")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    report = build_report()
    if args.write_report:
        REPORT_FILE.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
