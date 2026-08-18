from __future__ import annotations

import unittest

import pandas as pd

from scripts.run_architecture_conditioning_ablation import (
    CONDITION_SPECS,
    condition_by_label,
    expected_case_count,
    factorial_interaction_rows,
)


class ArchitectureConditioningAblationTests(unittest.TestCase):
    def test_design_is_a_fully_crossed_architecture_by_conditioning_matrix(self) -> None:
        self.assertEqual(
            set(CONDITION_SPECS),
            {"plain_candidate", "plain_zero", "residual_candidate", "residual_zero"},
        )
        self.assertEqual(
            {(spec.architecture, spec.candidate_conditioned) for spec in CONDITION_SPECS.values()},
            {
                ("plain", False),
                ("plain", True),
                ("residual", False),
                ("residual", True),
            },
        )

    def test_condition_specs_bind_features_and_planner_queries_consistently(self) -> None:
        self.assertEqual(condition_by_label("plain_candidate").feature_key, "x_intent")
        self.assertEqual(condition_by_label("plain_candidate").planner_method, "icgp_rvo_mpc")
        self.assertEqual(condition_by_label("plain_zero").feature_key, "x_passive")
        self.assertEqual(condition_by_label("plain_zero").planner_method, "passive_rvo_mpc")
        self.assertEqual(condition_by_label("residual_candidate").feature_key, "x_intent")
        self.assertEqual(condition_by_label("residual_zero").feature_key, "x_passive")

    def test_expected_case_count_tracks_all_factor_levels(self) -> None:
        self.assertEqual(
            expected_case_count(
                scenarios=("corridor", "narrow_gate"),
                robots=(4, 8),
                seed_indices=(0, 1, 2),
                noise_levels=("clean", "medium"),
            ),
            96,
        )

    def test_factorial_interaction_is_difference_of_conditioning_effects(self) -> None:
        rows = []
        values = {
            "plain_candidate": 0.60,
            "plain_zero": 0.50,
            "residual_candidate": 0.90,
            "residual_zero": 0.70,
        }
        for method, progress_ratio in values.items():
            rows.append(
                {
                    "noise_level": "medium",
                    "scenario": "corridor",
                    "robots": 4,
                    "seed_index": 0,
                    "method": method,
                    "progress_ratio": progress_ratio,
                }
            )
        interaction = factorial_interaction_rows(pd.DataFrame(rows), ("progress_ratio",))
        self.assertEqual(len(interaction), 1)
        self.assertAlmostEqual(float(interaction.loc[0, "plain_conditioning_effect"]), 0.10)
        self.assertAlmostEqual(float(interaction.loc[0, "residual_conditioning_effect"]), 0.20)
        self.assertAlmostEqual(float(interaction.loc[0, "interaction_delta"]), 0.10)


if __name__ == "__main__":
    unittest.main()
