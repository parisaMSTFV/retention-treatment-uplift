from __future__ import annotations

import unittest

from retention_uplift.stability import stability_row


class StabilityTests(unittest.TestCase):
    def test_stability_row_records_uncertainty_and_policy_comparison(self) -> None:
        row = stability_row(
            {
                "seed": 7,
                "selected_model": "T-learner linear",
                "test_customers": 3_000,
                "selected_dr_incremental_value": 2_000.0,
                "selected_ci_low": 100.0,
                "selected_ci_high": 3_900.0,
                "selected_true_incremental_value": 2_400.0,
                "risk_true_incremental_value": 1_600.0,
                "oracle_true_incremental_value": 2_800.0,
                "selected_treated_customers": 720,
            }
        )

        self.assertTrue(row["ci_excludes_zero"])
        self.assertTrue(row["beats_risk_only"])
        self.assertAlmostEqual(float(row["true_value_lift_vs_risk"]), 0.5)


if __name__ == "__main__":
    unittest.main()
