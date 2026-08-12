from __future__ import annotations

import unittest
from dataclasses import replace

from retention_uplift.config import ACTIONS, ProjectConfig
from retention_uplift.evaluation import overlap_diagnostics
from retention_uplift.simulation import simulate_experiment


class EvaluationTests(unittest.TestCase):
    def test_randomized_design_has_explicit_overlap_evidence(self) -> None:
        config = replace(ProjectConfig(), n_customers=1_500)
        experiment = simulate_experiment(config)
        diagnostics = overlap_diagnostics(experiment, config, "all")

        self.assertEqual(set(diagnostics["action"]), set(ACTIONS))
        self.assertTrue(diagnostics["support_pass"].all())
        self.assertEqual(int(diagnostics["customers_trimmed"].sum()), 0)
        self.assertAlmostEqual(
            float(diagnostics["configured_propensity"].min()),
            min(config.action_probabilities.values()),
        )
        self.assertAlmostEqual(
            float(diagnostics["max_inverse_probability_weight"].max()),
            1 / min(config.action_probabilities.values()),
        )

    def test_overlap_rule_flags_an_under_supported_arm(self) -> None:
        config = replace(
            ProjectConfig(),
            n_customers=1_500,
            action_probabilities={
                "control": 0.41,
                "reminder": 0.25,
                "voucher": 0.30,
                "service_call": 0.04,
            },
        )
        experiment = simulate_experiment(config)
        diagnostics = overlap_diagnostics(experiment, config, "all")
        failed = diagnostics.loc[~diagnostics["support_pass"]]

        self.assertEqual(failed["action"].tolist(), ["service_call"])
        self.assertEqual(int(failed.iloc[0]["customers_trimmed"]), len(experiment))


if __name__ == "__main__":
    unittest.main()
