from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from retention_uplift.config import ACTIVE_ACTIONS, ProjectConfig
from retention_uplift.policy import (
    greedy_uplift_policy,
    policy_cost,
    solve_policy,
    validate_policy,
)


class PolicyTests(unittest.TestCase):
    def test_policy_respects_budget_capacity_and_one_action(self) -> None:
        config = ProjectConfig()
        rng = np.random.default_rng(7)
        gains = pd.DataFrame(rng.normal(4, 3, size=(300, 3)), columns=ACTIVE_ACTIONS)
        policy = solve_policy(gains, config)
        validate_policy(policy, config)
        self.assertLessEqual(policy_cost(policy, config), config.budget_per_customer * len(policy))
        self.assertTrue(set(policy).issubset({"control", *ACTIVE_ACTIONS}))

    def test_negative_gain_customers_are_not_treated(self) -> None:
        config = ProjectConfig()
        gains = pd.DataFrame(-1.0, index=range(100), columns=ACTIVE_ACTIONS)
        policy = solve_policy(gains, config)
        self.assertTrue(policy.eq("control").all())

    def test_optimizer_and_greedy_baseline_share_stressed_constraints(self) -> None:
        config = ProjectConfig()
        rng = np.random.default_rng(19)
        gains = pd.DataFrame(rng.normal(5, 4, size=(300, 3)), columns=ACTIVE_ACTIONS)
        budget = config.budget_per_customer * len(gains) * 0.6
        for policy in (
            solve_policy(gains, config, budget=budget, capacity_multiplier=0.7),
            greedy_uplift_policy(
                gains,
                config,
                budget=budget,
                capacity_multiplier=0.7,
            ),
        ):
            validate_policy(
                policy,
                config,
                budget=budget,
                capacity_multiplier=0.7,
            )

    def test_capacity_multiplier_must_be_positive(self) -> None:
        config = ProjectConfig()
        gains = pd.DataFrame(1.0, index=range(20), columns=ACTIVE_ACTIONS)
        with self.assertRaises(ValueError):
            solve_policy(gains, config, capacity_multiplier=0)

    def test_non_finite_gains_and_negative_budget_are_rejected(self) -> None:
        config = ProjectConfig()
        gains = pd.DataFrame(1.0, index=range(20), columns=ACTIVE_ACTIONS)
        gains.loc[0, "voucher"] = np.nan
        with self.assertRaises(ValueError):
            solve_policy(gains, config)
        with self.assertRaises(ValueError):
            solve_policy(gains.fillna(1.0), config, budget=-1.0)

    def test_unknown_policy_action_and_invalid_config_are_rejected(self) -> None:
        config = ProjectConfig()
        gains = pd.DataFrame(1.0, index=range(20), columns=ACTIVE_ACTIONS)
        with self.assertRaises(ValueError):
            validate_policy(pd.Series(["control", "unknown"]), config)
        with self.assertRaises(ValueError):
            solve_policy(gains, replace(config, budget_per_customer=-0.1))
        with self.assertRaises(ValueError):
            replace(
                config,
                capacity_shares={
                    "reminder": 0.2,
                    "voucher": 1.1,
                    "service_call": 0.1,
                },
            ).validate()


if __name__ == "__main__":
    unittest.main()
