from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
