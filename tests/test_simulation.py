from __future__ import annotations

import unittest
from dataclasses import replace

import pandas as pd

from retention_uplift.config import ACTIONS, ProjectConfig
from retention_uplift.simulation import simulate_experiment, temporal_split


class SimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = replace(ProjectConfig(), n_customers=2_000)

    def test_simulation_is_reproducible_and_randomized(self) -> None:
        first = simulate_experiment(self.config)
        second = simulate_experiment(self.config)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(set(first["assigned_action"]), set(ACTIONS))
        self.assertTrue(first["assignment_probability"].between(0, 1).all())

    def test_temporal_partitions_do_not_overlap(self) -> None:
        frame = simulate_experiment(self.config)
        train, validation, test = temporal_split(frame, self.config)
        self.assertLess(train["wave"].max(), validation["wave"].min())
        self.assertLess(validation["wave"].max(), test["wave"].min())
        self.assertEqual(len(train) + len(validation) + len(test), len(frame))


if __name__ == "__main__":
    unittest.main()
