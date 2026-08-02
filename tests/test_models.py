from __future__ import annotations

import unittest
from dataclasses import replace

from retention_uplift.config import ACTIVE_ACTIONS, ProjectConfig
from retention_uplift.models import TLearner
from retention_uplift.simulation import simulate_experiment, temporal_split


class ModelTests(unittest.TestCase):
    def test_t_learner_returns_one_gain_per_active_action(self) -> None:
        config = replace(ProjectConfig(), n_customers=2_200)
        train, validation, _ = temporal_split(simulate_experiment(config), config)
        model = TLearner("linear", config.seed).fit(train)
        gains = model.predict_gains(validation)
        self.assertEqual(tuple(gains.columns), ACTIVE_ACTIONS)
        self.assertEqual(len(gains), len(validation))
        self.assertFalse(gains.isna().any().any())


if __name__ == "__main__":
    unittest.main()
