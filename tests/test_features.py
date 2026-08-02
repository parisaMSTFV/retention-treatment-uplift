from __future__ import annotations

import unittest
from dataclasses import replace

import pandas as pd

from retention_uplift.config import ProjectConfig
from retention_uplift.features import design_matrix
from retention_uplift.simulation import simulate_experiment


class FeatureTests(unittest.TestCase):
    def test_future_and_outcome_columns_cannot_change_features(self) -> None:
        frame = simulate_experiment(replace(ProjectConfig(), n_customers=1_200))
        original, columns = design_matrix(frame)
        changed = frame.copy()
        changed["net_value_60d"] = 999_999
        changed["truth_expected_net_voucher"] = -999_999
        rebuilt, _ = design_matrix(changed, columns)
        pd.testing.assert_frame_equal(original, rebuilt)


if __name__ == "__main__":
    unittest.main()
