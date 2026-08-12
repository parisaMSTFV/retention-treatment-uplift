from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from retention_uplift.config import ProjectConfig
from retention_uplift.pipeline import run_pipeline


class PipelineTests(unittest.TestCase):
    def test_small_pipeline_writes_auditable_outputs(self) -> None:
        config = replace(
            ProjectConfig(),
            n_customers=1_600,
            n_waves=12,
            train_end_wave=5,
            validation_end_wave=8,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = run_pipeline(root, config)
            self.assertIn(metrics["selected_model"], {
                "T-learner linear",
                "T-learner hist",
                "DR-learner hist",
            })
            self.assertTrue((root / "reports" / "policy_comparison.csv").exists())
            self.assertTrue((root / "reports" / "overlap_diagnostics.csv").exists())
            self.assertTrue((root / "reports" / "policy_sensitivity.csv").exists())
            self.assertTrue(
                (root / "reports" / "figures" / "policy_sensitivity.png").exists()
            )
            self.assertTrue((root / "reports" / "figures" / "qini_curves.png").exists())
            self.assertTrue((root / "data" / "sample" / "synthetic_policy_sample.csv").exists())


if __name__ == "__main__":
    unittest.main()
