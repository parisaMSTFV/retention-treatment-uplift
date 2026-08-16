from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from retention_uplift.criteo import (
    FEATURE_COLUMNS,
    file_sha256,
    scan_and_sample_criteo,
)
from retention_uplift.external_validation import BinaryTLearner, run_criteo_validation


def write_fixture(path: Path, rows: int = 7_000, seed: int = 17) -> None:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(rows, len(FEATURE_COLUMNS)))
    treatment = rng.binomial(1, 0.72, size=rows)
    base_visit = 1.0 / (1.0 + np.exp(-(-2.3 + 0.35 * features[:, 0])))
    effect = 0.025 + 0.045 * (features[:, 1] > 0)
    visit_probability = np.clip(base_visit + treatment * effect, 0.0, 0.95)
    visit = rng.binomial(1, visit_probability)
    conversion = visit * rng.binomial(1, 0.12, size=rows)
    exposure = treatment * rng.binomial(1, 0.25, size=rows)
    frame = pd.DataFrame(features, columns=FEATURE_COLUMNS)
    frame["treatment"] = treatment
    frame["conversion"] = conversion
    frame["visit"] = visit
    frame["exposure"] = exposure
    frame.to_csv(path, index=False, compression="gzip")


class CriteoContractTests(unittest.TestCase):
    def test_checksum_and_exact_sample_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fixture.csv.gz"
            write_fixture(source)

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                scan_and_sample_criteo(
                    source,
                    sample_size=5_000,
                    seed=9,
                    expected_sha256="0" * 64,
                    expected_bytes=None,
                    expected_rows=7_000,
                )

            result = scan_and_sample_criteo(
                source,
                sample_size=5_000,
                seed=9,
                expected_sha256=file_sha256(source),
                expected_bytes=source.stat().st_size,
                expected_rows=7_000,
                chunk_size=1_200,
            )
            self.assertEqual(result.source_rows, 7_000)
            self.assertEqual(len(result.sample), 5_000)
            self.assertEqual(
                set(result.sample["partition"]), {"train", "validation", "test"}
            )

    def test_post_assignment_exposure_is_not_a_model_feature(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fixture.csv.gz"
            write_fixture(source)
            frame = pd.read_csv(source)
            train = frame.iloc[:5_000].copy()
            test = frame.iloc[5_000:].copy()
            model = BinaryTLearner("logistic", seed=5).fit(train, "visit")

            original = model.predict_outcomes(test)
            test["exposure"] = 1 - test["exposure"]
            changed = model.predict_outcomes(test)

            np.testing.assert_allclose(original[0], changed[0])
            np.testing.assert_allclose(original[1], changed[1])

    def test_external_pipeline_writes_aggregate_uncertainty_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fixture.csv.gz"
            write_fixture(source)
            metrics = run_criteo_validation(
                root,
                source,
                sample_size=5_000,
                target_share=0.30,
                seed=13,
                expected_sha256=file_sha256(source),
                expected_bytes=source.stat().st_size,
                expected_rows=7_000,
            )

            reports = root / "reports" / "external_validation"
            self.assertEqual(metrics["data_mode"], "external_randomized_benchmark")
            self.assertIn(metrics["selected_model"], {"T-learner logistic", "T-learner hist"})
            self.assertIn("aipw_effect_among_targeted_ci_low", metrics["test_policy"])
            self.assertTrue((reports / "source_provenance.json").exists())
            self.assertTrue((reports / "external_run_metrics.json").exists())
            self.assertTrue((reports / "test_uplift_deciles.csv").exists())
            self.assertTrue((reports / "LICENSE.md").exists())
            summary = (reports / "run_summary.md").read_text(encoding="utf-8")
            self.assertIn("intention-to-treat", summary)
            self.assertIn("does not validate retention treatments", summary)
            self.assertFalse((reports / "sample.csv").exists())


if __name__ == "__main__":
    unittest.main()
