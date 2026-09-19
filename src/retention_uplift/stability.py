"""Repeated-seed evidence for the synthetic decision pipeline."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import pandas as pd

from retention_uplift.config import ProjectConfig
from retention_uplift.pipeline import run_pipeline
from retention_uplift.reporting import CSV_FLOAT_FORMAT

DEFAULT_STABILITY_SEEDS = (1, 7, 42, 123, 2026)


def stability_row(metrics: dict[str, object]) -> dict[str, object]:
    """Select decision-relevant metrics from one completed pipeline run."""
    truth = float(metrics["selected_true_incremental_value"])
    risk = float(metrics["risk_true_incremental_value"])
    ci_low = float(metrics["selected_ci_low"])
    ci_high = float(metrics["selected_ci_high"])
    return {
        "seed": int(metrics["seed"]),
        "selected_model": str(metrics["selected_model"]),
        "test_customers": int(metrics["test_customers"]),
        "dr_incremental_value": float(metrics["selected_dr_incremental_value"]),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_excludes_zero": ci_low > 0 or ci_high < 0,
        "true_incremental_value": truth,
        "risk_true_incremental_value": risk,
        "true_value_lift_vs_risk": (truth - risk) / max(abs(risk), 1.0),
        "beats_risk_only": truth > risk,
        "oracle_true_incremental_value": float(metrics["oracle_true_incremental_value"]),
        "treated_customers": int(metrics["selected_treated_customers"]),
    }


def run_seed_stability(
    run_root: Path,
    output_path: Path,
    *,
    seeds: Iterable[int] = DEFAULT_STABILITY_SEEDS,
    n_customers: int = 18_000,
) -> pd.DataFrame:
    """Run the full synthetic pipeline for each seed and write a compact evidence table."""
    seed_values = tuple(seeds)
    if len(seed_values) < 2 or len(set(seed_values)) != len(seed_values):
        raise ValueError("provide at least two distinct seeds")
    rows = []
    for seed in seed_values:
        config = replace(ProjectConfig(), seed=seed, n_customers=n_customers)
        metrics = run_pipeline(run_root / f"seed-{seed}", config)
        rows.append(stability_row(metrics))
    result = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, float_format=CSV_FLOAT_FORMAT)
    return result
