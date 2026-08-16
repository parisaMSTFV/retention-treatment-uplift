"""External validation on the randomized CRITEO-UPLIFTv2 benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from retention_uplift.criteo import (
    DATASET_BYTES,
    DATASET_FILENAME,
    DATASET_LICENSE,
    DATASET_LICENSE_URL,
    DATASET_NAME,
    DATASET_PAGE,
    DATASET_PAPER,
    DATASET_REPOSITORY,
    DATASET_REVISION,
    DATASET_ROWS,
    DATASET_SHA256,
    DATASET_URL,
    FEATURE_COLUMNS,
    SAMPLER_NAME,
    binary_difference_from_counts,
    scan_and_sample_criteo,
)


def _json_default(value: object) -> object:
    return value.item() if hasattr(value, "item") else str(value)


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def _interval(values: np.ndarray) -> dict[str, float]:
    estimate = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    return {
        "estimate": estimate,
        "standard_error": standard_error,
        "ci_low": estimate - 1.96 * standard_error,
        "ci_high": estimate + 1.96 * standard_error,
    }


def _frame_difference(frame: pd.DataFrame, outcome: str) -> dict[str, float | int]:
    counts: dict[str, int] = {}
    for treatment, label in ((0, "control"), (1, "treatment")):
        group = frame.loc[frame["treatment"].eq(treatment), outcome]
        counts[f"{label}_rows"] = len(group)
        counts[f"{label}_positive"] = int(group.sum())
    return binary_difference_from_counts(counts)


class BinaryTLearner:
    """Two outcome classifiers for randomized assignment versus control."""

    def __init__(self, family: str, seed: int) -> None:
        self.family = family
        self.seed = seed
        self.name = f"T-learner {family}"
        self.models: dict[int, object] = {}

    def _new_model(self, arm: int) -> object:
        if self.family == "logistic":
            return make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    max_iter=300,
                    random_state=self.seed + arm,
                ),
            )
        if self.family == "hist":
            return HistGradientBoostingClassifier(
                learning_rate=0.06,
                max_iter=100,
                max_leaf_nodes=15,
                min_samples_leaf=80,
                l2_regularization=2.0,
                random_state=self.seed + arm,
            )
        raise ValueError(f"unsupported external model family: {self.family}")

    def fit(self, frame: pd.DataFrame, outcome: str) -> BinaryTLearner:
        matrix = frame.loc[:, FEATURE_COLUMNS]
        for arm in (0, 1):
            mask = frame["treatment"].eq(arm)
            if mask.sum() < 500 or frame.loc[mask, outcome].nunique() != 2:
                raise ValueError(f"insufficient outcome support in randomized arm {arm}")
            model = self._new_model(arm)
            model.fit(matrix.loc[mask], frame.loc[mask, outcome])
            self.models[arm] = model
        return self

    def predict_outcomes(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if set(self.models) != {0, 1}:
            raise RuntimeError("external uplift model must be fitted before prediction")
        matrix = frame.loc[:, FEATURE_COLUMNS]
        control = self.models[0].predict_proba(matrix)[:, 1]
        treatment = self.models[1].predict_proba(matrix)[:, 1]
        return control, treatment


def _aipw_scores(
    frame: pd.DataFrame,
    mu_control: np.ndarray,
    mu_treatment: np.ndarray,
    outcome: str,
    treatment_probability: float,
) -> np.ndarray:
    treatment = frame["treatment"].to_numpy(dtype=float)
    observed = frame[outcome].to_numpy(dtype=float)
    return (
        mu_treatment
        - mu_control
        + treatment / treatment_probability * (observed - mu_treatment)
        - (1.0 - treatment)
        / (1.0 - treatment_probability)
        * (observed - mu_control)
    )


def _policy_metrics(
    frame: pd.DataFrame,
    mu_control: np.ndarray,
    mu_treatment: np.ndarray,
    *,
    outcome: str,
    treatment_probability: float,
    threshold: float,
) -> dict[str, float | int]:
    uplift = mu_treatment - mu_control
    policy = uplift >= threshold
    reach = float(np.mean(policy))
    scores = _aipw_scores(
        frame,
        mu_control,
        mu_treatment,
        outcome,
        treatment_probability,
    )
    policy_interval = _interval(policy.astype(float) * scores)
    targeted_interval = _interval(scores[policy])
    random_interval = _interval(reach * scores)
    incremental_interval = _interval((policy.astype(float) - reach) * scores)

    treatment = frame["treatment"].to_numpy(dtype=float)
    observed = frame[outcome].to_numpy(dtype=float)
    ipw_scores = (
        treatment * observed / treatment_probability
        - (1.0 - treatment) * observed / (1.0 - treatment_probability)
    )
    order = np.argsort(-uplift)
    x = np.arange(1, len(frame) + 1, dtype=float) / len(frame)
    cumulative = np.cumsum(ipw_scores[order]) / len(frame)
    ipw_qini = float(np.trapezoid(cumulative, x) - 0.5 * cumulative[-1])
    return {
        "customers": len(frame),
        "targeted_customers": int(policy.sum()),
        "targeted_share": reach,
        "score_threshold": threshold,
        "aipw_policy_effect_per_eligible": policy_interval["estimate"],
        "aipw_policy_effect_ci_low": policy_interval["ci_low"],
        "aipw_policy_effect_ci_high": policy_interval["ci_high"],
        "aipw_effect_among_targeted": targeted_interval["estimate"],
        "aipw_effect_among_targeted_ci_low": targeted_interval["ci_low"],
        "aipw_effect_among_targeted_ci_high": targeted_interval["ci_high"],
        "random_same_reach_effect_per_eligible": random_interval["estimate"],
        "incremental_over_random_same_reach": incremental_interval["estimate"],
        "incremental_over_random_ci_low": incremental_interval["ci_low"],
        "incremental_over_random_ci_high": incremental_interval["ci_high"],
        "ipw_qini_per_eligible": ipw_qini,
    }


def _uplift_deciles(
    frame: pd.DataFrame,
    mu_control: np.ndarray,
    mu_treatment: np.ndarray,
    outcome: str,
) -> pd.DataFrame:
    scored = frame.loc[:, ["treatment", outcome]].copy()
    scored["predicted_uplift"] = mu_treatment - mu_control
    scored["decile"] = (
        pd.qcut(scored["predicted_uplift"].rank(method="first"), 10, labels=False)
        + 1
    )
    rows: list[dict[str, float | int]] = []
    for decile, group in scored.groupby("decile"):
        result = _frame_difference(group, outcome)
        rows.append(
            {
                "decile": int(decile),
                "customers": len(group),
                "mean_predicted_uplift": float(group["predicted_uplift"].mean()),
                "observed_difference_in_means": result["difference_in_means"],
                "observed_ci_low": result["ci_low"],
                "observed_ci_high": result["ci_high"],
                "treated_customers": result["treatment_rows"],
                "control_customers": result["control_rows"],
            }
        )
    return pd.DataFrame(rows).sort_values("decile", ascending=False)


def _maximum_absolute_smd(frame: pd.DataFrame) -> float:
    treated = frame.loc[frame["treatment"].eq(1), FEATURE_COLUMNS]
    control = frame.loc[frame["treatment"].eq(0), FEATURE_COLUMNS]
    pooled = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2.0)
    smd = (treated.mean() - control.mean()) / pooled.replace(0.0, np.nan)
    return float(smd.abs().fillna(0.0).max())


def _treatment_prediction_auc(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> float:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=300, random_state=seed),
    )
    model.fit(train.loc[:, FEATURE_COLUMNS], train["treatment"])
    probability = model.predict_proba(test.loc[:, FEATURE_COLUMNS])[:, 1]
    return float(roc_auc_score(test["treatment"], probability))


def _plot_deciles(deciles: pd.DataFrame, path: Path) -> None:
    ordered = deciles.sort_values("decile")
    figure, axis = plt.subplots(figsize=(9, 5.2))
    estimates = ordered["observed_difference_in_means"].to_numpy(float)
    lower = estimates - ordered["observed_ci_low"].to_numpy(float)
    upper = ordered["observed_ci_high"].to_numpy(float) - estimates
    axis.errorbar(
        ordered["decile"],
        estimates,
        yerr=np.vstack([lower, upper]),
        fmt="o-",
        color="#2563EB",
        capsize=3,
        label="Observed randomized-arm contrast",
    )
    axis.plot(
        ordered["decile"],
        ordered["mean_predicted_uplift"],
        "o--",
        color="#0F766E",
        label="Mean predicted uplift",
    )
    axis.axhline(0.0, color="#64748B", linewidth=0.9)
    axis.set_xlabel("Predicted-uplift decile (10 = highest)")
    axis.set_ylabel("Two-week visit-rate difference")
    axis.set_title("External test-set uplift calibration with 95% intervals")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _write_external_summary(
    path: Path,
    metrics: dict[str, object],
    model_comparison: pd.DataFrame,
) -> None:
    selected = metrics["test_policy"]
    visit = metrics["full_source_outcomes"]["visit"]
    conversion = metrics["full_source_outcomes"]["conversion"]
    comparison = model_comparison.loc[
        :,
        [
            "model",
            "validation_targeted_share",
            "validation_incremental_over_random_same_reach",
            "validation_incremental_ci_low",
            "validation_incremental_ci_high",
            "selected",
        ],
    ].copy()
    for column in comparison.select_dtypes(include="number"):
        comparison[column] = comparison[column].map(lambda value: f"{value:.6f}")
    table = _markdown_table(comparison)
    visit_row = (
        f"| Visit | {visit['control_rate']:.4%} | {visit['treatment_rate']:.4%} | "
        f"{visit['difference_in_means']:.4%} | {visit['ci_low']:.4%} to "
        f"{visit['ci_high']:.4%} |"
    )
    conversion_row = (
        f"| Conversion | {conversion['control_rate']:.4%} | "
        f"{conversion['treatment_rate']:.4%} | "
        f"{conversion['difference_in_means']:.4%} | {conversion['ci_low']:.4%} to "
        f"{conversion['ci_high']:.4%} |"
    )
    text = f"""# External Validation — CRITEO-UPLIFTv2.1

## Evidence boundary

This run validates a **binary advertising-assignment uplift workflow** on the released Criteo
benchmark. It does not validate retention treatments, multi-action choice, customer value,
treatment costs, or transportability to another population.

The source is a collection of advertising incrementality tests in which a randomized holdout was
prevented from targeting. The estimand here is the intention-to-treat effect of randomized
assignment on a two-week visit outcome in the **released, non-uniformly subsampled benchmark**.
It is not the original advertisers' campaign effect. `exposure` is post-assignment and was excluded
from every model feature set.

## Source verification

- Dataset: {DATASET_NAME}
- Validated rows: {int(metrics['source_rows']):,}
- Validated bytes: {int(metrics['source_bytes']):,}
- SHA-256: `{metrics['source_sha256']}`
- License: [{DATASET_LICENSE}]({DATASET_LICENSE_URL}) — noncommercial use, attribution, and
  ShareAlike apply to the dataset and adapted material
- Modeling sample: {int(metrics['sample_rows']):,} rows selected by `{SAMPLER_NAME}` with seed
  `{int(metrics['seed'])}` after scanning and validating the complete file

## Full released-benchmark contrasts

| Outcome | Control rate | Assigned-treatment rate | Difference | 95% CI |
|---|---:|---:|---:|---:|
{visit_row}
{conversion_row}

These are unadjusted randomized-arm contrasts within the released benchmark. The narrow intervals
reflect its very large row count, not guaranteed external validity.

## Frozen test policy

The model was fitted on the training partition, selected and thresholded only on validation, then
evaluated once on the independent deterministic test partition. The selected model was
**{metrics['selected_model']}**. Its frozen threshold targeted
**{selected['targeted_share']:.1%}** of test rows.

- AIPW visit effect among targeted rows: **{selected['aipw_effect_among_targeted']:.4%}**
  (95% CI {selected['aipw_effect_among_targeted_ci_low']:.4%} to
  {selected['aipw_effect_among_targeted_ci_high']:.4%})
- AIPW policy effect per eligible test row: **{selected['aipw_policy_effect_per_eligible']:.4%}**
  (95% CI {selected['aipw_policy_effect_ci_low']:.4%} to
  {selected['aipw_policy_effect_ci_high']:.4%})
- Incremental effect over random targeting at the same realized reach:
  **{selected['incremental_over_random_same_reach']:.4%}**
  (95% CI {selected['incremental_over_random_ci_low']:.4%} to
  {selected['incremental_over_random_ci_high']:.4%})
- Treatment-prediction AUC on test: **{metrics['treatment_prediction_auc']:.4f}**;
  maximum absolute feature SMD: **{metrics['maximum_absolute_smd']:.4f}**

The AIPW calculation uses the pooled training assignment share because experiment/advertiser
strata and their propensities are not present in the public schema. That is an explicit limitation,
not evidence that every source experiment used an identical probability.

## Validation-only model selection

{table}

Validation intervals are descriptive diagnostics after model comparison; confirmatory uncertainty
is reported only on the untouched test partition.

## What this evidence does and does not support

The run supports that this repository can ingest a real randomized uplift benchmark, preserve a
clean train/validation/test boundary, rank a binary advertising treatment, and report sampling
uncertainty. It does not establish realized retention profit, optimal contact policy, temporal
stability, or individual causal truth. The file has no time key, experiment stratum, treatment
cost, or retention outcome, so those claims are deliberately omitted.
"""
    path.write_text(text, encoding="utf-8")


def _write_data_notice(path: Path) -> None:
    text = f"""# External Validation Data Notice

The aggregate files in this directory were produced from the {DATASET_NAME} dataset published by
Criteo AI Lab. The source dataset is licensed under
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International]({DATASET_LICENSE_URL}).

Source and attribution:

- Criteo AI Lab, [Criteo Uplift Prediction Dataset]({DATASET_PAGE})
- Eustache Diemert, Artem Betlei, Christophe Renaudin, and Massih-Reza Amini,
  *A Large Scale Benchmark for Uplift Modeling*, AdKDD 2018
- Exact source identity is recorded in [source_provenance.json](source_provenance.json).

Modification notice: the complete source was identity-checked and schema-validated, aggregate
randomized-arm statistics were calculated, and a deterministic sample was used for model
validation. The chart and tables are new aggregate outputs; raw source rows are not reproduced.

To the extent these aggregate reports or visualization constitute adapted material under the
dataset license, they are shared under {DATASET_LICENSE}. No raw or sampled dataset rows are
included. The repository's original source code remains covered by the root MIT license; neither
the MIT license nor this notice relicenses Criteo's source dataset.
"""
    path.write_text(text, encoding="utf-8")


def run_criteo_validation(
    root: str | Path,
    source_path: str | Path,
    *,
    sample_size: int = 300_000,
    target_share: float = 0.30,
    seed: int = 42,
    expected_sha256: str = DATASET_SHA256,
    expected_bytes: int | None = DATASET_BYTES,
    expected_rows: int | None = DATASET_ROWS,
) -> dict[str, object]:
    """Run a leakage-aware external uplift benchmark and write aggregate evidence."""

    if not 0.05 <= target_share <= 0.95:
        raise ValueError("target_share must be between 0.05 and 0.95")
    loaded = scan_and_sample_criteo(
        source_path,
        sample_size=sample_size,
        seed=seed,
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
        expected_rows=expected_rows,
    )
    sample = loaded.sample
    train = sample.loc[sample["partition"].eq("train")].copy()
    validation = sample.loc[sample["partition"].eq("validation")].copy()
    test = sample.loc[sample["partition"].eq("test")].copy()
    treatment_probability = float(train["treatment"].mean())
    if not 0.05 < treatment_probability < 0.95:
        raise ValueError("pooled training assignment share violates the support rule")

    candidates: list[tuple[BinaryTLearner, dict[str, float | int]]] = []
    for family in ("logistic", "hist"):
        model = BinaryTLearner(family, seed).fit(train, "visit")
        mu_control, mu_treatment = model.predict_outcomes(validation)
        threshold = float(np.quantile(mu_treatment - mu_control, 1.0 - target_share))
        result = _policy_metrics(
            validation,
            mu_control,
            mu_treatment,
            outcome="visit",
            treatment_probability=treatment_probability,
            threshold=threshold,
        )
        candidates.append((model, result))

    selected_index = int(
        np.argmax(
            [result["incremental_over_random_same_reach"] for _, result in candidates]
        )
    )
    comparison_rows = []
    for index, (model, result) in enumerate(candidates):
        comparison_rows.append(
            {
                "model": model.name,
                "validation_targeted_share": result["targeted_share"],
                "validation_policy_effect_per_eligible": result[
                    "aipw_policy_effect_per_eligible"
                ],
                "validation_incremental_over_random_same_reach": result[
                    "incremental_over_random_same_reach"
                ],
                "validation_incremental_ci_low": result[
                    "incremental_over_random_ci_low"
                ],
                "validation_incremental_ci_high": result[
                    "incremental_over_random_ci_high"
                ],
                "validation_ipw_qini_per_eligible": result["ipw_qini_per_eligible"],
                "score_threshold": result["score_threshold"],
                "selected": index == selected_index,
            }
        )
    model_comparison = pd.DataFrame(comparison_rows)
    selected_model, selected_validation = candidates[selected_index]
    test_mu_control, test_mu_treatment = selected_model.predict_outcomes(test)
    test_policy = _policy_metrics(
        test,
        test_mu_control,
        test_mu_treatment,
        outcome="visit",
        treatment_probability=treatment_probability,
        threshold=float(selected_validation["score_threshold"]),
    )
    deciles = _uplift_deciles(test, test_mu_control, test_mu_treatment, "visit")

    metrics: dict[str, object] = {
        "data_mode": "external_randomized_benchmark",
        "dataset": DATASET_NAME,
        "source_file": Path(source_path).name,
        "source_sha256": loaded.sha256,
        "source_bytes": loaded.file_bytes,
        "source_rows": loaded.source_rows,
        "sample_rows": len(sample),
        "sampler": SAMPLER_NAME,
        "seed": seed,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "outcome": "visit",
        "estimand": "pooled randomized-assignment ITT in the released benchmark",
        "target_share_requested": target_share,
        "pooled_training_assignment_share": treatment_probability,
        "selected_model": selected_model.name,
        "treatment_prediction_auc": _treatment_prediction_auc(train, test, seed + 800),
        "maximum_absolute_smd": _maximum_absolute_smd(sample),
        "full_source_outcomes": {
            outcome: binary_difference_from_counts(counts)
            for outcome, counts in loaded.full_counts.items()
        },
        "test_visit_difference_in_means": _frame_difference(test, "visit"),
        "test_conversion_difference_in_means": _frame_difference(test, "conversion"),
        "test_policy": test_policy,
    }

    root = Path(root)
    reports = root / "reports" / "external_validation"
    reports.mkdir(parents=True, exist_ok=True)
    provenance = {
        "dataset": DATASET_NAME,
        "official_page": DATASET_PAGE,
        "official_repository": DATASET_REPOSITORY,
        "download_url_pinned_revision": DATASET_URL,
        "repository_revision": DATASET_REVISION,
        "source_filename": DATASET_FILENAME,
        "sha256": loaded.sha256,
        "bytes": loaded.file_bytes,
        "rows": loaded.source_rows,
        "license": DATASET_LICENSE,
        "license_url": DATASET_LICENSE_URL,
        "paper": DATASET_PAPER,
        "sampler": SAMPLER_NAME,
        "sample_size": len(sample),
        "sample_seed": seed,
        "raw_or_sample_rows_committed": False,
    }
    (reports / "source_provenance.json").write_text(
        json.dumps(provenance, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    (reports / "external_run_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    model_comparison.to_csv(
        reports / "model_comparison.csv", index=False, float_format="%.12g"
    )
    deciles.to_csv(reports / "test_uplift_deciles.csv", index=False, float_format="%.12g")
    _plot_deciles(deciles, reports / "visit_uplift_deciles.png")
    _write_external_summary(reports / "run_summary.md", metrics, model_comparison)
    _write_data_notice(reports / "LICENSE.md")
    return metrics
