"""Uplift model, experiment, and policy evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from retention_uplift.config import ACTIONS, ACTIVE_ACTIONS, ProjectConfig
from retention_uplift.features import NUMERIC_FEATURES
from retention_uplift.policy import policy_cost


def _smd(treated: pd.Series, control: pd.Series) -> float:
    pooled = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2)
    return 0.0 if pooled == 0 else float((treated.mean() - control.mean()) / pooled)


def experiment_health(frame: pd.DataFrame, config: ProjectConfig) -> dict[str, float | int]:
    control = frame.loc[frame["assigned_action"].eq("control")]
    smds = []
    for action in ACTIVE_ACTIONS:
        treated = frame.loc[frame["assigned_action"].eq(action)]
        smds.extend(abs(_smd(treated[column], control[column])) for column in NUMERIC_FEATURES)
    expected = np.asarray([config.action_probabilities[action] * len(frame) for action in ACTIONS])
    observed = np.asarray([frame["assigned_action"].eq(action).sum() for action in ACTIONS])
    chi_square = float(np.sum((observed - expected) ** 2 / expected))
    return {
        "customers": len(frame),
        "max_abs_smd": max(smds),
        "assignment_chi_square": chi_square,
        **{f"n_{action}": int(count) for action, count in zip(ACTIONS, observed, strict=True)},
    }


def effect_accuracy(frame: pd.DataFrame, gains: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rows = []
    for action in ACTIVE_ACTIONS:
        truth = (
            frame[f"truth_expected_net_{action}"] - frame["truth_expected_net_control"]
        ).to_numpy(float)
        prediction = gains[action].to_numpy(float)
        rows.append(
            {
                "model": model_name,
                "action": action,
                "effect_rmse": float(np.sqrt(np.mean((prediction - truth) ** 2))),
                "effect_bias": float(np.mean(prediction - truth)),
                "rank_correlation": float(
                    pd.Series(prediction).corr(pd.Series(truth), method="spearman")
                ),
            }
        )
    return pd.DataFrame(rows)


def policy_effect(
    frame: pd.DataFrame,
    policy: pd.Series,
    outcome_predictions: pd.DataFrame,
    config: ProjectConfig,
    policy_name: str,
) -> tuple[dict[str, float | int | str], np.ndarray]:
    """Estimate incremental policy value with DR and compare to hidden simulation truth."""
    action_to_index = {action: index for index, action in enumerate(ACTIONS)}
    predicted = outcome_predictions.loc[frame.index, ACTIONS].to_numpy(float)
    policy_index = policy.map(action_to_index).to_numpy(int)
    observed_index = frame["assigned_action"].map(action_to_index).to_numpy(int)
    rows = np.arange(len(frame))
    mu_policy = predicted[rows, policy_index]
    mu_control = predicted[:, action_to_index["control"]]
    observed = frame["net_value_60d"].to_numpy(float)
    policy_probability = policy.map(config.action_probabilities).to_numpy(float)
    correction_policy = (observed_index == policy_index) / policy_probability * (
        observed - mu_policy
    )
    correction_control = (
        (observed_index == action_to_index["control"])
        / config.action_probabilities["control"]
        * (observed - mu_control)
    )
    dr_effect = mu_policy + correction_policy - mu_control - correction_control

    truth_matrix = frame.loc[:, [f"truth_expected_net_{action}" for action in ACTIONS]].to_numpy(
        float
    )
    true_effect = truth_matrix[rows, policy_index] - truth_matrix[:, action_to_index["control"]]
    standard_error = float(np.std(dr_effect, ddof=1) / np.sqrt(len(frame)))
    total = float(np.sum(dr_effect))
    scale = len(frame)
    result: dict[str, float | int | str] = {
        "policy": policy_name,
        "dr_incremental_value": total,
        "dr_value_per_customer": float(np.mean(dr_effect)),
        "ci_low": float((np.mean(dr_effect) - 1.96 * standard_error) * scale),
        "ci_high": float((np.mean(dr_effect) + 1.96 * standard_error) * scale),
        "true_incremental_value": float(np.sum(true_effect)),
        "treatment_cost": policy_cost(policy, config),
        "treated_customers": int(policy.ne("control").sum()),
        **{f"n_{action}": int(policy.eq(action).sum()) for action in ACTIONS},
    }
    return result, dr_effect


def uplift_rank_metrics(
    frame: pd.DataFrame,
    gains: pd.DataFrame,
    model_name: str,
    config: ProjectConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate treatment-specific IPW cumulative gain curves and Qini coefficients."""
    metric_rows = []
    curve_rows = []
    observed_action = frame["assigned_action"].to_numpy()
    observed_value = frame["net_value_60d"].to_numpy(float)
    n = len(frame)
    x = np.arange(1, n + 1) / n
    for action in ACTIVE_ACTIONS:
        score = gains[action].to_numpy(float)
        pseudo = (
            (observed_action == action) * observed_value / config.action_probabilities[action]
            - (observed_action == "control")
            * observed_value
            / config.action_probabilities["control"]
        )
        order = np.argsort(-score)
        cumulative_gain = np.cumsum(pseudo[order]) / n
        auuc = float(np.trapezoid(cumulative_gain, x))
        random_area = 0.5 * float(cumulative_gain[-1])
        qini = auuc - random_area
        metric_rows.append(
            {
                "model": model_name,
                "action": action,
                "auuc": auuc,
                "qini": qini,
                "full_sample_ipw_ate": float(cumulative_gain[-1]),
            }
        )
        for index in np.linspace(0, n - 1, 101, dtype=int):
            curve_rows.append(
                {
                    "model": model_name,
                    "action": action,
                    "targeted_share": float(x[index]),
                    "cumulative_gain_per_customer": float(cumulative_gain[index]),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(curve_rows)


def uplift_calibration(
    frame: pd.DataFrame,
    gains: pd.DataFrame,
    config: ProjectConfig,
    bins: int = 10,
) -> pd.DataFrame:
    rows = []
    for action in ACTIVE_ACTIONS:
        temporary = pd.DataFrame(
            {
                "prediction": gains[action],
                "assigned_action": frame["assigned_action"],
                "observed": frame["net_value_60d"],
            },
            index=frame.index,
        )
        temporary["bin"] = pd.qcut(
            temporary["prediction"].rank(method="first"), bins, labels=False
        )
        for bin_id, group in temporary.groupby("bin"):
            ipw_effect = np.mean(
                (group["assigned_action"].eq(action) * group["observed"])
                / config.action_probabilities[action]
                - (group["assigned_action"].eq("control") * group["observed"])
                / config.action_probabilities["control"]
            )
            rows.append(
                {
                    "action": action,
                    "bin": int(bin_id) + 1,
                    "predicted_gain": float(group["prediction"].mean()),
                    "observed_ipw_gain": float(ipw_effect),
                    "customers": len(group),
                }
            )
    return pd.DataFrame(rows)
