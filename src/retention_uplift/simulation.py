"""Synthetic randomized retention experiment with heterogeneous treatment effects."""

from __future__ import annotations

import numpy as np
import pandas as pd

from retention_uplift.config import ACTIONS, ProjectConfig


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -20.0, 20.0)))


def simulate_experiment(config: ProjectConfig) -> pd.DataFrame:
    """Generate randomized observations and hidden expected potential outcomes."""
    config.validate()
    rng = np.random.default_rng(config.seed)
    n = config.n_customers

    wave = rng.integers(0, config.n_waves, size=n)
    segment = rng.choice(
        ["New", "Growing", "Loyal", "At Risk"],
        size=n,
        p=[0.18, 0.29, 0.31, 0.22],
    )
    region_group = rng.choice(["Metro", "Large City", "Other"], size=n, p=[0.44, 0.34, 0.22])
    segment_order_shift = pd.Series(segment).map(
        {"New": -0.45, "Growing": 0.15, "Loyal": 0.65, "At Risk": -0.10}
    ).to_numpy()
    segment_recency_shift = pd.Series(segment).map(
        {"New": -2.0, "Growing": -7.0, "Loyal": -10.0, "At Risk": 24.0}
    ).to_numpy()

    orders_180d = np.maximum(0, rng.poisson(np.exp(0.75 + segment_order_shift))).astype(float)
    recency_days = np.clip(rng.gamma(2.2, 14.0, n) + segment_recency_shift, 1, 150)
    margin_180d = np.clip(
        orders_180d * rng.lognormal(mean=3.35, sigma=0.38, size=n)
        + rng.lognormal(mean=2.7, sigma=0.45, size=n),
        8,
        900,
    )
    not_perfect_rate = np.clip(
        rng.beta(1.4, 8.0, n)
        + (segment == "At Risk") * rng.uniform(0.02, 0.12, n),
        0,
        0.85,
    )
    voucher_sensitivity = np.clip(
        rng.beta(2.1, 2.8, n)
        + (segment == "New") * 0.12
        + (segment == "At Risk") * 0.06,
        0,
        1,
    )
    service_issue_score = np.clip(
        0.58 * not_perfect_rate + rng.beta(1.3, 5.5, n) * 0.55,
        0,
        1,
    )
    channel_engagement = np.clip(
        rng.beta(2.5, 2.2, n)
        + (segment == "Growing") * 0.08
        - (segment == "At Risk") * 0.10,
        0,
        1,
    )
    customer_tenure_months = np.clip(
        rng.gamma(2.8, 9.0, n) + (segment == "Loyal") * 16 - (segment == "New") * 15,
        1,
        84,
    )

    base_logit = (
        -0.55
        - 0.021 * recency_days
        + 0.28 * np.log1p(orders_180d)
        + 0.0011 * margin_180d
        - 1.35 * not_perfect_rate
        - 0.55 * service_issue_score
        + 0.42 * channel_engagement
        + 0.12 * np.sin(2 * np.pi * wave / 12)
    )
    base_probability = _sigmoid(base_logit)
    churn_risk = np.clip(
        1.0 - base_probability + rng.normal(0, 0.045, n),
        0.01,
        0.99,
    )

    reminder_effect = (
        -0.012
        + 0.105 * channel_engagement
        + 0.030 * (segment == "Growing")
        - 0.045 * service_issue_score
        - 0.020 * (recency_days > 95)
    )
    voucher_effect = (
        -0.025
        + 0.205 * voucher_sensitivity
        + 0.085 * churn_risk
        - 0.045 * (segment == "Loyal")
        - 0.020 * (margin_180d > 500)
    )
    service_call_effect = (
        -0.040
        + 0.255 * service_issue_score
        + 0.075 * (margin_180d > 260)
        + 0.060 * (segment == "At Risk")
        - 0.040 * (segment == "New")
    )
    probability_effects = {
        "control": np.zeros(n),
        "reminder": reminder_effect,
        "voucher": voucher_effect,
        "service_call": service_call_effect,
    }

    margin_if_retained = np.clip(
        18.0 + 0.24 * margin_180d + 8.0 * np.log1p(orders_180d),
        22,
        245,
    )
    expected_gross: dict[str, np.ndarray] = {}
    for action in ACTIONS:
        retained_probability = np.clip(base_probability + probability_effects[action], 0.02, 0.96)
        basket_multiplier = np.ones(n)
        if action == "voucher":
            basket_multiplier = 0.97 + 0.035 * voucher_sensitivity
        elif action == "service_call":
            basket_multiplier = 1.0 + 0.025 * service_issue_score
        expected_gross[action] = retained_probability * margin_if_retained * basket_multiplier

    assigned_action = rng.choice(
        ACTIONS,
        size=n,
        p=[config.action_probabilities[action] for action in ACTIONS],
    )
    observed_expectation = np.select(
        [assigned_action == action for action in ACTIONS],
        [expected_gross[action] for action in ACTIONS],
    )
    noise_scale = 8.0 + 0.08 * margin_if_retained
    gross_contribution = np.clip(observed_expectation + rng.normal(0, noise_scale), 0, None)
    assigned_cost = pd.Series(assigned_action).map(config.action_costs).to_numpy(float)
    net_value = gross_contribution - assigned_cost

    frame = pd.DataFrame(
        {
            "customer_id": [f"SYN-{index:06d}" for index in range(n)],
            "wave": wave,
            "segment": segment,
            "region_group": region_group,
            "recency_days": recency_days,
            "orders_180d": orders_180d,
            "margin_180d": margin_180d,
            "not_perfect_rate": not_perfect_rate,
            "voucher_sensitivity": voucher_sensitivity,
            "service_issue_score": service_issue_score,
            "channel_engagement": channel_engagement,
            "churn_risk": churn_risk,
            "customer_tenure_months": customer_tenure_months,
            "assigned_action": assigned_action,
            "assignment_probability": pd.Series(assigned_action)
            .map(config.action_probabilities)
            .to_numpy(float),
            "gross_contribution_60d": gross_contribution,
            "treatment_cost": assigned_cost,
            "net_value_60d": net_value,
        }
    )
    for action in ACTIONS:
        frame[f"truth_expected_net_{action}"] = (
            expected_gross[action] - config.action_costs[action]
        )
    return frame.sort_values(["wave", "customer_id"]).reset_index(drop=True)


def temporal_split(
    frame: pd.DataFrame,
    config: ProjectConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split experiment waves chronologically."""
    train = frame.loc[frame["wave"] <= config.train_end_wave].copy()
    validation = frame.loc[
        frame["wave"].between(config.train_end_wave + 1, config.validation_end_wave)
    ].copy()
    test = frame.loc[frame["wave"] > config.validation_end_wave].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("temporal split produced an empty partition")
    return train, validation, test
