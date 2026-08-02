"""Budget- and capacity-constrained retention policy construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, vstack

from retention_uplift.config import ACTIVE_ACTIONS, ProjectConfig


def solve_policy(
    gains: pd.DataFrame,
    config: ProjectConfig,
    budget: float | None = None,
) -> pd.Series:
    """Choose at most one positive-value action per customer under shared constraints."""
    missing = set(ACTIVE_ACTIONS) - set(gains.columns)
    if missing:
        raise ValueError(f"missing treatment gain columns: {sorted(missing)}")
    n = len(gains)
    m = len(ACTIVE_ACTIONS)
    budget = config.budget_per_customer * n if budget is None else budget
    values = gains.loc[:, ACTIVE_ACTIONS].to_numpy(float).reshape(-1)
    costs = np.tile([config.action_costs[action] for action in ACTIVE_ACTIONS], n)

    customer_rows = np.repeat(np.arange(n), m)
    customer_cols = np.arange(n * m)
    customer_matrix = coo_matrix(
        (np.ones(n * m), (customer_rows, customer_cols)),
        shape=(n, n * m),
    ).tocsr()
    budget_matrix = coo_matrix(
        (costs, (np.zeros(n * m, dtype=int), np.arange(n * m))),
        shape=(1, n * m),
    ).tocsr()
    capacity_rows = []
    for action_index in range(m):
        columns = np.arange(action_index, n * m, m)
        capacity_rows.append(
            coo_matrix(
                (np.ones(n), (np.zeros(n, dtype=int), columns)),
                shape=(1, n * m),
            ).tocsr()
        )
    constraint_matrix = vstack([customer_matrix, budget_matrix, *capacity_rows]).tocsr()
    upper = np.concatenate(
        [
            np.ones(n),
            [budget],
            [np.floor(config.capacity_shares[action] * n) for action in ACTIVE_ACTIONS],
        ]
    )
    constraints = LinearConstraint(
        constraint_matrix,
        lb=np.full(len(upper), -np.inf),
        ub=upper,
    )
    result = milp(
        c=-values,
        integrality=np.ones(n * m),
        bounds=Bounds(np.zeros(n * m), np.ones(n * m)),
        constraints=constraints,
        options={"time_limit": 30.0, "mip_rel_gap": 0.001},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"policy optimization failed: {result.message}")

    selected = result.x.reshape(n, m)
    policy = np.full(n, "control", dtype=object)
    chosen_rows, chosen_actions = np.where(selected > 0.5)
    policy[chosen_rows] = np.asarray(ACTIVE_ACTIONS, dtype=object)[chosen_actions]
    return pd.Series(policy, index=gains.index, name="recommended_action")


def segment_ate_gains(train: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """Smoothed segment-level experimental effects as an interpretable baseline."""
    control_overall = train.loc[train["assigned_action"].eq("control"), "net_value_60d"].mean()
    overall_effects = {
        action: train.loc[train["assigned_action"].eq(action), "net_value_60d"].mean()
        - control_overall
        for action in ACTIVE_ACTIONS
    }
    rows: list[dict[str, float | str]] = []
    for segment, group in train.groupby("segment"):
        control = group.loc[group["assigned_action"].eq("control"), "net_value_60d"]
        for action in ACTIVE_ACTIONS:
            treated = group.loc[group["assigned_action"].eq(action), "net_value_60d"]
            raw_effect = treated.mean() - control.mean()
            weight = min(len(treated), len(control)) / (min(len(treated), len(control)) + 100)
            rows.append(
                {
                    "segment": segment,
                    "action": action,
                    "gain": weight * raw_effect + (1 - weight) * overall_effects[action],
                }
            )
    lookup = pd.DataFrame(rows).pivot(index="segment", columns="action", values="gain")
    return pd.DataFrame(
        {
            action: target["segment"].map(lookup[action]).fillna(overall_effects[action])
            for action in ACTIVE_ACTIONS
        },
        index=target.index,
    )


def risk_only_gains(train: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """Rank by predicted loss risk and use one portfolio-average treatment."""
    control_mean = train.loc[train["assigned_action"].eq("control"), "net_value_60d"].mean()
    average_effects = {
        action: train.loc[train["assigned_action"].eq(action), "net_value_60d"].mean()
        - control_mean
        for action in ACTIVE_ACTIONS
    }
    best_action = max(average_effects, key=average_effects.get)
    priority = target["churn_risk"] * (20.0 + 0.35 * target["margin_180d"])
    gains = pd.DataFrame(0.0, index=target.index, columns=ACTIVE_ACTIONS)
    gains[best_action] = priority
    return gains


def policy_cost(policy: pd.Series, config: ProjectConfig) -> float:
    return float(policy.map(config.action_costs).sum())


def validate_policy(policy: pd.Series, config: ProjectConfig, budget: float | None = None) -> None:
    n = len(policy)
    budget = config.budget_per_customer * n if budget is None else budget
    if policy_cost(policy, config) > budget + 1e-6:
        raise ValueError("policy exceeds the treatment budget")
    for action in ACTIVE_ACTIONS:
        maximum = int(np.floor(config.capacity_shares[action] * n))
        if int(policy.eq(action).sum()) > maximum:
            raise ValueError(f"policy exceeds {action} capacity")
