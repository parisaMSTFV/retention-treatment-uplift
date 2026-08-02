"""Point-in-time feature preparation."""

from __future__ import annotations

import pandas as pd

NUMERIC_FEATURES = (
    "recency_days",
    "orders_180d",
    "margin_180d",
    "not_perfect_rate",
    "voucher_sensitivity",
    "service_issue_score",
    "channel_engagement",
    "churn_risk",
    "customer_tenure_months",
)
CATEGORICAL_FEATURES = ("segment", "region_group")


def design_matrix(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Create a stable numeric design matrix without outcome or assignment columns."""
    missing = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"missing model features: {sorted(missing)}")
    matrix = pd.get_dummies(
        frame.loc[:, NUMERIC_FEATURES + CATEGORICAL_FEATURES],
        columns=list(CATEGORICAL_FEATURES),
        dtype=float,
    )
    matrix = matrix.astype(float)
    if columns is None:
        columns = sorted(matrix.columns)
    return matrix.reindex(columns=columns, fill_value=0.0), list(columns)
