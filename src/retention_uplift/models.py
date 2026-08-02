"""Causal uplift estimators built on randomized experiment data."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from retention_uplift.config import ACTIONS, ACTIVE_ACTIONS, ProjectConfig
from retention_uplift.features import design_matrix


class UpliftModel(Protocol):
    name: str

    def fit(self, frame: pd.DataFrame) -> UpliftModel: ...

    def predict_gains(self, frame: pd.DataFrame) -> pd.DataFrame: ...

    def predict_outcomes(self, frame: pd.DataFrame) -> pd.DataFrame: ...


def _regressor(family: str, seed: int):
    if family == "linear":
        return make_pipeline(StandardScaler(), Ridge(alpha=8.0))
    if family == "hist":
        return HistGradientBoostingRegressor(
            learning_rate=0.065,
            max_iter=90,
            max_leaf_nodes=15,
            min_samples_leaf=55,
            l2_regularization=1.5,
            random_state=seed,
        )
    raise ValueError(f"unknown model family: {family}")


class TLearner:
    """Separate outcome model for each randomized arm."""

    def __init__(self, family: str, seed: int = 42) -> None:
        self.family = family
        self.seed = seed
        self.name = f"T-learner {family}"
        self.columns: list[str] | None = None
        self.models: dict[str, object] = {}

    def fit(self, frame: pd.DataFrame) -> TLearner:
        matrix, self.columns = design_matrix(frame)
        for index, action in enumerate(ACTIONS):
            mask = frame["assigned_action"].eq(action).to_numpy()
            if mask.sum() < 50:
                raise ValueError(f"too few observations in randomized arm: {action}")
            model = _regressor(self.family, self.seed + index)
            model.fit(matrix.loc[mask], frame.loc[mask, "net_value_60d"])
            self.models[action] = model
        return self

    def predict_outcomes(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.columns is None or not self.models:
            raise RuntimeError("model must be fitted before prediction")
        matrix, _ = design_matrix(frame, self.columns)
        return pd.DataFrame(
            {action: self.models[action].predict(matrix) for action in ACTIONS},
            index=frame.index,
        )

    def predict_gains(self, frame: pd.DataFrame) -> pd.DataFrame:
        outcomes = self.predict_outcomes(frame)
        return pd.DataFrame(
            {action: outcomes[action] - outcomes["control"] for action in ACTIVE_ACTIONS},
            index=frame.index,
        )


class DRLearner:
    """Cross-fitted doubly robust pseudo-outcome learner for each active action."""

    def __init__(self, config: ProjectConfig, seed: int = 42, folds: int = 3) -> None:
        self.config = config
        self.seed = seed
        self.folds = folds
        self.name = "DR-learner hist"
        self.columns: list[str] | None = None
        self.effect_models: dict[str, object] = {}
        self.outcome_model: TLearner | None = None

    def fit(self, frame: pd.DataFrame) -> DRLearner:
        matrix, self.columns = design_matrix(frame)
        cross_fit_outcomes = pd.DataFrame(index=frame.index, columns=ACTIONS, dtype=float)
        splitter = KFold(n_splits=self.folds, shuffle=True, random_state=self.seed)
        positions = np.arange(len(frame))
        for fold, (train_pos, holdout_pos) in enumerate(splitter.split(positions)):
            fold_model = TLearner("hist", self.seed + 10 * (fold + 1)).fit(
                frame.iloc[train_pos]
            )
            predictions = fold_model.predict_outcomes(frame.iloc[holdout_pos])
            cross_fit_outcomes.iloc[holdout_pos] = predictions.to_numpy()

        observed_action = frame["assigned_action"].to_numpy()
        observed_value = frame["net_value_60d"].to_numpy(float)
        mu_control = cross_fit_outcomes["control"].to_numpy(float)
        for index, action in enumerate(ACTIVE_ACTIONS):
            mu_action = cross_fit_outcomes[action].to_numpy(float)
            pseudo_effect = (
                mu_action
                - mu_control
                + (observed_action == action)
                / self.config.action_probabilities[action]
                * (observed_value - mu_action)
                - (observed_action == "control")
                / self.config.action_probabilities["control"]
                * (observed_value - mu_control)
            )
            model = _regressor("hist", self.seed + 100 + index)
            model.fit(matrix, pseudo_effect)
            self.effect_models[action] = model

        self.outcome_model = TLearner("hist", self.seed + 500).fit(frame)
        return self

    def predict_gains(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.columns is None or not self.effect_models:
            raise RuntimeError("model must be fitted before prediction")
        matrix, _ = design_matrix(frame, self.columns)
        return pd.DataFrame(
            {action: self.effect_models[action].predict(matrix) for action in ACTIVE_ACTIONS},
            index=frame.index,
        )

    def predict_outcomes(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.outcome_model is None:
            raise RuntimeError("model must be fitted before prediction")
        return self.outcome_model.predict_outcomes(frame)


def candidate_models(config: ProjectConfig) -> list[UpliftModel]:
    return [
        TLearner("linear", config.seed),
        TLearner("hist", config.seed),
        DRLearner(config, config.seed),
    ]
