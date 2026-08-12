"""End-to-end synthetic experiment, uplift modeling, and policy evaluation pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from retention_uplift.config import ACTIVE_ACTIONS, ProjectConfig
from retention_uplift.evaluation import (
    effect_accuracy,
    experiment_health,
    overlap_diagnostics,
    policy_effect,
    uplift_calibration,
    uplift_rank_metrics,
)
from retention_uplift.models import DRLearner, TLearner, UpliftModel, candidate_models
from retention_uplift.policy import (
    greedy_uplift_policy,
    risk_only_gains,
    segment_ate_gains,
    solve_policy,
    validate_policy,
)
from retention_uplift.reporting import write_reports
from retention_uplift.simulation import simulate_experiment, temporal_split


def _new_model(name: str, config: ProjectConfig) -> UpliftModel:
    if name == "T-learner linear":
        return TLearner("linear", config.seed)
    if name == "T-learner hist":
        return TLearner("hist", config.seed)
    if name == "DR-learner hist":
        return DRLearner(config, config.seed)
    raise ValueError(f"unsupported selected model: {name}")


def _truth_gains(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            action: frame[f"truth_expected_net_{action}"]
            - frame["truth_expected_net_control"]
            for action in ACTIVE_ACTIONS
        },
        index=frame.index,
    )


def _safe_sample(frame: pd.DataFrame, policy: pd.Series, gains: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "customer_id",
        "wave",
        "segment",
        "region_group",
        "recency_days",
        "orders_180d",
        "margin_180d",
        "churn_risk",
    ]
    sample = frame.loc[:, columns].copy()
    sample["recommended_action"] = policy
    for action in ACTIVE_ACTIONS:
        sample[f"predicted_net_gain_{action}"] = gains[action]
    return sample.sort_values("customer_id").head(30)


def _policy_sensitivity(
    development: pd.DataFrame,
    test: pd.DataFrame,
    gains: pd.DataFrame,
    outcome_predictions: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    """Evaluate decision policies on a compact budget-by-capacity operating grid."""
    risk_gains = risk_only_gains(development, test)
    rows: list[dict[str, object]] = []
    for budget_multiplier in (0.6, 1.0, 1.4):
        for capacity_multiplier in (0.7, 1.0, 1.3):
            budget = config.budget_per_customer * len(test) * budget_multiplier
            policies = {
                "Causal optimizer": solve_policy(
                    gains,
                    config,
                    budget=budget,
                    capacity_multiplier=capacity_multiplier,
                ),
                "Risk-only": solve_policy(
                    risk_gains,
                    config,
                    budget=budget,
                    capacity_multiplier=capacity_multiplier,
                ),
                "Greedy uplift rank": greedy_uplift_policy(
                    gains,
                    config,
                    budget=budget,
                    capacity_multiplier=capacity_multiplier,
                ),
            }
            for policy_name, policy in policies.items():
                validate_policy(
                    policy,
                    config,
                    budget=budget,
                    capacity_multiplier=capacity_multiplier,
                )
                result, _ = policy_effect(
                    test,
                    policy,
                    outcome_predictions,
                    config,
                    policy_name,
                )
                rows.append(
                    {
                        "budget_multiplier": budget_multiplier,
                        "capacity_multiplier": capacity_multiplier,
                        "budget_ceiling": budget,
                        "constraints_pass": True,
                        **result,
                    }
                )
    return pd.DataFrame(rows)


def run_pipeline(root: Path, config: ProjectConfig | None = None) -> dict[str, object]:
    config = ProjectConfig() if config is None else config
    config.validate()
    root = Path(root)
    (root / "data" / "generated").mkdir(parents=True, exist_ok=True)
    (root / "data" / "sample").mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(parents=True, exist_ok=True)

    experiment = simulate_experiment(config)
    train, validation, test = temporal_split(experiment, config)
    health = experiment_health(experiment, config)
    overlap = pd.concat(
        [
            overlap_diagnostics(train, config, "train"),
            overlap_diagnostics(validation, config, "validation"),
            overlap_diagnostics(test, config, "test"),
        ],
        ignore_index=True,
    )
    if not overlap["support_pass"].all():
        failed = sorted(overlap.loc[~overlap["support_pass"], "action"].unique())
        raise ValueError(f"overlap support rule failed for randomized arms: {failed}")
    experiment.to_csv(
        root / "data" / "generated" / "synthetic_experiment.csv",
        index=False,
        float_format="%.12g",
    )

    evaluator = TLearner("hist", config.seed + 900).fit(train)
    evaluator_outcomes = evaluator.predict_outcomes(validation)
    comparison_rows = []
    for model in candidate_models(config):
        model.fit(train)
        gains = model.predict_gains(validation)
        policy = solve_policy(gains, config)
        validate_policy(policy, config)
        result, _ = policy_effect(
            validation,
            policy,
            evaluator_outcomes,
            config,
            model.name,
        )
        accuracy = effect_accuracy(validation, gains, model.name)
        comparison_rows.append(
            {
                "model": model.name,
                "validation_dr_value": result["dr_incremental_value"],
                "validation_ci_low": result["ci_low"],
                "validation_ci_high": result["ci_high"],
                "mean_effect_rmse_simulation_only": accuracy["effect_rmse"].mean(),
            }
        )
    model_comparison = pd.DataFrame(comparison_rows).sort_values(
        "validation_dr_value", ascending=False
    )
    selected_name = str(model_comparison.iloc[0]["model"])
    model_comparison["selected"] = model_comparison["model"].eq(selected_name)

    development = pd.concat([train, validation], ignore_index=True)
    selected_model = _new_model(selected_name, config).fit(development)
    test_gains = selected_model.predict_gains(test)
    final_evaluator = TLearner("hist", config.seed + 950).fit(development)
    test_outcomes = final_evaluator.predict_outcomes(test)

    policies = {
        "No treatment": pd.Series("control", index=test.index, name="recommended_action"),
        "Risk-only": solve_policy(risk_only_gains(development, test), config),
        "Segment ATE": solve_policy(segment_ate_gains(development, test), config),
        "Selected causal policy": solve_policy(test_gains, config),
        "Simulation oracle": solve_policy(_truth_gains(test), config),
    }
    policy_rows = []
    for policy_name, policy in policies.items():
        validate_policy(policy, config)
        result, _ = policy_effect(test, policy, test_outcomes, config, policy_name)
        policy_rows.append(result)
    policy_comparison = pd.DataFrame(policy_rows)

    effect_metrics = effect_accuracy(test, test_gains, selected_name)
    rank_metrics, rank_curves = uplift_rank_metrics(
        test, test_gains, selected_name, config
    )
    calibration = uplift_calibration(test, test_gains, config)
    sensitivity = _policy_sensitivity(
        development,
        test,
        test_gains,
        test_outcomes,
        config,
    )
    sample = _safe_sample(test, policies["Selected causal policy"], test_gains)
    sample.to_csv(
        root / "data" / "sample" / "synthetic_policy_sample.csv",
        index=False,
        float_format="%.12g",
    )
    sample.to_csv(
        root / "reports" / "policy_assignments_sample.csv",
        index=False,
        float_format="%.12g",
    )

    selected_result = policy_comparison.loc[
        policy_comparison["policy"].eq("Selected causal policy")
    ].iloc[0]
    risk_result = policy_comparison.loc[policy_comparison["policy"].eq("Risk-only")].iloc[0]
    oracle_result = policy_comparison.loc[
        policy_comparison["policy"].eq("Simulation oracle")
    ].iloc[0]
    base_sensitivity = sensitivity.loc[
        sensitivity["budget_multiplier"].eq(1.0)
        & sensitivity["capacity_multiplier"].eq(1.0)
        & sensitivity["policy"].eq("Causal optimizer")
    ].iloc[0]
    metrics: dict[str, object] = {
        "seed": config.seed,
        "customers": len(experiment),
        "train_customers": len(train),
        "validation_customers": len(validation),
        "test_customers": len(test),
        "selected_model": selected_name,
        "max_abs_smd": health["max_abs_smd"],
        "assignment_chi_square": health["assignment_chi_square"],
        "selected_dr_incremental_value": selected_result["dr_incremental_value"],
        "selected_true_incremental_value": selected_result["true_incremental_value"],
        "risk_true_incremental_value": risk_result["true_incremental_value"],
        "oracle_true_incremental_value": oracle_result["true_incremental_value"],
        "selected_ci_low": selected_result["ci_low"],
        "selected_ci_high": selected_result["ci_high"],
        "selected_treatment_cost": selected_result["treatment_cost"],
        "selected_treated_customers": selected_result["treated_customers"],
        "mean_effect_rmse": effect_metrics["effect_rmse"].mean(),
        "mean_qini": rank_metrics["qini"].mean(),
        "minimum_configured_propensity": overlap["configured_propensity"].min(),
        "maximum_inverse_probability_weight": overlap[
            "max_inverse_probability_weight"
        ].max(),
        "overlap_support_failures": int((~overlap["support_pass"]).sum()),
        "sensitivity_scenarios": int(
            sensitivity[["budget_multiplier", "capacity_multiplier"]]
            .drop_duplicates()
            .shape[0]
        ),
        "sensitivity_policy_evaluations": len(sensitivity),
        "base_sensitivity_true_incremental_value": base_sensitivity[
            "true_incremental_value"
        ],
        **{key: value for key, value in health.items() if key.startswith("n_")},
    }
    write_reports(
        root,
        metrics,
        model_comparison,
        policy_comparison,
        effect_metrics,
        rank_metrics,
        rank_curves,
        calibration,
        overlap,
        sensitivity,
    )
    return metrics
