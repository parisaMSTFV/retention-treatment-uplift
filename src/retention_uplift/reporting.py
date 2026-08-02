"""Tables, charts, and narrative report outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    "reminder": "#2563EB",
    "voucher": "#D97706",
    "service_call": "#0F766E",
    "control": "#94A3B8",
}


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without an optional formatting dependency."""
    display = frame.copy()
    for column in display.select_dtypes(include=["number"]).columns:
        display[column] = display[column].map(lambda value: f"{value:.3f}")
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def plot_policy_value(policy_comparison: pd.DataFrame, path: Path) -> None:
    ordered = policy_comparison.sort_values("true_incremental_value")
    x = np.arange(len(ordered))
    width = 0.38
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar(
        x - width / 2,
        ordered["dr_incremental_value"],
        width,
        label="DR estimate",
        color="#2563EB",
    )
    axis.bar(
        x + width / 2,
        ordered["true_incremental_value"],
        width,
        label="Simulation truth",
        color="#0F766E",
    )
    axis.axhline(0, color="#334155", linewidth=0.9)
    axis.set_xticks(x, ordered["policy"], rotation=18, ha="right")
    axis.set_ylabel("Incremental net value")
    axis.set_title("Retention policy value on the untouched test period")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    _save_figure(figure, path)


def plot_calibration(calibration: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=False, sharey=False)
    for axis, (action, group) in zip(axes, calibration.groupby("action"), strict=True):
        group = group.sort_values("predicted_gain")
        axis.scatter(
            group["predicted_gain"],
            group["observed_ipw_gain"],
            color=COLORS[action],
            s=38,
        )
        low = min(group["predicted_gain"].min(), group["observed_ipw_gain"].min())
        high = max(group["predicted_gain"].max(), group["observed_ipw_gain"].max())
        axis.plot([low, high], [low, high], color="#64748B", linestyle="--", linewidth=1)
        axis.set_title(action.replace("_", " ").title())
        axis.set_xlabel("Predicted net gain")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Observed IPW net gain")
    figure.suptitle("Uplift calibration by predicted-gain decile", y=1.02)
    _save_figure(figure, path)


def plot_qini(curves: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    for action, group in curves.groupby("action"):
        axis.plot(
            group["targeted_share"],
            group["cumulative_gain_per_customer"],
            label=action.replace("_", " ").title(),
            color=COLORS[action],
            linewidth=2,
        )
    axis.axhline(0, color="#64748B", linewidth=0.8)
    axis.set_xlabel("Share of customers ranked for treatment")
    axis.set_ylabel("Cumulative IPW gain per customer")
    axis.set_title("Treatment-specific cumulative uplift curves")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    _save_figure(figure, path)


def plot_allocations(policy_comparison: pd.DataFrame, path: Path) -> None:
    ordered = policy_comparison.loc[
        policy_comparison["policy"].ne("No treatment")
    ].sort_values("true_incremental_value")
    figure, axis = plt.subplots(figsize=(9.5, 5.2))
    bottom = np.zeros(len(ordered))
    for action in ("reminder", "voucher", "service_call"):
        values = ordered[f"n_{action}"].to_numpy()
        axis.barh(
            ordered["policy"],
            values,
            left=bottom,
            label=action.replace("_", " ").title(),
            color=COLORS[action],
        )
        bottom += values
    axis.set_xlabel("Customers assigned an active treatment")
    axis.set_title("How each policy uses treatment capacity")
    axis.legend(frameon=False)
    axis.grid(axis="x", alpha=0.2)
    _save_figure(figure, path)


def write_narrative_reports(
    reports_dir: Path,
    metrics: dict[str, object],
    model_comparison: pd.DataFrame,
    policy_comparison: pd.DataFrame,
) -> None:
    selected_model = str(metrics["selected_model"])
    selected = policy_comparison.loc[
        policy_comparison["policy"].eq("Selected causal policy")
    ].iloc[0]
    risk = policy_comparison.loc[policy_comparison["policy"].eq("Risk-only")].iloc[0]
    oracle = policy_comparison.loc[policy_comparison["policy"].eq("Simulation oracle")].iloc[0]
    true_improvement = (
        (selected["true_incremental_value"] - risk["true_incremental_value"])
        / max(abs(risk["true_incremental_value"]), 1.0)
        * 100
    )
    oracle_regret = (
        (oracle["true_incremental_value"] - selected["true_incremental_value"])
        / max(abs(oracle["true_incremental_value"]), 1.0)
        * 100
    )
    model_table = _markdown_table(model_comparison)
    summary = f"""# Reproducible Run Summary

The pipeline simulated {int(metrics['customers']):,} customers across 24 randomized experiment
waves and kept the final four waves untouched for testing. Model selection used only the earlier
validation period. The selected estimator was **{selected_model}**.

## Experiment health

- Maximum absolute standardized mean difference: {float(metrics['max_abs_smd']):.3f}
- Train customers: {int(metrics['train_customers']):,}
- Validation customers: {int(metrics['validation_customers']):,}
- Test customers: {int(metrics['test_customers']):,}

## Test policy result

The selected causal policy generated **{selected['true_incremental_value']:,.0f}** synthetic
currency units of true incremental net value. Its doubly robust estimate was
**{selected['dr_incremental_value']:,.0f}**, with a 95% interval from
**{selected['ci_low']:,.0f}** to **{selected['ci_high']:,.0f}**.

Compared with risk-only targeting, the selected policy improved true incremental value by
**{true_improvement:.1f}%**. Its regret versus the simulation-only oracle was
**{oracle_regret:.1f}%**.

These values validate the workflow on synthetic data. They are not production performance claims.

## Validation model comparison

{model_table}
"""
    decision = f"""# Decision Note

## Recommendation

Use the **{selected_model}** ranking as the candidate policy for a prospective validation test,
subject to the documented budget and contact-capacity limits. Do not deploy it as a permanent
rule based only on this retrospective exercise.

## Why

- The policy was chosen on a chronological validation period and evaluated once on later waves.
- It creates {selected['true_incremental_value']:,.0f} units of simulated incremental net value,
  versus {risk['true_incremental_value']:,.0f} for risk-only targeting.
- It assigns {int(selected['n_reminder']):,} reminders, {int(selected['n_voucher']):,} vouchers,
  and {int(selected['n_service_call']):,} service calls without exceeding the shared budget.
- The estimate and simulation truth are directionally consistent, while the confidence interval
  makes remaining uncertainty visible.

## Guardrails before production

Confirm treatment eligibility, consent, operational capacity, realized costs, delayed outcomes,
and randomized holdouts. Monitor uplift calibration and policy value by experiment wave rather
than relying on response-rate lift alone.
"""
    (reports_dir / "run_summary.md").write_text(summary, encoding="utf-8")
    (reports_dir / "decision_note.md").write_text(decision, encoding="utf-8")
    (reports_dir / "run_metrics.json").write_text(
        json.dumps(
            metrics,
            indent=2,
            default=lambda value: value.item() if hasattr(value, "item") else str(value),
        ),
        encoding="utf-8",
    )


def write_reports(
    root: Path,
    metrics: dict[str, object],
    model_comparison: pd.DataFrame,
    policy_comparison: pd.DataFrame,
    effect_metrics: pd.DataFrame,
    rank_metrics: pd.DataFrame,
    rank_curves: pd.DataFrame,
    calibration: pd.DataFrame,
) -> None:
    reports_dir = root / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    model_comparison.to_csv(reports_dir / "model_comparison.csv", index=False)
    policy_comparison.to_csv(reports_dir / "policy_comparison.csv", index=False)
    effect_metrics.to_csv(reports_dir / "treatment_effect_metrics.csv", index=False)
    rank_metrics.to_csv(reports_dir / "uplift_rank_metrics.csv", index=False)
    calibration.to_csv(reports_dir / "uplift_calibration.csv", index=False)
    plot_policy_value(policy_comparison, figures_dir / "policy_value_comparison.png")
    plot_calibration(calibration, figures_dir / "uplift_calibration.png")
    plot_qini(rank_curves, figures_dir / "qini_curves.png")
    plot_allocations(policy_comparison, figures_dir / "treatment_allocation.png")
    write_narrative_reports(reports_dir, metrics, model_comparison, policy_comparison)
