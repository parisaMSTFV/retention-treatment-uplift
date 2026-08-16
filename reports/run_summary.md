# Reproducible Run Summary

The pipeline simulated 18,000 customers across 24 randomized experiment
waves and kept the final four waves untouched for testing. Model selection used only the earlier
validation period. The selected estimator was **T-learner linear**.

## Experiment health

- Maximum absolute standardized mean difference: 0.037
- Train customers: 12,091
- Validation customers: 2,966
- Test customers: 2,943
- Minimum configured treatment propensity: 0.15
- Maximum inverse-probability weight: 6.67
- Positivity support failures: 0

## Test policy result

The selected causal policy generated **2,870** synthetic
currency units of true incremental net value. Its doubly robust estimate was
**2,270**, with a 95% interval from
**392** to **4,148**.

Compared with risk-only targeting, the selected policy improved true incremental value by
**76.2%**. Its regret versus the simulation-only oracle was
**13.9%**.

These values validate the workflow on synthetic data. They are not production performance claims.

## Operating sensitivity

The same policies were evaluated across **9** combinations
of budget and channel capacity, producing
**27** matched-constraint policy evaluations. Every
reported allocation passed its budget and per-channel ceilings. The grid is a decision stress
test, not a claim that historical estimates automatically transport to a new operating regime.

## Validation model comparison

| model | validation_dr_value | validation_ci_low | validation_ci_high | mean_effect_rmse_simulation_only | selected |
| --- | --- | --- | --- | --- | --- |
| T-learner linear | 3911.476 | 2134.996 | 5687.956 | 1.673 | True |
| DR-learner hist | 2997.256 | 853.836 | 5140.676 | 2.727 | False |
| T-learner hist | 2828.555 | 716.878 | 4940.233 | 3.726 | False |
