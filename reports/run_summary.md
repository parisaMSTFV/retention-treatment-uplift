# Reproducible Run Summary

The pipeline simulated 18,000 customers across 24 randomized experiment
waves and kept the final four waves untouched for testing. Model selection used only the earlier
validation period. The selected estimator was **T-learner linear**.

## Experiment health

- Maximum absolute standardized mean difference: 0.037
- Train customers: 12,091
- Validation customers: 2,966
- Test customers: 2,943

## Test policy result

The selected causal policy generated **2,870** synthetic
currency units of true incremental net value. Its doubly robust estimate was
**2,291**, with a 95% interval from
**405** to **4,177**.

Compared with risk-only targeting, the selected policy improved true incremental value by
**76.2%**. Its regret versus the simulation-only oracle was
**13.9%**.

These values validate the workflow on synthetic data. They are not production performance claims.

## Validation model comparison

| model | validation_dr_value | validation_ci_low | validation_ci_high | mean_effect_rmse_simulation_only | selected |
| --- | --- | --- | --- | --- | --- |
| T-learner linear | 3888.662 | 2135.370 | 5641.953 | 1.673 | True |
| DR-learner hist | 2569.410 | 406.313 | 4732.507 | 2.803 | False |
| T-learner hist | 1862.299 | -213.681 | 3938.279 | 3.724 | False |
