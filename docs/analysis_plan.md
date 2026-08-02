# Analysis Plan

## Decision

Choose zero or one retention treatment for each eligible customer in a later experiment wave.
The policy must maximize expected incremental 60-day net value while respecting a shared budget
and treatment-specific capacity ceilings.

## Population and assignment

The synthetic experiment contains 18,000 customers assigned independently to control, reminder,
voucher, or service call with known probabilities of 0.40, 0.25, 0.20, and 0.15. Features are
measured before assignment. The observed outcome is 60-day contribution after fixed action cost.

## Temporal split

- Train: waves 0–15
- Validation: waves 16–19
- Test: waves 20–23

The test period remains untouched until the estimator and policy logic have been selected.

## Candidate estimators

1. Linear T-learner
2. Histogram-gradient-boosting T-learner
3. Histogram-gradient-boosting DR-learner with three-fold cross-fitted pseudo-outcomes

The primary model-selection measure is doubly robust incremental policy value on validation.
The simulation-only effect RMSE is reported but cannot determine the selected model.

## Policies compared

- No treatment
- Risk-only targeting using one portfolio-average action
- Smoothed segment-level average treatment effects
- Selected customer-level causal policy
- Simulation oracle based on hidden expected potential outcomes

Every active policy is solved under the same budget and channel capacities.

## Primary test metric

Incremental 60-day net value of the selected policy versus assigning control to everyone,
estimated with a doubly robust off-policy estimator and a normal-approximation 95% interval.

## Secondary metrics

- true expected policy value and regret in simulation only;
- treatment-effect RMSE, bias, and Spearman rank correlation;
- AUUC and Qini for each active treatment;
- uplift calibration by predicted-gain decile;
- budget use, treatment counts, and constraint violations;
- experiment arm counts and maximum absolute standardized mean difference.

## Decision rule

Recommend prospective validation only when the selected policy has positive validation value,
respects every operational constraint, and remains directionally positive on the untouched test.
The policy is not considered production-ready from this synthetic retrospective run.
