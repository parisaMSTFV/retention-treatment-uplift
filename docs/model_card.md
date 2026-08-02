# Model Card

## Intended use

Demonstrate how randomized retention experiments can support heterogeneous treatment-effect
estimation and constrained customer-level action selection. The project is a portfolio case
study, not a deployable model.

## Inputs

Pre-treatment lifecycle, purchase, margin, experience, engagement, sensitivity, tenure, segment,
and region features. Assignment, outcomes, treatment cost, and hidden potential outcomes are not
model features.

## Outputs

- predicted net gain for reminder, voucher, and service call relative to control;
- one recommended action or control for each eligible customer;
- policy-level value, uncertainty, treatment mix, and constraint audit.

## Selected estimator

The committed run selects a linear T-learner by doubly robust validation policy value. Separate
regularized outcome models are fitted for the four randomized arms. The selected estimator is
refitted on train plus validation before one test evaluation.

## Evaluation

The final policy is evaluated on later untouched experiment waves. Doubly robust estimation is
the primary method. Hidden simulation truth supports effect-error and oracle-regret diagnostics
but is excluded from selection and optimization.

## Known limitations

- Synthetic covariates and treatment mechanisms simplify real retention behavior.
- Known propensities and complete assignment logs avoid common production data failures.
- Fixed treatment costs omit redemption variability and operational queues.
- The outcome horizon is 60 days and may miss longer-term customer or brand effects.
- Small treatment arms produce noisy calibration and off-policy estimates.
- No fairness or protected-group analysis is included because the generator does not create
  protected attributes.

## Production controls required

Prospective randomized validation, eligibility and consent enforcement, treatment overlap checks,
cost reconciliation, calibration and drift monitoring, fairness review, contact-frequency limits,
fallback rules, and persistent holdout measurement.
