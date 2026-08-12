# Decision Note

## Recommendation

Use the **T-learner linear** ranking as the candidate policy for a prospective validation test,
subject to the documented budget and contact-capacity limits. Do not deploy it as a permanent
rule based only on this retrospective exercise.

## Why

- The policy was chosen on a chronological validation period and evaluated once on later waves.
- It creates 2,870 units of simulated incremental net value,
  versus 1,629 for risk-only targeting.
- It assigns 647 reminders, 56 vouchers,
  and 24 service calls without exceeding the shared budget.
- The estimate and simulation truth are directionally consistent, while the confidence interval
  makes remaining uncertainty visible.
- All randomized arms pass the pre-declared 5% propensity support rule; no observations are
  trimmed in this synthetic experiment.
- Budget and channel-capacity stress tests compare the optimizer with risk-only and greedy
  uplift-ranked policies under the same constraints.

## Guardrails before production

Confirm treatment eligibility, consent, operational capacity, realized costs, delayed outcomes,
and randomized holdouts. Monitor uplift calibration and policy value by experiment wave rather
than relying on response-rate lift alone.
