# Interview Guide

## Thirty-second explanation

This project closes the gap between churn prediction and retention action. A churn model ranks
who may leave, while an uplift model estimates who will respond because of a specific treatment.
I used synthetic randomized experiment data, compared three heterogeneous-effect estimators, and
optimized one treatment per customer under budget and channel-capacity constraints. On a later
untouched period, the selected policy created 76.2% more simulated incremental net value than
risk-only targeting.

## Why use randomized data?

Customers who receive retention offers in historical operations are usually different before
treatment. Random assignment makes treatment and control comparable in expectation and gives the
effect models a credible causal target. Uplift modeling does not repair an invalid experiment.

## Why subtract cost before optimization?

A treatment can increase retention and still destroy value. Voucher and service costs must be
charged to the action before ranking. The optimizer therefore maximizes incremental net value,
not response probability or gross lift.

## Why did the linear T-learner win?

Selection was based on validation policy value, not model complexity or hidden simulator truth.
The synthetic effects are relatively smooth, and the linear model generalized better than the
nonlinear candidates. This is evidence for using the simplest model that wins the decision
metric, not a claim that linear uplift is always best.

## What does Doubly Robust mean here?

The policy evaluator combines an outcome model with inverse-probability corrections from the
randomized assignment. The estimate remains consistent when either the outcome model or the
propensity model is correct under the standard assumptions. In this simulation, assignment
probabilities are known.

## Why can the DR estimate differ from simulation truth?

The estimate uses only observed randomized outcomes and therefore has sampling noise, especially
for smaller treatment arms. The simulator can calculate every customer's expected potential
outcome, which is unavailable in real data. The confidence interval communicates that uncertainty.

## What prevents leakage?

Features are explicitly selected from pre-treatment columns. Train, validation, and test are
separated by experiment wave. A test mutates observed outcomes and hidden potential outcomes and
confirms that the design matrix remains identical.

## How would this move to production?

Start with clean exposure and eligibility logs, reconcile realized action cost, define delayed
outcome windows, and run a prospective policy-versus-business-as-usual experiment. Keep a control
group, monitor calibration and value by wave, enforce contact limits, and retrain only after enough
new randomized evidence accumulates.

## What would you improve next?

Add cross-treatment interference checks, variable voucher redemption cost, uncertainty-aware
policy optimization, longer-term value, contact-fatigue state, protected-group fairness review,
and a production monitoring contract.
