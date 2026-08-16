# External Validation — CRITEO-UPLIFTv2.1

## Evidence boundary

This run validates a **binary advertising-assignment uplift workflow** on the released Criteo
benchmark. It does not validate retention treatments, multi-action choice, customer value,
treatment costs, or transportability to another population.

The source is a collection of advertising incrementality tests in which a randomized holdout was
prevented from targeting. The estimand here is the intention-to-treat effect of randomized
assignment on a two-week visit outcome in the **released, non-uniformly subsampled benchmark**.
It is not the original advertisers' campaign effect. `exposure` is post-assignment and was excluded
from every model feature set.

## Source verification

- Dataset: CRITEO-UPLIFTv2.1
- Validated rows: 13,979,592
- Validated bytes: 311,422,618
- SHA-256: `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc`
- License: [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) — noncommercial use, attribution, and
  ShareAlike apply to the dataset and adapted material
- Modeling sample: 300,000 rows selected by `splitmix64-bottom-k-v1` with seed
  `42` after scanning and validating the complete file

## Full released-benchmark contrasts

| Outcome | Control rate | Assigned-treatment rate | Difference | 95% CI |
|---|---:|---:|---:|---:|
| Visit | 3.8201% | 4.8543% | 1.0342% | 1.0056% to 1.0629% |
| Conversion | 0.1938% | 0.3089% | 0.1152% | 0.1085% to 0.1219% |

These are unadjusted randomized-arm contrasts within the released benchmark. The narrow intervals
reflect its very large row count, not guaranteed external validity.

## Frozen test policy

The model was fitted on the training partition, selected and thresholded only on validation, then
evaluated once on the independent deterministic test partition. The selected model was
**T-learner logistic**. Its frozen threshold targeted
**30.3%** of test rows.

- AIPW visit effect among targeted rows: **1.4034%**
  (95% CI 0.2785% to
  2.5283%)
- AIPW policy effect per eligible test row: **0.4251%**
  (95% CI 0.0843% to
  0.7658%)
- Incremental effect over random targeting at the same realized reach:
  **0.2968%**
  (95% CI 0.0536% to
  0.5400%)
- Treatment-prediction AUC on test: **0.5005**;
  maximum absolute feature SMD: **0.0419**

The AIPW calculation uses the pooled training assignment share because experiment/advertiser
strata and their propensities are not present in the public schema. That is an explicit limitation,
not evidence that every source experiment used an identical probability.

## Validation-only model selection

| model | validation_targeted_share | validation_incremental_over_random_same_reach | validation_incremental_ci_low | validation_incremental_ci_high | selected |
| --- | --- | --- | --- | --- | --- |
| T-learner logistic | 0.300005 | 0.003289 | 0.000855 | 0.005724 | True |
| T-learner hist | 0.300909 | 0.003225 | 0.000908 | 0.005542 | False |

Validation intervals are descriptive diagnostics after model comparison; confirmatory uncertainty
is reported only on the untouched test partition.

## What this evidence does and does not support

The run supports that this repository can ingest a real randomized uplift benchmark, preserve a
clean train/validation/test boundary, rank a binary advertising treatment, and report sampling
uncertainty. It does not establish realized retention profit, optimal contact policy, temporal
stability, or individual causal truth. The file has no time key, experiment stratum, treatment
cost, or retention outcome, so those claims are deliberately omitted.
