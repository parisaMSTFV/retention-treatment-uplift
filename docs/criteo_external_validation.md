# CRITEO-UPLIFTv2.1 External Validation Contract

This path validates the repository's binary uplift mechanics against a public randomized
advertising benchmark. It is deliberately separate from the synthetic multi-action retention
decision pipeline.

## Primary-source identity and license

- Publisher: [Criteo AI Lab](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)
- Official verified organization mirror:
  [criteo/criteo-uplift](https://huggingface.co/datasets/criteo/criteo-uplift)
- Pinned mirror revision: `2424920019e49d52d72c13ac1143ec5d53af276b`
- File: `criteo-research-uplift-v2.1.csv.gz`
- Bytes: `311422618`
- Rows: `13979592`
- SHA-256: `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc`
- License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- Design paper:
  [A Large Scale Benchmark for Uplift Modeling](https://ama.liglab.fr/~amini/Publis/large-scale-benchmark.pdf)

The license permits use and adaptation only for noncommercial purposes and requires attribution
and ShareAlike when licensed material or adapted material is shared. The raw file and all sampled
rows remain uncommitted. The aggregate external reports include attribution and a separate license
notice; the repository's original code remains MIT-licensed.

The publisher's legacy `go.criteo.net` download returned HTTP 404 during the 2026-08-14
verification. The verified Criteo organization mirror returned the pinned file above, whose size
and SHA-256 matched its published file metadata. The downloader therefore uses the pinned mirror.

## Accepted schema

The adapter requires exactly these columns:

| Columns | Contract | Analysis role |
|---|---|---|
| `f0` … `f11` | Finite numeric values | Pre-assignment anonymized model features |
| `treatment` | Binary `0/1` | Randomized assignment indicator |
| `visit` | Binary `0/1` | Primary two-week outcome |
| `conversion` | Binary `0/1`, implies visit | Secondary aggregate outcome |
| `exposure` | Binary `0/1`, cannot occur in control | Post-assignment audit field; never a feature |

The loader verifies the full compressed-file checksum and byte size before parsing, then validates
every row and the official row count. It stops on missing/extra columns, non-finite features,
non-binary indicators, `conversion > visit`, `exposure > treatment`, or identity mismatch.

## Sampling and split

The whole 13.98-million-row file is scanned. Modeling uses the exact 300,000 rows with the lowest
SplitMix64 keys derived from source-row position and seed 42 (`splitmix64-bottom-k-v1`). A second
independent deterministic key assigns approximately 60%/20%/20% to train, validation, and test.
The file has no public time key or experiment identifier, so this is an independent random split,
not a temporal or cross-experiment transport test.

## Estimand and uncertainty

The primary estimand is the intention-to-treat effect of randomized advertising assignment on a
two-week visit outcome **within the released, non-uniformly subsampled benchmark**. It is not the
effect in the original advertiser populations.

Two T-learners are fitted on training only. Model family and a 30% targeting threshold are selected
on validation. The frozen model and threshold are evaluated once on test. Reports include:

- full-file treatment/control differences in visit and conversion rates with normal 95% intervals;
- test AIPW policy effect and effect among targeted rows with influence-function intervals;
- the difference versus random targeting at the same realized reach with a paired interval;
- test uplift deciles with randomized-arm contrasts and intervals;
- treatment-prediction AUC and maximum absolute feature SMD as randomization diagnostics.

The AIPW estimator uses the pooled training assignment share. Experiment/advertiser strata and
their assignment probabilities are absent from the public schema, so equal propensity across all
source tests is not claimed.

## Reproduce

Review the license before explicitly accepting it:

```bash
python scripts/download_criteo_uplift.py \
  --accept-license CC-BY-NC-SA-4.0

retention-uplift \
  --external-criteo data/external/criteo-research-uplift-v2.1.csv.gz \
  --external-sample-size 300000 \
  --external-target-share 0.30 \
  --seed 42 \
  --project-root .
```

See the committed [external run summary](../reports/external_validation/run_summary.md) and
[machine-readable provenance](../reports/external_validation/source_provenance.json).
