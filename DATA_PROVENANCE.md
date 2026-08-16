# Data Provenance

The core multi-action retention case study is synthetic and generated from scratch by
`src/retention_uplift/simulation.py`. A separate, explicitly bounded external validation uses the
public CRITEO-UPLIFTv2.1 randomized advertising benchmark. No employer data is used.

## What is simulated

- anonymous customer IDs prefixed with `SYN-`;
- 24 experiment waves;
- behavioral, value, experience, channel, and lifecycle features;
- randomized assignment to control, reminder, voucher, or service call;
- 60-day gross contribution and treatment cost;
- hidden expected potential outcomes used only for simulation diagnostics.

Feature distributions and treatment-effect equations are fictional. Treatment costs, allocation
probabilities, budget limits, channel capacities, dates, labels, and all reported results are
case-study assumptions.

## What is committed

The full row-level experiment is written to `data/generated/`, which is excluded from Git. The
repository includes only aggregate reports and a 30-row synthetic policy sample without hidden
potential outcomes.

The external-validation directory contains aggregate metrics, a chart, a narrative boundary, and
machine-readable provenance generated from the checksum-verified public benchmark. It contains no
raw or sampled Criteo rows.

## External randomized benchmark

Criteo AI Lab describes CRITEO-UPLIFTv2.1 as data assembled from advertising incrementality tests
where a randomized part of the population was prevented from targeting. The released unbiased
version contains 13,979,592 rows with 12 anonymized features, assignment, visit, conversion, and
exposure indicators.

- Official page: <https://ailab.criteo.com/criteo-uplift-prediction-dataset/>
- Official verified organization mirror:
  <https://huggingface.co/datasets/criteo/criteo-uplift>
- Pinned revision: `2424920019e49d52d72c13ac1143ec5d53af276b`
- Size: `311422618` bytes
- SHA-256: `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc`
- License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

The license permits noncommercial use and requires attribution and ShareAlike for shared licensed
or adapted material. The raw download is excluded by `.gitignore`. Aggregate external reports
carry a [separate data notice](reports/external_validation/LICENSE.md); original repository code
remains MIT-licensed.

The external analysis uses randomized assignment (`treatment`) and pre-assignment anonymized
features. It excludes `exposure` from model inputs because exposure occurs after assignment. It
reports assignment ITT within the released, non-uniformly subsampled benchmark, not the original
advertisers' effect and not a retention-treatment result. Full assumptions and reproduction steps
are in [`docs/criteo_external_validation.md`](docs/criteo_external_validation.md).

## What is excluded

No employer customer, order, campaign, employee, warehouse, database, server, dashboard,
credential, or internal business rule is used. The Criteo source is anonymized public benchmark
data under its own license. The public-file scanner checks common secret, connection,
private-network, and internal-domain markers during CI.
