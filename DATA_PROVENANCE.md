# Data Provenance

All data in this repository is synthetic and generated from scratch by
`src/retention_uplift/simulation.py`.

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

## What is excluded

No real customer, order, campaign, employee, company, warehouse, database, server, dashboard,
credential, or internal business rule is used. The public-file scanner checks common secret,
connection, private-network, and internal-domain markers during CI.
