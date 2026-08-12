# Metric Dictionary

| Metric | Definition | Decision use |
|---|---|---|
| Individual predicted net gain | Predicted outcome under an active action minus predicted control outcome | Rank customer-action pairs |
| DR incremental policy value | Doubly robust estimate of policy outcome minus all-control outcome | Primary policy evaluation |
| True incremental value | Expected policy value from hidden simulator truth minus all-control truth | Simulation validation only |
| 95% confidence interval | DR mean effect ± 1.96 standard errors, scaled to the test population | Show estimation uncertainty |
| Effect RMSE | Root mean squared error between predicted and hidden true individual net effects | Synthetic model diagnostic |
| Effect bias | Mean predicted individual effect minus mean hidden true effect | Detect systematic over- or underprediction |
| Rank correlation | Spearman correlation between predicted and hidden treatment effects | Test prioritization quality in simulation |
| AUUC | Area under the IPW cumulative gain curve | Evaluate uplift ranking |
| Qini | AUUC minus the random-ranking area | Measure ranking improvement over random |
| Budget used | Sum of fixed costs for assigned actions | Enforce the shared spending ceiling |
| Treatment capacity | Assigned customers for an action divided by eligible customers | Enforce operational limits |
| Regret versus oracle | Oracle true value minus selected policy true value, divided by oracle value | Measure remaining decision gap in simulation |
| Maximum absolute SMD | Largest arm-versus-control standardized mean difference across numeric pre-treatment features | Randomization health check |
| Configured propensity | Known randomization probability for an experiment arm | Verify design-based positivity and calculate IPW exposure |
| Maximum inverse-probability weight | Reciprocal of an arm's configured propensity | Flag unstable off-policy corrections before evaluation |
| Support pass | Configured propensity is at least the pre-declared 0.05 threshold | Decide whether an arm comparison is evaluable or must be excluded |
| Sensitivity policy value | DR and simulation-only incremental value after re-solving under a budget/capacity grid point | Identify binding operating constraints and fragile decisions |

All currency values are fictional synthetic units.
