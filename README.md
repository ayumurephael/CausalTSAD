# CausalTSAD

Known-normal-DAG, zero-shot time-series anomaly detection and root cause analysis.

## Branches

`main` contains shared contracts, data preparation and research protocols. Baselines live on `baseline/<name>-vN`; proposals live on `proposal/cXX-<name>-vN`. Each starts from a recorded shared-base commit. Failed implementations, runs and results are retained. A new proposal never replaces another proposal's implementation. Combined proposals require an explicitly named integration branch and references to their component commits.

Only source code, configurations, compact aggregate results and provenance are published. Datasets, checkpoints, per-point arrays, credentials and private machine administration are excluded. Remote runs use immutable copies of a committed snapshot and record its local and GitHub commit hashes.

## Information contract

Main model inputs: observed values, legal-point mask, normal directed topology. True normal lags are a separately labelled enhanced comparison. Fault location/type, normal twins, coefficients, noise and equations are excluded from target inputs. Source normal twins and actual injection metadata may supervise training. Undefined effect/anomaly_channel/anomaly_time labels remain null. Target arrays never participate in gradient updates; no target checkpoint/hyperparameter selection.

The 250M synthetic benchmark is split by underlying system: 50M development, 50M validation, 150M locked test. The locked test is not used during proposal development. Real-data transfer is a separate question because a real system may not provide a true DAG.

See [branch policy](docs/BRANCH_POLICY.md) and [pilot registry](experiments/registry.json). No performance improvement or SOTA is claimed before matched evaluation.
