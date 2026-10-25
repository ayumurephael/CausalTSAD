# C09: conditional residual tails for known-DAG zero-shot TSAD

Independent branch `proposal/c09-conditional-tail-v1`, parallel to C08. Frozen C02 causal normal predictor + shared source-trained tail distribution + source-trained residual correction to the frozen self-trained TimeRCD raw logit. No target optimizer, true coefficients/noise/lags or root labels as inputs.

[Mechanism,4arms x3seeds, fixed budgets and decision rules](docs/C09_CONDITIONAL_TAIL.md). [Configuration](experiments/c09_tail_v1.json). [Dependency lineage](experiments/dependencies_c09.json).

Three normal-tail passes then three fusion passes, all fixed before results. Gaussian/marginal-quantile/conditional-quantile/base-only-calibration controls. Same nominal network slots, disclosed effective-capacity differences. Source coverage/pinball/context-swap, full100development original4metrics/AP/FPR and conditional ready-region RCA. C1 and first64points exactly original multi detector. Original raw-logit baseline is preserved; validation/locked data closed.

This is a candidate solution under evaluation, not a proven improvement, new conformal guarantee, complete foundation pretraining or claimed SOTA. Data/checkpoints/private administration/SSH credentials are excluded from Git. GPU G0 must pass before the bounded `tools/controller_c09.py` pipeline can optimize.
