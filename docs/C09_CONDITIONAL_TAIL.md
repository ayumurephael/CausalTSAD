# C09: source-trained causal conditional tails, independent of C08

This branch starts common main `5510edec73d4309e64110e63c86ed7502d233409`. C08 runs on H100; C09 runs concurrently on A100 and does not read or select C08 weights/results. Earlier proposals remain on their own branches.

## Question and mature mechanism

A residual of the same size may be routine under high uncertainty and exceptional under low uncertainty. Test whether better conditional residual tails improve TSAD, rather than assuming Gaussian calibration or declaring any better NLL a detector improvement. This develops Research Card C09, drawing on the already fully read R2 conditional-density comparisons and R3 tail-probability interpretation. Quantile/pinball regression, monotone quantiles and exponential tail extrapolation are mature tools; no novelty claim is made merely for using them. R3's causal identifiability/Shapley claims are not automatically inherited.

Use the same frozen C02 `parents_current_s348` Gaussian normal predictor in every arm. It sees own64-point past and normal-DAG direct parents, including explicitly authorized same-time parents, with per-channel block-start past normalization. It never receives own current query, future, true lag/coefficient/noise. Predictions are time-chunked for memory while retaining exact history/phase indices. Original TimeRCD raw logits are reused from a SHA-verified6MB reference bridge, zero new encoder work.

New shared tail head:12 legal features ->64 ->64 ->9 outputs. Context contains mean/logvariance of the frozen predictor, own history summaries, phase/coverage, log parent degree and log C. Context is clipped+-50; queried residual values are not clipped. This is legitimate contextual prediction, not fitting a model to the target sequence. Source normal twins train tails, but the anomalous/observed branch always obtains features from its own observed sequence; R is not assumed clean.

Nine quantiles at .001/.01/.05/.1/.5/.9/.95/.99/.999 are ordered by positive gaps. Interpolate CDF between quantiles, use exponential lower/upper extrapolation and compute two-sided log-tail surprise without float32 probability saturation; log1p compresses evidence for fusion. Gaussian uses stable log_ndtr. The Gaussian arm learns residual location/logscale; quantile arms use pinball. Compare **common normal pinball and coverage**, not unmatched loss numbers. This does not establish conformal guarantees for dependent time series.

## Four arms x three optimization seeds

| Arm | Learned tail distribution | Fusion input |
|---|---|---|
| gaussian | contextual Gaussian | learned tail evidence + original logit |
| marginal_quantile | same9-output MLP, context zero | learned marginal tail evidence + original logit |
| conditional_quantile | learned contextual monotone quantiles | conditional tail evidence + original logit |
| base_cal | same contextual quantile training as preceding arm | zero extra evidence, original logit/C only |

Every arm uses the same frozen causal teacher, source cases, initial architecture/data order/seeds and nominal dense Linear slots. Gaussian uses2/9 outputs and marginal context is zero, so **effective capacity is not perfectly equal**; record rather than hide the difference. Three seeds share one teacher, not three independent pretraining replicas. A frozen same-C wrong-source-context intervention is a source-only diagnostic; no target metadata is used to choose contexts.

Fusion is the same5->16->1 zero-initialized residual adapter for all arms: mean/max/std of node-tail evidence, base logit, log C. Train it only after freezing the tail head. Unlike C08 it has no root objective, graph posterior or null/root class coupling; RCA here is conditional ranking of node-tail evidence. Thus C08 and C09 answer distinct questions and can run concurrently.

## Budget, warmup and input boundaries

Audited312 M training systems,123 M source-validation systems. Three predetermined normal-tail passes, then three predetermined observed+normal fusion passes.936 optimizer steps in each stage/run. Per pass3,977,216 full raw scalars are seen; only3,728,640 query scalars at t>=64 receive loss. First64 points retain original TimeRCD exactly in every arm, and remain in the full target metric point set. C1 always original common-multi scores, no C1 training; not a uni-checkpoint comparison.

Across12runs:

- Normal-tail optimization134,231,040 query scalars.
- Observed-fusion optimization134,231,040 query scalars.
- Additional normal-fusion optimization134,231,040 query scalars.
- Each stage full-input/overlapping history, frozen teacher cache, source evaluation/context swaps, target forwards, health and G0 compute are counted separately. These query figures are not total compute.

Source first-pass tail calibration is a predetermined learning-curve checkpoint; final3+3 passes are compulsory.3h/run-stage and10h campaign technical limits. No CE/NLL-based extension or target checkpoint selection. New errors stop new C09 jobs, preserve finished work and do not stop other users/C08. Original2.5B/250M/checkpoints are never overwritten. Root/windows are evaluation-only for C09; effect/anomaly_channel/anomaly_time stay null, old1D labels unchanged.

## Evaluation and interpretation

Initial, density-pass1, density-final, fusion-final source results: per-system coverage/common pinball, swapped-context calibration, AP, root ranking. Fixed100development final evaluation uses original4metrics/AP/common raw-logit baseline and source-normal99th-percentile thresholds. Primary paired comparisons: conditional vs Gaussian, conditional vs marginal, conditional vs base_cal and original TimeRCD. Normal FPR, C1, tracks/families/faults and declines mandatory. All target validation/locked arrays stay closed and all target arrays never backpropagate.

C09 source/target RCA excludes t<64 and C1; full anomaly detection metrics still include all original points. Target normal twins are evaluated separately for FPR and never replace observed input. Source thresholds are M-only; C1 uses the same M-base threshold for original/fused. Do not infer real-world SOTA from known-DAG synthetic development results. Better coverage without AP/VUS preservation or an FPR-only recall loss does not establish success. If marginal and conditional are equal, retain simple calibration and remove the conditional-context-necessity claim; do not reject the entire tail direction.
