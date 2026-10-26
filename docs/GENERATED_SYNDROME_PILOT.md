# C04i companion: frozen generated-development transfer (not source training)

## Problem
Does the source-supported raw causal-innovation model transfer to the pre-existing
100 development systems (80 multivariate, 20 univariate) of the retained250M
benchmark? Detection: original four formulas, AP/AUC and normal FPR; RCA: only
conditional on the annotated true event window. No TimeRCD RCA adaptation.

## Evidence
On32 independent source validation systems, full true signatures achieve matched
AP.891507/conditional rootTop1.890625; wrong graph AP.634545, empty.503911.
Initial seven-parameter priors already give AP.891612. This supports causal
context inference, not a learned foundation-model claim or target performance.

## Failure Mechanism
Earlier C04 neural scores transferred poorly (full development M AP.05827 vs
TimeRCD.45832, same >=320 points). Global predictor and anomaly-family mismatch
are hypotheses. Source gains alone do not establish robustness to different
scales, short faults, nonlinear dynamics, noisy prefixes or normal switches.

## Key Insight
Compare the fixed latent-context model and its graph/signature controls on the
same target suffix. Preserve all negative cases; do not select variants by target
labels or silently average the source 75%-positive eight-view task with this task.

## Novel Mechanism
No new model claim. Deploy the unchanged7a4674e raw kernel with its fixed source
priors. Prefix256 gives robust closed-form latent regression; W16/L8/FIRST320,
raw+tanh features and scales(.5,2,8) stay fixed. This is per-system forward latent
inference, not zero per-system computation. No target gradient or parameter fitting
via SGD. Models receive only observed values and known normal topology.

## Closest Prior Work
R19 Bayesian fault-hypothesis filtering and R15 likelihood-ratio/background
comparison motivate competition between physical and measurement innovations.
This is a local Gaussian approximation with approximate predictive diagonal noise,
not their complete method, exact Kalman filtering, or a novel foundation model.

## Falsifiable Prediction
If source improvement transfers, true_signature should improve paired M AP and
conditional root ranking over wrong/empty graphs without unacceptable normal FPR.
Compare true_self with the identical normal predictor: a detection improvement is
different from a root-localization improvement. If target deterioration persists,
stop this frozen prototype and analyze track/family/C/duration/prefix failures;
do not tune target thresholds or train on these arrays. No locked-test promotion
from a single favorable subgroup or AP alone.

## Minimal Experiment and frozen controls
- Reuse the historical pilot100 selection (no new label-driven selection), all
  development; validation/locked arrays remain unopened.
- true_signature/true_self/wrong_signature/empty_signature × source seeds348–350.
  Source fitting unchanged:32 training,16 calibration,32 validation systems. Reuse
  existing evidence; target computes each mode/view once then applies3 tiny heads.
- Each observed and normal case independently conditions its own prefix. The
  normal twin is only a separate FPR evaluation, never observed-case repair/input.
  Saved graph coefficients/noise/lag/root/normal twin never enter observed calls.
- All methods including cached self-trained TimeRCD use the same >=320 labels,
  fixed original metrics and score representation (logit). Read source-only
 99th-percentile thresholds; recompute B0 threshold on the identical16 source cal
  systems from its existing cached normal scores. No target calibration.
- C1 is exact cached native selftrained_uni passthrough for every method, excluded
  from causal RCA; this is preservation by bypass, not learned C1 capability.
- Conditional RCA averages posterior over each applied true event's annotated
  window intersect >=320; empty windows/C1 excluded with explicit denominators.
- Freeze source checkpoint/input/code SHA, historical row list and metric runtime.
  Budget:80 M systems ×4 modes ×2 views, actual sum(T-320)*C and full sum(T*C)
  recorded separately;3 prior applications are not3 new evidence passes. CPU
  original-metric jobs are extra computation, no new TimeRCD forwards. 4h GPU
  deadline then at most4h CPU metrics. One1GiB process per available GPU, sharing
  only with measured >=1792MiB free; never stop another user's process.
- Global prior SHA must remain unchanged; source augmentation hashes are separate
  from old25亿 and benchmark. Fresh source/target data overlap audits are not
  invented: previous retained-benchmark full-array hashes0 matches; no new full
  rescan of old25亿 for the augmented views has been performed.

This pilot is explored development, not blind150M test, real-data SOTA, full
pretraining, or proof that graph direction is uniquely causal.
