# C02: available parent information, source-only probe v1

This is an independent branch, not a modification of C01/C03 or a claimed
TimeRCD improvement. It starts at common local main
`5b96fd63fb5ed7e0b21dcdc6e80f35d3fadcb8f6`. On2026-10-25 the user explicitly
authorized a separately labelled same-time-parent branch; own current/future
values remain forbidden in predictive inputs. No true lag/coef/noise is input.

## Research Insight Card

| Field | Content |
|---|---|
| Problem | Determine whether a causal head failed to benefit from G because required parent observations were unavailable, rather than because G is useless. New systems, known normal topology, no target updates. Primary paired source-system NLL; observed-source AP and conditional RCA separately. |
| Evidence | C01's graph variants have similar target scores; C03's fixed earlier moment features do not improve source NLL and swapping source systems does not hurt. Neither proves the cause. Source code samples lag0–7; our earlier head predicts16 future points from histories ending at block start. R1/R2/R5 fully read normal conditional models allow temporal conditioning, with R1 including contemporary parents. |
| Failure Mechanism | For point t=t0+h and parent lag L, a head restricted to <t0 misses the exact structural parent value when h≥L. At L=0 it misses every point. Autocorrelation may compensate, so availability alone does not prove a useful predictive effect. This is a limitation of our pilot interface, not an established original-TimeRCD failure cause. |
| Key Insight | Supply the causal evidence at the time at which the mechanism uses it; distinguish data availability from learning the graph or its unknown parameters. |
| Novel Mechanism | Five matched normal predictors vary only G and the parent time view; own history, normalizer, architecture and dense computations stay fixed. This is a causal availability probe, not yet learned lag inference, latent context or adaptive patching. |
| Closest Prior Work | R1/R2 causal conditionals and R5 time-parent masks are direct precedents. Temporal conditioning/same-time parents are established; no novelty is claimed for the probe. The later research question is cross-system fixed-weight TSAD/RCA with unknown mechanisms. |
| Falsifiable Prediction | Correct and timely parents should improve normal prediction and useful source anomaly/root evidence over stale/empty/wrong controls. If only any current channel helps, direction-specific benefit is not shown. Empty graphs must be an exact timing null. |
| Minimal Experiment | 5arms×3seeds, each one audited independent normal-source pass4,976,640scalars; total74,649,600normal scalars, anomalous observed gradient0. First isolate source mechanism before any TimeRCD integration or new target test. |

## Matched arms

| Arm | Graph | Parent observations used for point t |
|---|---|---|
| none | Empty | Past view computed but unused; block/past/current are exact nulls at identical weights |
| parents_block | True normal DAG |64points ending just before t0=floor(t/16)×16 |
| parents_past | True normal DAG |64points ending just before t |
| parents_current | True normal DAG |64points ending at t, only for actual direct parents |
| rewired_current | Label-free node-permuted normal DAG |Same current view; intentionally wrong graph control |

Every own branch uses64values strictly before t. Every variant uses the **same
block-start own per-channel center/scale**, computed before t0, for both histories
and queries. This prevents a normalization-refresh change from explaining the
parent-refresh contrast. Empty initial history uses center0/scale1; others have
scale≥.001. All timing views are constructed in all arms. All evaluate all dense
C² messages and16output slots per point, selecting the same phase slot t mod16.
Capacity and Linear forward FLOPs are matched. Compared with older pilots,
per-point prediction is costlier; its full extra computation is recorded.

There is only one message layer and no cross-channel normalization/hidden-state
mixing before the graph mask. A current target scalar can feed its children but
cannot feed its own prediction. Direct own-current perturbation, future and
nonparent invariance, permutation, finite backward, C1 constants and empty-graph
nulls are G0 checks. The wrong graph is a diagnostic, not a true causal model;
an actual descendant may be a fake parent, so good normal prediction with it
would not establish causal direction or good RCA.

## Fixed data and budget

Only the already replayed independent source pool is used:
SHA256 `bd24005add8469c106b89ee69a68bf1f6a20e7070d4dcf7f6a4e3b74dd4c2ac6`.
Original-generator mechanism family; this is not previously unseen pretraining
mechanisms. No2.5B/250M data, old checkpoint or other branch is overwritten.
Every run: one434update pass, normal-query exposure4,976,640; AdamW .0005,
batch8192scalars, order57348, clip1, seeds348/349/350. Anomalous source values
are only used in frozen evaluation. Source validation contains513systems,
123multivariate; normal/anomalous branches are paired units.

Initial/final source evaluation each uses main normal+observed and three frozen
normal timing interventions, total19,681,280extra query scalar forwards per run,
295,219,200for15runs. Those are separate from74,649,600normal training scalars.
Repeated64point history uses and dense Linear FLOPs are reported, not equated
with unique observations or full training FLOPs. There is no encoder forward.
Seeds stay on one GPU type/node across all five arms. Node types can differ
between optimization seeds and are reported; three seeds are not independent
data systems.

One isolated oracle audit replays32label-free-selected source-validation systems
and verifies full float64 normal/observed hashes. Actual lags count the required
parent observations available in each view over t≥64. Oracle lags and other
generation metadata never reach training or prediction. Extra CPU regeneration
is accounted separately. This audit does not grant the main model true lags.

## Stop and interpretation

Fixed final checkpoint,2h/run,8h/campaign; stop new jobs on G0/hash/nonfinite/error.
No target arrays/labels select checkpoints, hyperparameters or extra steps.
Ready NLL uses the same t≥64point set for every arm; full NLL remains reported.
Conditional source RCA uses actual injection windows after prediction, excludes
C1 and is not end-to-end RCA. No effect/channel/time threshold labels are added.

Do not require a global AUC≥.85 gate. A useful local prediction/innovation effect
can justify a bounded next integration; lack of one stops this prototype rather
than all causal work. NLL improvement alone is not TSAD performance, and any
next TimeRCD fusion must include original-logit/score-representation controls and
matched graph controls before using the locked benchmark. C01/C03 results stay
as separate immutable evidence.
