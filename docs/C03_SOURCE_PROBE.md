# C03: source-only mechanism-context probe v1

This independent branch starts from shared local main
`7b7d7f6` (full base recorded by Git at creation). It does not modify C01 or
claim an already improved TimeRCD. The C01 graph-control campaign runs separately
on H100; this experiment uses A100s without waiting for target results.

## Research Insight Card

| Field | Fixed interpretation |
|---|---|
| Problem | With the same normal DAG, different systems have different normal conditional mechanisms. Test whether earlier observable context helps a shared predictor on new source systems. Primary: normal Gaussian NLL per independent source system, full and predeclared context-ready points. AP is a separate source anomaly diagnostic. |
| Evidence | R1/R2/R5, fully read in the parent research folder, model causal conditionals but generally train per system. The source generator varies coefficients/dynamics across systems. C01 results were not used to choose this prototype. Whether its current representation averages incompatible mechanisms remains a hypothesis. |
| Failure Mechanism | A single shared mapping without a system-specific mechanism description can average opposite/weak/strong parent effects. This is not established by the earlier B1 negative result, and covariance need not identify a nonlinear causal mechanism. |
| Key Insight | The graph supplies who may explain whom; earlier observed parent/child history may supply a description of this particular normal relationship, without updating target parameters. |
| Novel Mechanism | Pilot: deterministic parent-specific past cross-moments feed a shared conditional head. Compare zero/mean/median context with identical head capacity, initial weights and normal exposure. It is an engineered neural probe, not the full stochastic latent-context C03 design. |
| Closest Prior Work | R1 causal conditional models, R2 conditional-density comparisons, R5 causally constrained probabilistic forecasting. These are substantive precedents; moment features, temporal conditioning and median aggregation are established techniques. No novelty claim follows from combining them. |
| Falsifiable Prediction | On identical independent source systems, correct earlier context improves normal NLL over zero context, and exchanging context with another source system degrades the improvement. If swaps do not matter, useful mechanism identification is not demonstrated. Normal NLL improvement alone does not establish anomaly detection. |
| Minimal Experiment | Three equal-capacity arms × seeds348/349/350; one fixed normal-source pass each. No TimeRCD, benchmark arrays, true lag/coefficients or parameter fitting on target. Decide on a later explicit C01+C03 integration branch only after source evidence. |

## Interface and information boundary

At block start t, the predictor sees the most recent 64 values per channel and
predicts the next16 values. Recent histories are independently standardized per
channel. Empty history uses center0/scale1; other scales floor at .001. Queries
are normalized with the same own-history statistics and are only used in the
likelihood or frozen score, never in predictor inputs.

Earlier context ends at t−64 and contains up to192 previous points, split into
three64-point chunks. For each target/parent pair, compute correlations at
lags0/1/2/4/8, both channels' own lag1 correlations, and coverage. These are
observable statistics, **not true edge lags or causal discoveries**. Mean context
averages chunk statistics; median context uses their componentwise median.
Short contexts (<16points) produce zeros. The fixed lag grid/window is a pilot
feature interface, not a learned universal range or an adaptive patch proposal.

Only permitted graph parents contribute pair messages. A fixed shared neural
mapping uses the context; no optimizer, closed-form regression fit or persistent
parameter update runs at inference. All three arms compute dense C² message
slots; the zero arm supplies zero context to the same model parameterization.
All preprocessing centering/scaling is per channel, and a pair's moments depend
only on that pair. Nonparent perturbation and permutation checks cover both
preprocessing and the model. C1 has no edges and cannot directly use the context.

The median arm is a modest robust-statistic probe. It does not implement the
full posterior uncertainty, learned contamination mixture or full context
identification described in the broader C03 Card. Observed anomaly scoring uses
that observed sequence's own earlier context, which may contain faults. The
normal twin is never substituted for the observed context.

## Data, budget and controls

Use only the existing audited B1 independent source pool, SHA256
`bd24005add8469c106b89ee69a68bf1f6a20e7070d4dcf7f6a4e3b74dd4c2ac6`.
1,801 systems total, source train4,976,640 and source validation1,968,128 scalars.
It shares the original generator family with pretraining, not new held-out
dynamics. No new generation, encoder inference or target arrays are needed.

Each run optimizes only **4,976,640 normal-twin query scalars**, one pass434updates.
Nine runs total44,789,760 normal optimization scalars. Anomalous original observed
scalars in gradients are **zero**, rather than misleadingly calling this45M
original-observation exposure. All context/raw histories are source input views;
the cache records repeated recent/earlier scalar uses separately. The normal and
observed source-validation queries, initial/final evaluations and context swaps
are extra frozen computation, recorded per run. Trainable parameters, initial
hash, source/cache/config SHA, final checkpoint SHA, Linear forward FLOPs, runtime
and peak CUDA are recorded. Nonlinear/cache/backward costs are not included in
the Linear-only FLOPs count.

All arms use the same normal targets, scalar batch8192, data order57348,
AdamW .0005 and clip1, seed-matched initialization, fixed final checkpoint.
Ready points require at least128 earlier context points; this mask depends only
on indices, not labels, and is the same point set for every arm. Report full NLL
as well, including initial history difficulties. Source-system context swaps
use a cyclic other source-validation system of the same C, matching the same
block index. These **source-normal oracle diagnostics** are not deployment input
or a claim that reference data are clean at target.

Stop at one pass, 2h per run, 8h campaign, or nonfinite/hard information failure.
No automatic extra epochs/lr search. A small-budget failure stops this prototype,
not the causal-context direction. C01's original59,719,680-scalar budget is not
expanded by this separate44,789,760-normal-scalar probe. The original2.5B corpus,
weights,250M benchmark and C01 outputs remain unchanged. No new channel/effect
labels are generated. RCA and target TSAD improvement have not been tested here.
