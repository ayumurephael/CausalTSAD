# C01 fixed-budget pilot v1

This is a new experiment, not a successful result. B1's negative result does not
reject C01. All C01 code remains on `proposal/c01-parent-innovation-v1`.

## Question and controls

Does a source-trained normal parent-conditional innovation improve the existing
self-trained TimeRCD detector on unseen generated systems? Four arms, each with
optimization seeds 348/349/350:

| Arm | Conditional history | Graph |
|---|---|---|
| pretrained_none | Frozen self-trained TimeRCD, each channel independently | Empty |
| pretrained_parents | Same frozen histories | True normal parent topology |
| pretrained_rewired | Same frozen histories | Label-free node-permuted topology |
| raw_parents | Zero-padded normalized raw history of the same feature dimension | True topology |

All arms retain exactly the same original frozen full-attention TimeRCD
detection score. All have the same trainable density/correction parameterization,
same initialization within optimization seed, data order, one pass and dense C²
message slots before masking. The raw arm has lower native encoder computation;
it is a representation control, **not an equal end-to-end FLOPs control**. Rewired
graphs preserve edge counts and the degree multiset, not each named node's degree.
C1 graph arms all have empty graphs; C1 differences after training can arise from
the shared multivariate training gradients.

## Prediction and leakage boundary

At block start t, each channel's 64 past values at times <t are independently
normalized/encoded. A shared head predicts a diagonal Gaussian for the next 16
values before seeing any query value. The 16-step horizon amortizes the frozen
encoder and supervises every real point, including the last partial block. It is
a specified pilot interface, not adaptive patching or C03 context inference.
The empty initial history uses zero features, center 0 and scale 1. Other scales
are legal-history standard deviations floored at .001. Raw features use exactly
the same history and normalization. No global normalization enters this branch.

The original detector separately keeps its own original global normalization,
5000-point windows, last-row tail padding and ChannelMean logits. It may see the
offline query sequence; its representations never enter the density head. This
original path is not claimed online or parent-only. A source-trained zero-output
residual classifier corrects its logits using detached log-innovation mean/max/
standard deviation, original logit and log(C). CE cannot train the normal density.
At initialization, the fused logit equals the original logit exactly. All source
normal and anomalous views are independently encoded; an anomalous history is
never paired with a clean query for the normal mechanism likelihood.

Inputs: observed sequence, valid history and normal directed topology. No normal
lag, coefficients, noise, fault/root metadata or target normal twin is supplied
to a prediction. Source normal twins supervise density, and source 1D labels
supervise fusion. Roots are not trained in this pilot. Target normal cases are
separate FPR evaluation cases, not clean references fed to anomalous inference.

## Budget and accounting

Reuse the audited independent B1 source pool: 4,976,640 training and 1,968,128
source-validation original scalars, manifest `bd24005add8469c106b89ee69a68bf1f6a20e7070d4dcf7f6a4e3b74dd4c2ac6`.
Same original-generator family as pretraining, new independent systems/seeds;
not a claim that this mechanism family was unseen by the backbone. All source
systems were replayed and benchmark full-array overlap was zero. No new pool
generation or old data modification occurs.

Each run: one pass, AdamW .0005, fixed final checkpoint, original exposure
4,976,640 plus the same normal-twin query supervision. Twelve runs total
59,719,680 original optimization scalars plus the same extra normal queries.
Original model parameters never enter an optimizer. Cache computation is shared
once: both views, overlapping history slots, actual patch tokens and original
detector padded scalars are recorded separately per source/development split.
Repeated head validation, extra technical forwards, parameters, dense Linear
FLOPs, time and CUDA peak are counted. Forward FLOPs are not total training FLOPs.
Source feature caching is not a second optimization exposure.

Stop: 3h per training run, 6h per cache/evaluation stage, 12h campaign, or nonfinite
loss/gradient, information-boundary failure, changed frozen hashes, input mismatch
or worker failure. A failure stops new launches; existing output/logs remain.
No target-driven extension of epochs/lr or best-checkpoint selection.

## Checks, evaluation and interpretation

Preflight uses only random/source arrays and index metadata: query/future block
blindness, nonparent and invalid-history invariance, channel permutation, empty
graphs, C1, CE-density gradient isolation, zero-residual exact preservation, and
original-implementation base-score equivalence (<1e-4, including a padded tail).
Frozen model state digests before/after caching must match. Immutable tracked
code, source/config/cache and original checkpoint SHA bind the run.

Only the same label-free fixed100 development systems as B1 are opened, 10 per
track/C. Target validation and locked arrays remain closed. Report original four
metrics plus AP, source-normal-calibrated FPR, all seeds, both tracks, C1, C>1,
mechanisms and declines. Paired units are systems, not event count or optimization
seeds. Compare fused vs original and true graph vs empty/rewired; report density
separately without choosing the best score after seeing target labels. Conditional
RCA ranks innovations in the true injection interval, excludes C1 and is not
end-to-end RCA. No new effect/anomaly_channel/anomaly_time targets are created.

Expand only if the actual source/development evidence supports a useful local or
downstream gain without an unexplained FPR/C1 collapse. A small-budget negative
result stops this prototype; it does not reject all causal mechanisms or C03.
No fixed global AUC gate, invented novelty or SOTA claim is used.
