# C08b: train the readout repair, and separate it from gradient surgery

This is a separate revision/branch from C08 v1, preserving all failed and successful cases. The original24run pilot completed and all197compact files, weights,936updates/exposure/init/dense Linear counts were audited. Original root-CE model improved source root prediction and some conditional target RCA, but M AP fell .47940->.47348; normal FPR rose .00331->.00399 under M-source calibration. DAG vs local RCA is not established as a universal graph benefit: ID Top1 .60 vs .50, heldout .2417 vs .2750.

The completed12run frozen coupling diagnostic found:

- Exact identity `logmeanexp(r)=mean(r)+KL(U||softmax(r))`. Root certainty is added to anomaly evidence by construction; mathematical identity alone does not prove failure cause.
- Removing the per-point confidence term gave source AP .82246->.82717 and M-development AP .47348->.47931; constant subtraction preserved old AP, shuffled subtraction did not recover it. M paired AP recovery+.005834, bootstrap95%[.000897,.012121], FPR decreased. Original TimeRCD AP .47940 remains slightly higher; this is recovery from our own regression, not SOTA.
- Source-training final gradients conflicted in61.98% of192 seed/system probes; root/detection norm median1.71. These are full-vector diagnostics including detector-private calibration; no shared-only or high-curvature claim. Gradient evidence supports a limited optimization comparison, not proof that PCGrad solves the target failure.

## Frozen factorial design

| Readout | Sum loss gradients | Shared-only2task PCGrad |
|---|---|---|
| Original LSE | **Reuse** original C08dag_joint3seeds, no duplicate training | lse_pcgrad3seeds |
| Mean node evidence | mean_sum3seeds | mean_pcgrad3seeds |

Same13,250parameters, seeds348/349/350, original cache/teacher/base, source systems, data order and3passes per new run. Mean readout is original S minus KL; p(null) and joint-root mass are recomputed consistently. No extra coefficients/temperatures tuned on target. C1 remains original multi exactly. Source root CE only actual active windows; propagated tails retain old1D detection labels. No new channel impact labels.

PCGrad follows the fully read R11 *Gradient Surgery for Multi-Task Learning*, arXiv2001.06782v4,27pages including appendices, and fixed author TensorFlow code commitc5fbd7c856526373828074f06875230f7f3ee79e. Implement the mature2task projection in our own PyTorch code, **only on shared coordinates**; private detector calibration keeps its own gradient, None/zero gradients are handled explicitly. Do not copy the author's discarded `tf.random.shuffle(loss)` call or skip-None flattening into mismatched task vectors. With two tasks, ordering has no nontrivial random-choice effect. Negative cosine alone is not the paper's full high-curvature/magnitude condition, and the supervised appendix discusses regularization effects.

Nine new fixed-final runs; each11,931,648 original observed query scalars plus same normal queries, total107,384,832+107,384,832. Three old baseline runs are reused exactly. PCGrad uses2task backward calls/update vs1sum; report this extra work and projection operations, not equal-total-FLOPs. Forward dense slots and original exposure match. Cached encoder/teacher work reused,0newencoder/cache. Initial/pass1/final source curves; final only100development x original4metrics/AP/FPR/RCA. No target gradients/validation/locked access.

Technical2h/train,3hstage,8hcampaign; fixed3pass936updates, no extra steps based on CE or target success. G0: original forward exact, mean/KL identity, mass normalization, C1, analytic projection conflict/nonconflict/private/zero, finite actual-head backward. Stop new jobs on technical failure, preserve originals and completed runs. Fixed source budgets do not prove full foundation learning.

Decision: mean_sum vs reused LSE-sum isolates training the corrected interface; PCGrad effects evaluated in each readout. A method must improve AP/VUS beyond original TimeRCD, retain RCA and avoid FPR tradeoff before expansion. If only recovery, retain it as an implementation correction and separate-root component, not a new SOTA. If effects are optimization/extra calculation rather than causal information, name them correctly. No automatic scale-up.
