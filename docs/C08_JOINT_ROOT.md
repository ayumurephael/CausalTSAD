# C08 v1: a candidate TSAD/RCA solution, tested by a controlled pilot

This is a new branch from common main `e41ebcb3ad1164cf87041e94360fb3c1b5353332`. It does not modify C01/C02/C03. It is not yet an effective final method, and is not full foundation-model pretraining.

## Mechanism and exact model

Reuse the frozen self-trained TimeRCD raw binary logit B(t). Reuse the frozen C01 **univariate/empty-graph** density teacher `pretrained_none_s348`; discard its old global correction. All arms and optimization seeds use this same teacher, isolating graph information in the new posterior readout rather than changing the teacher at the same time.

The teacher's distribution is generated from the previous64 points before each16-point query block. No query enters that distribution. Current values then produce signed standardized residuals/energy, logvariance, normalized query/mean, history coverage, and causal8/32/64-point moving averages:12 local features, compressed/clipped at+-50. These are posterior evidence, not a new forecaster. Mean/std remain per channel, and the observed branch uses its own observed context; a normal twin never replaces its input.

The new head is shared across nodes and channel counts:32-wide node embedding, two directed graph message layers, separate parent/child aggregates, a zero-initialized node output and a zero-initialized base-logit calibrator. Every arm evaluates all dense C-squared pair slots before masking. With node evidence r_i and calibrator a:

`S = B + a(B,log C) + logsumexp(r_i) - log C`

`p(null)=sigmoid(-S); p(root=i)=sigmoid(S)*softmax(r)_i`.

The root evidence participates directly in detection; RCA is not just a detached extra ranking. These are model probabilities trained with weighted BCE, not a claim of calibrated Bayesian posteriors or identifiable interventions. Actual predictive calibration is evaluated separately. A source-normal99th-percentile threshold is fixed before target evaluation.

The classifier/posterior may see current innovations and parent/child symptoms. It must not be described as a parent-only query-blind density. The original B is an offline detector and may see the whole sequence. Equivariance/future checks hold B fixed and validate the new branch, not the full original TimeRCD. C1 returns B exactly, and C1 never enters optimization. C1 here uses common **multi** weights; it is not a uni checkpoint experiment.

## Eight arms x three seeds

| Arm | Topology used by readout | Local values | Extra root objective |
|---|---|---|---|
| local_det | empty | yes | no |
| local_joint | empty | yes | true root |
| dag_det | normal DAG | yes | no |
| dag_joint | normal DAG | yes | true root |
| rewired_joint | node-permuted wrong DAG | yes | true root |
| base_cal | empty | zero | no |
| prior_joint | normal DAG | zero | true root |
| dag_random_root | normal DAG | yes | fixed wrong source root |

Architecture/initialization/data order/raw scalar exposure/dense Linear FLOPs match within seed. Empty/zero inputs reduce active information and effective capacity; identical nominal parameter count is not identical effective capacity. Root losses entail extra computation and supervision, explicitly counted. Rewiring preserves edge count and degree multiset, not per-named-node degree; the wrong source target is a deterministic nonzero offset from the real root. No target labels generate a graph, grouping or threshold.

## Source supervision, budget, and stopping

Audited shared pool:312 M training systems (3,977,216 observed scalars/pass) and123 M source-validation systems. Original2.5B remains unchanged. No new source mechanism family is claimed; the teacher's pretraining saw the original generator family. Reuse of a seed348 teacher makes3 new seeds optimization seeds, not3 independent pretraining replicates.

Each run: **3 predetermined passes /936 updates /11,931,648 observed scalars**, plus same number normal-twin head queries.24runs total **286,359,552 observed optimization scalars +286,359,552 extra normal scalars**. Cache preparation, source initial/pass1/final evaluation, random G0 and fixed100development forwards are additional and separately recorded; no new encoder runs. Source pass1 is a fixed learning-curve diagnostic, never target checkpoint selection. Final3-pass endpoint is compulsory. Per-run2h /stage3h /campaign8h technical limits; a failure halts new jobs, preserves finished work and never touches unrelated tasks. No extension because CE falls or target results look promising.

Global weighted BCE uses the original1D labels, including propagated tails. Conditional root CE/logC is applied only where source `root_window & labels` is true. The normal class is supervised via BCE and p(null). Tail anomalies without an active injection do not receive invented root/effect targets. New effect/anomaly_channel/anomaly_time stay null. Roots/windows are labels, not model arguments. Ground-truth DAG topology is an authorized model input; no true lag/coefficient/noise is supplied.

## Evaluation and decision

Initial/pass1/final source AP and RCA by independent system; same fixed100development (40 ID-M,40 heldout-M,20 C1) for all24finals, original4metric formulas, **common raw logits** for both B and S. No target validation/locked access or gradients. Original probability-protocol historical results remain available in C01; score precision is not counted as a causal-model gain. Source thresholds are M-only; the protected C1 branch uses the same baseline M threshold for both methods, so this is not new C1 FPR calibration.

Primary causal contrasts: dag_joint-local_joint, dag_joint-rewired_joint, and graph-by-root-loss interaction `(dag_joint-dag_det)-(local_joint-local_det)`. Also compare base_cal, graph-prior-only and wrong-root controls. Report per-C/track/family/fault/root-depth positives and declines, paired-system uncertainty and seed variation; no one global .85 gate. Normal FPR must accompany AP/VUS. A change in root scores alone is **RCA-only evidence**, not TSAD success.

RCA is reported both conditional on true injection windows and detection+root event-matched (missed events score0); this is not free-running alarm precision or proof of unique causal identifiability. C1 root scores excluded. No benchmark event window is ever passed to inference.

Success for expanding this prototype requires reproducible source root signal and nontrivial AP/VUS improvement beyond same-logit base/calibration controls without a meaningful normal-FPR tradeoff. Confidence intervals, individual systems and interactions determine interpretation; tiny-budget null findings do not reject the whole causal direction. Do not train longer automatically. A graph-only or shuffled-root win revises the explanation rather than being hidden.

## Closest papers and boundaries

[DejaVu FSE2022](https://netman.aiops.org/wp-content/uploads/2022/11/DejaVu-paper.pdf), full13pages read: supervised shared graph localization, but fault discovery is explicitly outside its scope and FDG is undirected. Paper says cross-system transfer is not currently supported. [CDRL4AD2025v1](https://arxiv.org/abs/2510.11084v1), full20pages read: Eq16 averages per-node prediction/reconstruction error, Eq17 ranks it for RCA; no injected-root supervision. Its released `CausalAttentionLayer` also accepts current target y, so it is not our query-blind predictive baseline. Mature shared GNN supervision is adopted, not claimed novel. Our test concerns source-root supervision, explicit null state, cross-system variable-C deployment and whether root evidence improves a frozen foundation detector under known normal DAG inputs.

Fixed author commits and actual code-reading scope are in the private project reading log, not silently vendored external execution. The copied `dependencies/c01_model.py` is our unchanged earlier C01 implementation, used only to load the frozen teacher; source/file hashes are bound in the experiment dependency manifest. Data/weights/private administration/SSH credentials never enter Git.
