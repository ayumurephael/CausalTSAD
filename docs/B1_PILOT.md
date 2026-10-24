# Standalone B1: first fixed-budget causal baseline pilot

This branch is a baseline, not a C01/C03 implementation. It uses no TimeRCD weights. The independent C01 branch starts from the same shared main revision and will not replace this experiment.

Four graph modes (no cross-channel edges, true parents, all other channels, label-free rewiring), three optimization seeds (348–350), identical architecture/initialization within each seed and fixed data order. Every pair message is densely computed before masking, so graph sparsity does not reduce the allocated computation. Both the node's history and candidate parent's history are available to a shared message network. There is no access to current query values until the conditional distribution has been predicted.

A 64-point legal history is normalized using its own observed past per channel. Initial missing history is masked; the first query is predicted without using its own value. A shared Gaussian head models normal dynamics. Normal source twins supervise density; old one-dimensional anomaly labels supervise a detached detector, so detection gradients cannot alter the normal mechanism. Report classifier and density AP separately; do not infer mechanism learning solely from BCE. Innovation ranking supplies only conditional RCA in this pilot.

Source pool: original replayable generator, new system RNG, attribute mode on, activation off, length 1024, C=1/8/12/16/20. Each C receives nearly 20% of the scalar budget. Whole-system flooring gives **4,976,640 original training scalars** and **1,968,128 source-validation scalars**, 1,801 systems total. Training adds an equal normal-twin forward pass. Each of 12 runs performs exactly one pass: 59,719,680 original optimized scalars total, plus 59,719,680 normal-twin training scalars. Initial/final source validation, preflight backward and target inference are additional and recorded separately. No target arrays are optimized.

All seeds are excluded from the original corpus forbidden-seed registry. Full float64 observed and normal arrays are checked against all benchmark array hashes. The preparation manifest stores all seed/input/source-code hashes. No claim is made that original-corpus full-array hashes were rescanned here; isolation from that corpus is by reserved seed and system namespace. Mechanism exposure is original-generator ID, not the held-out dynamics.

Final checkpoint is the fixed one-pass endpoint. No best epoch, benchmark tuning or automatic budget extension. Two-hour training limit and three-hour stage deadline. Nonfinite values, failed information-boundary checks or source/data mismatch stop the affected project queue. All failures and partial curves remain in their original run directory.

Before training: GPU nonparent/query/future invariance, permutation equivariance, constant/C1 finiteness, CE gradient isolation, chunk-vs-full prediction equivalence, source hashes and no mixed development/locked shard checks. This is a validity gate, not an all-task AUC threshold.

After training: fixed 100 development systems (10 label-free IDs per track/C), both normal and anomalous views, original four metrics, AP, source-calibrated 1% FPR and conditional RCA. All final seeds/modes are retained. Validation and locked arrays remain unopened. Source normal calibration is frozen for all target inputs. This subset is a development probe, not a locked result or SOTA evaluation. Full Table 1 / real transfer require a separately frozen protocol.

Large checkpoints and per-point scores stay on the server. Compact progress and result files reference immutable code/source/weight hashes. GPU queues use available memory and do not stop other users' processes.
