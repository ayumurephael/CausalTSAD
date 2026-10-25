# Frozen C08 coupling diagnosis, not another performance claim

Parent implementation `5fc60f5faf7d2e6789a03a9c02ef3a1c5d8d54e4`, original24finals immutable. All have completed100development each. Correct-DAG joint model M AP .47348 vs baseline .47940, VUS .41264 vs .41551, FPR .00399 vs .00331 under this stage's M-only source thresholds. Root Top1 .42083 vs local-joint .38750; source root Top1 improves with true supervision. Thus better root prediction did not improve detection. It does not prove that root supervision or causal information is fundamentally harmful.

Two candidate explanations, to test rather than assume:

1. Source detection/root gradients may compete. Measure dot/cosine/norms at the exact common initial state and fixed final states, on64 source **training** systems (16/C selected by key SHA, no label selection). No source-validation or target gradients. No optimizer steps or saved modified parameters. A negative cosine alone is insufficient proof that it caused target failure; neighboring multitask optimization methods are candidates only after full reading/diagnosis.
2. The readout couples root confidence and anomaly evidence. Exact algebra for root logits r and p=softmax(r):

   `S=B+a+logmeanexp(r)=B+a+mean(r)+KL(U_C || p)`.

   Root confidence introduces a nonnegative additive term even though the model may be confidently choosing an arbitrary root on normal data. This identity establishes an interface property, not the reason for our metric decrease. Test original S, S minus the per-point KL, S minus a source-normal mean KL (constant/rank null), and S minus a label-free time-shuffled KL (temporal-association control). All controls fixed before scores are inspected; no coefficient tuning.

Four arm types dag_joint/local_joint/dag_det/prior_joint x3seeds. Source-validation123M systems calibrate each transformed score to normal99th percentile and provide paired AP; fixed100development AP/FPR only, no VUS rescan/new encoder/target optimization. C1 exact identity and original-score null against saved arrays must hold. Paired source/target/seed positives and negatives all retained. No locked/validation arrays. Parameter/weight/cache/code SHA before/after; frozen forwards and training-only gradient backward counts separate. Two hours/job,four hours/campaign, zero optimizer budget.

If removal fails to beat constant/shuffled controls, revise the confidence explanation. This frozen edit is not the same as training a properly decoupled head, and a null result cannot reject that broader direction. No new solution is claimed from a post-hoc edit alone. Root/effect labels are not invented; actual roots remain source targets/assessment only.
