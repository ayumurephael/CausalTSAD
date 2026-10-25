# 已完成结果与范围

C05六个冻结模型×123源M系统全部完成，0optimizer、未读目标数组。65有子系统的child分数均值差CI跨0，超源阈值额外症状只1系统/1seed；没有普遍测量污染failure证据，也没有实现双层潜状态solution。

见[数值与全部正反系统](results/fixed_pilot/summary.json)及[完整报告](results/fixed_pilot/README.md)。以下为原实现说明，训练代码版本在不可变launch中固定；新的文档提交不改变已跑模型。

---

# C05 measurement-isolation source mechanism probe

Separate branch from common main5510ede. C08b training runs on H100; this source-only no-training diagnostic uses released A100 resources. It is **not the complete C05 two-layer latent-state solution** and not a new performance claim.

Known source root is used only to construct a synthetic measured-sensor fault: normal observations in every nonroot channel, exactly the original observed root trajectory in the root channel. It never enters the model call. Evaluate normal, original, and root-isolated measurement-only views using frozen C02 univariate and normal-parent predictors,3optimization seeds/6model jobs. Actual parent same-time branch remains authorized, own query/future excluded by the inherited predictor.

Question: can a damaged parent measurement create large child innovations even while child observations are exactly normal? Check score/rank changes and existing source-calibrated threshold exceedance, with no-graph nonroot and parent-model unrelated-node exact nulls. Roots need not be graph sources; root degree and parent observations are recorded. Original cases are not blindly called physical state faults; isolated cases are not claimed to be do-interventions or regenerated latent states. This is a controlled measurement fault construction. Original input/normal/isolated array hashes and matched-root checks saved; undefined effect/anomaly_channel/anomaly_time labels remain null and original1D labels unchanged.

123independent multivariate source-validation systems,6frozen teacher jobs,28,237,824extra query scalar forwards,0optimizer/0newencoder. Weights/inputs/code SHA bound. Scope includes conditional full/ready RCA; no target benchmark/locked/validation arrays. Every null/reference check is retained; cross-model differences alone do not identify causal mechanism. Same-time bad-parent sensitivity plus known-normal child observations supplies the direct probe. Further dual-state learning requires its own audited data/controls/budget.

Closest fully read papers: R1 Yang2022 measurement vs intervention anomalies, R8 AERCA2025 exogenous-intervention limits, R2 density-estimation failure. Mature mechanisms welcome, no novelty claim for residuals. Fixed2h deadline, failure halts new C05jobs and preserves original artifacts/other projects. No automatic training or expansion.
