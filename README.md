# C03v2已完成

九组全部完成，源历史机制信号成立，目标收益区间跨0且集中两例、误报增。见[完整数值](results/fixed_pilot/summary.json)与[解释](results/fixed_pilot/interpretation.md)。下一轮因果信息对照独立在c03-causal-specificity-v3，旧模型不改。

---

# C03v2：学习因果条件机制上下文

当前分支实现新的C03v2集成，不覆盖C09或旧C03。详细架构、信息边界、9组预算与停止条件见[实验说明](docs/C03_LEARNED_CONTEXT_V2.md)；实际阶段以部署记录为准，尚无新效果结论。

---

以下是保留的C09基础实现说明：

# C09: conditional residual tails for known-DAG zero-shot TSAD

Independent branch `proposal/c09-conditional-tail-v1`, parallel to C08. Frozen C02 causal normal predictor + shared source-trained tail distribution + source-trained residual correction to the frozen self-trained TimeRCD raw logit. No target optimizer, true coefficients/noise/lags or root labels as inputs.

[Mechanism,4arms x3seeds, fixed budgets and decision rules](docs/C09_CONDITIONAL_TAIL.md). [Configuration](experiments/c09_tail_v1.json). [Dependency lineage](experiments/dependencies_c09.json).

Three normal-tail passes then three fusion passes, all fixed before results. Gaussian/marginal-quantile/conditional-quantile/base-only-calibration controls. Same nominal network slots, disclosed effective-capacity differences. Source coverage/pinball/context-swap, full100development original4metrics/AP/FPR and conditional ready-region RCA. C1 and first64points exactly original multi detector. Original raw-logit baseline is preserved; validation/locked data closed.

This is a candidate solution under evaluation, not a proven improvement, new conformal guarantee, complete foundation pretraining or claimed SOTA. Data/checkpoints/private administration/SSH credentials are excluded from Git. GPU G0 must pass before the bounded `tools/controller_c09.py` pipeline can optimize.
