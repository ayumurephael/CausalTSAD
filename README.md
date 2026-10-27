# I13 查询条件上下文与证据分解

本分支实现已落地，实验是否完成以项目根目录STATUS和实际回执为准；此处不宣称性能已通过。

分支：`proposal/i13-query-context-decomposition-v1`。共同引擎继承I12，但本分支只承担下列实验，不修改旧proposal分支。

[完整覆盖、输入边界、预算与停止规则](docs/IMPROVEMENT3_PROTOCOL.md)。主要代码在 `proposals/improvement3/`；来源入口 `tools/run_improvement3.py`；目标入口 `tools/transfer_improvement3.py` 与 `tools/metrics_improvement3.py`。

训练臂：context_nll, query_nll, query_shuffled_nll, query_rank, decompose_learned, generic_calibrator。

| Research Insight Card | 内容 |
|---|---|
| Problem | 受污染上下文有多个正常解释；旧selector对查询工况不敏感，平均能量混合观测偏离和解释分歧。 |
| Evidence | I08 oracle有可恢复信号；I09失败、I11局部成功及负迁移已记录。A+D是恒等分解，D是否有害尚无直接结论。 |
| Failure Mechanism | 同一候选在不同查询下外推可信度可能不同；高分歧可能是无知，也可能是异常信息。 |
| Key Insight | 用合法查询工况区分外推不可信与异常偏离，保留而不是先验删除分歧。 |
| Novel Mechanism | 同容量20输入选择器的query/context/shuffle与关系监督；显式系数一致性和0参数规则；A/D与同容量普通校准。 |
| Closest Prior Work | ANP、ALPaCA、I09–I11；query attention和加权混合不是新贡献。 |
| Falsifiable Prediction | query超过context与shuffled才支持查询作用；若A校准不胜普通校准，不宣称解释分解有必要。 |
| Minimal Experiment | 六臂×3优化seed×真错图，384更新固定终点；与旧I11九冻结checkpoint同来源对照。 |

原始权重、数据、逐系统结果不包含在公开仓库。来源数据、上下文闭式估计、额外计算均需披露；无目标反传，TimeRCD不测RCA。


## r2：事件指标 JSON 保存修复

2026-10-27：r1在初始评估保存时遇到NumPy整数无法写入JSON，尚未执行优化步。r2仅把事件边界转为Python整数，并加入严格JSON回读检查；模型、损失、指标数值和固定预算不变。旧快照/错误保留，已完成缓存先核验SHA再复用，原技术截止时间不延长。本修复不构成新的性能结果。
