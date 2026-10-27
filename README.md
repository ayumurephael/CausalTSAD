# I17 来源冻结后的432系统迁移

本分支实现已落地，实验是否完成以项目根目录STATUS和实际回执为准；此处不宣称性能已通过。

分支：`diagnostic/i17-frozen-development-transfer-v1`。共同引擎继承I12，但本分支只承担下列实验，不修改旧proposal分支。

[完整覆盖、输入边界、预算与停止规则](docs/IMPROVEMENT3_PROTOCOL.md)。主要代码在 `proposals/improvement3/`；来源入口 `tools/run_improvement3.py`；目标入口 `tools/transfer_improvement3.py` 与 `tools/metrics_improvement3.py`。

训练臂：本分支无全局参数训练。

| Research Insight Card | 内容 |
|---|---|
| Problem | 来源机制有效不代表生成开发性能改善，且原新评测点集不同。 |
| Evidence | E1–E4后尚无新I系列目标迁移，旧最好结果仍只能按既定后缀解释。 |
| Failure Mechanism | 源目标的动态、故障、尺度/背景和评测范围可能改变排名。 |
| Key Insight | 先在来源确定方案与阈值，再以共同可评分点检验迁移和互补。 |
| Novel Mechanism | 有SHA冻结文件门控的开发专用推断与原四指标入口；每点32..T-1恰好输出一次，锁定/验证拒绝。 |
| Closest Prior Work | 原TimeRCD与E1–E4，采用既有指标实现；本分支是评测，不包装成新模型。 |
| Falsifiable Prediction | 同时报告留出边故障、原机制、正常误报及下降例；只有来源好不能宣称真实SOTA。 |
| Minimal Experiment | 待来源完成后固定全432系统，目标零梯度、零目标选参；互补oracle不进入路由。 |

原始权重、数据、逐系统结果不包含在公开仓库。来源数据、上下文闭式估计、额外计算均需披露；无目标反传，TimeRCD不测RCA。


## r2：事件指标 JSON 保存修复

2026-10-27：r1在初始评估保存时遇到NumPy整数无法写入JSON，尚未执行优化步。r2仅把事件边界转为Python整数，并加入严格JSON回读检查；模型、损失、指标数值和固定预算不变。旧快照/错误保留，已完成缓存先核验SHA再复用，原技术截止时间不延长。本修复不构成新的性能结果。
