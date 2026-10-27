# I16 原始TimeRCD路径与有界因果增量

本分支实现已落地，实验是否完成以项目根目录STATUS和实际回执为准；此处不宣称性能已通过。

分支：`proposal/i16-protected-causal-increment-v1`。共同引擎继承I12，但本分支只承担下列实验，不修改旧proposal分支。

[完整覆盖、输入边界、预算与停止规则](docs/IMPROVEMENT3_PROTOCOL.md)。主要代码在 `proposals/improvement3/`；来源入口 `tools/run_improvement3.py`；目标入口 `tools/transfer_improvement3.py` 与 `tools/metrics_improvement3.py`。

训练臂：protected, raw_adapt, concat_adapt。

| Research Insight Card | 内容 |
|---|---|
| Problem | 直接拼接因果残差可能改变TimeRCD输入分布，需要保护原有检测能力。 |
| Evidence | E4目标局部有收益但区间跨零；I06 zero/wrong拼接也可改善，不足以说明真实因果贡献。 |
| Failure Mechanism | 新增token同时改变位置、归一化和注意力；收益可能是一般再校准而非DAG信息。 |
| Key Insight | 先保持原始观测编码与分数，只允许来源学习一个可归因的有界校正。 |
| Novel Mechanism | 原TimeRCD参数/input不变，零初始化±2 logit修正；与同容量raw适配、拼接适配及错图对照。 |
| Closest Prior Work | C04b保护融合和成熟残差适配；新证据必须来自当前结构/上下文接口及严格对照。 |
| Falsifiable Prediction | protected须超过raw_adapt并保住短故障/正常FPR；zero init只能保证初始等价，不保证训练后AP。 |
| Minimal Experiment | 三臂×3seed×真错图；来源384步固定；TimeRCD不输出RCA，定位另由因果分支承担。 |

原始权重、数据、逐系统结果不包含在公开仓库。来源数据、上下文闭式估计、额外计算均需披露；无目标反传，TimeRCD不测RCA。


## r2：事件指标 JSON 保存修复

2026-10-27：r1在初始评估保存时遇到NumPy整数无法写入JSON，尚未执行优化步。r2仅把事件边界转为Python整数，并加入严格JSON回读检查；模型、损失、指标数值和固定预算不变。旧快照/错误保留，已完成缓存先核验SHA再复用，原技术截止时间不延长。本修复不构成新的性能结果。
