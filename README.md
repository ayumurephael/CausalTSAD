# I12 边结构化残差与未知起点扫描

本分支实现已落地，实验是否完成以项目根目录STATUS和实际回执为准；此处不宣称性能已通过。

分支：`proposal/i12-edge-segment-source-v1`。共同引擎继承I12，但本分支只承担下列实验，不修改旧proposal分支。

[完整覆盖、输入边界、预算与停止规则](docs/IMPROVEMENT3_PROTOCOL.md)。主要代码在 `proposals/improvement3/`；来源入口 `tools/run_improvement3.py`；目标入口 `tools/transfer_improvement3.py` 与 `tools/metrics_improvement3.py`。

训练臂：本分支无全局参数训练。

| Research Insight Card | 内容 |
|---|---|
| Problem | 节点幅度可能不能分辨传感器观测故障、过程扰动与边改变；原关系评测含case区分成分。 |
| Evidence | 旧I11关系AP59.31/条件Top142.45仅是已探索来源32系统。外部线性代理提示边覆盖可能有效，但未证明对当前神经模型有效。 |
| Failure Mechanism | 沿父贡献方向累积的残差可能呈现不同故障签名；度数、窗口长度、整体能量也可能解释提升。 |
| Key Insight | 在同一个正常预测下分开测量节点偏差、方向对齐和时间持续性，才有机会识别有用的结构证据。 |
| Novel Mechanism | 精确提取神经父块贡献，配套尾随多尺度、结构签名、同窗口节点/时间错配边/degree-only定位控制。 |
| Closest Prior Work | C04r、CIRCA、Chen–Liu；结构残差和时间扫描成熟，不把基本算子称新颖。 |
| Falsifiable Prediction | 真边证据应超过同窗口节点和时间错配边，并在未知起点/新系统保留；反之撤回方向性解释。 |
| Minimal Experiment | 64训练/16正常校准/32确认系统的冻结前向；I12自身0参数更新，三个故障成因分列。 |

原始权重、数据、逐系统结果不包含在公开仓库。来源数据、上下文闭式估计、额外计算均需披露；无目标反传，TimeRCD不测RCA。
