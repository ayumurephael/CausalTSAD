# I14 低维时间背景与结构方向控制

本分支实现已落地，实验是否完成以项目根目录STATUS和实际回执为准；此处不宣称性能已通过。

分支：`proposal/i14-context-time-basis-v1`。共同引擎继承I12，但本分支只承担下列实验，不修改旧proposal分支。

[完整覆盖、输入边界、预算与停止规则](docs/IMPROVEMENT3_PROTOCOL.md)。主要代码在 `proposals/improvement3/`；来源入口 `tools/run_improvement3.py`；目标入口 `tools/transfer_improvement3.py` 与 `tools/metrics_improvement3.py`。

训练臂：本分支无全局参数训练。

| Research Insight Card | 内容 |
|---|---|
| Problem | 合法慢背景下关系检测和定位仍弱，需要不使用目标标签的背景接口。 |
| Evidence | I11 smooth关系AP53.36/Top120.83；代理同样较弱，尚不能证明唯一原因。 |
| Failure Mechanism | 固定正常机制无法表示加性慢背景，但去背景也可能移除持续故障。 |
| Key Insight | 背景基必须在受保护上下文拟合，并检验它与因果设计的混淆。 |
| Novel Mechanism | 四个傅里叶列；普通追加与上下文上对合法设计残差化两版本，所有输入仍无真实系数/噪声。 |
| Closest Prior Work | C04趋势商反证、ALPaCA上下文回归；增加傅里叶基本身不新。 |
| Falsifiable Prediction | smooth改善同时plain保持；若plain下降>2pp或边故障被吸收则该固定接口不通过。 |
| Minimal Experiment | 与无时间基共享来源系统、冻结基函数、扫描规则和正常阈值池；0全局训练。 |

原始权重、数据、逐系统结果不包含在公开仓库。来源数据、上下文闭式估计、额外计算均需披露；无目标反传，TimeRCD不测RCA。


## r2：事件指标 JSON 保存修复

2026-10-27：r1在初始评估保存时遇到NumPy整数无法写入JSON，尚未执行优化步。r2仅把事件边界转为Python整数，并加入严格JSON回读检查；模型、损失、指标数值和固定预算不变。旧快照/错误保留，已完成缓存先核验SHA再复用，原技术截止时间不延长。本修复不构成新的性能结果。
