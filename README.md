# I15 最终统计量监督与故障签名定位

本分支实现已落地，实验是否完成以项目根目录STATUS和实际回执为准；此处不宣称性能已通过。

分支：`proposal/i15-segment-signature-supervision-v1`。共同引擎继承I12，但本分支只承担下列实验，不修改旧proposal分支。

[完整覆盖、输入边界、预算与停止规则](docs/IMPROVEMENT3_PROTOCOL.md)。主要代码在 `proposals/improvement3/`；来源入口 `tools/run_improvement3.py`；目标入口 `tools/transfer_improvement3.py` 与 `tools/metrics_improvement3.py`。

训练臂：stat_pair, stat_temporal, stat_typed。

| Research Insight Card | 内容 |
|---|---|
| Problem | 关系case分类改善不等于找到故障时刻；单一节点最大残差也可能混淆传播与起因。 |
| Evidence | I11关系改善伴随短故障/长定位下降；外部建议提供边级时间签名假说。 |
| Failure Mechanism | 跨视图排序可走整条case捷径；不同故障成因对节点及入出边的作用不同。 |
| Key Insight | 监督实际部署的时间统计，并让根因解释故障产生的联合模式。 |
| Novel Mechanism | 同容量pair vs pair+within-view；正常NLL0.05正则；独立故障类型/RCA头含无解释类，不回写检测读出。 |
| Closest Prior Work | C04s私有头、CIRCA、Chen–Liu；故障签名须以跨类型控制验证，不能靠名称拼接。 |
| Falsifiable Prediction | within-view应改善同视图AP及起点/漏检组合；typed应在相同检测下提高定位，不能仅靠root入度先验。 |
| Minimal Experiment | 三臂×3seed×真错图；384步、固定最终权重，报告条件Top1及报警后联合成功率。 |

原始权重、数据、逐系统结果不包含在公开仓库。来源数据、上下文闭式估计、额外计算均需披露；无目标反传，TimeRCD不测RCA。
