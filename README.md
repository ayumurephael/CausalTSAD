# CausalTSAD第一档E1–E4：上下文、相对读出与因果输入

本分支`diagnostic/improvement-tier1-v1`保留完整实验实现和结果说明。**288/288冻结推断任务、240/240目标四指标已完成并校验；全局权重更新0次。**后补说明与科学快照分开：本地科学commit e211ff9cd22bc528d8c13bd77ed4569936bb35e6，对应GitHub730c094beaa660ab4929e39a6353b6d03fbe6320，tree一致。

## 先看什么

1. [完整结果与通俗方法说明](docs/E1_E4_RESULTS.md)：数据范围、具体实现、正反结论、配对区间和成本。
2. [运行前协议与I01–I04卡](docs/IMPROVEMENT_TIER1.md)：固定条件、控制、种子和终点，未按结果重写。
3. [所有条件/分层](docs/E1_E4_ALL_CONDITIONS.md)、[逐系统正反结果](docs/E1_E4_PAIRED_CASES.md)。
4. 源码按[context_crossfit.py](proposals/context_crossfit.py) → [relative_context_scores.py](proposals/relative_context_scores.py) → [run_improvement_tier1.py](tools/run_improvement_tier1.py)阅读。

## 最重要的结果

80多变量开发系统，76个后缀有效异常、80正常FPR；3套冻结因果权重先在系统内平均。AP不是准确率；条件Top1已给真实事件窗口。百分数，FPR越低越好。

| 方法 | AP | VUS-PR | FPR↓ | 条件Top1 |
|---|---:|---:|---:|---:|
| 旧前缀＋因果残差 | 15.589 | 14.946 | 21.706 | 45.614 |
| E1交叉拟合＋因果残差 | 23.677 | 21.451 | 11.839 | 58.333 |
| E2＋E3周期/父历史与相对读出 | 32.185 | 28.706 | 3.439 | 57.895 |
| 同后缀原观测→自训练TimeRCD multi | 47.687 | 41.203 | 0.784 | 不评 |
| E4真图[x,r]→自训练multi | 50.010 | 44.743 | 1.837 | 不评 |

E1 AP配对+8.088pp有支持；[x,r]相对原multi AP+2.324pp的95%区间跨0，误报增加，留出轨几乎没提升，C8下降。保留因果输入的局部线索，但不能称稳定胜出或SOTA。TimeRCD遵照用户要求只评检测。

主输入仅观测/valid/正常DAG，真实系数、噪声、root、正常孪生不参与目标预测。允许父同刻；E1离线保护块外未来上下文、ACF和case统计均披露，不冒充严格在线。目标数据不反传、锁定150M不打开。下一源机制诊断与有限训练尚未在本分支实现。

历史C04s及其它proposal分别保留独立分支，继承文件不意味着本分支又运行那些实验。大数组、checkpoint和私有部署信息不在公共仓库；完整SHA审计与运行记录由主工作区保存。复现需要配置自己的路径和已有数据/权重，不能直接重启旧控制器。
