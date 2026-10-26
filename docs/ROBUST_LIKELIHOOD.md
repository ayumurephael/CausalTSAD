# C04k：误差过大时，是否只是Gaussian似然过度自信？

| Research Insight Card | 内容 |
|---|---|
| Problem | 同一固定因果正常预测器在纯正常生成case上大量报警。目标是检验分布尾部失配，指标AP/AUC、正常FPR、条件RCA，以及已知持续故障是否仍可辨。 |
| Evidence | C04h生成80多变量AP11.79%/正常FPR50.23%，原机制轨FPR82.43%；来源同方差diag控制并不退化。正常误差失准是实测，尾部过窄只是可能解释，不作为既定原因。 |
| Failure Mechanism | 正常误差稍偏离固定Gaussian时，大误差的平方项可能给故障假设过高证据；但也可能是均值完全错误，换尾部无济于事，甚至隐藏真实故障。 |
| Key Insight | 保持正常均值、预测协方差、图、故障模板和冻结7参数完全一致，只给正常/故障两解释相同的径向尺度不确定性，隔离“结构失配”和“尾部过自信”。 |
| Novel Mechanism | 使用协方差K参数化的joint Student-t，固定nu4；正常K=D，故障K=D+beta MMᵀ，Mahalanobis分母nu-2，保持两模型与原Gaussian各自协方差相同。通过Woodbury和正值稳定二次型求密度比。不是把scale错当variance，也不是独立每点t噪声。 |
| Closest Prior Work | R20 Shah/Wilson/Ghahramani2014全文9＋补充4页，主p3/4视觉复核。使用其成熟协方差参数化/尺度混合理解；不实现完整TP核学习、预测过程或IW过程，不称完整复现。作者站/会议页未找到算法源码；对本有限公式做独立dense参考。R19故障竞争已读，旧source先验来自C04h。 |
| Falsifiable Prediction | 如果尾部过自信是关键，固定均值不变时应改善独立来源正常背景下校准，并保留matched关系故障/根因。如果只压低所有证据、重新校准后无收益，或本来均值错误未改善，则此原型不支持扩大；共享径向t还可能让一个大误差遮蔽其它节点，保留负例。 |
| Minimal Experiment | 与C04j共同冻结48来源校准/验证系统×原样/平滑背景×真图/空图、80生成多变量诊断×3冻结head；不新增SGD或手调df网格。dense t、同协方差、nu→∞Gaussian极限、null、巨大误差有限性、未来/自身当前隔离和C20先预检。预算/数据/阈值/停止同本轮固定计划，代码独立分支保存。 |

本原型的背景与目标配置由固定协议给出，目标标签不用于选择nu、窗口或checkpoint；250M全部不反传，validation/locked仍封存。7参数来自原Gaussian来源训练并冻结，若接口有证据，后续另做来源训练，不能把本轮称新预训练。含每case前256点闭式潜参数推断。

[原始论文](https://proceedings.mlr.press/v33/shah14.html)。主文含旧隐藏文本层，视觉复核以显示的主文Eq5/6为准；补充条件均值符号与主文有差异，不照抄。R21邻近全文也已读；本分支不实现BOCPD。不要以Bayes+t+DAG本身声称首创，原始生成TSAD/RCA增益仍待实测。
