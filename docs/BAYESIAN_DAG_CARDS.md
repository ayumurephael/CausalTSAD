# 独立路线：DAG如何通过先验增强TSAD与RCA

用户已澄清：本路线研究DAG增强检测本身，**不以复用C04或保护C04收益为条件**。C04关系破坏监督另行保存和验证。两线最终在同一生成基准协议比较，可能整合但不预设。

## 核心理解

设H包含故障源r、故障类型m、起始时刻τ和传播时滞d。正常DAG规定各假设能够解释哪些后续证据。观测到当前数据e≤t后，

`p(H | e≤t,G) ∝ p(e≤t | H,G) p(H | G)`。

子通道在未来发生可观测偏离的先验，来自对这些假设的边缘化；未来观测到来后再更新。它可以升高，也会因为预期症状没有发生而降低。传感器读数异常不必影响物理系统；子通道受到上游故障影响也不意味着它自身是新的根因。DAG仅有拓扑不提供传播概率、时滞、故障率或正常/故障观测分布；这些量要从独立来源学习或作为明确的敏感性假设，不能从测试真值输入。

## B01 / C06v1 — 对时滞、故障类型与根因共同边缘化

| 字段 | 内容 |
|---|---|
| Problem | 多个局部证据弱且先后出现时，已知DAG能否提升检测和根因后验？看AP/AUC、正常FPR、Brier、条件根因Top1/Top3/MRR和未来8步症状预测。 |
| Evidence | **事实：**R17工业TNBN利用相对事件时间作诊断/预测；R16说明DBN的先验预测与似然更新及共享祖先相关。**推测：**跨新图边缘化未知时滞比固定同时刻图关联更有效。现有C04是否有效不作为本路线必要条件。 |
| Failure Mechanism | 仅把父异常概率加到孩子分数上，忽略自身证据、时滞不确定性和正常反证，也可能把相同祖先经过多条路径的证据重复计数。 |
| Key Insight | DAG应定义一组可竞争的时序解释，而不是强制每个子通道报警；对解释做后验推断可同时完成向前预测与向后定位。 |
| Novel Mechanism | 首版枚举“正常/单一持续故障源×物理或测量×onset×共享lag×噪声强度”假设，用规范Gaussian似然和源拟合参数逐时更新。同一故障假设统一解释所有节点，避免先把每条路径当独立证据再累加。零参数更新可用于新C/新G。**这是受限模型的可行性验证，不是新foundation model或一般精确DBN算法。** |
| Closest Prior Work | R17 Temporal Nodes Bayesian Network（UAI1999）、R16 Factored Frontier（UAI2001）已明确包含这类时序贝叶斯思想。我们不能以“Bayes+DAG”本身宣称新颖；未来实质增量需证明跨系统共享、时滞不确定、混合故障与原始TS证据学习。 |
| Falsifiable Prediction | 真图延迟混合模型应超过无图、同时传播、反图/错图；对同样的根传感器扰动，观测到下游响应与未响应应改变物理/测量后验；无信息似然必须退化为先验，改变未来数据不能改过去后验。 |
| Minimal Experiment | 3个独立来源拟合seed×128个新图系统（C8/12/16/20）×8case；每组4normal/2measurement/2physical，共同噪声下两类故障根观测严格一致。7个模型控制：independent/instant/physical_only/dag_delay_mixture/reverse/wrong/oracle_lag。最后一个明确为增强oracle，不参加主结论。源拟合64图/seed仅正常样本与实际注入根窗口，0affected标签。 |

## B02 — 物理传播与观测故障的竞争解释

| 字段 | 内容 |
|---|---|
| Problem | 父传感器坏了但系统没坏时，阻止错误向下游广播，提高正常FPR、根因Top1和故障类型校准。 |
| Evidence | **事实：**R17把外生事件与后续状态变化分开；已有C05卡片记录坏父观测可能污染条件残差。**推测：**显式两层潜状态比单一“异常”状态更可靠。 |
| Failure Mechanism | 把测量异常当物理状态异常，会预测不存在的下游故障；把传播症状当新根因，会错定位到幅度最大节点。 |
| Key Insight | 先问“系统状态变了，还是读数变了”，再决定是否沿G传播概率。 |
| Novel Mechanism | 完整方案分别建模物理状态z、观测x及source故障m，正常G约束z的转移，测量故障只改变观测似然。v1仅在B01有限假设中加入measurement/physical竞争，对照physical_only；**尚未实现一般连续潜状态模型**，后续独立branch。 |
| Closest Prior Work | 状态空间模型/动态BN早已有状态与观测分离；R17及C05是接口先例。可信增量需跨图共享和未知故障类型下零样本定位，不是简单增加模式变量。 |
| Falsifiable Prediction | 根读数完全匹配的measurement/physical组应有不同模式后验；只给父报警、不见预期下游变化时，measurement解释提高，而不是强制子报警。 |
| Minimal Experiment | B01相同噪声配对，比较有/无measurement假设；报告类型后验与测量案例误报，不能只看总体AP。原三类通道影响标签仍null。 |

## B03 — 共享的可学习传播核与稀疏事件后验

| 字段 | 内容 |
|---|---|
| Problem | 原始时序存在未知lag、恢复、反馈、符号/幅度变化；有限枚举不能直接扩到长序列foundation任务。需测试未知C、图、机制和长序列的性能/成本。 |
| Evidence | **事实：**R16指出即使图稀疏，历史共同原因仍使精确联合状态指数增长；FF因子化可加速但丢依赖。**推测：**共享事件核+少量高概率根事件可兼顾速度和表达。 |
| Failure Mechanism | 固定lag或粗暴全因子化会错配先后关系或过度自信；把当前判别后验再次作为似然还会重复使用先验/历史证据。 |
| Key Insight | 学习的是跨系统可复用的时间转移规则与条件观测似然，不是每张目标图的参数。 |
| Novel Mechanism | 源训练共享normal/fault emission和时滞/恢复核，测试只做前向条件化与消息更新；小图精确枚举校验、一般图粒子/结构化近似，单独记录近似误差/额外计算。当前仅proposal，不把有限v1称该完整实现。 |
| Closest Prior Work | R16 FF/BK/LBP与R17事件化时间节点均是直接先例；CNP/ANP可作共享条件机制。未来需要实测跨系统原始TSAD/RCA增益来支撑贡献。 |
| Falsifiable Prediction | 换错图/时间打乱应损害校准与定位；多路径diamond图的后验不应比小图精确参考无端自信；target无梯度仍能跨C工作。 |
| Minimal Experiment | B01通过后，再建独立原始SCM训练池，估计局部观测密度与共享核；graph/empty/reverse/wrong、fixed/learned lag三seed，同容量/曝光控制，保留正常反证与测量故障。先小训练，不按250M locked调参。 |

## 首轮v1边界、预算与停止

- v1输入是**人为生成的Gaussian证据流**，不是原250M原始TS；故障持续至窗末、单一根、所有边共享一个潜在lag、传播为确定性首次到达。这个受限模型中假设边缘化是精确的，不能说已经解决工业SCM或foundation任务。
- T192，源64图×3拟合seed；新评估128图、每图8case、3fit seed，7模型控制。192个源图独立于128评估图；64位namespace随机种子与原32位训练数据不同，所有新seed记录，不读取旧benchmark数组。
- Gaussian正常/故障方差只从独立源normal与已知注入根窗口拟合。模式/故障率与onset经验先验经固定Dirichlet平滑，lag先验均匀[1,2,4,8]，强度先验均匀两源成分；不是输入每个新系统真实噪声/lag/coeff。actual lag仅oracle_lag单列。没有target拟合/梯度、没有C04依赖。
- 每次预测只用e≤t，future8为真正先验预测。oracle_activation仅概率仿真审计未来Brier，不进入拟合/推理，不转成effect/anomaly_channel/anomaly_time标签。这三项仍null；一维label按直接故障开始置1。
- 原TimeRCD和C04不加载/不改；CPU源MLE是封闭统计估计，不伪称深度预训练；GPU只加速大量假设边缘化。每worker≤1.5GiB/2h、阶段4h；记录实际输入重复曝光与hypothesis×time单元，不能只报原数据点掩盖额外枚举计算。
- 预检必须通过：标量枚举参考、未来值不影响过去、G/X共同置换等变、无信息似然等于先验、root归一化。失败保留错误，停止新增本项目任务，不影响他人。
- 正信号门槛是相对无图/错图/instant的系统配对证据、测量负例不恶化和概率校准；不以任意全组AUC≥.85否定局部机制。若只有oracle真实lag有效，下一步问题是lag学习，不宣布topology-only成功。即便v1全通过，也只获得原始TS小训练资格。

## 完整阅读及作者代码

R16全文8页、R17全文8页逐页内存阅读；R16页3/5、R17页5/6已渲染复核，无TXT。R17用专家结构、模拟器参数与时间区间，800/200记录验证；不是跨系统foundation方法。它将时间事件（含不改变）相对父事件条件化，直接支持提出这个方向，但不是我们的创新证明。

作者BNT固定commit bebba5f437b4e1e29169f0f3669df59fb5392e62。完整读取ff_inf_engine.m、filter_evidence.m、noisyORtoTable.m：先验乘父边缘→边缘化→观测似然→归一化；该FF代码限制观测leaf与前一slice隐藏父，不是任意DBN精确推断。noisy-OR包含inhibition/leak。没有执行/复制第三方Matlab算子，本v1使用独立有限假设算法，不能叫FF完整复现。

[R16全文](https://arxiv.org/pdf/1301.2296)、[R17全文](https://arxiv.org/pdf/1301.6675)、[作者工具箱](https://github.com/bayesnet/bnt/tree/bebba5f437b4e1e29169f0f3669df59fb5392e62)。
