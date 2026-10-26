# C04r：不知道故障边和系数时，稀疏因果备择是否仍有用

| Research Insight Card | 内容 |
|---|---|
| Problem | 正常DAG已知、故障位置/参数未知，能否识别一条边关系改变？主要看边故障AP与条件Top1，正常FPR为必要对照。 |
| Evidence | C04o全父×9lag Gaussian备择未胜能量/非父。C04q同已知根与16点窗口后，256点系数实际备择LR仍比能量高23.139pp，但该上界知道真实故障边/系数，不能部署。短故障KL弱，不能要求全部高分。 |
| Failure Mechanism | dense备择同时给许多父/lag方向分配变化先验，可能为真实单边变化付出多余复杂度惩罚；自身历史残差化可能移走一部分变化信号。这些是候选解释。 |
| Key Insight | 正常图不只约束正常预测，也能缩小“哪些机制可能变了”的备择空间；未知系数应积分，未知边/lag应混合，而不是读取生成真值。 |
| Novel Mechanism | 每条合法父边×lag0..8的单特征rank1 Gaussian系数扰动，tau²=.0025/.04/.64均匀积分，节点内归一化候选数；与同窗口方差膨胀能量按.5/.5混合、根先验1/C。均为working diagonal-noise evidence，不称真实已校准因果后验。 |
| Closest Prior Work | R23 FANS全文/作者核心已读，残差—父依赖可揭示函数机制变化，但它每目标拟合与环境样本设定不同；R22 ALPaCA全文/核心已读，权重空间Gaussian积分是成熟接口。此处没有新Bayes公式或首次DAG异常检测声明。 |
| Falsifiable Prediction | 相同partial特征/W16下rank1应胜旧dense；真实父应胜同数量非父。去掉own残差化若改善，不能只归功于稀疏先验；W64/256必须与相同窗口能量比。非父同样好、只靠更长窗口或正常误报增加，均反驳因果收益解释。 |
| Minimal Experiment | 只复用C04n32来源/16校准系统的3seed true创新缓存，0新骨干、0优化、0目标。11新读出：energy16/64/256、parent/nonparent partial16、parent/nonparent raw16/64/256；旧dense parent/nonparent16缓存原样作配对。2h技术期限，固定配方，不按目标挑窗口。 |

rank1解析logBF=`.5*(tau² U²/(1+tau² S)-log(1+tau² S))`，其中U为窗口内Σ(rφ/v)，S为Σ(φ²/v)。给定同一残差和方差，分子不是实际故障参数；候选φ只从本case观测和正常拓扑构造。partial严格沿用C04o前256点own过去回归残差和缩放，raw只作前缀稳健标准化。父同刻值允许，lag均匀候选而非真实lag。非父按旧key/node固定抽样，包含可能后代，是关系特异性控制，不称可用因果父。

主机制对照为线性族长系数故障(duration>=64)的父−非父、rank1 partial16−dense partial16；全部32系统/三族/四故障/五持续时间都报告。来源已探索，不冒称确认盲测。局部增益系统区间>0且FPR恶化<=2pp可支持后续独立来源学习；不同C/族和所有下降例保留，不因总体负例直接否定局部方向。未取得局部支持不做该固定头目标迁移，已知备择上界不计入可用方案排名。

G0：rank1单特征等价dense单特征<=1e-9、energy16回放旧缓存<=1e-7、zero特征BF0、未来扰动不改变此前证据；正常均值/方差完全复用不更新。每系统输入/图/缓存SHA核验，分数/每候选计算量/运行时间保存。统计单元是系统，三seed先系统内平均，条件RCA窗口只用于指标。

250M任何数组不读取；原25亿/旧模型/结果不改。source labels仍为直接注入时间，不生成effect/anomaly_channel/anomaly_time。图、root、系数等审计记录不进入模型接口；无新数据生成/重放，仅观测特征重建和缓存读出计算。
