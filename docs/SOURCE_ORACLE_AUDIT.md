# C04q：边变化究竟有多少可检测信号（仅oracle审计）

| Research Insight Card | 内容 |
|---|---|
| Problem | C04o父特征头在32源系统的边变化任务未胜能量/非父。区分信号过弱、正常机制估计不准、证据头接口不合适；不把单个失败头解释为图无用。 |
| Evidence | 边变化AP差parent−energy −0.082pp，区间[−0.298,+0.139]，parent−nonparent −0.006pp，均无稳定优势。源系数扰动仅25%，很多短窗口可能低信噪比。R23 FANS全文及作者fans.py/模块已读：其判据依赖正确的正常噪声映射，论文使用每系统40k正常训练，与我们224点闭式上下文不同。 |
| Failure Mechanism | 弱机制变化在极短窗口可能无足够统计证据；错误正常映射本身也制造父—残差依赖；9lag×多父的边缘似然可能被复杂度惩罚稀释。三者需要区别，不能只降低成功阈值。 |
| Key Insight | 先问“知道真正正常机制时是否能看见故障”，再问学习/近似推断损失了多少信号。 |
| Novel Mechanism | 明确privileged oracle：用真实rho、正常coeff/lag/noise得到边故障的真实正常创新，比较其检测排序、残差—父特征依赖与原估计。另给实际故障root/edge/扰动的条件LR上界，**对全部查询时间算LR**，真实区间只用于评估/KL。 |
| Closest Prior Work | R23 FANS的功能/噪声分离、R8外生创新、经典Gaussian已知备择LR。不是新算法，不迁移FANS定理保证；i.i.d/正确SCM/可逆性/支持重叠在时序和误设模型中不自动满足。 |
| Falsifiable Prediction | 若oracle仍近随机且累计条件KL小，应把部分失败归为任务弱信号；oracle明显好而估计差，支持正常机制误差；oracle LR好而oracle能量差，支持读出/备择接口不足。各情况均不直接证明因果图足够。 |
| Minimal Experiment | 只重用C04n的32源系统/640边case与1152缓存中的96真图估计缓存，0新骨干前向、0训练、0目标数组。CPU每节点最多一个主进程（向量运算），与C04p并行。条件窗口dcorr描述统计，不用i.i.d置换p值冒充时序保证。 |

真实系数、lag、noise、root和实际异常参数仅在本oracle审计出现；输出不进入C04p特征、训练loss、阈值或主目标模型。条件已知root/edge LR不能称未知根因RCA，未评TimeRCD RCA。lag变化的oracle依赖特征含真实旧/新lag，单独标明增强。源测量故障的观测值不等于物理状态，本轮oracle只评边故障，避免把坏父观测当真实状态噪声。

预算预先限定：oracle正常预测6,938,624查询标量（每系统22view）、复用估计20,815,872 slot应用，另计dcorr矩阵元素和LR读出应用；无额外真实数据曝光。所有case/SHA/原计划保留，不覆盖旧原型。4h技术期限，出现真实错误保留日志，新revision修复；不停止他人。
