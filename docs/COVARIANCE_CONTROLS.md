# C04i：固定边际的协方差机制反证

| Research Insight Card | 内容 |
|---|---|
| Problem | C04h真图来源AP .89151、条件rootTop1 .89063，收益来自跨通道相关，还是只来自孩子误差方差/自身时间结构？只在同一80来源系统上隔离。 |
| Evidence | full比true_self matched AP +.01711，CI[.00570,.03021]；但两个模型的孩子边际方差也不同。原七参数训练几乎没改善AP，不把收益归因额外训练。 |
| Failure Mechanism | 可能误归因：图告诉模型哪些孩子误差变大，即使不使用相关也能定位。仅full/self/unit/wrong/empty不能证明跨通道相关必要性。 |
| Key Insight | 全协方差→按通道分块→纯对角；前两者保留每条通道整个时间窗的边际Gaussian分布，三者逐点方差相同，逐层拆分跨通道与单通道时间证据。 |
| Novel Mechanism | 原H的Sigma=D+beta MMᵀ不变，分别只清除跨通道块，或清除全部非对角。正常预测器、M、强度、数据划分与7参数prior相同。成熟Gaussian控制不是新算法。 |
| Closest Prior Work | 已全文R19 covariance likelihood竞争、R15背景/结构消除。属于C04h主张的反证，不冒称新FKF。 |
| Falsifiable Prediction | full稳定优于channel-block才支持跨通道相关；channel-block仅优于diagonal支持自身时间结构；若三者相近，保留正常图条件预测/边际残差模式解释。过程独立创新的三种协方差应相同，是null。 |
| Minimal Experiment | 新增true_diagonal/true_channelblock两对照，各3seed固定32updates。复用原80系统源文件与五模式既有结果，不重算full。先独立dense参考、边际保持、原full路径等价预检。 |

source仍32train/16cal/32val，原matched quartet与新模式对分别评。新增10,723,328 query-equivalent raw前向/15,597,568完整输入slots；15,138,816缓存优化query曝光，34,062,336缓存评价曝光。diagonal不做Gaussian Cholesky，channel-block按C个16维边际做，额外矩阵分解计数分开；不声称端到端相同FLOPs。worker1GiB、stage4h，0目标数组/梯度、locked关闭。全球prior可学习只在source，局部前缀机制为冻结程序的闭式条件化。全部旧代码/权重/结果保留，两个新对照独立branch。
