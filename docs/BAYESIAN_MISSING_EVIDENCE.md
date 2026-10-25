# B01-v2：DAG是否能利用下游证据弥补看不见的根因？

本研究独立于C04关系监督。不预设它必须修复C04。上一版受控证据实验AP .93383 vs独立 .93431，差CI跨0；条件root Top1 .97656 vs .76823，未来8步latent Brier .00954 vs .03051。高定位不等于检测改善，也不等于原250M rawTSAD性能。

| Research Insight Card | 内容 |
|---|---|
| Problem | 如果根因传感器不可用，正常DAG能否把下游的弱证据汇合为故障后验？分别报告检测AP、正常FPR、条件根因Top1/MRR和8步latent Brier。 |
| Evidence | v1根因总可见；不用图已可直接检测，AP没有真图收益。DAG显著影响定位，但数据本身按该图传播，不能当真实系统证据。 |
| Failure Mechanism | 候选解释：强根因自证据使传播先验对“是否异常”近乎冗余。反之，传感器测量故障并不沿物理图传播；强制传播会降低似然、误定位或误报。 |
| Key Insight | DAG是缺失证据下的时空约束，不是把父异常机械广播到子。必须让“物理传播”和“局部测量故障”竞争，并允许正常假设。 |
| Novel Mechanism | 沿用有限Bayesian假设模型；独立可用性mask把未观测项的log-likelihood ratio置0，隐状态仍可传播。未知根、起点、模式、lag、方差成分边缘化。成熟贝叶斯缺失处理不宣称原创。 |
| Closest Prior Work | 全文R16 Factored Frontier/UAI2001与R17 Temporal Bayesian Networks/UAI1999；作者BNT核心已读。此处不是FF代码实现或任意DBN精确算法，是已限定单永久故障模型的有限枚举。 |
| Falsifiable Prediction | 根因缺失且至少一个后代可见时，真图应改善physical AP/定位，超过独立、反图、错图。根因缺失的纯测量故障没有任何变化可观测，不能因图而虚构成功。若仅root排名改善而AP不改善，就删“提升检测”结论。 |
| Minimal Experiment | 固定128独立图×3来源fit重复×7对照，沿v1同一eval系统，新增每系统随机半通道不可见。mask RNG与root/graph/label独立；分组统计root可见/不可见及可见后代数，不挑正例。 |

正常/测量/物理八case和root观测配对均不变。7方法：independent、instant、physical_only、dag_delay_mixture（主）、reverse、wrong、oracle_lag（独立增强）。wrong是固定通道置换图，不叫逐节点degree匹配。原fit参数JSON逐字节复用，零新增估计/SGD，不用目标normal重校准，阈值固定后验.5。full-observed结果复用v1，不重跑。

预算：57,802,752分配输入标量slots、28,901,376实际可见证据标量前向；无优化，hypothesis×time计算另记，不冒称与无图严格FLOPs相等。每worker1.5GiB/2h、stage4h，最多4A100+6H100共享，实际显存检查；不停止他人。G0包括与原无mask输出逐位相等、all-valid逐位相等、缺失位置任意污染不变、未来隔离、通道置换。输入/图/mask/标签审计/分数SHA在实际工作时记录，不继承v1遗漏输入SHA的记录方式。

主支持检验是root不可见且可见后代>0的physical AP相对independent配对系统bootstrap CI下界>0，且正常FPR增加≤.01；反/错图用于特异性、RCA另报。若AP不通过不扩大为rawTSAD检测方案，但保留定位/预测的局部证据。128是独立系统，3fit重复不是384独立系统。不给full组AP>=.85机械门槛。假设misspecification（恢复/多根/异质edge lag）未覆盖，后续需独立反例，不称工业SOTA或foundation模型。

250M全部数组、原25亿与C04/TimeRCD权重均不读取；labels_detail三类仍null。oracle latent只算前瞻Brier审计，不作为模型输入或新增阈值标签；lag仅oracle_lag分支传入。

技术revision r1：初版两节点均在G0停止、未启动评估worker。相同输入重复实验score误差0，root误差2.55e−15；缺失位置污染score误差0、root2.11e−15。原duplicate-index CUDA index_add的浮点累加顺序不能逐位复现，改为按meshgrid中连续root块归约。原source/log/诊断保留，不放宽逐位mask检查；新root=r1。完整CUDA逐操作数学确定性不宣称已经证明。
