# C04h / B03-small：原始时序的因果误差模式与竞争故障解释

| Research Insight Card | 内容 |
|---|---|
| Problem | 根观测完全相同，测量错误不改变物理后代、过程扰动会改变后代；能否依据正常DAG和原始观测同时检测与定位？看matched measurement quartet AP、physical quartet AP、源正常FPR、条件rootTop1/MRR；整体8case AP单列。 |
| Evidence | C04匹配源任务真图AP .59286，证明关系监督有局部用途；迁移失败。C04f局部更新保护仅恢复.00401 AP，C04g去趋势仅.50745。独立Bayes实验只在物理/缺失证据局部改善，说明需读原始条件关系，不能只广播父异常。 |
| Failure Mechanism | 候选：单看残差幅度，测量坏父节点也会制造孩子预测误差；过程故障则可由正常传播解释孩子。把两类误差混在一个异常分数里，可能既丢检测，也错把症状当根因。 |
| Key Insight | 在线性近似下，A(L)=I−B(L)。过程故障f给残差u+f；测量错误m给u+A(L)m。DAG与前缀条件机制给出不同的时滞“误差指纹”，通过竞争解释而非最大残差定位。 |
| Novel Mechanism | 仅从每条观测前256点前向推断局部ridge-Gaussian机制后验近似（3次固定robust重加权、无网络参数更新）；只给正常拓扑，不给真实lag/coeff/noise。16点因果窗口中比较过程/测量×3方差强度的Gaussian协方差比，用源匹配关系监督学共享6混合权重+1bias，根因/检测来自同一假设后验。 |
| Closest Prior Work | R19 Field Kalman2017全文11页：已知模型/故障目录的有限概率滤波、log-odds稳定化；R17 DBN时间诊断，R8 AERCA创新；成熟故障指纹/BLR/协方差检验不称原创。未找到R19作者公开算法源码，不冒称复现其滤波器。可能的贡献须由跨系统前向条件化+边际匹配关系监督+测量/过程区分的TSAD/RCA结果建立。 |
| Falsifiable Prediction | 真实graph-signature应优于相同true预测器的unit-signature，以及degree匹配wrong与empty；若仅normal NLL改善或unit同样好，则不能称指纹有用。物理/测量根观测相同，模式后验应因非根证据不同而改变；误差足够小时应承认不可辨识。 |
| Minimal Experiment | 固定32train/16sourcecal/32sourceval系统，来自原C04独立来源池各C的前8/4/8个；不读250M。原2normal+2matched测量，再加2测量+2物理模式对：新根为另一正常根轨迹×1.5、两种模式的根观测逐位相等；physical按同SCM、同噪声传播，真参数只在builder内。5证据对照×3个7参数prior初始化seed，32updates固定一遍。 |

**范围。**这是新的原始时序小基线，不是完整foundation训练，也不强制把独立Bayes路线接到C04旧模型。原C04权重完全保留，不加载/覆盖。前缀R可能受污染，固定3次Student权重只是robust近似，不能假定R干净或声称精确Bayesian回归。Gaussian协方差比在给定对角噪声/指纹时精确；真实时间/节点预测误差相关、参数后验完整传播、非线性局部化均近似。不是R19完整FKF或未知故障目录已解。

模型输入严格白名单(X,G)，输出含query>=320；normal predictor不读自己当前或未来，父同刻分支已获授权。固定基底own lag1..8、父raw/tanh lag0..8、intercept/time；不传family标签，不含sin基底，sourceval sin未直接给公式但已经在历史实验中探索，不能叫全新盲测。prefix后局部系数latent state不再更新，全球7参数只source训练。测试前向条件化属于冻结模型推断，不是针对测试重训。模型不接收root/annotation/normal twin/audit参数。

**预算与控制。**五证据：true_signature / true_self / true_unit / wrong_signature / empty_signature。true_self保留自身AR测量指纹、去掉跨节点指纹，排除收益仅来自单通道时间结构。wrong保留每节点入/出度，可能有环；对照是局部残差空间近似，不能宣称全部wrong图是合法DAG生成密度。unit模式混合有重复成分，名义参数匹配、有效信息容量不等。每SCM共享前缀只按输入内容SHA复用，不用正常孪生修复。缓存26,808,320 ready-query等效标量、38,993,920 full-input槽位；额外prefix线性求解/矩阵分解计数单列。3×5个prior一遍32updates，共37,847,040缓存query等效曝光；不把缓存重复曝光叫新原始数据。先技术预检/独立Woodbury-vs-dense/未来隔离/当前目标盲/通道置换/归一化/参数不变，再跑pilot，8系统生成复核另计。worker2GiB、stage4h、先到停止，所有文件SHA/seed/源曝光记录，不影响他人。

源cal只normal2case选99%阈值，sourceval完整8case（3/4正例）与原matched quartet/new measurement quartet/physical quartet分开，不能和旧C04 1/2 paired AP混比。学习loss按normal/matched故障/new模式故障=.5/.25/.25加权，root监督只fault。若physical仅换一条同分布正常源根轨迹，会有分布不可辨识的“假故障”；因此新模式对加1.5幅度变化，同时明确它不是精确边际匹配，原matched4view独立保留。先比较原measurement quartet的graph/self/unit控制，再看physical、calibration、负例和sin；不以全组AUC>=.85一票否决。局部反证不通过不扩大缓存或加步；通过后才拟定原始benchmark开发评估，锁定150M关闭。所有effect/anomaly_channel/anomaly_time仍null。
