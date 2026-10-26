# C04n：短故障的分数是否被时间平滑稀释

| Research Insight Card | 内容 |
|---|---|
| Problem | 给正常DAG的跨系统TSAD+条件RCA；目标生成开发80系统出现短尖峰检测远差于TimeRCD。固定正常预测，检验时间读出是否造成部分失败；AP/FPR/条件Top1为诊断指标。 |
| Evidence | 完成的C04l-T真图normal_only AP14.36%、VUS13.39%、Top1 30.70%；TimeRCD同点集AP48.46%、VUS42.31%。一个downward spike的AP为6.50%，而TimeRCD为100%；实现固定尾随平均16点。事实是差距和平均算子，稀释是待证假说。 |
| Failure Mechanism | 单点创新经过尾随平均可能让随后15个正常点也得高分，损害精确定位；长事件又可能需要累积证据。源旧故障覆盖整个704点查询，未能检验这种差异。 |
| Key Insight | 先区分“因果预测不准”与“预测残差转分数时把时间支持弄错”，不能把原始创新稀疏的合理性等同于最终检测区间合理性。 |
| Novel Mechanism | 不新训练/换正常均值，仅并列单点能量与原尾随16点；源故障按1/4/16/64/256持续时间、measurement/physical/系数/lag改变分组。不是adaptive patch或新foundation架构。 |
| Closest Prior Work | 已全文读R8 AERCA明确区分外生干预发生时间和下游传播时间；作者models/aerca.py、utils已读并回看。原创新定位、滑窗证据是成熟方法；当前只检验源任务长度与目标的接口错配，无原创性宣称。 |
| Falsifiable Prediction | 源短故障point相对trailing16应改善AP，而长关系变化可能相反；若只在单个目标好看、源短故障无优势，则不支持稀释解释。若AP改善但正常FPR仍差，则正常建模失败未解。 |
| Minimal Experiment | 32独立源系统×42case、3冻结normal_only种子、真/空/错图；目标仅原已探索80系统，同点集t>=320，TimeRCD缓存复用。16旧源cal×两背景×两正常固定99%阈值规则；无新目标选参。新源32系统不训练，也不是locked。 |

源新namespace348004000000，C8/12/16/20各8系统，家族linear/tanh/sin。观测故障不传播，物理故障以创新脉冲进入SCM后继续自然传播；系数/lag改变只在注入区间发生。**一维标签为直接注入时间，尾部传播不私自扩成effect标签**，真实根只用于条件RCA。测量与物理的根波形没有强制完全匹配。prefix沿用25%污染概率，R不保证正常。builder保存参数，但模型只见cases/正常graph。

本轮固定全部预算：新source173,408,256完整输入slots、旧sourcecal8,257,536、目标78,172,128，共259,837,920冻结前向；0优化/反传，额外技术预检独立记录。与C04o共用真图残差，无额外骨干前向；C04o头只评source，不评目标。原trailing16目标分数逐点对旧缓存<=1e-8是G0。source和target按同一系统内3seed平均，保留单种子结果、时长/家族/C/正反例。

G0：旧头等价、权重不变、无目标参数更新/信息泄漏/非有限数。G1：新来源短事件duration<=4的配对AP改善均值>=2pp、95%系统bootstrap下界>0，且源正常FPR恶化不超过2pp，取得独立有限训练资格；较长事件与全部失败必须一起报告，不要求全方向都AUC>.85。没通过则停止当前point原型；不会据此否定因果整体方向。4h技术期限，10GPU按空余显存>=3GiB共享，每worker上限2GiB；指标本轮AP/AUC/FPR/条件RCA，暂不重复昂贵原四指标。

原TimeRCD只做TSAD，无RCA。80目标被探索多次，不是新盲测；150Mlocked不开。原source32与全部250M数组不反传。输出分数、残差/方差cache及SHA保留，未来只换头优先复用cache，避免重复GPU。
