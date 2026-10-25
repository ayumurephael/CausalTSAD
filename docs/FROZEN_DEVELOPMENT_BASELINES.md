# 官方与自训练TimeRCD：合成开发集TSAD基线

用户明确确认：全部development，单列已探索100系统；不评原版TimeRCD的RCA。完整开发集540底层系统/1080正常与含异常case，恰50,000,000观测标量，两轨、C1/8/12/16/20各108系统。validation/locked数组不打开，0优化，不生成新数据/新影响标签。

两个原生检测器：官方checkpoint与自训练checkpoint。C1用各自uni，C>1用各自multi；另报C1的common-multi附表，专门与此前因果pilot保护分支对齐，不能把uni/multi偷换成同一行。加载checkpointSHA绑定，原Transformer/全注意力/ChannelMean原封不动，无图/根因/事件窗输入，不计算逐通道RCA排名。

## 两个明确的数值口径

1. **native**：保留生成数据的存储精度做原版全序列逐通道zscore，再转Torch FP32、5000窗口与末窗复制padding；与上游zero_shot等价。
2. **causal_compat**：先转FP32再做相同zscore，匹配此前因果pilot的model_inputs。两者可能存在舍入差，不能把它当模型改进。

每个口径同时保存原softmax异常概率与未饱和raw logit（先ChannelMean再取class1−class0），来自同一次前向，不重新跑网络来换评分表示。原四指标/辅助AP、ROC-AUC分别计算，绝不把概率与logit混比。周期推断也各按该输入口径沿原find_length_rank；不按标签调period。只有归一化后的FP32输入逐字节相同才能复用另一口径前向，实际复用和曝光全部登记。

正常FPR两种：固定原概率0.5（logit0）诊断；独立源normal-validation 99%分位数主口径。源390C1/123M系统，共513，source normal原输入就是FP32，所以native/compat源前向严格相同，可精确复用。阈值每checkpoint/score表示独立固定，M阈值用于common-multi C1附表；不读取目标normal来拟合阈值。正常case只评误报，没有正例的AP/F1记未定义，不编0。检测指标按底层系统宏平均，两轨/各C/族/事件、100旧子集单列。

## 运行与计算

GPU只负责推理，完成后进程退出释放GPU，再由CPU池算指标，避免CPU指标阻塞下一批GPU工作。使用现有A100/H100中>=10GiB余量设备，每worker最多8GiB，允许与他人共享、不停他人任务。模型不重新训练，无伪造三优化seed重复；官方历史训练exposure未知，不能宣称全部机制未见。

每节点运行不可变代码与原Runtime参考快照，checkpoint/source/index/score/metric代码SHA绑定。预检四checkpoint对上游zero_shot概率<=1e-6、raw<=1e-5；5017×20长输入检验全注意力与复制末窗、无参数变化。随机预检原生reference4,626、适配205,306 scalar-forwards另计；源正常校准、所有原始/正常开发前向实际累计，不能把它们忽略。

新结果单独目录，不覆盖原baseline或任何proposal。历史100系统自训练原logit记录仍保留用于数值/预处理核查，不将旧数值当官方已测结果。所有基线只TSAD，RCA字段明确未评/不适用；联合B2 RCA放独立方案表。
