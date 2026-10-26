# C04g：先消除正常趋势，再检验图关系判别

| Research Insight Card | 内容 |
|---|---|
| Problem | 能否让关系分支对每通道线性趋势不敏感，同时保留真图相对错图的关系检测和定位？看source paired AP、正常FPR、条件rootTop1、linear/tanh/sin分层。 |
| Evidence | C04e已证明增强改善漂移但损害clean能力；这不是漂移必然可修复的证明。C04模型固定前256统计，对后续趋势很敏感。 |
| Failure Mechanism | 候选解释：合法低频边际变化进入创新/节点特征，压过关系证据。我们测试移除该变化是否真的保留关系信息，而不是调阈值。 |
| Key Insight | 以相同因果线性滤波器作用于全部通道，先把线性趋势等价类消掉。LTI系统中差分与各边滞后线性算子可交换；非线性系统不保证此性质，必须作为负例报告。 |
| Novel Mechanism | 固定Delta²x_t=x_t−2x_(t−1)+x_(t−2)，float64在进入网络前计算，首2点固定0；原C04结构/正常DAG/损失不变，参数全由原checkpoint继续。不是adaptive patch，不把差分本身称原创，也不是完整幅度异常检测器。 |
| Closest Prior Work | 成熟时间序列差分/趋势消除、C04原始关系任务；与R15背景消除目的相关，但不是其生成LLR实现。后续贡献必须是跨系统因果关系检测的稳健性，不是首次差分。 |
| Falsifiable Prediction | 线性漂移输入在滤波后和原样输出近等价，真图仍优于degree/wrong；clean关系性能若崩掉，说明消除趋势同时丢失重要机制或接口/预算不适合。tanh/sin可能更差，不能由linear结果推广。 |
| Minimal Experiment | 3graph×3seed=9run，原C04对应checkpoint、同256train单遍/两分支/八view，与C04e曝光/容量/优化设置相同。clean与随机linear增强都执行，不因滤波后重复偷省计算。旧clean-only/augmented原型为固定参照。 |

已知反例：持续level shift在差分后只有边界脉冲，内部为0，不能把这一分支替代保留幅度信息的TimeRCD。预检明确验证该盲点。非线性SCM的处理后图不保证仍是相同有向条件机制；单通道可逆差分缺少初始状态时也不能宣称因果识别。

预算每run256updates、20,185,088 query等效曝光、29,360,128 raw完整输入；9run181,665,792/264,241,152。额外source eval25231360query/run另计；3次标量前端线性算术/slot另记，nominal神经网络Linear/attention/图聚合与C04e相同，CPU/前端不是相同FLOPs。每worker2GiB、≥3GiB空余，run3h/stage4h。原64source clean校准、128sourceval四条件固定最终评估，按C/族/污染分层，不按target标签选择。linear稳健性和图控制正证据只是局部资格；若clean全组退化，保留局部结果不包装成通用改进。250M/原25亿不读，原权重不改，新三标签仍null。
