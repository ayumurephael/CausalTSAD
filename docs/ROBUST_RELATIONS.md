# C04e：在正常漂移下保留关系破坏监督

本方案与用户提出的独立Bayesian DAG路线分开。旧C04/C04b/C04c/C04d/漂移诊断均保留，不覆盖。

| 字段 | 内容 |
|---|---|
| Problem | C04来源真图能检测关系破坏，但正常趋势导致高误报和定位丢失。要降低漂移FPR并保留clean paired AP、条件rootTop1和真图特异性。 |
| Evidence | 128系统×3冻结seed：clean AP=.59286/FPR=.00968；linear2 AP=.51325/FPR=.66711、AP差CI[−.09106,−.06856]；linear8 AP=.50120/FPR=.87531。来源干预确立局部敏感性，尚未证明覆盖全部生成基准退化。 |
| Failure Mechanism | **候选解释：**模型把不同于平稳来源的合法时变边际视为故障，压过关系信息。损害正常控制不等于证明所有目标异常都由漂移解释。 |
| Key Insight | 不必抛弃关系监督；在正负类完整单通道轨迹多重集保持完全相同时，把多种正常边际变化纳入关系训练，要求区分关系破坏与合法非平稳性。 |
| Novel Mechanism | 保持原C04网络和4视图关系任务，增加一个干净分支和一个共享逐通道确定性漂移分支。对照相同预算干净继续训练。每通道同一h(t)作用于四视图，FP64/FP32类多重集不变；无channel threshold labels。成熟augmentation不称原创，待证明贡献是精确匹配关系监督的跨系统非平稳稳健性。 |
| Closest Prior Work | 已全文阅读R14 NCE强调对比噪声结构，R15 LLR区分背景与结构，但本方案不是其生成似然比实现；C04是直接基础。不能把“添加趋势”包装成首次数据增强。 |
| Falsifiable Prediction | augmented应在相同真图继续训练对照上改善linear2/8，而clean AP保留；真图仍超过degree/wrong。若仅阈值变大或无图同样好，则不能说保留了关系机制；quadratic4为未参与增强的负例检查。 |
| Minimal Experiment | 2arm×3graph×3seed=18run；各从对应已训练C04 final出发，原256训练系统单遍，每组两分支×4view。每run256updates、20,185,088 query曝光、29,360,128完整输入；总363,331,584/528,482,304，所有额外来源评估另计。 |

原normal NLL/root损失权重保留，lr=1e−4、weight decay沿原配置，梯度裁剪1、root hidden仍detach。同seed两arm权重init完全一致，数据顺序与增强计划跨图一致；continued_clean也执行两分支，无相同前向偷省以冒称FLOPs匹配。名义参数/Linear/attention/图聚合前向匹配，mask有效信息与梯度路径不同，backward/nonlinear不宣称严格相等。

每次第二分支25% identity，其余线性强度U[0,8]前缀σ，正负符号按系统/seed固定。模型输入仍仅时序和正常G；增强生成参数不输入模型。模型和25亿旧权重/数据不覆盖。原C04 sourcecalibration64组保持clean，所有clean/linear2/linear8/quadratic4 FPR都用同一clean threshold，不能用stress重校准掩盖误报。初始source结果复用checkpoint与已完成漂移诊断；fixed-final四条件报告，不挑best。

每run≤3h含最终评估，阶段≤8h、worker4GiB/分派时≥6GiB空余，允许共享但不停止他人。18run原单遍预算固定，不无限加步。来源通过（clean AP较continued_clean下降≤.02、两线性条件AP改善与FPR降低、真图相对degree/wrong配对有正证据）才冻结到生成development；锁定关闭。即使成立，也只是该增强原型，不等于所有非平稳性、raw工业或SOTA成功。
