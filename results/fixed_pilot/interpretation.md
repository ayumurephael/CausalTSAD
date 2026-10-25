

## C03v2九组完成后的更新

全部9组训练与100development评测已取回89文件，优化936+936步/每阶段有效8,948,736标量/初始化/参数9129+113/Linear FLOPs均核验；两节点saved base数组逐字节一致、AP和FPR严格一致，仅部分VUS的CPU末位求和差<=1.11e-16。原始结果未改，分析器原过严dict断言保留为r0，另有baseline_equivalence_audit.json证明模型G0没有放宽。

源123M独立系统：no_context正常pinball .098320→learned_mean .089658，降低约8.81%，122/123改善，差CI[-.009276,-.008056]；只换错历史+.027000，123/123退化。说明使用系统历史，不直接证明因果父子信息。污染历史代价约.000065，块中位数并未明显胜均值。

目标80M宏平均：原AP.479401/VUS.415507/FPR.003309；learned_mean .480866/.419726/.003761。VUS差+.004219(约+.422个百分点)，95%CI[-.000492,.010986]；正常FPR差+.000453，CI为正。**两个mechanism_holdout/nonlinear_ar2_dag/measurement系统贡献净VUS增益约95.3%，其余78平均差约+.000204**；仅为事后敏感性核查，不删掉系统或改主表，不按这些目标例子调参。ID略退、留出部分提升仍需全体正反例。

当前条件ready RCA：mean Top1=39.47%、median42.11%，但排除前256点后只有76个目标M事件、源105个事件有效；不能与C08完整80事件的40.42%直接排序或合并成一套成绩。完整检测指标仍含全部时间点，C1完全保留。

下一步C03v3独立分支：图值×上下文成组对照，复用真图已有no_context/mean×3seed六组，新增degree_only/wrong_degree×无/有学习摘要×3seed十二组。真实度数在全部视图匹配，wrong仅非因果预测对照，可能有环/后代，不能冒称正确DAG。区分正常波动校准与真实父信息用途，网络/seed/source/querymask/预算保持。新有效密度/观测融合/额外正常融合各107,384,832，teacher新缓存前向39,555,520单列。运行真实阶段见research_status.json；validation/locked继续关闭。

## B2联合基线已经固化

独立baseline/joint-tsad-rca-v1，使用已完成C08b mean+PCGrad，不新增训练或按目标选seed。统一`JointTSADRCA.predict(X,G)`和`tools/predict_joint.py`输出raw异常分数、null/联合root概率、条件根因排名；自动事件按源阈值连续段产生，不需要真实事件窗口。真实原始输入C1/C8/C20共29,696观测标量在H100重放，通过feature/base/fused/root逐点误差0、异常概率误差<9e-8、全部模块hash不变。额外历史encoder114,688 padded slots、base29,696、referencehead29,696单列，0optimizer。

同一联合模型三seed平均AP48.0171%、VUS41.6150%、FPR0.3496%、**真实窗口条件RCA Top1=40.4167%**，检出+正确定位的事件匹配28.75%。自动事件接口已可运行，但自由运行事件precision/recall/定位成绩尚未测，不能把条件40.4%移贴过去。它是可复现基线，不是已证明超越原模型的SOTA。原C08b测量结果、所有checkpoint与其他proposal保留。
