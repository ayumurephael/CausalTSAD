# C03v3：上下文收益来自父值还是单通道尺度？

C03v2九组完成：123独立源系统122个pinball改善；学习均值.089658 vs无上下文.098320；换错历史+.027000，证明使用历史条件，但**尚不证明利用真实父子关系**。目标80M VUS=.419726 vs原.415507，95%差CI[-.000492,.010986]、正常FPR增。ID略降、机制留出部分提升，两个measurement系统贡献较大，不能只报平均分。

保持网络、冻结C02 teacher、自训练TimeRCD、优化与数据不变，增加图值×上下文对照。true图已有no_context/learned_mean×3seed六组直接复用，不重复训练。新degree_only/wrong_degree各no_context/learned_mean×3seed共12组。均值按源pinball优于median(103/123系统)和更简单计算保留，非目标挑最高。

**degree_only**：真实入度在teacher和最终12维条件特征都保留，但所有父值消息置零。它仍知道节点度数，不能误称完全无图；检验新增父观测的作用。**wrong_degree**：按system_id哈希固定假父集，逐目标入度保持、排除自身同刻；有替代父时至少换一条。它可有环、可含实际后代，只是明确的非因果预测对照，不作为真实DAG主模型；不声称不符合真实父集的负对照仍是严格因果预测。

ValueGraphTeacher保持所有原参数名/shape/weightSHA；只把消息掩码与真实度数分开。true算子随机非零权重逐点严格等于原teacher，own-query/future/degree-only非父值null必须通过。额外teacher前向39,555,520观测标量（两新图模式，正常/观测含既定source与100development），0新encoder；不是训练曝光。真实cache逐字节复用，自有新cache独立、跨节点SHA相同后才开训练。

每新run固定3正常密度+3冻结密度融合，各936步。12run有效正常密度107,384,832、原观测融合107,384,832、额外正常融合107,384,832；完整源输入每run每阶段11,931,648，额外验证/重叠上下文/前向FLOPs另计。新头9129+113参数，同seed初始化/次序/标签mask/名义容量/denseFLOPs匹配，无父值和无上下文有效信息/容量不同须披露。仅运行2GiB memory cap改变资源上限，不改变网络/精度/批量；同seed三图控制用同GPU型号，seed348/349 H100、350 A100。

全部12固定final自动评估同100development原四指标/AP/FPR/ready条件RCA；C1原common-multi不变、前256保留原分数，完整指标点集不裁掉。source训练312M、验证123M，错上下文和污染诊断沿v2。源阈值同1%，真实normal/root/系数不进目标模型，真实lag仍不用。目标validation/locked关闭、0目标梯度。

判读：图×上下文交互优先，并看true对degree_only、degree_matched_wrong的源预测/误报/TSAD差、实际父边替换率；若degree_only同样有效，收益更像单通道动态校准，应保留有效方法但删除“因果关系带来提升”的claim。错父集可能含替代预测信息，不能把它强弱简单解释为真实因果识别。仅真图优且独立源反证吻合，再考虑C04机制保持/破坏配对或原骨干有限微调；不因为每个局部单元都未到某个固定AUC就阻止学习，也不自动加轮数。

本阶段有受控的已知图输入，因果结构/degree由正常G提供；没有新因果发现算法，没有证明干预识别，没有最终SOTA或novelty。CNP/分位数机制来自已全文阅读的R12等，不增加未读论文声明。技术非有限/G0/SHA失败停止该节点新任务，保留其他已启动任务和旧产物。6h缓存/3h训练评测stage/10h训练campaign、固定预算先到止。
