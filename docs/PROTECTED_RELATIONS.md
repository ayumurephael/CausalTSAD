# C04f：保护已有关系判别方向，再学非平稳性

| Research Insight Card | 内容 |
|---|---|
| Problem | 保留C04真图paired AP/root定位，同时改善正常漂移误报。主看独立来源128系统AP/FPR/条件rootTop1，不把CE降低当关系改善。 |
| Evidence | C04e18run强漂移FPR .86793→.05295，但clean AP .59589→.56718、root .42839→.30729；124/128系统AP下降。说明当前增强有保留代价，不证明梯度冲突就是唯一原因。 |
| Failure Mechanism | 假设增强更新与已学clean关系方向干扰；原AdamW预条件后实际位移也可能不满足原始梯度的非冲突条件。用逐步clean/aug梯度dot及实际位移dot直接检查。负cos不是泛化失败的充分证据。 |
| Key Insight | 不丢弃关系任务；把clean关系任务作为更新方向的局部约束，检查能否保住它再学习正常背景变化。约束是loss的一阶局部近似，不是AP/泛化保证。 |
| Novel Mechanism | 对实际AdamW下降位移delta=theta_old−theta_candidate，若delta·g_clean<0，则投影到非负半空间；shared/root私有块分别处理，避免用一种任务改善补偿另一块损害。float32舍入保守内移另记。不是发明GEM，也不称原论文完整复现；Adam状态不投影。 |
| Closest Prior Work | 全文R18 A-GEM(ICLR2019)20页及作者固定45421499483b28935491251e9e821c55e8b3c089核心create_stochastic_gem_ops与reference sampling已读；R11 PCGrad也已读。A-GEM作者使用SGD原始梯度，我们使用AdamW实际位移、并有相同干净/增强数据，不是原来的顺序task协议。 |
| Falsifiable Prediction | 若确有方向干扰，保护臂应减少clean退化并保留漂移收益，真图相对degree/wrong仍有正证据。若投影触发却AP/root不保留，loss局部方向不足以解释/修复，停止此原型而不否定关系监督。 |
| Minimal Experiment | 2arm(unprotected_split/protected_step)×3graph×3seed=18run；均从原C04对应checkpoint起，原256系统一遍，每组clean与aug各四view。与C04e相同增强plan/lr/损失/曝光，新增投影向量计算另记。 |

每run256updates/20,185,088观测query等效曝光、29,360,128完整输入；总363,331,584/528,482,304，额外四条件sourcecal/val25231360query/run另计。两臂同容量、同8前向/反向和名义Linear/attention/图聚合FLOPs；投影CPU/GPU向量成本不冒称匹配。对照也独立执行split backward，避免把浮点累加顺序变化当算法增益。没有新教师或target输入。

clean校准64系统保持，linear2/8/quadratic4也用clean阈值，fixed-final不选best。每worker2GiB、显存余量≥3GiB，单run3h/stage4h先到停止，共享但不停止他人。clean AP较unprotected改善且相对clean-only保留≤.02、root保留≤.05，linear2/8优于clean-only，并有真图控制证据才考虑冻结目标迁移；不只凭约束满足推广。250M数组不读、0target梯度，原三新标签null。若预算失败不调lr加步。
