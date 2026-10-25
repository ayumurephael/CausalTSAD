# C04c：保留关系证据方向与配对排序

## 八字段研究卡

| 字段 | 内容 |
|---|---|
| Problem | C04已学到受控关系破坏检测和条件RCA，但如何将其用于自训练TimeRCD而不压掉信号？先比较来源ready AP/正常FPR，主方案通过后才做540生成开发系统原四指标/AP/RCA。 |
| Evidence | C04b 12run固定来源训练审计通过，关系teacher新验证AP=.59286，B0=.50205，融合=.48453。冻结CPU诊断中99.801%的查询点对teacher logit呈负局部响应；正例平均修正3.258、正常3.321。标签oracle上限诊断AP=.99937，只限来源审计、禁止模型输入。 |
| Failure Mechanism | **事实：**该融合器反向响应一个有用的关系分数。**待反证：**自由多特征混合BCE在两类来源的不同先验/基线尺度下，优先学习来源偏移，破坏了同系统正常/故障排序。不是整个TimeRCD或所有因果融合的失败机制；负偏导也不单独证明全局因果原因。 |
| Key Insight | 同一系统四视图的相对异常证据应保留，不能只优化不同来源的整体分类校准。限制关系分数的条件响应方向，并用匹配四视图消去共同截距，检验该解释。 |
| Novel Mechanism | 新8参数有界线性残差，零初始化严格B0；相同特征与尺度，教师logit系数自由/投影非负，分别训练点BCE或BCE+四个正负视图配对logistic ranking。正常/故障孪生只在独立来源构造监督，测试输入仍X/G。成熟校准与排序不是原创；待证明贡献为跨系统关系破坏监督及证据保留接口。 |
| Closest Prior Work | Gutmann/Hyvarinen AISTATS2010 NCE全文8页，Ren等NeurIPS2019 LLR主文12+附录9页均读完。NCE Eq3–9解释特定二元对比分类与密度比；我们的经验四视图未知噪声密度，不能直接套一致性定理。LLR Eq1–5依赖背景近似抵消；degree判别logit不等于归一化背景log-density，禁止相减后冒称严格LLR。CNP/ANP与旧C04提供条件表示。 |
| Falsifiable Prediction | 在同8参数骨架内，positive比free保留方向，pair比point提高真图来源配对AP，收益应随真图而非错图/去父信息出现；若只有约束导数成立而检测/误报没改善，机制修复失败。若四臂均可用，说明非线性混合更可能是C04b问题，而不能单独归功约束或排序。 |
| Minimal Experiment | 4variant×3graph×3seed+3basecal=39个极小缓存头，全部既有9teacher特征复用，新teacher/TimeRCD前向0。每run568update/15,561,216查询标量等效曝光，总606,887,424，完整输入等效882,745,344；每head8参数，Linear forward每时点14FLOPs，point和pair分支均执行，非线性/投影额外成本单列。 |

## 固定协议

- **唯一预注册主方案positive_pair**。其余仅拆分归因，不按开发集挑赢家。四个variant的同seed数据顺序、特征标准化、teacher初始化/训练来源相同；3seed包含teacher种子和小头数据顺序改变，不是独立SCM。
- 旧312训练组、新256训练组；旧/新组总权重50:50，旧trust=.25，AdamW lr=.01/decay=.0001，固定单遍568updates，不加步或扫学习率。8参数与434参数C04b不等容量，因此归因只看本轮2×2内对照。线性只为诊断融合接口，不称foundation预训练。
- 所有variant共同投影B0特征系数≥0（总分对B0不反向）；positive另投影teacher logit系数≥0。NLL/MSE三特征系数仍自由，不能保证所有关联方向变化时总分也单调。basecal屏蔽因果四特征。
- Pair只在新四视图同一系统/同一时点计算：两异常×两正常4个差，平均softplus(−差)，权重1。point也计算同一pair图但乘0，前向/反向图名义匹配。bias在pair差中代数消去，其他来源特征不必消去，不宣称已完全去掉来源捷径。
- Source normal阈值沿用C04b同一池；旧val正常误报是校准内，新val独立。图/根因/标签不进入融合头（只有已允许X/G产生的冻结特征）；根因监督仍在旧teacher，新头只检测。
- 继续门槛：主方案真图相对degree/wrong来源AP系统配对bootstrap95%下界>0；旧ready AP下降≤.01、新验证正常FPR较配对B0增加≤.01。此处是有限集成门槛，不把通用AUC≥.85强加局部机制。
- 固定1h/head、4h阶段，真实失败停止新增，原错误/结果不覆盖。39源头训练固定final；若主方案未过则停止此接口，不自动挑其他variant测benchmark。通过才冻结到生成开发集；150M锁定不打开。
- C1/前320点硬identity；正常拓扑可以输入，滞后/系数/噪声/root/normal孪生不是目标输入。R不保证干净，旧一维labels保持，三类影响标签null。
- RCA依然是C04冻结root后验，Top1=.42448仅来源条件后缀结果，不代表目标性能，也不编造TimeRCD原生RCA。

## 全文和作者源码核查

R14原PDF SHA9f05eba5ed29ee20575bed070aaff0a1431efce5097d660c8cba2f69450dbf75；R15主文343d67588bd81f50b81ab3aec4818d7f769600896a8db740eccc790363e5aa44，附录e1b878e58d309b2703824bade047761650587f9d13f4e2cda4edc09477f462c9。按原PDF逐页内存阅读，公式/结果页渲染复核，无TXT派生。

R15作者google-research/google-research commit5b09c22d73a9d35eb6c5d2a99b95677a45053466，generative.py、generative_eval.py、utils.py完整读取，未执行。其AR目标错一位避免自身泄漏；背景模型训练变异序列，评估做同样本前景减背景loglikelihood；utils从其余三种符号取值（附录伪码允许四值，存在实现细节差异），time.time设种子不适合我们的固定复现。compute_auc先构建ys后滤NaN，不能直接拷贝其缺失处理。我们不直接执行/移植TensorFlow训练器，不把两个判别器logit差称该论文LLR实现。

[NCE原始全文](https://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf)，[LLR原始主文](https://proceedings.neurips.cc/paper/2019/file/1e79596878b2320cac26dd792a6c51c9-Paper.pdf)，[作者代码](https://github.com/google-research/google-research/tree/5b09c22d73a9d35eb6c5d2a99b95677a45053466/genomics_ood)。
