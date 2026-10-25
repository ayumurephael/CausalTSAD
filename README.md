# 已知正常 DAG 的 TimeRCD：异常检测与根因定位

B1是基线；C01是候选检测方案；C02/C03只是各自较大idea的受控源探针。此前结果尚未建立稳定TSAD增益，不能据此结束整个因果方向。

本轮新增完整阅读DéjàVu13页、CDRL4AD20页及作者核心实现，已更新[Insight Cards](Research_Insight_Cards.md)和[全文记录](papers/全文阅读记录.md)。C08实现p(null)/p(root)联合读出，源根因监督直接约束检测汇总。24组（8控制×3seed），固定3遍M源数据，共286,359,552原优化标量，额外正常同量；5共享H100已实际优化。原TimeRCD/创新teacher冻结，C1原logit精确保留。

[独立C08分支](https://github.com/ayumurephael/CausalTSAD/tree/proposal/c08-joint-root-readout-v1)。后台自动做全部固定100开发系统评估，尚无新性能结论。validation/locked不读取，目标不反传。当前真实阶段见research_status.json与TASK.md顶部。

---

研究目标：利用正常系统有向图及可观察因果上下文，改进既有自训练TimeRCD在独立合成系统的检测，并尽可能定位实际故障起始通道。原P/S路线已按用户要求删除，原数据、权重及P0/P1展示保留。

## 当前结论

| 实验 | 已完成工作 | 实际结论 |
|---|---|---|
| B1 | 4图×3种子独立小baseline | 未建立稳定检测收益，保留阴性结果。 |
| C01 | 12训练＋100开发系统评测；12冻结共同logit对照 | 同表示下VUS43.34%→43.46%、AP47.710%→47.717%，区间跨0、误报增加；未建立可靠因果增益。 |
| C03 | 3上下文×3种子，源验证513系统 | 固定交叉矩上下文没有改善多变量预测，错误系统上下文也未变差；不扩大这个原型。 |
| C02 | 5父信息/图对照×3种子，源验证123多变量系统 | 真父图条件RCA约65%，空/错图约63%；及时更新父值额外收益不稳定。仍未证明改进目标TSAD。 |

原概率表示把部分不同分数压成相同的1，影响排序。仅原模型改用已有logit，AP就从44.16%到47.71%；因此不能把混合表示的涨幅当模型贡献。原概率表、统一logit表均保留。VUS也受表示和阈值采样影响，不能把所有变化都叫舍入。

上述旧阶段均已完成，C08联合读出是当前正在运行的新实验。它区分图信息、额外根因监督、额外容量/校准，检验定位信号能否转化成检测收益；没有自动增加C01/C03轮数。

## 结果与方案

- [C01原概率结果](results/c01_pilot_v1/三种子完整开发结果.md)、[统一logit公平对照](results/c01_pilot_v1/统一logit公平对照.md)
- [C02源侧结果](results/c02_source_v1/源侧三种子结果.md)、[C03源侧结果](results/c03_source_v1/源侧三种子结果.md)
- [十张Insight Cards](Research_Insight_Cards.md)、[实验协议](实验协议.md)、[全文阅读记录](papers/全文阅读记录.md)：10篇161页，含附录，未转换TXT。
- [CausalTSAD仓库](https://github.com/ayumurephael/CausalTSAD)：baseline/shared-causal-v1、proposal/c01-parent-innovation-v1、proposal/c02-parent-timing-v1、proposal/c03-mechanism-context-v1独立保存，不覆盖彼此。

## 信息与保留边界

模型可见观测、valid、正常有向拓扑。用户已允许同刻直接父观测的单列分支；自己的当前/未来值不能进入自己的条件预测分布。真lag仅增强/隔离审计，真系数/噪声/方程/故障/正常孪生不是目标输入。正常孪生与真实根因可作源监督，目标正常case只单独测FPR。effect/anomaly_channel/anomaly_time继续null，旧一维labels不改。

250M基准按系统双轨隔离：50M开发、50M验证、150M锁定，2720系统/5440case。C01/C08只读固定100开发系统；C02/C03均source-only。所有目标数组不反传，validation/locked均未打开。原2.5B、旧权重、基准DAG及P0/P1展示不覆盖。此前RCA为真实事件窗条件定位，C08额外测检出+定位事件匹配；尚非完整自由运行报警RCA或SOTA结论，C1排名排除。

主服务器稳定根目录 `/home2/xuhaijie/TimeRCD_reproduction`，基准在 `evaluation_data/synthetic_benchmark_v1_r1/data`，权重在 `reference_checkpoints/causal_baselines`。每一阶段保存独立代码、输入/权重SHA、种子和额外计算；CPU指标/缓存额外开销与优化查询曝光分开。详细状态见research_status.json、TASK.md、各分支配置及results。

旧自动化已删除，未新建。`admin/reset_remote.py`是已执行的旧清理工具，禁止重跑。当前完成的队列不轮询、不重复launch；下一阶段应新建独立实现与不可变运行目录。
