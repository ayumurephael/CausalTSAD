**结论：真图有有限源侧定位信号；及时更新父信息没有稳定的额外收益。尚未回接TimeRCD或证明目标集检测提高。下面为完整源侧结果。**

# C02 父信息时间对齐：源侧三种子结果

15run固定终点完成；每run434updates、4,976,640正常查询标量、权重SHA与同种子初始SHA/容量/Linear FLOPs/硬件类型核验通过。没有目标基准读取。

| 范围 | 方法 | 正常NLL↓ | ready NLL↓ | 观测AP↑ | 条件RCA Top1↑ |
|---|---|---:|---:|---:|---:|
| all | none | 10.349738 | 1.300915 | 0.281274 | 0.634146 |
| all | parents_block | 9.971441 | 1.304054 | 0.280459 | 0.655827 |
| all | parents_past | 10.037828 | 1.300610 | 0.281316 | 0.655827 |
| all | parents_current | 10.028976 | 1.300162 | 0.280728 | 0.653117 |
| all | rewired_current | 10.329179 | 1.302615 | 0.281595 | 0.628726 |
| multi | none | 1.404030 | 1.311899 | 0.231685 | 0.634146 |
| multi | parents_block | 1.386861 | 1.301498 | 0.234372 | 0.655827 |
| multi | parents_past | 1.388779 | 1.301710 | 0.237450 | 0.655827 |
| multi | parents_current | 1.387411 | 1.300754 | 0.234563 | 0.653117 |
| multi | rewired_current | 1.404561 | 1.310941 | 0.233698 | 0.628726 |
| C1 | none | 13.171077 | 1.297451 | 0.296913 | — |
| C1 | parents_block | 12.678886 | 1.304860 | 0.294994 | — |
| C1 | parents_past | 12.765605 | 1.300263 | 0.295151 | — |
| C1 | parents_current | 12.754393 | 1.299975 | 0.295288 | — |
| C1 | rewired_current | 13.143866 | 1.299989 | 0.296700 | — |

## 结构性父观测的可用性（仅审计，不输入模型）

{'block': 0.22393433179723501, 'past': 0.8824884792626728, 'current': 1.0}

32个源系统完整float64重放哈希一致。该统计计算所需特定父观测是否落入输入窗口，不代表全部可预测信息或性能必增；自相关可能替代缺失信息。

## 多变量配对结果

- fresh_past_minus_block/ready_NLL: {'mean': 0.00021208414863441695, 'per_seed': [-0.00012396748473004598, 0.001109314158679993, -0.00034909422804669637], 'hierarchical_seed_system_bootstrap95': [-0.001430287323185422, 0.002339301619749405], 'systems': 123, 'negative_systems': 73, 'positive_systems': 50}
- fresh_past_minus_block/observed_AP: {'mean': 0.0030782275217949937, 'per_seed': [0.0031899909537404488, 0.003436077443879192, 0.002608614167765342], 'hierarchical_seed_system_bootstrap95': [-0.0013393423866885172, 0.008593976869277733], 'systems': 123, 'negative_systems': 52, 'positive_systems': 54}
- fresh_past_minus_block/RCA_Top1: {'mean': 0.0, 'per_seed': [0.0, 0.0, 0.0], 'hierarchical_seed_system_bootstrap95': [0.0, 0.0], 'systems': 123, 'negative_systems': 0, 'positive_systems': 0}
- current_minus_past/ready_NLL: {'mean': -0.0009559848444248602, 'per_seed': [-0.0002189351291191287, -0.00187266406004991, -0.0007763553441055422], 'hierarchical_seed_system_bootstrap95': [-0.0024316687046027764, 0.0002671753810996283], 'systems': 123, 'negative_systems': 100, 'positive_systems': 23}
- current_minus_past/observed_AP: {'mean': -0.002886687820087312, 'per_seed': [-0.001896447911227277, -0.004343167610559962, -0.002420447938474698], 'hierarchical_seed_system_bootstrap95': [-0.008537431331284255, 0.0010561135201428145], 'systems': 123, 'negative_systems': 56, 'positive_systems': 50}
- current_minus_past/RCA_Top1: {'mean': -0.0027100271002710027, 'per_seed': [-0.008130081300813009, 0.0, 0.0], 'hierarchical_seed_system_bootstrap95': [-0.016260162601626018, 0.0], 'systems': 123, 'negative_systems': 1, 'positive_systems': 0}
- current_minus_block/ready_NLL: {'mean': -0.0007439006957904433, 'per_seed': [-0.0003429026138491747, -0.0007633499013699167, -0.0011254495721522385], 'hierarchical_seed_system_bootstrap95': [-0.0024721333893333993, 0.0011755788794701016], 'systems': 123, 'negative_systems': 84, 'positive_systems': 39}
- current_minus_block/observed_AP: {'mean': 0.0001915397017076814, 'per_seed': [0.0012935430425131717, -0.0009070901666807695, 0.0001881662292906437], 'hierarchical_seed_system_bootstrap95': [-0.006306771699818272, 0.006052423503767874], 'systems': 123, 'negative_systems': 52, 'positive_systems': 55}
- current_minus_block/RCA_Top1: {'mean': -0.0027100271002710027, 'per_seed': [-0.008130081300813009, 0.0, 0.0], 'hierarchical_seed_system_bootstrap95': [-0.016260162601626018, 0.0], 'systems': 123, 'negative_systems': 1, 'positive_systems': 0}
- current_minus_none/ready_NLL: {'mean': -0.011144769224047984, 'per_seed': [-0.00899015887966001, -0.013780989782596991, -0.01066315900988695], 'hierarchical_seed_system_bootstrap95': [-0.024590299736839648, 0.005196687191482481], 'systems': 123, 'negative_systems': 88, 'positive_systems': 35}
- current_minus_none/observed_AP: {'mean': 0.0028783964033758055, 'per_seed': [-0.0014969817578325323, 0.0009347324961742868, 0.009197438471785665], 'hierarchical_seed_system_bootstrap95': [-0.01259277677063815, 0.017400363335849014], 'systems': 123, 'negative_systems': 45, 'positive_systems': 64}
- current_minus_none/RCA_Top1: {'mean': 0.018970189701897018, 'per_seed': [0.016260162601626018, 0.016260162601626018, 0.024390243902439025], 'hierarchical_seed_system_bootstrap95': [0.0, 0.04065040650406504], 'systems': 123, 'negative_systems': 0, 'positive_systems': 5}
- current_minus_wrong/ready_NLL: {'mean': -0.010186537979094962, 'per_seed': [-0.009917209788066586, -0.01271754503250122, -0.007924859116717083], 'hierarchical_seed_system_bootstrap95': [-0.02659819393461636, 0.006847977246533886], 'systems': 123, 'negative_systems': 81, 'positive_systems': 42}
- current_minus_wrong/observed_AP: {'mean': 0.0008651108486118692, 'per_seed': [0.0007445899449260157, -0.0032098247482118284, 0.005060567349121418], 'hierarchical_seed_system_bootstrap95': [-0.012247274142016113, 0.013415189367999677], 'systems': 123, 'negative_systems': 52, 'positive_systems': 57}
- current_minus_wrong/RCA_Top1: {'mean': 0.024390243902439025, 'per_seed': [0.016260162601626018, 0.024390243902439025, 0.032520325203252036], 'hierarchical_seed_system_bootstrap95': [0.005420054200542005, 0.051490514905149054], 'systems': 123, 'negative_systems': 0, 'positive_systems': 7}

## 限制

三种子并非三个独立数据集；同种子五臂在同类硬件/节点上，seed348/349 A100、350 H100。全部差异使用相同点集；full与ready不能直接相减作因果解释。

C1无父边，跨模型差异来自共享训练；同一空图模型在不同parent timing下必须是exact null。RCA仅为源真实注入窗条件定位，不是端到端RCA。当前父观测分支已由用户授权，模型没有真实lag/coef/noise输入；没有TimeRCD整合或目标TSAD提升结论。

固定一步源试验只检验该原型；不自动延长预算，也不以整体AP≥.85作为局部机制学习的硬门槛。
