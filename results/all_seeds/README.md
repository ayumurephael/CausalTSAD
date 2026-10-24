# B1 三种子完整开发结果

12个固定终点run全部完成；4张A100，3种子×4图对照。每run原观测曝光4,976,640，正常孪生额外同量。固定100个开发系统；未读取目标validation/locked。
| 范围 | 方法 | VUS-PR | AP | 正常FPR | 条件RCA Top1 |
|---|---|---:|---:|---:|---:|
| all | none | 0.03857 | 0.03298 | 0.00135 | 0.34583 |
| all | parents | 0.03800 | 0.03294 | 0.00118 | 0.35833 |
| all | full | 0.03851 | 0.03266 | 0.00098 | 0.35000 |
| all | rewired | 0.03934 | 0.03303 | 0.00110 | 0.35000 |
| id_multi | none | 0.03653 | 0.03677 | 0.00000 | 0.56667 |
| id_multi | parents | 0.03670 | 0.03703 | 0.00000 | 0.59167 |
| id_multi | full | 0.03584 | 0.03604 | 0.00000 | 0.57500 |
| id_multi | rewired | 0.03647 | 0.03675 | 0.00000 | 0.56667 |
| heldout_multi | none | 0.02408 | 0.02438 | 0.00000 | 0.12500 |
| heldout_multi | parents | 0.02455 | 0.02479 | 0.00000 | 0.12500 |
| heldout_multi | full | 0.02493 | 0.02546 | 0.00000 | 0.12500 |
| heldout_multi | rewired | 0.02442 | 0.02473 | 0.00000 | 0.13333 |
| C1 | none | 0.07162 | 0.04262 | 0.00677 | — |
| C1 | parents | 0.06752 | 0.04105 | 0.00588 | — |
| C1 | full | 0.07104 | 0.04029 | 0.00490 | — |
| C1 | rewired | 0.07493 | 0.04221 | 0.00552 | — |

## 真父图减无边图的配对差异

- all: {'mean': -0.0005667491852172531, 'per_seed': [-0.0005918569325228067, 0.0011210461046020682, -0.002229436727731021], 'hierarchical_seed_system_bootstrap95': [-0.002849979282194785, 0.001073368545104375], 'note': 'Three optimization seeds remain a small sample; systems are paired and never tripled as independent units'}
- id_multi: {'mean': 0.00016457458024611837, 'per_seed': [0.0003332049173719728, 0.00025827892638753837, -9.776010302115575e-05], 'hierarchical_seed_system_bootstrap95': [-0.0006095298036842493, 0.0010194918682837081], 'note': 'Three optimization seeds remain a small sample; systems are paired and never tripled as independent units'}
- heldout_multi: {'mean': 0.0004692743236538139, 'per_seed': [-0.0002560138932770199, 0.0011309595306541855, 0.0005328773335842764], 'hierarchical_seed_system_bootstrap95': [-0.0003949034405283297, 0.00148770248530037], 'note': 'Three optimization seeds remain a small sample; systems are paired and never tripled as independent units'}
- C1: {'mean': -0.0041014437338861304, 'per_seed': [-0.0031136667108039393, 0.0028267536089268937, -0.012017418099781345], 'hierarchical_seed_system_bootstrap95': [-0.014154148171431532, 0.0024903559482917703], 'note': 'Three optimization seeds remain a small sample; systems are paired and never tripled as independent units'}

## 结论边界

先按上述配对结果判断该小baseline，不根据检测BCE下降判断因果机制。C1各组推理时均为空图，差异来自多变量训练造成的共享权重差异。RCA仅为给定真实注入窗口的定位，不能称端到端RCA；C1定位排除。
此独立baseline没有使用TimeRCD权重，不能把它的成绩写成TimeRCD改进。一次短源曝光和小模型的结果也不能否定C01/C03或因果建模方向。所有模式/种子/负例和固定终点均保留，未挑选最好checkpoint。
