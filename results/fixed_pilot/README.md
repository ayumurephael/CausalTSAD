**结论更新：请同时阅读 [COMMON_LOGIT.md](COMMON_LOGIT.md)。原概率小涨不能单独作为改进证据；共同logit下检测增益不稳定且误报增加。下面保留原概率口径完整结果。**

# C01 三种子固定预算结果

12run，固定100开发系统；全部权重SHA、434updates、原标量曝光和种子内初始SHA/头Linear FLOPs核验通过。原检测基线在全部run数值一致。

| 范围 | 分支 | VUS-PR | AP | 正常FPR | 条件RCA Top1 |
|---|---|---:|---:|---:|---:|
| all | 原自训练TimeRCD | 0.420957 | 0.441622 | 0.003877 | — |
| all | pretrained_none | 0.426997 | 0.447319 | 0.004272 | 0.412500 |
| all | pretrained_parents | 0.427633 | 0.447997 | 0.004241 | 0.412500 |
| all | pretrained_rewired | 0.427750 | 0.447721 | 0.004271 | 0.420833 |
| all | raw_parents | 0.427980 | 0.447810 | 0.004205 | 0.383333 |
| multi | 原自训练TimeRCD | 0.418419 | 0.472207 | 0.001925 | — |
| multi | pretrained_none | 0.423960 | 0.477642 | 0.002458 | 0.412500 |
| multi | pretrained_parents | 0.424756 | 0.478463 | 0.002420 | 0.412500 |
| multi | pretrained_rewired | 0.424913 | 0.478141 | 0.002457 | 0.420833 |
| multi | raw_parents | 0.425157 | 0.478229 | 0.002367 | 0.383333 |
| id_multi | 原自训练TimeRCD | 0.647512 | 0.749708 | 0.002321 | — |
| id_multi | pretrained_none | 0.656866 | 0.758337 | 0.003094 | 0.458333 |
| id_multi | pretrained_parents | 0.656991 | 0.759198 | 0.003022 | 0.458333 |
| id_multi | pretrained_rewired | 0.656893 | 0.758236 | 0.003092 | 0.450000 |
| id_multi | raw_parents | 0.656551 | 0.758762 | 0.002918 | 0.491667 |
| heldout_multi | 原自训练TimeRCD | 0.189326 | 0.194706 | 0.001528 | — |
| heldout_multi | pretrained_none | 0.191054 | 0.196947 | 0.001822 | 0.366667 |
| heldout_multi | pretrained_parents | 0.192522 | 0.197728 | 0.001818 | 0.366667 |
| heldout_multi | pretrained_rewired | 0.192932 | 0.198045 | 0.001822 | 0.391667 |
| heldout_multi | raw_parents | 0.193762 | 0.197695 | 0.001817 | 0.275000 |
| C1 | 原自训练TimeRCD | 0.431109 | 0.319281 | 0.011689 | — |
| C1 | pretrained_none | 0.439143 | 0.326030 | 0.011529 | — |
| C1 | pretrained_parents | 0.439142 | 0.326132 | 0.011525 | — |
| C1 | pretrained_rewired | 0.439100 | 0.326044 | 0.011525 | — |
| C1 | raw_parents | 0.439275 | 0.326136 | 0.011556 | — |

## 预先固定的配对比较

- all / parents_fused_minus_base: {'mean': 0.006676602451623365, 'per_seed': [0.003185403062996784, 0.009618956185764227, 0.0072254481061090866], 'systems': 100, 'hierarchy_seed_system_bootstrap95': [-3.984862233545821e-05, 0.01734147649840547], 'improved_systems': 49, 'declined_systems': 28, 'note': 'FPR is lower-better. Other keys higher-better. Three optimization seeds are not independent systems.'}
- all / parents_minus_none: {'mean': 0.0006368893082374834, 'per_seed': [0.00019489841233892837, -4.100126186914511e-06, 0.0017198696385604363], 'systems': 100, 'hierarchy_seed_system_bootstrap95': [-0.0002895384360481468, 0.003717713529399234], 'improved_systems': 33, 'declined_systems': 36, 'note': 'FPR is lower-better. Other keys higher-better. Three optimization seeds are not independent systems.'}
- all / parents_minus_rewired: {'mean': -0.00011650720683090337, 'per_seed': [-0.002064294852395816, -4.899337558414596e-05, 0.001763766607487252], 'systems': 100, 'hierarchy_seed_system_bootstrap95': [-0.0028245537137558657, 0.002622145242641295], 'improved_systems': 34, 'declined_systems': 34, 'note': 'FPR is lower-better. Other keys higher-better. Three optimization seeds are not independent systems.'}
- all / pretrained_minus_raw: {'mean': -0.00034682022557613694, 'per_seed': [-0.0009777759593204629, 7.296805266430255e-05, -0.00013565277007225097], 'systems': 100, 'hierarchy_seed_system_bootstrap95': [-0.002797099281822458, 0.000997586361976974], 'improved_systems': 39, 'declined_systems': 35, 'note': 'FPR is lower-better. Other keys higher-better. Three optimization seeds are not independent systems.'}
- id_multi / parents_fused_minus_base: {'mean': 0.009478720162885956, 'per_seed': [0.005743893063348003, 0.014664238806620078, 0.00802802861868979], 'systems': 40, 'hierarchy_seed_system_bootstrap95': [-0.003227210277077217, 0.034851454636796814], 'improved_systems': 10, 'declined_systems': 11, 'note': 'FPR is lower-better. Other keys higher-better. Three optimization seeds are not independent systems.'}
- id_multi / parents_minus_none: {'mean': 0.00012488830784497076, 'per_seed': [0.0004973294348773676, -0.00014085167364993474, 1.818716230747936e-05], 'systems': 40, 'hierarchy_seed_system_bootstrap95': [-0.00036621312023666004, 0.000863750761348395], 'improved_systems': 10, 'declined_systems': 9, 'note': 'FPR is lower-better. Other keys higher-better. Three optimization seeds are not independent systems.'}
- id_multi / parents_minus_rewired: {'mean': 9.730568934009448e-05, 'per_seed': [0.0004999625741233434, -0.0001402309214989469, -6.781458460411315e-05], 'systems': 40, 'hierarchy_seed_system_bootstrap95': [-0.00041665796100285387, 0.0008347146228657242], 'improved_systems': 11, 'declined_systems': 8, 'note': 'FPR is lower-better. Other keys higher-better. Three optimization seeds are not independent systems.'}
- id_multi / pretrained_minus_raw: {'mean': 0.0004393755656023127, 'per_seed': [0.001767620215804113, -0.0003322531048621728, -0.00011724041413500234], 'systems': 40, 'hierarchy_seed_system_bootstrap95': [-0.000513869291161462, 0.002435215794852545], 'improved_systems': 11, 'declined_systems': 10, 'note': 'FPR is lower-better. Other keys higher-better. Three optimization seeds are not independent systems.'}
- heldout_multi / parents_fused_minus_base: {'mean': 0.003196047687057439, 'per_seed': [-0.0003908758093048399, 0.005438336829318311, 0.004540682041158844], 'systems': 40, 'hierarchy_seed_system_bootstrap95': [-0.0027736585584521947, 0.013979532959266565], 'improved_systems': 26, 'declined_systems': 11, 'note': 'FPR is lower-better. Other keys higher-better. Three optimization seeds are not independent systems.'}
- heldout_multi / parents_minus_none: {'mean': 0.0014677364982060407, 'per_seed': [2.568796194798768e-05, 0.00010057028862000177, 0.004276951244050134], 'systems': 40, 'hierarchy_seed_system_bootstrap95': [-0.0006586853838540946, 0.009303314808185087], 'improved_systems': 18, 'declined_systems': 16, 'note': 'FPR is lower-better. Other keys higher-better. Three optimization seeds are not independent systems.'}
- heldout_multi / parents_minus_rewired: {'mean': -0.00040975167718581353, 'per_seed': [-0.005764440411967964, -5.959317924061677e-06, 0.004541144698334586], 'systems': 40, 'hierarchy_seed_system_bootstrap95': [-0.00768194639232196, 0.006636392923579301], 'improved_systems': 17, 'declined_systems': 16, 'note': 'FPR is lower-better. Other keys higher-better. Three optimization seeds are not independent systems.'}
- heldout_multi / pretrained_minus_raw: {'mean': -0.001239787323350757, 'per_seed': [-0.004265407585661427, 0.0006824641839171757, -0.00013641856830801937], 'systems': 40, 'hierarchy_seed_system_bootstrap95': [-0.008695819913532365, 0.0011464775434282197], 'improved_systems': 21, 'declined_systems': 14, 'note': 'FPR is lower-better. Other keys higher-better. Three optimization seeds are not independent systems.'}
- C1 / parents_fused_minus_base: {'mean': 0.008033476558230035, 'per_seed': [0.005220980806897592, 0.00788962965694434, 0.010989819210848174], 'systems': 20, 'hierarchy_seed_system_bootstrap95': [0.0019326586329886989, 0.01626140623757139], 'improved_systems': 13, 'declined_systems': 6, 'note': 'FPR is lower-better. Other keys higher-better. Three optimization seeds are not independent systems.'}
- C1 / parents_minus_none: {'mean': -8.030709146064659e-07, 'per_seed': [-7.154273195606881e-05, 6.006213912529344e-05, 9.071380086955977e-06], 'systems': 20, 'hierarchy_seed_system_bootstrap95': [-0.0005136509344399067, 0.0004628567743607271], 'improved_systems': 5, 'declined_systems': 11, 'note': 'FPR is lower-better. Other keys higher-better. Three optimization seeds are not independent systems.'}
- C1 / parents_minus_rewired: {'mean': 4.235594153692075e-05, 'per_seed': [0.00020748141371015874, 4.741360092528734e-05, -0.00012782719002468382], 'systems': 20, 'hierarchy_seed_system_bootstrap95': [-0.0002661669931886438, 0.0004845762658494999], 'improved_systems': 6, 'declined_systems': 10, 'note': 'FPR is lower-better. Other keys higher-better. Three optimization seeds are not independent systems.'}
- C1 / pretrained_minus_raw: {'mean': -0.0001332776123837974, 'per_seed': [0.00010669494311231256, -0.0003355818947884931, -0.0001709458854752117], 'systems': 20, 'hierarchy_seed_system_bootstrap95': [-0.0008475764237949357, 0.0004908631612617647], 'improved_systems': 7, 'declined_systems': 11, 'note': 'FPR is lower-better. Other keys higher-better. Three optimization seeds are not independent systems.'}

## 判读边界

三种子并非三个独立数据集；表中为系统宏平均，事件先在系统内平均。FPR越低越好，其他指标越高越好。若融合有益但真图不胜空图/错图，不能归因于因果方向。只报告CE下降不足以证明关系学会。

根因仅在真实注入窗口内定位，未实现端到端RCA。raw控制的可训练头容量相同、原检测路径相同，但条件编码FLOPs较低。开发集用于探索，锁定与validation均未读。没有选最好seed/分数/checkpoint；所有下降与机制分层在analysis.json保留。

## 正反案例（按三种子平均、仅描述，不据此重选配置）

- 最大上升: timercd-synthetic-benchmark-v1/id/development/c08/sys00048 (id, C=8, ['outlier']) ΔVUS=0.386508
- 最大上升: timercd-synthetic-benchmark-v1/mechanism_holdout/development/c16/sys00009 (mechanism_holdout, C=16, ['measurement']) ΔVUS=0.133333
- 最大上升: timercd-synthetic-benchmark-v1/id/development/c01/sys00044 (id, C=1, ['continuous upward spike']) ΔVUS=0.049661
- 最大上升: timercd-synthetic-benchmark-v1/id/development/c01/sys00029 (id, C=1, ['outlier']) ΔVUS=0.038592
- 最大上升: timercd-synthetic-benchmark-v1/id/development/c12/sys00024 (id, C=12, ['sudden increase']) ΔVUS=0.037128
- 最大下降: timercd-synthetic-benchmark-v1/mechanism_holdout/development/c12/sys00026 (mechanism_holdout, C=12, ['self_dynamics']) ΔVUS=-0.037445
- 最大下降: timercd-synthetic-benchmark-v1/id/development/c20/sys00052 (id, C=20, ['rapid rise followed by slow decline']) ΔVUS=-0.032321
- 最大下降: timercd-synthetic-benchmark-v1/mechanism_holdout/development/c08/sys00027 (mechanism_holdout, C=8, ['self_dynamics']) ΔVUS=-0.015323
- 最大下降: timercd-synthetic-benchmark-v1/id/development/c16/sys00030 (id, C=16, ['decrease after downward spike']) ΔVUS=-0.014763
- 最大下降: timercd-synthetic-benchmark-v1/id/development/c16/sys00046 (id, C=16, ['shake']) ΔVUS=-0.012105
