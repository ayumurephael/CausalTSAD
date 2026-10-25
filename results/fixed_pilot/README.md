# C08b 读出与梯度四臂结果

9个新run与3个复用基线完成、固定3pass终点；以下是共同raw-logit比较。不能用训练损失或单独RCA提升代替TSAD收益。

| 方法 | M AP | M VUS-PR | M正常FPR | 条件Top1 | 检出+定位Top1 |
|---|---:|---:|---:|---:|---:|
| base | 0.479401 | 0.415507 | 0.003309 | — | — |
| dag_joint | 0.473481 | 0.412635 | 0.003986 | 0.420833 | 0.316667 |
| lse_pcgrad | 0.475101 | 0.413911 | 0.003950 | 0.429167 | 0.300000 |
| mean_sum | 0.479140 | 0.414588 | 0.003470 | 0.420833 | 0.300000 |
| mean_pcgrad | 0.480171 | 0.416150 | 0.003496 | 0.404167 | 0.287500 |

## 核心配对比较

{
  "mean_sum - dag_joint": {
    "AP": {
      "mean": 0.005659245223230404,
      "per_seed": [
        0.0032418572453646625,
        0.005714748622171287,
        0.008021129802155264
      ],
      "hierarchical_seed_system_bootstrap95": [
        0.0007597478198518839,
        0.011756713249481777
      ],
      "systems": 80,
      "positive_systems": 40,
      "negative_systems": 26
    },
    "VUS-PR": {
      "mean": 0.0019531099670574454,
      "per_seed": [
        0.0009726959248038261,
        0.0017868296494703024,
        0.0030998043268982076
      ],
      "hierarchical_seed_system_bootstrap95": [
        -0.0009532587780149496,
        0.005667512472581227
      ],
      "systems": 80,
      "positive_systems": 29,
      "negative_systems": 25
    },
    "normal_FPR": {
      "mean": -0.0005150942698127019,
      "per_seed": [
        -0.0004736579693349325,
        -0.00039879913400409013,
        -0.0006728257060990832
      ],
      "hierarchical_seed_system_bootstrap95": [
        -0.0014938356725819138,
        8.100993527411825e-07
      ],
      "systems": 80,
      "positive_systems": 10,
      "negative_systems": 26
    }
  },
  "mean_pcgrad - mean_sum": {
    "AP": {
      "mean": 0.0010308712115952767,
      "per_seed": [
        0.0008480772768211537,
        0.0013408795582351455,
        0.0009036567997295307
      ],
      "hierarchical_seed_system_bootstrap95": [
        0.00017234494627936148,
        0.002284884814180055
      ],
      "systems": 80,
      "positive_systems": 31,
      "negative_systems": 31
    },
    "VUS-PR": {
      "mean": 0.0015616939214280875,
      "per_seed": [
        0.0010217573418355697,
        0.003449504143421659,
        0.00021382027902703323
      ],
      "hierarchical_seed_system_bootstrap95": [
        0.00010857183521920466,
        0.005284361713214966
      ],
      "systems": 80,
      "positive_systems": 26,
      "negative_systems": 25
    },
    "normal_FPR": {
      "mean": 2.604518485319899e-05,
      "per_seed": [
        3.64250336693421e-05,
        4.101949253101203e-05,
        6.910283592428403e-07
      ],
      "hierarchical_seed_system_bootstrap95": [
        -3.9611237760568385e-05,
        8.547172078732476e-05
      ],
      "systems": 80,
      "positive_systems": 16,
      "negative_systems": 5
    }
  },
  "lse_pcgrad - dag_joint": {
    "AP": {
      "mean": 0.0016200159452447896,
      "per_seed": [
        0.001540370702441273,
        0.002055981429120046,
        0.0012636957041730495
      ],
      "hierarchical_seed_system_bootstrap95": [
        -0.0021047907249839297,
        0.0054012606174497334
      ],
      "systems": 80,
      "positive_systems": 37,
      "negative_systems": 29
    },
    "VUS-PR": {
      "mean": 0.001275346336602795,
      "per_seed": [
        0.0022057466566649635,
        0.0015189791672402387,
        0.00010131318590318305
      ],
      "hierarchical_seed_system_bootstrap95": [
        -0.0029781451919647,
        0.006535291270979364
      ],
      "systems": 80,
      "positive_systems": 29,
      "negative_systems": 27
    },
    "normal_FPR": {
      "mean": -3.59193277074077e-05,
      "per_seed": [
        -5.7123847936238614e-05,
        3.0130924099366366e-05,
        -8.076505928535084e-05
      ],
      "hierarchical_seed_system_bootstrap95": [
        -0.00019985698397333043,
        7.165955222542416e-05
      ],
      "systems": 80,
      "positive_systems": 14,
      "negative_systems": 17
    }
  },
  "mean_pcgrad - lse_pcgrad": {
    "AP": {
      "mean": 0.005070100489580892,
      "per_seed": [
        0.0025495638197445434,
        0.0049996467512863855,
        0.007661090897711745
      ],
      "hierarchical_seed_system_bootstrap95": [
        -3.3153631896783975e-05,
        0.01164114153124855
      ],
      "systems": 80,
      "positive_systems": 34,
      "negative_systems": 31
    },
    "VUS-PR": {
      "mean": 0.002239457551882738,
      "per_seed": [
        -0.00021129339002556782,
        0.003717354625651723,
        0.0032123114200220577
      ],
      "hierarchical_seed_system_bootstrap95": [
        -0.003364311891749868,
        0.008838515578907661
      ],
      "systems": 80,
      "positive_systems": 26,
      "negative_systems": 28
    },
    "normal_FPR": {
      "mean": -0.00045312975725209526,
      "per_seed": [
        -0.0003801090877293519,
        -0.0003879105655724445,
        -0.0005913696184544895
      ],
      "hierarchical_seed_system_bootstrap95": [
        -0.00130893697318908,
        1.7932774070140106e-05
      ],
      "systems": 80,
      "positive_systems": 13,
      "negative_systems": 20
    }
  },
  "dag_joint - base": {
    "AP": {
      "mean": -0.0059195964741138255,
      "per_seed": [
        -0.0032241066964525888,
        -0.006288105590357251,
        -0.008246577135531636
      ],
      "hierarchical_seed_system_bootstrap95": [
        -0.012703559989159256,
        -0.00013903221302375777
      ],
      "systems": 80,
      "positive_systems": 32,
      "negative_systems": 34
    },
    "VUS-PR": {
      "mean": -0.002871400189961144,
      "per_seed": [
        -0.0019007756313095495,
        -0.0027629046708591357,
        -0.003950520267714748
      ],
      "hierarchical_seed_system_bootstrap95": [
        -0.007504568849244589,
        0.0010948074611931708
      ],
      "systems": 80,
      "positive_systems": 27,
      "negative_systems": 28
    },
    "normal_FPR": {
      "mean": 0.00067668078976039,
      "per_seed": [
        0.0006081039546474798,
        0.0005845165271428358,
        0.0008374218874908548
      ],
      "hierarchical_seed_system_bootstrap95": [
        0.00010521818501474697,
        0.0016852844341170956
      ],
      "systems": 80,
      "positive_systems": 32,
      "negative_systems": 6
    }
  },
  "lse_pcgrad - base": {
    "AP": {
      "mean": -0.004299580528869034,
      "per_seed": [
        -0.0016837359940113158,
        -0.004232124161237206,
        -0.006982881431358584
      ],
      "hierarchical_seed_system_bootstrap95": [
        -0.011347742729571893,
        0.0017554050846727526
      ],
      "systems": 80,
      "positive_systems": 33,
      "negative_systems": 33
    },
    "VUS-PR": {
      "mean": -0.001596053853358349,
      "per_seed": [
        0.0003049710253554139,
        -0.001243925503618897,
        -0.003849207081811565
      ],
      "hierarchical_seed_system_bootstrap95": [
        -0.00864811767974683,
        0.005149280894969075
      ],
      "systems": 80,
      "positive_systems": 28,
      "negative_systems": 28
    },
    "normal_FPR": {
      "mean": 0.0006407614620529824,
      "per_seed": [
        0.0005509801067112411,
        0.0006146474512422021,
        0.0007566568282055038
      ],
      "hierarchical_seed_system_bootstrap95": [
        8.972540780603848e-05,
        0.0015555257379471503
      ],
      "systems": 80,
      "positive_systems": 28,
      "negative_systems": 6
    }
  },
  "mean_sum - base": {
    "AP": {
      "mean": -0.0002603512508834205,
      "per_seed": [
        1.775054891207358e-05,
        -0.0005733569681859648,
        -0.00022544733337637076
      ],
      "hierarchical_seed_system_bootstrap95": [
        -0.002389648717254274,
        0.0021461602203210048
      ],
      "systems": 80,
      "positive_systems": 37,
      "negative_systems": 26
    },
    "VUS-PR": {
      "mean": -0.0009182902229036993,
      "per_seed": [
        -0.000928079706505724,
        -0.000976075021388833,
        -0.0008507159408165406
      ],
      "hierarchical_seed_system_bootstrap95": [
        -0.0026069079917845025,
        0.0005439012178213465
      ],
      "systems": 80,
      "positive_systems": 29,
      "negative_systems": 24
    },
    "normal_FPR": {
      "mean": 0.00016158651994768812,
      "per_seed": [
        0.00013444598531254717,
        0.00018571739313874553,
        0.00016459618139177154
      ],
      "hierarchical_seed_system_bootstrap95": [
        4.7403941769789974e-05,
        0.0002948118813611328
      ],
      "systems": 80,
      "positive_systems": 21,
      "negative_systems": 6
    }
  },
  "mean_pcgrad - base": {
    "AP": {
      "mean": 0.0007705199607118562,
      "per_seed": [
        0.0008658278257332273,
        0.0007675225900491807,
        0.0006782094663531602
      ],
      "hierarchical_seed_system_bootstrap95": [
        -0.001474395097070133,
        0.0038239584725881107
      ],
      "systems": 80,
      "positive_systems": 34,
      "negative_systems": 29
    },
    "VUS-PR": {
      "mean": 0.0006434036985243884,
      "per_seed": [
        9.367763532984611e-05,
        0.0024734291220328266,
        -0.0006368956617895074
      ],
      "hierarchical_seed_system_bootstrap95": [
        -0.0012952935880133874,
        0.004219842315552077
      ],
      "systems": 80,
      "positive_systems": 27,
      "negative_systems": 26
    },
    "normal_FPR": {
      "mean": 0.0001876317048008871,
      "per_seed": [
        0.0001708710189818893,
        0.00022673688566975754,
        0.00016528720975101437
      ],
      "hierarchical_seed_system_bootstrap95": [
        4.832962595245109e-05,
        0.00035101847414309195
      ],
      "systems": 80,
      "positive_systems": 23,
      "negative_systems": 6
    }
  },
  "readout_x_gradient_interaction": {
    "AP": {
      "mean": -0.0005891447336495128,
      "per_seed": [
        -0.0006922934256201191,
        -0.0007151018708849003,
        -0.0003600389044435188
      ],
      "hierarchical_seed_system_bootstrap95": [
        -0.003899814152879063,
        0.0028932147451290286
      ],
      "systems": 80,
      "positive_systems": 29,
      "negative_systems": 38
    },
    "VUS-PR": {
      "mean": 0.00028634758482529225,
      "per_seed": [
        -0.0011839893148293942,
        0.0019305249761814203,
        0.00011250709312385026
      ],
      "hierarchical_seed_system_bootstrap95": [
        -0.003815329955488039,
        0.0042408731150011866
      ],
      "systems": 80,
      "positive_systems": 24,
      "negative_systems": 31
    },
    "normal_FPR": {
      "mean": 6.196451256060667e-05,
      "per_seed": [
        9.354888160558069e-05,
        1.0888568431645656e-05,
        8.145608764459368e-05
      ],
      "hierarchical_seed_system_bootstrap95": [
        -3.367274657186889e-05,
        0.00022503539060514792
      ],
      "systems": 80,
      "positive_systems": 22,
      "negative_systems": 9
    }
  }
}

## 源学习曲线

{
  "dag_joint": {
    "initial": {
      "AP": 0.8220650891633393,
      "RCA_Top1": 0.08130081300813008
    },
    "pass1": {
      "AP": 0.8166499822595644,
      "RCA_Top1": 0.6368563685636855
    },
    "final": {
      "AP": 0.822463793016246,
      "RCA_Top1": 0.6883468834688347
    }
  },
  "lse_pcgrad": {
    "initial": {
      "AP": 0.8220650891633393,
      "RCA_Top1": 0.08130081300813008
    },
    "pass1": {
      "AP": 0.8183510394199748,
      "RCA_Top1": 0.6558265582655828
    },
    "final": {
      "AP": 0.8245604238158609,
      "RCA_Top1": 0.6964769647696477
    }
  },
  "mean_sum": {
    "initial": {
      "AP": 0.8220650891633393,
      "RCA_Top1": 0.08130081300813008
    },
    "pass1": {
      "AP": 0.8214851350868665,
      "RCA_Top1": 0.6314363143631437
    },
    "final": {
      "AP": 0.8264629550887909,
      "RCA_Top1": 0.6829268292682927
    }
  },
  "mean_pcgrad": {
    "initial": {
      "AP": 0.8220650891633393,
      "RCA_Top1": 0.08130081300813008
    },
    "pass1": {
      "AP": 0.821247145184017,
      "RCA_Top1": 0.6531165311653117
    },
    "final": {
      "AP": 0.8267386905491566,
      "RCA_Top1": 0.7046070460704607
    }
  }
}

所有正反系统、各C/两轨/族/故障类型在analysis.json。层级bootstrap只描述当前系统与优化seed，不代表多重比较确认性检验或独立模型预训复制。无自动加预算。