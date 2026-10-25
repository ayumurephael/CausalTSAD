# C08 完整联合读出结果

全部24run完成、固定3pass终点；以下是共同raw-logit比较。不能用训练损失或单独RCA提升代替TSAD收益。

| 方法 | M AP | M VUS-PR | M正常FPR | 条件Top1 | 检出+定位Top1 |
|---|---:|---:|---:|---:|---:|
| base | 0.479401 | 0.415507 | 0.003309 | — | — |
| local_det | 0.479883 | 0.416679 | 0.003436 | 0.350000 | 0.237500 |
| local_joint | 0.475058 | 0.411534 | 0.004178 | 0.387500 | 0.283333 |
| dag_det | 0.479821 | 0.416624 | 0.003428 | 0.250000 | 0.154167 |
| dag_joint | 0.473481 | 0.412635 | 0.003986 | 0.420833 | 0.316667 |
| rewired_joint | 0.474511 | 0.410838 | 0.003833 | 0.354167 | 0.270833 |
| base_cal | 0.479401 | 0.415507 | 0.003478 | 0.100000 | 0.062500 |
| prior_joint | 0.479401 | 0.415507 | 0.003516 | 0.104167 | 0.050000 |
| dag_random_root | 0.479301 | 0.415499 | 0.003507 | 0.050000 | 0.020833 |

## 核心配对比较

{
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
  "dag_joint - local_joint": {
    "AP": {
      "mean": -0.0015773360651454458,
      "per_seed": [
        -0.0016710261519992816,
        -0.004159462213216404,
        0.0010984801697793475
      ],
      "hierarchical_seed_system_bootstrap95": [
        -0.008848643878848744,
        0.006455891047966137
      ],
      "systems": 80,
      "positive_systems": 30,
      "negative_systems": 36
    },
    "VUS-PR": {
      "mean": 0.0011013031560834867,
      "per_seed": [
        0.0008750308132172973,
        -0.0005344879027431473,
        0.0029633665577763093
      ],
      "hierarchical_seed_system_bootstrap95": [
        -0.002448093778052354,
        0.005940069388355818
      ],
      "systems": 80,
      "positive_systems": 29,
      "negative_systems": 26
    },
    "normal_FPR": {
      "mean": -0.00019281761123508695,
      "per_seed": [
        -8.390894246897807e-06,
        -0.0002586942874957831,
        -0.00031136765196258
      ],
      "hierarchical_seed_system_bootstrap95": [
        -0.0006292102645550938,
        6.236589984949155e-05
      ],
      "systems": 80,
      "positive_systems": 24,
      "negative_systems": 16
    }
  },
  "dag_joint - dag_det": {
    "AP": {
      "mean": -0.006339937773761004,
      "per_seed": [
        -0.003993917812896085,
        -0.006430950056610854,
        -0.008594945451776073
      ],
      "hierarchical_seed_system_bootstrap95": [
        -0.013047625580769166,
        -0.0006121818389559269
      ],
      "systems": 80,
      "positive_systems": 28,
      "negative_systems": 39
    },
    "VUS-PR": {
      "mean": -0.003988938765598789,
      "per_seed": [
        -0.004882452115320554,
        -0.0029471702212096225,
        -0.00413719396026619
      ],
      "hierarchical_seed_system_bootstrap95": [
        -0.00929955915766193,
        0.00037576004517357825
      ],
      "systems": 80,
      "positive_systems": 26,
      "negative_systems": 31
    },
    "normal_FPR": {
      "mean": 0.0005572378714079387,
      "per_seed": [
        0.0004380249108988689,
        0.0004685720845048022,
        0.0007651166188201448
      ],
      "hierarchical_seed_system_bootstrap95": [
        3.267908564963716e-05,
        0.0015680125854604383
      ],
      "systems": 80,
      "positive_systems": 28,
      "negative_systems": 10
    }
  },
  "dag_joint - rewired_joint": {
    "AP": {
      "mean": -0.001030392541342416,
      "per_seed": [
        0.0014757837856311852,
        -0.003811765999038498,
        -0.0007551954106199349
      ],
      "hierarchical_seed_system_bootstrap95": [
        -0.007202736311598224,
        0.004666414439949501
      ],
      "systems": 80,
      "positive_systems": 36,
      "negative_systems": 32
    },
    "VUS-PR": {
      "mean": 0.001797330905021629,
      "per_seed": [
        0.0031028569470807612,
        -0.0005796737865488076,
        0.0028688095545329317
      ],
      "hierarchical_seed_system_bootstrap95": [
        -0.000810614014873872,
        0.006470363223223521
      ],
      "systems": 80,
      "positive_systems": 30,
      "negative_systems": 24
    },
    "normal_FPR": {
      "mean": 0.00015204675251397493,
      "per_seed": [
        0.00031238752739391865,
        7.534555434099755e-05,
        6.840717580700856e-05
      ],
      "hierarchical_seed_system_bootstrap95": [
        -1.7919037959309045e-05,
        0.0004686142873883576
      ],
      "systems": 80,
      "positive_systems": 23,
      "negative_systems": 16
    }
  },
  "dag_joint - base_cal": {
    "AP": {
      "mean": -0.005919596474113827,
      "per_seed": [
        -0.0032241066964525888,
        -0.006288105590357256,
        -0.008246577135531636
      ],
      "hierarchical_seed_system_bootstrap95": [
        -0.012703559989159261,
        -0.00013903221302375777
      ],
      "systems": 80,
      "positive_systems": 32,
      "negative_systems": 34
    },
    "VUS-PR": {
      "mean": -0.0028713996316276267,
      "per_seed": [
        -0.0019007756313095495,
        -0.0027629029958585816,
        -0.003950520267714748
      ],
      "hierarchical_seed_system_bootstrap95": [
        -0.007504568821327913,
        0.0010948074751515086
      ],
      "systems": 80,
      "positive_systems": 27,
      "negative_systems": 28
    },
    "normal_FPR": {
      "mean": 0.0005070728241523913,
      "per_seed": [
        0.0004237613642553988,
        0.00041089440675121607,
        0.0006865627014505589
      ],
      "hierarchical_seed_system_bootstrap95": [
        -2.0973145000637166e-05,
        0.0015058318874015748
      ],
      "systems": 80,
      "positive_systems": 27,
      "negative_systems": 10
    }
  },
  "dag_joint - prior_joint": {
    "AP": {
      "mean": -0.005919596474113827,
      "per_seed": [
        -0.00322410669645259,
        -0.006288105590357253,
        -0.008246577135531636
      ],
      "hierarchical_seed_system_bootstrap95": [
        -0.012703559989159261,
        -0.00013903221302376007
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
      "mean": 0.0004696000419358475,
      "per_seed": [
        0.0003860689054851068,
        0.0003678582938096939,
        0.0006548729265127418
      ],
      "hierarchical_seed_system_bootstrap95": [
        -5.9786654494812045e-05,
        0.0014701055428005794
      ],
      "systems": 80,
      "positive_systems": 26,
      "negative_systems": 11
    }
  },
  "dag_joint - dag_random_root": {
    "AP": {
      "mean": -0.005819818750193648,
      "per_seed": [
        -0.003262447472475575,
        -0.006183440382684877,
        -0.00801356839542049
      ],
      "hierarchical_seed_system_bootstrap95": [
        -0.012563455864047876,
        -9.039130417164812e-05
      ],
      "systems": 80,
      "positive_systems": 33,
      "negative_systems": 34
    },
    "VUS-PR": {
      "mean": -0.002863419133386625,
      "per_seed": [
        -0.002023357288034417,
        -0.0028178385042570094,
        -0.003749061607868448
      ],
      "hierarchical_seed_system_bootstrap95": [
        -0.0074195731112407124,
        0.0010740894174264585
      ],
      "systems": 80,
      "positive_systems": 28,
      "negative_systems": 27
    },
    "normal_FPR": {
      "mean": 0.0004787871296322643,
      "per_seed": [
        0.00039708155513849125,
        0.00038710491933507127,
        0.0006521749144232302
      ],
      "hierarchical_seed_system_bootstrap95": [
        -5.6440147069346964e-05,
        0.0014711837334968547
      ],
      "systems": 80,
      "positive_systems": 28,
      "negative_systems": 10
    }
  },
  "local_joint - local_det": {
    "AP": {
      "mean": -0.004824592344125204,
      "per_seed": [
        -0.0023823098323721924,
        -0.002400775981262866,
        -0.009690691218740553
      ],
      "hierarchical_seed_system_bootstrap95": [
        -0.014841128540254789,
        0.0023886451371700566
      ],
      "systems": 80,
      "positive_systems": 30,
      "negative_systems": 36
    },
    "VUS-PR": {
      "mean": -0.005144708292771194,
      "per_seed": [
        -0.005731365155984054,
        -0.002475206179328593,
        -0.007227553543000935
      ],
      "hierarchical_seed_system_bootstrap95": [
        -0.012574954807820324,
        -6.980389803296105e-05
      ],
      "systems": 80,
      "positive_systems": 27,
      "negative_systems": 30
    },
    "normal_FPR": {
      "mean": 0.0007419927036753991,
      "per_seed": [
        0.0004699092305124518,
        0.0007056642144932306,
        0.001050404666020515
      ],
      "hierarchical_seed_system_bootstrap95": [
        3.580611681486529e-05,
        0.0020901763096860615
      ],
      "systems": 80,
      "positive_systems": 26,
      "negative_systems": 13
    }
  },
  "graph_x_root_interaction": {
    "AP": {
      "mean": -0.0015153454296358013,
      "per_seed": [
        -0.0016116079805238927,
        -0.00403017407534799,
        0.0010957457669644778
      ],
      "hierarchical_seed_system_bootstrap95": [
        -0.00872876492394466,
        0.006413794195185645
      ],
      "systems": 80,
      "positive_systems": 31,
      "negative_systems": 36
    },
    "VUS-PR": {
      "mean": 0.0011557695271724063,
      "per_seed": [
        0.0008489130406635001,
        -0.00047196404188102833,
        0.0030903595827347467
      ],
      "hierarchical_seed_system_bootstrap95": [
        -0.0023812668303284704,
        0.006011153063258621
      ],
      "systems": 80,
      "positive_systems": 28,
      "negative_systems": 27
    },
    "normal_FPR": {
      "mean": -0.00018475483226746052,
      "per_seed": [
        -3.1884319613583e-05,
        -0.0002370921299884283,
        -0.00028528804720037026
      ],
      "hierarchical_seed_system_bootstrap95": [
        -0.0006117191902078986,
        7.151723577596033e-05
      ],
      "systems": 80,
      "positive_systems": 21,
      "negative_systems": 19
    }
  }
}

## 源学习曲线

{
  "local_det": {
    "initial": {
      "AP": 0.8220650891633393,
      "RCA_Top1": 0.08130081300813008
    },
    "pass1": {
      "AP": 0.8201196190576664,
      "RCA_Top1": 0.4336043360433604
    },
    "final": {
      "AP": 0.8202273682588036,
      "RCA_Top1": 0.42818428184281837
    }
  },
  "local_joint": {
    "initial": {
      "AP": 0.8220650891633393,
      "RCA_Top1": 0.08130081300813008
    },
    "pass1": {
      "AP": 0.8161684905190715,
      "RCA_Top1": 0.6124661246612466
    },
    "final": {
      "AP": 0.8212999185965751,
      "RCA_Top1": 0.6856368563685638
    }
  },
  "dag_det": {
    "initial": {
      "AP": 0.8220650891633393,
      "RCA_Top1": 0.08130081300813008
    },
    "pass1": {
      "AP": 0.8202986975363133,
      "RCA_Top1": 0.23848238482384823
    },
    "final": {
      "AP": 0.8202466843845135,
      "RCA_Top1": 0.2682926829268293
    }
  },
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
  "rewired_joint": {
    "initial": {
      "AP": 0.8220650891633393,
      "RCA_Top1": 0.08130081300813008
    },
    "pass1": {
      "AP": 0.8173527125729473,
      "RCA_Top1": 0.6314363143631436
    },
    "final": {
      "AP": 0.8241775768473584,
      "RCA_Top1": 0.6639566395663957
    }
  },
  "base_cal": {
    "initial": {
      "AP": 0.8220650891633393,
      "RCA_Top1": 0.08130081300813008
    },
    "pass1": {
      "AP": 0.8220650891633393,
      "RCA_Top1": 0.08130081300813008
    },
    "final": {
      "AP": 0.8220650891633393,
      "RCA_Top1": 0.08130081300813008
    }
  },
  "prior_joint": {
    "initial": {
      "AP": 0.8220650891633393,
      "RCA_Top1": 0.08130081300813008
    },
    "pass1": {
      "AP": 0.8220650891633393,
      "RCA_Top1": 0.0840108401084011
    },
    "final": {
      "AP": 0.8220650891633393,
      "RCA_Top1": 0.09756097560975611
    }
  },
  "dag_random_root": {
    "initial": {
      "AP": 0.8220650891633393,
      "RCA_Top1": 0.08130081300813008
    },
    "pass1": {
      "AP": 0.8212846706596317,
      "RCA_Top1": 0.03523035230352304
    },
    "final": {
      "AP": 0.8216044694744596,
      "RCA_Top1": 0.021680216802168025
    }
  }
}

所有正反系统、各C/两轨/族/故障类型在analysis.json。层级bootstrap只描述当前系统与优化seed，不代表多重比较确认性检验或独立模型预训复制。无自动加预算。