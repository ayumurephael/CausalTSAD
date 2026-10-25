# C05测量隔离源机制验证

6个冻结模型、123独立源M系统、28,237,824前向标量、0优化。所有源输入/权重SHA与不相关通道null检查通过。

{
  "status": "complete",
  "groups": {
    "all": {
      "none": {
        "normal": {
          "Top1_ready": 0.08888888888888889,
          "child_score_mean": 0.3396500825100912
        },
        "original": {
          "Top1_ready": 0.6361111111111111,
          "child_score_mean": 0.37335633578099636
        },
        "measurement_only": {
          "Top1_ready": 0.6444444444444445,
          "child_score_mean": 0.3396500825100912
        }
      },
      "parents_current": {
        "normal": {
          "Top1_ready": 0.10277777777777777,
          "child_score_mean": 0.39757220301991114
        },
        "original": {
          "Top1_ready": 0.6611111111111111,
          "child_score_mean": 0.4381177017264655
        },
        "measurement_only": {
          "Top1_ready": 0.6694444444444444,
          "child_score_mean": 0.41054581528529666
        }
      }
    },
    "root_has_children": {
      "none": {
        "normal": {
          "Top1_ready": 0.07179487179487179,
          "child_score_mean": 0.3396500825100912
        },
        "original": {
          "Top1_ready": 0.5743589743589743,
          "child_score_mean": 0.37335633578099636
        },
        "measurement_only": {
          "Top1_ready": 0.5897435897435898,
          "child_score_mean": 0.3396500825100912
        }
      },
      "parents_current": {
        "normal": {
          "Top1_ready": 0.07179487179487179,
          "child_score_mean": 0.39757220301991114
        },
        "original": {
          "Top1_ready": 0.5948717948717949,
          "child_score_mean": 0.4381177017264655
        },
        "measurement_only": {
          "Top1_ready": 0.6102564102564103,
          "child_score_mean": 0.41054581528529666
        }
      }
    },
    "root_has_no_children": {
      "none": {
        "normal": {
          "Top1_ready": 0.10909090909090909,
          "child_score_mean": null
        },
        "original": {
          "Top1_ready": 0.7090909090909091,
          "child_score_mean": null
        },
        "measurement_only": {
          "Top1_ready": 0.7090909090909091,
          "child_score_mean": null
        }
      },
      "parents_current": {
        "normal": {
          "Top1_ready": 0.1393939393939394,
          "child_score_mean": null
        },
        "original": {
          "Top1_ready": 0.7393939393939394,
          "child_score_mean": null
        },
        "measurement_only": {
          "Top1_ready": 0.7393939393939394,
          "child_score_mean": null
        }
      }
    }
  },
  "child_effects": {
    "all": {
      "child_score_change_from_bad_parent": {
        "mean": 0.012973609225045014,
        "per_seed": [
          0.020154645137881743,
          0.0052282715207324,
          0.0135379110165209
        ],
        "hierarchical_seed_system_bootstrap95": [
          -0.00700446509454592,
          0.05415916263776636
        ],
        "systems": 65,
        "positive_systems": 35,
        "negative_systems": 30
      },
      "child_exceedance_delta": {
        "mean": 3.975352812562115e-05,
        "per_seed": [
          0.0,
          0.00011926058437686345,
          0.0
        ],
        "hierarchical_seed_system_bootstrap95": [
          0.0,
          0.0002385211687537269
        ],
        "systems": 65,
        "positive_systems": 1,
        "negative_systems": 0
      }
    },
    "root_has_children": {
      "child_score_change_from_bad_parent": {
        "mean": 0.012973609225045014,
        "per_seed": [
          0.020154645137881743,
          0.0052282715207324,
          0.0135379110165209
        ],
        "hierarchical_seed_system_bootstrap95": [
          -0.006879753528461524,
          0.054356760631934974
        ],
        "systems": 65,
        "positive_systems": 35,
        "negative_systems": 30
      },
      "child_exceedance_delta": {
        "mean": 3.975352812562115e-05,
        "per_seed": [
          0.0,
          0.00011926058437686345,
          0.0
        ],
        "hierarchical_seed_system_bootstrap95": [
          0.0,
          0.0002385211687537269
        ],
        "systems": 65,
        "positive_systems": 1,
        "negative_systems": 0
      }
    }
  },
  "all_root_with_children_RCA_contrasts": [
    {
      "system_id": "c8/1",
      "channels": 8,
      "outdegree": 2,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c8/3",
      "channels": 8,
      "outdegree": 1,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c8/5",
      "channels": 8,
      "outdegree": 2,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c8/6",
      "channels": 8,
      "outdegree": 1,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c8/7",
      "channels": 8,
      "outdegree": 4,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c8/8",
      "channels": 8,
      "outdegree": 2,
      "Top1_delta": 0.6666666666666667
    },
    {
      "system_id": "c8/9",
      "channels": 8,
      "outdegree": 1,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c8/13",
      "channels": 8,
      "outdegree": 2,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c8/14",
      "channels": 8,
      "outdegree": 1,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c8/15",
      "channels": 8,
      "outdegree": 1,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c8/17",
      "channels": 8,
      "outdegree": 1,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c8/18",
      "channels": 8,
      "outdegree": 1,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c8/19",
      "channels": 8,
      "outdegree": 1,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c8/20",
      "channels": 8,
      "outdegree": 3,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c8/23",
      "channels": 8,
      "outdegree": 3,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c8/25",
      "channels": 8,
      "outdegree": 1,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c8/26",
      "channels": 8,
      "outdegree": 2,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c8/27",
      "channels": 8,
      "outdegree": 1,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c8/28",
      "channels": 8,
      "outdegree": 1,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c8/29",
      "channels": 8,
      "outdegree": 1,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c8/30",
      "channels": 8,
      "outdegree": 3,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c8/31",
      "channels": 8,
      "outdegree": 2,
      "Top1_delta": 0.6666666666666667
    },
    {
      "system_id": "c8/33",
      "channels": 8,
      "outdegree": 1,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c8/36",
      "channels": 8,
      "outdegree": 1,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c8/38",
      "channels": 8,
      "outdegree": 1,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c8/39",
      "channels": 8,
      "outdegree": 1,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c8/40",
      "channels": 8,
      "outdegree": 2,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c8/43",
      "channels": 8,
      "outdegree": 2,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c12/1",
      "channels": 12,
      "outdegree": 1,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c12/6",
      "channels": 12,
      "outdegree": 1,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c12/7",
      "channels": 12,
      "outdegree": 1,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c12/10",
      "channels": 12,
      "outdegree": 1,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c12/12",
      "channels": 12,
      "outdegree": 2,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c12/14",
      "channels": 12,
      "outdegree": 2,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c12/16",
      "channels": 12,
      "outdegree": 3,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c12/18",
      "channels": 12,
      "outdegree": 1,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c12/20",
      "channels": 12,
      "outdegree": 3,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c12/21",
      "channels": 12,
      "outdegree": 2,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c12/24",
      "channels": 12,
      "outdegree": 2,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c12/25",
      "channels": 12,
      "outdegree": 2,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c12/27",
      "channels": 12,
      "outdegree": 2,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c12/29",
      "channels": 12,
      "outdegree": 1,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c12/30",
      "channels": 12,
      "outdegree": 1,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c16/1",
      "channels": 16,
      "outdegree": 2,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c16/2",
      "channels": 16,
      "outdegree": 2,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c16/4",
      "channels": 16,
      "outdegree": 4,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c16/5",
      "channels": 16,
      "outdegree": 1,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c16/6",
      "channels": 16,
      "outdegree": 1,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c16/7",
      "channels": 16,
      "outdegree": 1,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c16/11",
      "channels": 16,
      "outdegree": 2,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c16/12",
      "channels": 16,
      "outdegree": 1,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c16/14",
      "channels": 16,
      "outdegree": 2,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c16/15",
      "channels": 16,
      "outdegree": 2,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c16/16",
      "channels": 16,
      "outdegree": 2,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c16/17",
      "channels": 16,
      "outdegree": 2,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c16/20",
      "channels": 16,
      "outdegree": 3,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c16/21",
      "channels": 16,
      "outdegree": 2,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c16/22",
      "channels": 16,
      "outdegree": 3,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c20/1",
      "channels": 20,
      "outdegree": 1,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c20/3",
      "channels": 20,
      "outdegree": 1,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c20/6",
      "channels": 20,
      "outdegree": 1,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c20/9",
      "channels": 20,
      "outdegree": 1,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c20/10",
      "channels": 20,
      "outdegree": 1,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c20/16",
      "channels": 20,
      "outdegree": 2,
      "Top1_delta": 0.0
    },
    {
      "system_id": "c20/18",
      "channels": 20,
      "outdegree": 2,
      "Top1_delta": 0.0
    }
  ],
  "original_root_parents_exactly_normal_systems": 123,
  "original_nonroot_response_systems": 39,
  "extra_frozen_query_scalars": 28237824,
  "optimizer_updates": 0,
  "interpretation": "Measurement-only construction holds nonroot observations normal and root observations fixed. Increased child score is spurious child evidence in this probe, not an effect label or proof that all original samples are physical state faults. Root-window conditional source RCA only."
}

这不是完整C05潜状态模型，也未测目标TSAD收益。