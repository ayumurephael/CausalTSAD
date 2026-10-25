# C08根因置信度耦合与源训练梯度反证

12组、0optimizer，旧分数null<1e-6且权重SHA均不变。KL代数恒等式不等于实际失败原因；下面为固定干预，而非训练新模型。

| 模型 | 读出 | M开发AP | M正常FPR |
|---|---|---:|---:|
| dag_joint | original | 0.473481 | 0.003986 |
| dag_joint | remove_uniform_to_root_KL | 0.479315 | 0.003459 |
| dag_joint | subtract_source_normal_mean_KL | 0.473481 | 0.003986 |
| dag_joint | subtract_time_shuffled_KL | 0.472062 | 0.003965 |
| local_joint | original | 0.475058 | 0.004178 |
| local_joint | remove_uniform_to_root_KL | 0.480571 | 0.003416 |
| local_joint | subtract_source_normal_mean_KL | 0.475058 | 0.004178 |
| local_joint | subtract_time_shuffled_KL | 0.473908 | 0.004155 |
| dag_det | original | 0.479821 | 0.003428 |
| dag_det | remove_uniform_to_root_KL | 0.479824 | 0.003438 |
| dag_det | subtract_source_normal_mean_KL | 0.479821 | 0.003428 |
| dag_det | subtract_time_shuffled_KL | 0.480171 | 0.003430 |
| prior_joint | original | 0.479401 | 0.003516 |
| prior_joint | remove_uniform_to_root_KL | 0.479401 | 0.003514 |
| prior_joint | subtract_source_normal_mean_KL | 0.479401 | 0.003516 |
| prior_joint | subtract_time_shuffled_KL | 0.479132 | 0.003498 |

## 源训练梯度

{
  "dag_joint": {
    "initial": {
      "samples": 192,
      "conflict_fraction": 0.6041666666666666,
      "mean_cosine": -0.02328505386756979,
      "root_to_detection_norm_median": 2.5208551448239023,
      "PCGrad_angle_magnitude_condition_fraction": 0.11979166666666667
    },
    "final": {
      "samples": 192,
      "conflict_fraction": 0.6197916666666666,
      "mean_cosine": -0.1614731470381251,
      "root_to_detection_norm_median": 1.7100665019828216,
      "PCGrad_angle_magnitude_condition_fraction": 0.2708333333333333
    }
  },
  "local_joint": {
    "initial": {
      "samples": 192,
      "conflict_fraction": 0.5833333333333334,
      "mean_cosine": -0.021518551121074345,
      "root_to_detection_norm_median": 2.460140477327629,
      "PCGrad_angle_magnitude_condition_fraction": 0.13020833333333334
    },
    "final": {
      "samples": 192,
      "conflict_fraction": 0.7395833333333334,
      "mean_cosine": -0.1874112068845045,
      "root_to_detection_norm_median": 0.9998389094572475,
      "PCGrad_angle_magnitude_condition_fraction": 0.296875
    }
  },
  "dag_det": {
    "initial": {
      "samples": 192,
      "conflict_fraction": 0.6041666666666666,
      "mean_cosine": -0.02328505386756979,
      "root_to_detection_norm_median": 2.5208551448239023,
      "PCGrad_angle_magnitude_condition_fraction": 0.11979166666666667
    },
    "final": {
      "samples": 192,
      "conflict_fraction": 0.5677083333333334,
      "mean_cosine": -0.08908228743265058,
      "root_to_detection_norm_median": 3.3680484057452933,
      "PCGrad_angle_magnitude_condition_fraction": 0.3541666666666667
    }
  },
  "prior_joint": {
    "initial": {
      "samples": 192,
      "conflict_fraction": 0.5677083333333334,
      "mean_cosine": -0.0017570311532838512,
      "root_to_detection_norm_median": 0.4489507120406131,
      "PCGrad_angle_magnitude_condition_fraction": 0.03125
    },
    "final": {
      "samples": 192,
      "conflict_fraction": 0.4791666666666667,
      "mean_cosine": 0.020655395813564903,
      "root_to_detection_norm_median": 0.71465837600141,
      "PCGrad_angle_magnitude_condition_fraction": 0.046875
    }
  }
}

全部配对区间/正反系统/源与两轨分层在analysis.json。负cosine不能单独证明PCGrad一定有效，高曲率未测；固定去KL失败不能否定重新训练的解耦读出。