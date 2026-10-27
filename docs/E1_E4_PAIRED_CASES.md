# E4逐系统正反结果



完整系统ID公共前缀为timercd-synthetic-benchmark-v1/，表中省略development。仅列76个后缀含有效异常的系统；三seed先平均。按真图[x,r]减原multi AP升序，所有下降例均保留。单条的巨大提升不能当总体普遍提升。



| 系统ID（省略公共前缀） | 异常类型 | 原multi AP | 真图[x,r] AP | 错图[x,r] AP | 真图−原multi (pp) |
| --- | --- | --- | --- | --- | --- |
| id/c20/sys00010 | upward spike | 95.000 | 53.748 | 35.997 | -41.252 |
| id/c08/sys00013 | outlier | 81.250 | 44.074 | 6.247 | -37.176 |
| id/c08/sys00037 | upward spike | 100.000 | 64.459 | 76.481 | -35.541 |
| id/c08/sys00005 | downward spike | 100.000 | 78.836 | 91.667 | -21.164 |
| id/c08/sys00008 | upward spike | 100.000 | 83.333 | 83.333 | -16.667 |
| id/c20/sys00052 | rapid rise followed by slow decline | 60.865 | 45.823 | 45.616 | -15.043 |
| id/c16/sys00052 | outlier | 68.537 | 57.455 | 47.275 | -11.082 |
| id/c20/sys00018 | shake | 95.989 | 85.683 | 86.964 | -10.306 |
| mechanism_holdout/c12/sys00026 | self_dynamics | 12.681 | 5.700 | 11.075 | -6.980 |
| id/c16/sys00010 | wide downward spike | 99.892 | 93.430 | 89.894 | -6.462 |
| id/c12/sys00040 | increase after downward spike | 95.078 | 88.930 | 96.055 | -6.147 |
| mechanism_holdout/c16/sys00009 | measurement | 83.036 | 79.306 | 75.347 | -3.730 |
| mechanism_holdout/c08/sys00027 | self_dynamics | 64.516 | 60.816 | 58.821 | -3.700 |
| id/c20/sys00003 | continuous downward spike | 97.619 | 93.981 | 24.508 | -3.638 |
| id/c12/sys00051 | slow decline followed by rapid rise | 99.902 | 96.563 | 97.191 | -3.338 |
| id/c20/sys00028 | increase after downward spike | 100.000 | 96.760 | 99.356 | -3.240 |
| mechanism_holdout/c08/sys00033 | edge_lag | 5.838 | 3.712 | 4.040 | -2.125 |
| id/c16/sys00000 | downward convex | 68.767 | 67.186 | 67.530 | -1.581 |
| id/c20/sys00034 | wide upward spike | 99.987 | 99.061 | 93.445 | -0.926 |
| id/c12/sys00011 | sudden decrease | 3.020 | 2.325 | 2.783 | -0.695 |
| id/c08/sys00011 | slow decline followed by rapid rise | 99.127 | 98.438 | 98.407 | -0.688 |
| mechanism_holdout/c20/sys00053 | measurement | 2.155 | 1.547 | 1.580 | -0.608 |
| mechanism_holdout/c16/sys00016 | self_dynamics | 2.623 | 2.262 | 2.413 | -0.361 |
| mechanism_holdout/c12/sys00039 | measurement | 100.000 | 99.926 | 99.901 | -0.074 |
| mechanism_holdout/c16/sys00038 | measurement | 0.220 | 0.151 | 0.154 | -0.069 |
| id/c12/sys00032 | decrease after upward spike | 99.419 | 99.360 | 99.198 | -0.060 |
| mechanism_holdout/c08/sys00017 | edge_lag | 0.241 | 0.205 | 0.208 | -0.036 |
| mechanism_holdout/c20/sys00006 | measurement | 100.000 | 99.976 | 99.976 | -0.024 |
| mechanism_holdout/c20/sys00011 | edge_lag | 0.249 | 0.226 | 0.261 | -0.023 |
| mechanism_holdout/c20/sys00008 | measurement | 0.162 | 0.150 | 0.149 | -0.012 |
| mechanism_holdout/c16/sys00020 | measurement | 0.167 | 0.159 | 0.170 | -0.008 |
| mechanism_holdout/c08/sys00032 | measurement | 100.000 | 99.998 | 99.998 | -0.002 |
| id/c16/sys00024 | outlier | 100.000 | 100.000 | 100.000 | +0.000 |
| id/c12/sys00027 | outlier | 100.000 | 100.000 | 100.000 | +0.000 |
| id/c08/sys00034 | outlier | 100.000 | 100.000 | 100.000 | +0.000 |
| id/c16/sys00031 | outlier | 100.000 | 100.000 | 100.000 | +0.000 |
| id/c08/sys00019 | outlier | 100.000 | 100.000 | 100.000 | +0.000 |
| id/c08/sys00043 | upward spike | 100.000 | 100.000 | 100.000 | +0.000 |
| id/c08/sys00048 | outlier | 100.000 | 100.000 | 100.000 | +0.000 |
| mechanism_holdout/c16/sys00045 | self_dynamics | 0.066 | 0.066 | 0.065 | +0.000 |
| mechanism_holdout/c08/sys00011 | self_dynamics | 0.014 | 0.014 | 0.014 | +0.000 |
| mechanism_holdout/c20/sys00021 | self_dynamics | 0.013 | 0.013 | 0.013 | +0.000 |
| mechanism_holdout/c08/sys00019 | edge_coefficient | 0.046 | 0.051 | 0.054 | +0.005 |
| mechanism_holdout/c20/sys00049 | edge_coefficient | 0.426 | 0.433 | 0.428 | +0.008 |
| id/c16/sys00045 | shake | 99.990 | 99.999 | 99.999 | +0.009 |
| mechanism_holdout/c16/sys00039 | measurement | 0.177 | 0.186 | 0.196 | +0.009 |
| mechanism_holdout/c16/sys00007 | edge_lag | 0.075 | 0.086 | 0.089 | +0.011 |
| mechanism_holdout/c12/sys00022 | self_dynamics | 2.308 | 2.330 | 2.343 | +0.022 |
| mechanism_holdout/c08/sys00021 | edge_lag | 0.144 | 0.173 | 0.146 | +0.028 |
| mechanism_holdout/c12/sys00007 | self_dynamics | 0.476 | 0.514 | 0.518 | +0.038 |
| mechanism_holdout/c16/sys00026 | measurement | 0.358 | 0.401 | 0.423 | +0.043 |
| id/c12/sys00019 | outlier | 0.100 | 0.144 | 0.084 | +0.044 |
| mechanism_holdout/c20/sys00045 | edge_lag | 0.063 | 0.133 | 0.125 | +0.070 |
| id/c20/sys00043 | shake | 99.923 | 99.995 | 99.995 | +0.071 |
| mechanism_holdout/c08/sys00031 | edge_coefficient | 0.572 | 0.644 | 0.657 | +0.072 |
| mechanism_holdout/c12/sys00025 | measurement | 99.830 | 99.936 | 99.928 | +0.106 |
| mechanism_holdout/c12/sys00050 | self_dynamics | 0.388 | 0.602 | 0.559 | +0.214 |
| mechanism_holdout/c12/sys00027 | edge_coefficient | 1.552 | 1.767 | 1.839 | +0.215 |
| mechanism_holdout/c12/sys00029 | edge_lag | 0.413 | 0.677 | 0.743 | +0.264 |
| mechanism_holdout/c16/sys00041 | edge_lag | 1.796 | 2.380 | 2.329 | +0.584 |
| mechanism_holdout/c16/sys00001 | edge_coefficient | 2.714 | 3.301 | 3.114 | +0.587 |
| mechanism_holdout/c20/sys00032 | edge_lag | 9.501 | 10.149 | 9.607 | +0.648 |
| mechanism_holdout/c12/sys00008 | edge_coefficient | 6.443 | 7.134 | 6.911 | +0.692 |
| mechanism_holdout/c08/sys00048 | measurement | 49.679 | 50.474 | 54.391 | +0.795 |
| id/c08/sys00044 | increase after upward spike | 96.518 | 99.976 | 99.360 | +3.458 |
| mechanism_holdout/c08/sys00009 | measurement | 60.437 | 66.095 | 59.252 | +5.658 |
| id/c20/sys00000 | outlier | 0.052 | 5.999 | 1.185 | +5.947 |
| id/c12/sys00050 | outlier | 89.091 | 98.897 | 98.317 | +9.807 |
| id/c16/sys00015 | outlier | 81.000 | 91.540 | 100.000 | +10.540 |
| id/c12/sys00002 | outlier | 60.147 | 71.410 | 76.289 | +11.262 |
| mechanism_holdout/c20/sys00038 | self_dynamics | 25.757 | 37.531 | 42.429 | +11.774 |
| id/c16/sys00007 | slow decline followed by rapid rise | 76.926 | 97.598 | 95.553 | +20.672 |
| id/c20/sys00030 | upward spike | 0.029 | 50.020 | 0.220 | +49.991 |
| id/c12/sys00042 | continuous upward spike | 10.012 | 97.055 | 0.963 | +87.042 |
| id/c16/sys00046 | shake | 10.390 | 99.463 | 7.589 | +89.072 |
| id/c20/sys00031 | outlier | 0.412 | 100.000 | 1.007 | +99.588 |




全部E1–E4主对比的逐系统差值位于analysis.json的paired/*/all_system_differences；追加对比位于exploratory_followup.json的paired。正常FPR包含另外4个无有效后缀异常的系统，因此不能从这76行重算80系统FPR。
