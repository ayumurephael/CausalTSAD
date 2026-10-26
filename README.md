# C04s：原始因果残差锚点与独立检测/定位头

本分支是来源小实验的可复现实现，**不是已在完整生成基准超过TimeRCD的最终模型**。2026-10-26：18run已完成、SHA核验并分析；尚未开展本版生成迁移。继承的旧TASK、research_status和历史README叙述不能作为本分支当前运行状态。

## 先按这个顺序读代码

1. [正常因果预测器](proposals/causal_meta_basis.py)：自身过去32点、父当前/过去8点的共享神经基函数；前256点做三轮稳健ridge，输出正常均值/方差/残差。R可能污染，目标无全局权重更新。
2. [正常表示训练](tools/train_meta_basis.py)及[原C04l协议](docs/CAUSAL_META_BASIS.md)：后续使用normal_only三个固定final，表示训练和新头训练是两阶段。
3. [特征](proposals/temporal_innovation_head.py)中的features()、[缓存](tools/cache_temporal_innovations.py)：从残差/方差构建1/4/16/64/256统计量。CausalMetaBasis.forward()['score']是旧W16输出，**C04s最终分数必须继续走新头**。
4. [当前模型](proposals/raw_anchor_task_heads.py)：零初始化MLP修正原始Gaussian单点能量。multi_dual的检测/定位各自MLP；multi_shared2是同参数/MAC的共享最终证据对照。
5. [训练](tools/train_raw_anchor_heads.py)、[G0预检](tools/preflight_raw_anchor_heads.py)、[有限队列](tools/controller_raw_anchor_heads.py)。读[六臂计划/八字段卡](docs/RAW_ANCHOR_TASK_HEADS.md)再运行。

## 方法与结果

共有64来源训练、16校准、32验证系统；六臂×seed348/349/350，每run64步固定final。132,120,576等价原训练曝光，0新正常预测器前向，0目标数组。单头1217参数，双头2434；只在同容量shared2/dual间归因任务解耦。

| 来源两背景宏平均 | AP | 条件根因Top1 |
|---|---:|---:|
| 原始单点能量 | 31.629% | 41.367% |
| 当前特征共享小头 | 32.238% | 40.547% |
| 同容量shared2 | 31.718% | 40.495% |
| 独立dual | 31.788% | 41.406% |

当前特征头AP相对原能量＋0.609pp，95%系统区间[0.400,0.832]，定位下降。dual相对shared2定位＋0.911pp，[0.221,1.562]；dual检测和det-only在288个逐点数组完全相同。说明冻结表示/私有参数下可以保护检测，不是共享可训练骨干完全无冲突的证明。新头尚无生成开发成绩、无自动检出→定位主评测、无SOTA。

恢复原始能量后，多尺度比同容量单点低0.472pp；旧压缩基础下的多尺度优势不通用。不能把来源数字与旧80生成系统的AP15.59%画成同数据提升。

## 运行与数据边界

训练器要求固定job_plan与SHA缓存；不随意生成新计划复现“最好”结果。正常DAG可输入，真实故障边/lag/coeff/noise/root/twin不作目标特征；父当前允许，自身当前/未来不能用于正常预测。根因窗口仅作来源训练目标/评测，条件Top1不含漏检/误报。

权重、大缓存、GPU连接凭据不在公共代码树。正式实验的不可变源码是dd8118062fdf0780522f49a0ccf1bac87461b259；本次后补README只改文档，不改变已部署科学代码。分支继承的其他模块不是本实验入口。
