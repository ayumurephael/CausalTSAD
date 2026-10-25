# 已完成结果与范围

当前分支是C08冻结读出/源梯度反证，12组全部完成、0optimizer。LSE包含root后验集中度项；去除此逐点项可恢复多数检测退化，常量与错时序对照不支持相同解释。它不是新训练模型，也不是SOTA证明。

见[数值与全部正反系统](results/fixed_pilot/summary.json)及[完整报告](results/fixed_pilot/README.md)。以下为原实现说明，训练代码版本在不可变launch中固定；新的文档提交不改变已跑模型。

---

# C08: joint no-fault / root readout for known-DAG TSAD

Independent branch `proposal/c08-joint-root-readout-v1`, from common main. Earlier proposals are preserved on their own branches. This is a candidate solution under a fixed controlled pilot, not a proven improvement.

Frozen self-trained TimeRCD + frozen own-history innovation teacher + a shared directed-graph posterior. Detection and RCA use the same node evidence.24runs:8controls x3optimization seeds, fixed3source passes each. Original and normal-twin optimized head queries are separately counted; no target gradients or locked-set access.

[Exact mechanism, controls, supervision, budget and stopping rules](docs/C08_JOINT_ROOT.md). [Frozen configuration](experiments/c08_joint_v1.json). [Dependency hashes](experiments/dependencies.json).

Run the source-free GPU G0 first, then `tools/controller_c08.py` with the documented local paths and immutable committed snapshot. `tools/train_c08.py` accepts only the source-cache split; `tools/evaluate_c08.py` opens only the existing100development systems after inference. Credentials/private administration/data/weights are not distributed in Git. Source thresholds are fixed before target evaluation. Readout-only pilot, not full foundation pretraining or a claim of SOTA.

Baseline B1 is a baseline, C01 a candidate TimeRCD solution, C02/C03 restricted source mechanism probes. Their negative results do not complete or invalidate every broader Insight Card. This C08 implementation adds genuine source-root supervision and tests whether it helps detection, with base calibration, wrong graph, graph prior and wrong-root controls.
