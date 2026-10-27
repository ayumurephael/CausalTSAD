# I11：因果上下文训练目标对齐

本分支是三目标×两图×三种子的有限来源实验；尚无结果，不称成功模型。

先读[八字段方案、预算与判读规则](docs/I11_OBJECTIVE_ALIGNMENT.md)，再读[损失与预测白名单](proposals/objective_alignment.py)、[训练入口](tools/run_i11_objective.py)。

旧分支不覆盖。主输入仅观测、有效mask、正常DAG；正常孪生/根只用于明确标注的来源loss。目标无SGD，2.5亿基准与锁定数据不读取。
