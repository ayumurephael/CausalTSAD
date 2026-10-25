# C04独立检测器：生成开发集迁移

C04b和C04c来源融合失败后，停止有界加法接口，不继续调lr/epoch或按目标挑头。新分支独立保存，直接检验已训练C04关系模型在生成基准上的TSAD/RCA迁移；这是必要的模型基线，不是新算法或SOTA声明。

固定9个C04 final（true/degree/wrong×348/349/350），0训练。全部540开发系统、1080case、50M观测标量，单列432M/108C1和固定100；C04只在M前向，C1明确标为原生selftrained_uni passthrough，不能说因果模型学会C1。原TimeRCD full主表保留。

**主配对点集统一t≥320。**C04需要历史上下文，不把TimeRCD前320分数与C04不同尺度生拼。所有模型，包括B0，均在相同ready标签点集算原四指标/AP/AUC/FPR；这些后缀数值与既有全序列主表不混比。真实事件窗∩ready点集的C04 root Top1/Top3/MRR另报，事件无ready点与C1剔除并保存分母；不做TimeRCD RCA，不称自动报警RCA。

M正常阈值统一基于C04来源calibration的64系统/128normalcase；C04用各自原checkpoint已固定阈值，B0从相同来源normal缓存取99%分位。C1仍用原uni来源阈值。0目标校准、标签选参、验证/locked数组、目标梯度。正常孪生只独立输入用于误报评估，不参与observed预测。

逐文件观测/normal/B0缓存和9checkpoint SHA绑定；B0从已完成v1缓存复用，新TimeRCD前向0。teacher流式chunk256复用已通过双GPU的旧full等价G0；同权重保持原theta，不更新参数。新目标接口先实际C1+M预检并复用其结果。代码重构不称新训练。

总冻结C04查询曝光按实际ready长度×C×两view×三graph×三seed累计（上限900M，C1跳过，前320扣除），CPU同节点不同seed复用B0后缀指标。两节点各计算一次B0，明确重复CPU计数，0重复encoder。逐文件scores与root数组保留。每GPU worker1GiB、两节点尽量共享所有余量≥2GiB的卡；3h推理/10h技术终点，失败保留数据，停止本项目新增分派，不影响其他任务。CPU阶段释放GPU进程后运行spawn避免嵌套fork死锁。

预注册判读：先报告所有9模型与共同B0，按底层系统配对、3优化seed汇总、两轨/机制/C/异常时长/全部下降例。真图是否优于错图/degree与是否优于B0是两个问题；只其中一个成立不能说整体提升。没有在目标上选择seed、checkpoint、融合强度或阈值；本开发集用于探索，不是全新盲测。仅来源机制成功不保证生成/真实迁移，锁定150M始终关闭。
