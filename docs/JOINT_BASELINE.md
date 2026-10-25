# B2：可运行的联合TSAD/RCA基线

这是已经完成训练/开发评测的C08b mean+PCGrad模型的部署封装，**不是另训一个B2**。固定3seed348/349/350各自终点，不按目标挑seed或ensemble。代表用348只是预设示例，所有报告均三seed宏平均。

输入仅`values[T,C]`和正常`adjacency[C,C]`(source-row,target-column)。载入原自训练multi、C01固定own-history创新teacher348、当前seed联合head。原TimeRCD全序列归一化/5000点窗口/末尾复制padding与ChannelMean保持，历史encoder/teacher转换逐式复用原cache；当前节点证据→父/子图消息→含null状态的根因后验。离线base本来可读未来，整体模型不能冒称在线因果预测器。三类影响标签仍null，root/真实窗口/正常孪生/系数/噪声均不在推理API。

`predict`输出原与融合raw logit、异常概率、null概率、条件root posterior、联合root概率、源阈值报警mask。度量排序用raw logit，避免概率饱和。`alarm_events`按连续报警点输出自动事件段与根因排名，不读取真实窗口；没有额外合并间隔/目标调阈值。C1保持原common-multi分数，其根因排名平凡，不用于RCA结论。

## 已测量能力与界限

固定100 development中80M系统，三优化seed平均：AP48.0171%、VUS-PR41.6150%、正常点FPR0.3496%；原自训练基线AP47.9401%、VUS41.5507%、FPR0.3309%。提升区间跨0且FPR增，不称稳定优越或SOTA。

同一套权重的**条件RCA Top1=40.4167%**(97/240事件-优化seed评估，80个独立底层系统)，给定真实注入区间对posterior平均后排根因。检出且定位正确的事件匹配Top1=28.75%(漏检记0)，但仍依赖真实事件匹配口径，**不是自由运行报警精度**。新自动事件API尚须独立事件级precision/recall/定位评测，不把条件Top1改名当它的成绩。

## 使用

在GPU上准备NPZ，只有values与adjacency两个键即可。权重保留服务器，不推GitHub；checkpoint路径/SHA见provenance manifest，原运行环境与原TimeRCD依赖保持。

```bash
python tools/predict_joint.py --input input.npz --output new_prediction \
  --original-code /absolute/original_runtime_v1 \
  --base-checkpoint /absolute/selftrained_multi.pth \
  --teacher-checkpoint /absolute/pretrained_none_s348/final.pth \
  --head-checkpoint /absolute/mean_pcgrad_s348/final.pth
```

接口预检需从原始X/G重新前向，与既有保存分数/后验比对，G0最大绝对误差<=1e-6、权重前后hash不变；技术重放算力另记，0优化/0重新训练。不能只检查import就声称部署一致。
