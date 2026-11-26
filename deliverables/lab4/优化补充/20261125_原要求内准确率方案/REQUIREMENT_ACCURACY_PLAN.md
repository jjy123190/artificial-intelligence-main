# 实验四：在原要求内提高准确率的优先方案

> 2026-11-25。本轮重新读取了原会话中的课程要求，而非仅依据已实现代码定义“合规”。正式test仍为56.47%/68.81%；新研究分数全部为val或小规模pilot，不能混用。

## 1. 原要求允许做什么，不允许偷换什么

原文要求：mini-ImageNet小样本图像分类、**轻量ResNet12**、特征均值方差校准＋原型自适应、**交叉熵＋分布对齐正则**、Baseline/MAML/校准比较、消融、端侧部署。没有规定通道数或“只能使用130160参数”，也没有95%等必须达到的数字。

由此可做、且不改变题目的内容：

| 方案 | 与原要求的关系 | 必须保留的边界 |
|---|---|---|
| 同0.51M骨干独立初始化、改善训练 | 最稳妥，网络仍为12层/同参数 | 不能把更多epochs/更强增强的收益归于校准模块 |
| 中等宽度ResNet12（约2.02M） | 原文未限定通道数，可作为轻量工程候选 | “轻量”不能只靠称呼；要报参数、内存、CPU延迟。教师若另有限制需遵守 |
| Meta-Baseline式5-way任务训练 | 在CE+alignment外加训练目标，符合研究方向 | 保留真正二阶MAML对照，不把这个方法冒充MAML |
| 学习型特征校准小模块 | 与“特征分布校准单元”直接一致 | 不用query真值训练；新增参数/训练预算单列 |
| 局部特征匹配/低维二阶统计 | ResNet12不变、头部增强，可作为补充模块 | 仍保留原均值方差/原型主方案；论文大骨干成绩不套用 |
| 本地自训教师的自蒸馏 | 不引入外部训练数据；可作扩展对照 | 教师/学生训练预算、训练数据、部署学生大小都要披露 |
| 官方外部教师/大模型 | 原文未明文禁止，但是否接受作主成果无法由代码推断 | 仅独立参考或另行确认允许；**不作为本轮首选主线** |
| ImageNet-1K预训练直接替换 | mini-ImageNet来自ImageNet，存在类/图像重叠风险 | 没有排除重叠前，不用于可比主成绩 |

这次不建议继续把题目“变成更大预训练模型演示”；优先把**自己的轻量ResNet12训练好**。

## 2. 新发现：当前训练存在明确的可疑瓶颈

### 2.1 扩宽增加了参数，却未充分增加不同特征

当前wide/continued来自`widen_resnet12`的相邻通道复制，随后只加了很小噪声。读取train特征和卷积权重后发现：

- 原图train缓存中，相邻复制特征的Pearson相关系数平均 **0.994735**，最小约0.900。
- 128维特征的95%方差由约 **43个主成分**解释；熵有效秩约29.61。
- 末块3×3卷积的相邻输出权重余弦均值约0.999。
- 64类**未增强基础训练图片**的分类准确率仅 **63.80%**，并非只是在强增强输入上低分。

这些是训练内诊断，说明通道多样性、容量/优化值得优先研究，**不是数学证明“低准确率全由复制通道造成”**，也不是修复后必增益。低维语义表示可能本身合理，所以必须配对试验，而不是按特征秩选test模型。

建议的正式路径：从独立随机初始化训练0.51M和2.02M两个ResNet12；不要继续把同一套高度相关权重反复扩宽、低学习率续训。复用本地模型时，可尝试零和fan-out去对称，但pilot若不能改善，就不要无限堆小修补。

### 2.2 算力事实需要纠正

本机**确实有NVIDIA GeForce RTX 4050 Laptop GPU**，不是物理上没有独显。但当前：

- Windows PnP/VideoController：`Status=Error`，`ConfigManagerErrorCode=43`；
- CUDA driver：`cuInit=100`，`no CUDA-capable device is detected`；
- 当前Python环境只有`torch 2.10.0+cpu`；
- `nvidia-smi`失败，提示权限不足，不能据此推定可用显存或性能。

所以限制是“**显卡当前不可用＋CPU版环境**”，不是“这台机器没有GPU”。本次只读检查，没有禁用/启用设备、卸载驱动、重启或替换当前环境。Code43可能涉及驱动或硬件，不能保证装CUDA版PyTorch就好。[Microsoft说明](https://support.microsoft.com/en-us/windows/fix-graphics-device-problems-with-error-code-43-6f6ae1ec-0bbe-a848-142e-0c6190502842)建议按情况更新/回退/重装设备驱动；这类系统操作应另行处理。

恢复设备后，创建**独立环境**安装合适CUDA版PyTorch并做实际GPU前后向，再决定训练预算；不要覆盖现有可运行网站的CPU环境。GPU只用于训练，最终仍可CPU部署，符合端侧要求。

## 3. 新检索的方法与如何借鉴

### A. Meta-Baseline式“先表示预训练，再任务微调”——优先

[论文](https://arxiv.org/abs/2003.04390) · [官方实现](https://github.com/yinboc/few-shot-meta-baseline)

先用基础64类CE训练充分，再用5-way任务按评测度量微调。当前训练是16-way批内2-shot metric，与5-way1/5-shot不是同一任务分布；改为交替1/5-shot有明确动机。

注意官方示例`train_meta_mini.yaml`含` tval_dataset: split: test`用于诊断。**本项目绝不照搬这个test读取路径**；训练只用train64，选择只用val16。

### B. 独立初始化的中等宽度ResNet12＋合理训练配方——优先

保留12个3×3卷积与全局池化，建议固定比较：
- `[16,32,64,128]`，509,856参数；
- `[32,64,128,256]`，2,018,048参数。

不直接换成12.4M论文模型。增加容量和重新训练**单列容量/训练预算实验**，不是原校准消融。自训可参考RFS的SGD .05、momentum .9、wd .0005、100轮，60/80衰减、独立初始化、crop/flip/color；从当前AdamW续训得到的结论不能预先替代这项完整实验。

GroupNorm先保持不变以隔离训练/容量；BatchNorm/DropBlock若纳入，另设条件且test时冻结BN统计，不能混入同一归因。MAML保留GN和真正二阶适应。未完整训练前不引用论文accuracy作为本地目标保证。

### C. 低维二阶分布或局部特征——作为次级模块，不盲装头

- [FRN，CVPR2021](https://arxiv.org/abs/2012.01506)用support局部描述子重建query特征图，无需大型新增网络；[官方实现](https://github.com/Tsingularity/FRN)包含**训练**而非仅换推断头。
- [DeepBDC，CVPR2022](https://arxiv.org/abs/2204.04567)建模联合分布/二阶信息；[官方代码](https://github.com/Fei-Long121/DeepBDC)以ResNet12报告结果，但它的通道/训练预算与本模型不同。
- [DN4](https://github.com/WenbinLee/DN4)也提供ResNet12的局部image-to-class匹配。

可在更强自训轻量表示上尝试把128通道先降到32维，再计算可归一化的二阶描述子，避免直接产生8256维向量及昂贵校准。必须做GAP对照、降维/局部头消融和CPU计时。**本轮冻结旧骨干直接换FRN式头没有成功**，不能据论文宣称现成可用。

### D. 冗余抑制/自蒸馏——配套而非首要保证

[Net2Net](https://arxiv.org/abs/1511.05641)说明函数保留扩宽的设计；复制后仍需让新通道学到不同表征。本次独立实现了对输入通道对的`+noise/-noise`fan-out扰动：严格相等的通道下前向保持、上游梯度不同，有单元测试；实际已训练通道只近似相等，故必须保留未扰动epoch0回退。

[VICReg](https://arxiv.org/abs/2105.04906)为方差/协方差冗余抑制提供启发；这里用的相邻通道相关惩罚不是VICReg复现。只能作train正则，不能以test特征秩选系数。

自蒸馏可在基础train上的本地强教师与小学生之间开展，避免外部来源争议；已有外部教师代码保持单列，不强推其进入课程主结论。

## 4. 本轮实际运行，而非只写建议

### 4.1 学习型特征校准——已完成，未通过

冻结原0.51M骨干，但新增可学习线性/残差校准头（16,384/32,896参数）。用**全部38,400张train特征**训练各12轮：基础CE＋交替1/5-shot的5-way任务损失＋原图/翻转的均值方差一致＋表征锚定。epoch0可选；160任务/shot选择epoch，图片互斥确认池600任务/shot。无外部权重，无新的test使用。

| 相同确认任务 | 1-shot | 5-shot |
|---|---:|---:|
| 当前continued配置 | 56.55% | 69.10% |
| 线性校准 | 56.35% | 69.20% |
| 非线性残差校准 | 56.59% | 69.37% |

残差头相对当前为 **+0.038±0.677pp / +0.264±0.370pp**，两区间均跨零，也未达到+1pp最小实用增益。不是有效提分，未发布。说明在已有冻结特征上学一个小头仍不足；应优先改变编码器训练。

### 4.2 冻结骨干的局部重建——已完成，未通过

保留`continued.encoder`的128×5×5空间特征，不做GAP；搜索5个预注册归一化/ridge组合，每shot160搜索＋600确认任务。无新增可学习参数、无query交互、无外部训练。任务/图片固定。

最佳确认 **45.92% / 60.94%**，低于同任务当前头 **56.13% / 68.81%**。这只是FRN式头在**未接受局部重建训练的旧encoder**上的负结果，不代表训练FRN或所有局部方法均不行。两种shot的完整原始结果与审计见证据目录。

### 4.3 真正CNN训练的通道多样性pilot

两条分支都读取相同6144张train图片（64类×96），相同初始化、增强、顺序、2轮、AdamW；共有CE＋分布对齐＋5-way任务损失，仅候选多了fan-out去对称和pair相关正则。固定1280张val图、200任务/shot，使用归纳原型选择，**不是网站双视图校准分数**。

这个pilot只能检查方案能否快速产生迹象，不能代替38,400张全量训练/正式test。开始前登记所有预算，原模型epoch0始终可选。不因为结果不理想而改任务、延长某一分支或删除低分任务。最终两条分支都选回**未训练的epoch0**（同任务归纳原型46.53%/62.81%），没有得到改进。因此不把“发现冗余”夸大为“去对称已解决低分”。结果见证据目录`channel_pilot/result.json`与审计；全部训练记录保留。

## 5. 我建议下一步执行的主方案

**推荐：“独立初始化的中等轻量ResNet12 + 两阶段训练 + 原均值方差/原型校准”。** 若严格要求维持0.51M，则先执行同容量独立初始化版本。

1. 先处理GPU Code43，验证设备可用；不能修复则迁移到可用训练机，端侧部署仍保留Windows CPU。
2. 固定0.51M与2.02M容量候选、3个训练种子，使用train64完整38,400图片；从头独立初始化，不再复制旧通道作为唯一方案。
3. 阶段一：100轮CE与配对增强alignment；同预算λ=0对照。固定日程与增强，不在test上挑epoch。
4. 阶段二：额外20轮5-way1/5-shot交替episode训练，保留CE/alignment以防遗忘；单列额外预算，并做无episode微调对照。
5. 保持训练统计的均值/方差校准、原型自适应和可选图传播；先冻结骨干再确定下游配置，不把所有改动一次混成“校准提升”。
6. 新模型在相同val任务/shot上两种shot均≥+1pp且配对区间下界>0，才锁定一次test。公开test以前用过，必须披露；3个训练seed方差单独报，不能被600任务CI替代。
7. 报告参数、模型文件、CPU编码/适应时间、内存和轻量性。Baseline/MAML/校准主对照与原消融继续保留，平台提交/指定设备另行验收。

此处不承诺70%/80%或其他数字，但这比继续对旧特征换头更有证据支撑。**没有强方案被真实训练并过验证前，不应把高accuracy写到网页上。**

## 6. 证据和复现

- `outputs/lab04/next_study/requirement_accuracy_20261125/`：train诊断、GPU只读诊断、联网来源、完整原始结果及审计副本。
- `learned_calibration.py` / `study_learned_calibration.py`：学习型校准。
- `local_reconstruction.py` / `study_local_reconstruction.py`：局部重建。
- `channel_diversity.py` / `pilot_channel_diversity.py`：去对称pilot。
- `requirement_study_audit.py`：重新计算逐任务、epoch选择、配对差和门槛，不读新test。

```powershell
rtk proxy python -B -X utf8 -m experiments.lab04.requirement_study_audit --kind learned --directory .local/lab04-learned-calibration-20261125
rtk proxy python -B -X utf8 -m experiments.lab04.requirement_study_audit --kind local --directory .local/lab04-local-reconstruction-20261125
rtk proxy python -B -X utf8 -m experiments.lab04.requirement_study_audit --kind pilot --directory .local/lab04-channel-pilot-20261125
```

上轮“无GPU”描述应按本文件纠正为“GPU设备Code43且当前环境为CPU版本”。旧检查记录保留，不修改旧模型或实验哈希伪装从未出错。
