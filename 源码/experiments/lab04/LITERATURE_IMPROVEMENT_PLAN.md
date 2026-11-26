# 实验四：联网调研后的准确率改进方案

> 2026-11-25。本文件区分**已验证事实、完成但无增益的实验、下一阶段待训练方案**。不承诺固定准确率，不将论文分数或外部模型成绩归入自训模型。

## 1. 结论与调整方向

当前自训最优模型 `continued` 的正式历史测试仍为 **56.4733% / 68.8133%**（5-way 1/5-shot，每类15 query，600个配对任务/shot）。本次不改写这组成绩。

下一步优先级调整为：

1. **优先改善特征编码器训练，再谈推断微调。** 不再把少量续训或图传播调参当作主要提分路线。
2. **保留课程的自训轻量主线，并增加明确独立的强参考/教师分支。** 教师不能伪装成本地训练的轻量网络。
3. **在轻量学生上做训练级特征/关系蒸馏，而非仅对冻结特征做线性映射。** 已准备可运行的训练原型和配对无蒸馏对照；完整训练尚未执行。
4. 没有通过验证门槛的推断方法不进入测试、不替换网页，不为了展示更高分而换掉评价口径。

## 2. 本次实际联网核验了什么

检索入口包含 arXiv 的 `transductive few-shot learning`（按时间排序）、GitHub 的 `PT-MAP` 仓库搜索、OpenAlex 的2023年后方法发现；再逐一打开论文/作者代码核验。Google 简版未返回有效内容、DuckDuckGo 要求验证、GitHub 搜索 API 限流，均未当作检索证据。成功内容及失败请求保存在 `outputs/lab04/next_study/literature_20261125/verified_sources.json`。本次核验不是“穷尽最新SOTA”的宣称。

| 来源 | 核实到的有用内容 | 对实验四的用途/限制 |
|---|---|---|
| **RFS，ECCV 2020** [论文](https://arxiv.org/abs/2003.11539) · [官方代码](https://github.com/WangYueFt/rfs) | 强监督表示加蒸馏；官方ResNet12宽度`[64,160,320,640]`；SGD、100轮mini-ImageNet训练；论文Table 1的simple为62.02/79.64，distill为64.82/82.14 | 支持把重点转到表示学习/蒸馏；这些是论文数据，不是本实验预期保证。官方模型远大于我们的轻量版，分类器和训练预算也不同。 |
| **TIM，NeurIPS 2020** [论文](https://arxiv.org/abs/2008.11297) · [官方代码](https://github.com/mboudiaf/TIM) | 冻结特征，最大化query预测互信息；官方GD使用`.1 CE - H(mean p) + .1 mean H(p)`；mini脚本1000步 | 已独立实现并验证；无需查询标签，仍是传导方法。作者ResNet18约73.9/85.0是另一骨干下的论文成绩。 |
| **PT-MAP** [论文](https://arxiv.org/abs/2006.03806) · [官方代码](https://github.com/yhu01/PT-MAP) | Power Transform、分别中心化support/query、Sinkhorn平衡分配、阻尼原型更新；官方示例`beta=.5, lambda=10, alpha=.2, steps=20` | 已实现论文公式级候选。官方WRN成绩82.92/88.82不迁移到本项目；现有LeakyReLU特征有负值，使用**显式signed-power变体**，不冒称精确复现。 |
| **SimpleShot** [论文](https://arxiv.org/abs/1911.04623) · [官方代码](https://github.com/mileyan/simple_shot) | 简单的训练均值中心化、L2标准化和最近邻/原型是必要强基线 | 用于同head、同图片的特征对照；不要把“复杂元学习”默认视作更强。 |
| **Distribution Calibration，ICLR 2021** [论文](https://arxiv.org/abs/2101.06395) · [官方代码](https://github.com/ShuoYang-1998/Few_Shot_Distribution_Calibration) | 从相似基础类借均值/协方差，采样伪特征，再训练logistic regression；上游依赖S2M2_R/WRN | 与课程校准主题最贴合，但不能直接在我们的带符号/已白化特征上照搬平方根、`alpha=.21`及论文成绩。保留为下一阶段非传导对照。 |
| **FEAT，CVPR 2020** [论文](https://openaccess.thecvf.com/content_CVPR_2020/html/Ye_Few-Shot_Learning_via_Embedding_Adaptation_With_Set-to-Set_Functions_CVPR_2020_paper.html) · [官方代码](https://github.com/Sha-Lab/FEAT) | task-specific embedding adaptation；官方提供mini-ImageNet ResNet12权重与划分 | 使用已有作者权重作**外部编码器参考**，仅encoder、不含FEAT attention。因此不能命名为“复现FEAT完整结果”。 |
| **UNEM，CVPR 2025** [论文](https://openaccess.thecvf.com/content/CVPR2025/html/Zhou_UNEM_UNrolled_Generalized_EM_for_Transductive_Few-Shot_Learning_CVPR_2025_paper.html) · [官方代码](https://github.com/ZhouLong0/UNEM-Transductive) | 把广义EM展开成网络，在验证数据上学习步长/熵等参数；视觉分支支持mini-ImageNet、tiered-ImageNet、CUB | 已核验较新方向，不盲追新论文。它会额外使用验证标签训练参数，需要专门的meta-training/selection隔离；本次没有宣称已复现或直接套用其参数。 |

下载的第三方PDF/网页/代码是**本地阅读缓存**。打包脚本排除这些正文，仅保留原创说明、引文链接、来源元数据及本实验数值证据；不可把第三方阅读内容当自有成果再分发。

## 3. 代码/训练的真实瓶颈

### 3.1 “ResNet12”不是同一模型预算

| 分支 | 通道 | 参数量 | 训练来源 |
|---|---|---:|---|
| 原自训窄网络 | `[8,16,32,64]` | 130,160（含分类头） | 本地mini-ImageNet train64 |
| 当前自训最优 | `[16,32,64,128]` | 509,856（含分类头），501,600（encoder） | 本地warm/分支/扩宽/续训，GroupNorm |
| 建议中等容量对照 | `[32,64,128,256]` | 2,018,048（含分类头） | **待训练**，用于容量预算曲线 |
| 官方FEAT参考 | `[64,160,320,640]` | 12,424,320（encoder） | 作者权重、BatchNorm，非本地自训 |

同为12层并不意味着同容量。比较encoder时，外部参考约为当前模型 **24.77倍**（若拿含分类头的当前总量比较则约24.37倍）。从官方大骨干换来高分，必须说明容量、归一化、训练及来源同时改变。

当前40轮warm start、分支训练、扩宽训练和16轮续训并不等于“官方大ResNet12从头100轮”；新通道通过扩宽产生、训练目标和优化器也不同。续训的增强训练准确率仍约61%，支持继续调查容量/优化问题，但**增强训练准确率不能单独证明欠拟合或排除数据/配方问题**。

### 3.2 原方法已经用了较强传导先验

当前方案包含双视图、类内协方差白化、原型更新、图传播及查询均衡先验。PT-MAP的平衡传输和原型更新与它存在功能重叠，不能预设叠加就有收益。1-shot与5-shot分开选型，当前5-shot本来就关闭图传播。

带符号特征不能不加说明地用`x**0.5`。本次原始cache统计：学生train负特征约18.71%、val约16.03%；教师约16.51%/15.30%。signed power只是适配，不等于论文在正特征上的Tukey变换。

## 4. 本次完成的验证研究

完整自动生成数据表见 [LITERATURE_VALIDATION_RESULTS.md](LITERATURE_VALIDATION_RESULTS.md)。以下所有新分数属于**验证集**，不与历史test值直接做差。

### 4.1 同一轻量模型的推断改进：完成，未通过

冻结`continued.pt`、两视图特征、训练统计及现有配置；验证每类200张搜索、400张确认（两池图片不相交，类别相同）；160个搜索任务/shot，搜索前固定5个候选，前2名进入600个确认任务/shot。

| 同样的确认任务 | 1-shot | 5-shot |
|---|---:|---:|
| 现有continued | 55.83% | 69.77% |
| MAP + 原白化特征 | 55.56% | 69.86% |

配对差分别为 **−0.276±0.238pp / +0.087±0.129pp**（逐任务95%区间半宽）；5-shot区间跨零，1-shot反而下降。TIM及signed PT-MAP未进入前2名，原始搜索负结果全部保存。

门槛预先固定为：两种shot均配对下界>0且均值至少+1pp。这个+1pp是**最小实用增益门槛**，不是用户已取消的固定最终准确率目标。结果未通过，故**未接触新的测试数据，未发布候选**。

### 4.2 教师冻结特征的线性几何迁移：完成，未通过

用相同6400张基础train图片，在128维学生与640维教师之间拟合ridge，再将`WWᵀ`压回128维度量矩阵。6个预注册组合；和4.1使用同任务选择规则与确认池。

最优ridge变体确认分数 **53.33% / 66.62%**，低于55.83%/69.77%。这说明**该线性迁移方案**不奏效，不能解释成“训练级蒸馏无效”；它根本没有改变编码器，无法自动补齐丢失的语义信息。

### 4.3 外部强编码器的同任务诊断

单独用两个cache共同覆盖的3200张val图片、600任务/shot，所有方法support/query ID完全相同。匹配中心化头只用相同6400张train图片求均值；教师不使用query标签。详细成绩和配对差在自动生成结果表。

这个对照用于判断**表示学习方向是否值得投入**，不是同训练预算容量消融，也不是未经接触的测试。外部模型同时改变容量、BatchNorm、训练和来源，不能把差值全部归因于“多了参数”。原FEAT attention模块未使用。

数值异常也保留：外部PT-MAP在原300次Sinkhorn上限触发质量守卫，原研究记录失败并停止。新补充方案只将上限改为作者代码的1000次，对师生同时生效，保持相同任务、温度、步数、幂变换；不删除失败任务、不根据正确率放松残差阈值。仍失败则整组报告“数值失败”，不提供筛掉难例后的准确率。

## 5. 修订后的实验四架构

### 路线A：严格无外部权重、自训主线（课程主结论）

保留窄骨干Baseline/真正全网二阶MAML/原完整特征校准的公平对照。新增容量研究单列：0.51M与2.02M学生，不与同骨干校准消融混合。

训练配方以RFS为参考：固定train64；random crop、颜色扰动、水平翻转；SGD momentum .9、weight decay .0005、100 epochs、lr .05于60/80衰减，batch64或记录等效batch的gradient accumulation。**这是一项待验证配方，不是说把AdamW改成SGD必定更好**。网络初始化、BN/GN、DropBlock和loss也应分实验，不在一个结果里同时改完再归因。

建议先完成2.02M模型的CE基线，再做相同初始化/样本顺序/预算下的`CE+alignment`，然后加episode metric项；选择最佳epoch后下游依次对照原型、训练统计校准和固定图方法。每个条件至少3个预注册训练种子时才讨论训练方差；600 task CI不能替代训练seed方差。

### 路线B：外部教师 → 0.51M轻量学生（推荐的高收益探索）

教师为已核验来源的FEAT ResNet12 encoder，**冻结、eval、无梯度**。学生仍为当前`[16,32,64,128]`，部署时保留学生，删除训练投影头。教师训练数据由作者声明为train64，不能独立证明整个历史；有隐含外部模型训练预算，单独列为“外部教师蒸馏”。

推荐目标：

\[
L = L_{CE} + 1.0 L_{feature} + 0.5 L_{relation} + 0.5 L_{align} + 0.25 L_{proto}.
\]

`L_feature`是128→640训练投影后的归一化方向匹配；`L_relation`匹配同一基础train minibatch中的余弦关系矩阵（去掉对角）；`L_align`保留两增强视图均值/方差一致；`L_proto`保留基础类episode损失。**教师没有被加载的64类classifier，不虚构教师logits做KL。** 此方法是受RFS启发的特征/关系蒸馏，不是RFS论文精确复现。

新增实现：
- `feature_distillation.py`：loss和可移除投影头，teacher强制stop-gradient。
- `train_feature_distillation.py`：train/val限定、原始图片字节指纹、固定验证任务、epoch0可选、16轮固定预算、optimizer/RNG断点恢复、CPU/CUDA可选、无测试/发布入口。
- `--method control`与`distill`从同一初始化、同一种子和两视图流开始；唯一loss开关是feature/relational KD。不能拿不同batch/视图/epochs的旧训练冒充无KD公平对照。

两边必须看到**同一个增强后RGB视图**，再各自归一化；不能把随机裁剪学生特征对齐到原图教师cache。为节省CPU可预缓存确定性同视图教师输出，但必须记录augmentation seed/ID，不能偷换视图。

训练脚本的默认行为只是登记，**没有`--run`不会启动训练**。本次已完成随机输入前后向与安全测试，并已登记`control`/`distill`两份16轮计划；两组初始权重、种子、train/val样本与48,000张原始图片字节指纹一致。证据见`literature_20261125/distillation_registered/status.json`；实际训练轮数仍为0。**未完成真实蒸馏训练，尚无学生提分结果**。

```powershell
# 源码目录；先登记，不进行训练
rtk proxy python -B -X utf8 -m experiments.lab04.train_feature_distillation `
  --method control --output .local/lab04-kd-control-20261125
rtk proxy python -B -X utf8 -m experiments.lab04.train_feature_distillation `
  --method distill --output .local/lab04-kd-student-20261125

# 准备好算力后显式开始；GPU方案应登记到新的目录，勿改已有plan
rtk proxy python -B -X utf8 -m experiments.lab04.train_feature_distillation `
  --method control --output .local/lab04-kd-control-20261125 --run
rtk proxy python -B -X utf8 -m experiments.lab04.train_feature_distillation `
  --method distill --output .local/lab04-kd-student-20261125 --run
```

### 路线C：推断/分布校准进阶（先备选，不继续盲搜）

1. 在更强的**已冻结**学生特征上重新检验SimpleShot、原校准、TIM/PT-MAP，不把本次弱特征上的负结果外推到所有模型。
2. 对Distribution Calibration提供非传导版本：base-only统计、PSD协方差正则、明确带符号变换和伪特征采样随机种子。先在train-held-out class构造任务选基本超参，不采样/拟合query标签。
3. 如采用UNEM，在train-held-out class任务学习展开参数，把val用于选择，测试只执行锁定层数；不要把已反复观察的val训练再称“独立验证”。
4. 加入不均衡query诊断及单query inductive fallback。平衡查询方法的高分不等于真实边端零批量场景性能。

## 6. 算力与推进边界

本机PyTorch为CPU版本、无CUDA。随机输入的固定小样本计时见`literature_20261125/cost_probe/result.json`：0.51M单视图CE每8张中位0.082s；2.02M约0.420s；轻量学生+大教师双视图约1.169s。按38400张线性外推约6.5/33.6/93.5分钟每轮，**不是实训ETA**（未含I/O、优化器、验证，batch及机器负载也不同）。

这解释了为何本次先完成缓存验证与方案/代码准备，而没有暗中启动十几至几十小时CPU训练。优先在GPU环境运行成对训练，或先实现严格同视图的教师缓存。不要把待运行的计划说成已经提升。

## 7. 评价、消融、发布要求

- 原题保持：mini-ImageNet、轻量ResNet12、Baseline、真正二阶MAML、均值/方差/原型校准、CE+分布一致性、边端成本。
- 对照轴分开：同权重推断、同容量训练配方、容量扩大、外部教师/官方模型。
- 全部搜索次数、失败、负结果和计算预算记录。既有test已经被多轮观察，后续测试需披露而不能称全新benchmark。
- 新模型须先对现有continued做同ID验证：两shot均值≥+1pp且配对区间下界>0，才允许冻结选型；该门槛可反对明显退步，不消除反复验证的选择偏差。
- 真正发布前还要：固定原test task一次、独立重算、线上/离线一致、时延/参数/外部来源、失败fallback、报告与演示版本同步。未经这些步骤不得把研究分数填进网页“正式成绩”。
- 不设置95%等固定终点；目标是有可复核证据的准确率—成本改进。

## 8. 可复核文件与命令

```powershell
rtk test python -B -X utf8 -m unittest tests.test_lab04_literature tests.test_lab04_literature_study tests.test_lab04_teacher_geometry tests.test_lab04_feature_distillation tests.test_lab04_feature_distillation_runner -v
rtk proxy python -B -X utf8 -m experiments.lab04.study_literature --phase audit
rtk proxy python -B -X utf8 -m experiments.lab04.study_teacher_geometry --phase audit
rtk proxy python -B -X utf8 -m experiments.lab04.complete_literature_diagnostic --audit-only
rtk proxy python -B -X utf8 -m experiments.lab04.train_feature_distillation --synthetic-smoke
```

原第一阶段的`diagnostic_failure.json`不能删；`study_literature --phase audit`只重算已完成记录/选型，并不声称那个中断诊断已完成。完整外部参考诊断以单独的`literature_diagnostic_20261125/diagnostic.json`和审计为准。

## 后续核查勘误：物理GPU与最新优先级

更深入的只读设备检查发现本机有RTX4050 Laptop，但Windows报告Code43，CUDA初始化失败；当前训练环境是CPU版PyTorch。前文“无CUDA”应理解为**当前不可用**，不是机器物理上没有GPU。未擅自更改驱动。更严格的课程主线优先级、扩宽通道冗余诊断与新的自训方案见[REQUIREMENT_ACCURACY_PLAN.md](REQUIREMENT_ACCURACY_PLAN.md)。这不改写上轮实验和归档报告的原始数据。
