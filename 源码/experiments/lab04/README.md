# 实验4：小样本 ResNet12 特征校准

该模块在标准 mini-ImageNet 的 64/16/20 类划分上运行真实图像实验。训练类、验证类、测试类完全分离，episode 内的 support 和 query 图像互不重叠。

## 模型和对照

- ResNet12-Narrow：4 个残差块，每块 3 层 3×3 卷积，宽度为 `[8,16,24,32]`，32 维特征，GroupNorm。它是约 5 万参数的 CPU 轻量版本，不能拿其结果直接对标常见 640 通道 ResNet12 论文成绩。
- Baseline：交叉熵训练后的骨干，使用 support 的最近原型分类。
- MAML：真实全网络二阶元学习，内循环更新全部骨干与分类头参数。测试时使用 support 做 3 步更新；query 标签不参与适应。
- 特征校准：基础类近邻先验提供均值与方差收缩，1-shot 用先验方差避免零方差；类原型可通过无标签 query 的软分配迭代更新。
- 对齐损失：对同一训练图像的两次增强，约束特征均值与方差一致。对照分支和校准分支使用同一初始权重、相同样本顺序、增强、交叉熵步数和训练轮数，仅对齐系数不同。

`none` 消融保留校准分支骨干但关闭推理校准，因此能区分训练对齐与推理改进。`mean` 仅均值，`variance` 仅方差，`prototype` 仅原型更新，`full` 使用全部模块。原型更新属于转导推理，会利用整批无标签 query；均值和方差策略不利用其他 query 的内容。

## 运行

在 `源码` 目录运行以下命令。Python 3.11+，依赖 `torch>=2.5`、`numpy`、`Pillow`。数据目录为 `train/wnid/*.jpg`、`val/wnid/*.jpg`、`test/wnid/*.jpg`，`class_names.json` 提供 WNID 名称。网站入口为 `#/labs/lab04`，接口沿用现有平台的 `/api/labs/lab04/*`。

```powershell
python -m experiments.lab04.prepare_data --root .local/mini-imagenet
python -m experiments.lab04.experiment --data-root .local/mini-imagenet --output outputs/lab04 --train-per-class 600 --eval-per-class 100 --epochs 10 --branch-epochs 2 --maml-tasks 500 --episodes 100 --seed 20261122
python -m unittest experiments.lab04.test_lab04 -v
python main.py lab04 verify
python main.py lab04 demo --method calibrated --shots 1 --seed 20261122
```

训练使用全部 38,400 张训练图像；验证和测试各固定采样每类 100 张，测试采用 5-way 1-shot/5-shot，每类 15 张 query，每种 shot 100 个配对 episode。所有推理方法使用相同测试 episode。超参数只在验证类选择，测试集在参数锁定后评估。实际配置、采样路径摘要、完整指标、逐 episode 结果与权重 SHA256 保存在结果目录，不从讲稿或 UI 文案反推分数。

```python
from experiments.lab04.inference import run_episode
result = run_episode("outputs/lab04", shots=1, seed=20261122,
                     method="calibrated", calibration="full")
```

返回 support、query、逐样本预测、概率、真实推理耗时、原型前后坐标和权重摘要。校准改变类原型与距离度量，样本特征不移动；图形界面应保持散点位置，仅表现真实原型变化。MAML 的前后特征来自网络实际适应。

## 结果解释

准确率为 0 到 1 的比例，95% CI 为比例尺度的半宽；配对差值为 -1 到 1 的有符号比例，展示时乘以 100 得到百分点。CI 用 episode 级标准误的正态近似，描述给定一次训练权重后的抽样波动，不涵盖不同训练种子的波动。表格中的缓存特征推理耗时不包含图像编码，不能直接与 MAML 全流程耗时作速度优劣比较；网站实测耗时包含本次图像编码。

演示图像池来自留出的 20 个测试类，每类 40 张，仅用于非商业教育演示。完整 ImageNet 数据不随本模块发布。特征校准是教学改进方案，正负结果均保留，不承诺提升。

## 本次实测

| 方法 | 1-shot | 5-shot |
| --- | ---: | ---: |
| Baseline | 40.39% ± 1.93 | 57.13% ± 1.58 |
| MAML | 26.87% ± 1.36 | 33.29% ± 1.73 |
| 完整校准 | 40.68% ± 2.10 | 53.24% ± 1.67 |
| 仅原型更新 | 43.43% ± 2.19 | 57.37% ± 1.56 |

± 为95%置信区间半宽，单位百分点。完整方案1-shot的配对差为+0.29个百分点且区间跨零，5-shot下降3.89个百分点；仅原型更新1-shot的配对提升为3.04 ± 1.01个百分点。当前方差先验对新类适配不佳，有限元训练预算下的MAML也较弱；不将这些结果写成普遍改进。逐任务原始记录在 `outputs/lab04/episodes.json`，所有汇总可由其重算。

与实验一至三相比，本次新增独立的视觉分类任务，以满足课程明确指定的 mini-ImageNet/ResNet12 要求，同时沿用原网站导航、错误处理和成果目录。模型单测覆盖二阶梯度、支持/查询隔离、1-shot数值稳定及元数据继承，网站检查覆盖输入限制、可用性、旧实验入口和真实推理。

## 方法出处

- Vinyals et al., *Matching Networks for One Shot Learning*, 2016，mini-ImageNet 的提出。
- Ravi & Larochelle, *Optimization as a Model for Few-Shot Learning*, 2017，采用的类别划分。
- Finn et al., *Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks*, 2017。
- Snell et al., *Prototypical Networks for Few-shot Learning*, 2017。
- Yang et al., *Free Lunch for Few-shot Learning: Distribution Calibration*, ICLR 2021，分布校准参考；本实现是显式记录的轻量变体，不宣称复现该论文分数。
