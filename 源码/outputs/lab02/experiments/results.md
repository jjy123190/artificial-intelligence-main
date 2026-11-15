# 第二次汇报：本机实测结果

> 全部分类标签均为合成标签，不能解释为真实企业风险准确率。

配置：随机种子 [7, 19, 43]；每个种子训练 1000、验证 300、测试 600。
各压力场景使用相同测试企业，标签不变；模型在测试前冻结。均值±标准差按种子计算。

## 无额外噪声测试集

| 方法 | Accuracy | F1 | Brier↓ | 单样本微秒 |
|---|---:|---:|---:|---:|
| rules | 0.7300±0.0073 | 0.7252±0.0142 | 0.2700 | 192.9 |
| bayes_manual | 0.8617±0.0088 | 0.7767±0.0055 | 0.0878 | 3.2 |
| bayes_learned | 0.8939±0.0173 | 0.8575±0.0180 | 0.0707 | 2.6 |
| fuzzy | 0.6922±0.0175 | 0.7000±0.0200 | 0.2156 | 172.4 |
| hybrid_fixed | 0.9006±0.0140 | 0.8720±0.0131 | 0.0923 | 361.0 |
| hybrid_tuned | 0.8939±0.0173 | 0.8575±0.0180 | 0.0707 | 2.9 |
| hybrid_no_rule | 0.8939±0.0173 | 0.8575±0.0180 | 0.0707 | 2.9 |
| hybrid_no_fuzzy | 0.8939±0.0173 | 0.8575±0.0180 | 0.0707 | 4.0 |
| hybrid_no_bayes（3/3种子回退为fuzzy基线） | 0.6922±0.0175 | 0.7000±0.0200 | 0.2156 | 142.1 |
| fuzzy_learned | 0.7000±0.0153 | 0.7053±0.0193 | 0.2121 | 148.8 |
| hybrid_adaptive | 0.8939±0.0173 | 0.8575±0.0180 | 0.0707 | 3.9 |
| adaptive_no_rule | 0.8939±0.0173 | 0.8575±0.0180 | 0.0707 | 3.9 |
| adaptive_no_fuzzy | 0.8939±0.0173 | 0.8575±0.0180 | 0.0707 | 3.1 |
| adaptive_no_bayes（3/3种子回退为fuzzy基线） | 0.7000±0.0153 | 0.7053±0.0193 | 0.2121 | 165.4 |

## 严重噪声与缺测（σ=0.35，缺测率40%）

| 方法 | Accuracy | F1 | Brier↓ |
|---|---:|---:|---:|
| rules | 0.6444 | 0.6214 | 0.3454 |
| bayes_manual | 0.7194 | 0.5302 | 0.1874 |
| bayes_learned | 0.7244 | 0.6174 | 0.1887 |
| fuzzy | 0.5994 | 0.6109 | 0.2385 |
| hybrid_fixed | 0.7183 | 0.6400 | 0.1875 |
| hybrid_tuned | 0.7244 | 0.6174 | 0.1887 |
| hybrid_no_rule | 0.7244 | 0.6174 | 0.1887 |
| hybrid_no_fuzzy | 0.7244 | 0.6174 | 0.1887 |
| hybrid_no_bayes（3/3种子回退为fuzzy基线） | 0.5994 | 0.6109 | 0.2385 |
| fuzzy_learned | 0.6022 | 0.6091 | 0.2384 |
| hybrid_adaptive | 0.7244 | 0.6174 | 0.1887 |
| adaptive_no_rule | 0.7244 | 0.6174 | 0.1887 |
| adaptive_no_fuzzy | 0.7244 | 0.6174 | 0.1887 |
| adaptive_no_bayes（3/3种子回退为fuzzy基线） | 0.6022 | 0.6091 | 0.2384 |

## 验证集选择的权重（Bayes / Fuzzy / Rule）

- seed 7: {'bayes': 1.0, 'fuzzy': 0.0, 'rule': 0.0}；验证集 Brier=0.079318。
  此种子退化为单组件模型；不能声称该次实验验证了融合增益。
- seed 19: {'bayes': 1.0, 'fuzzy': 0.0, 'rule': 0.0}；验证集 Brier=0.074404。
  此种子退化为单组件模型；不能声称该次实验验证了融合增益。
- seed 43: {'bayes': 1.0, 'fuzzy': 0.0, 'rule': 0.0}；验证集 Brier=0.073834。
  此种子退化为单组件模型；不能声称该次实验验证了融合增益。

## 组件移除与显式回退基线

下表逐种子说明实际操作。回退项是替代基线对照，不是纯删除后的组件边际贡献；模型参数不按测试结果重选。

| 种子 | 方法 | 对照基准 | 实际操作 | 权重（Bayes/Fuzzy/Rule） |
|---|---|---|---|---|
| 7 | hybrid_no_rule | hybrid_tuned | 零权重组件，结果不变 | 1/0/0 |
| 7 | hybrid_no_fuzzy | hybrid_tuned | 零权重组件，结果不变 | 1/0/0 |
| 7 | hybrid_no_bayes | hybrid_tuned | 回退为 fuzzy 基线 | 0/1/0 |
| 7 | adaptive_no_rule | hybrid_adaptive | 零权重组件，结果不变 | 1/0/0 |
| 7 | adaptive_no_fuzzy | hybrid_adaptive | 零权重组件，结果不变 | 1/0/0 |
| 7 | adaptive_no_bayes | hybrid_adaptive | 回退为 fuzzy 基线 | 0/1/0 |
| 19 | hybrid_no_rule | hybrid_tuned | 零权重组件，结果不变 | 1/0/0 |
| 19 | hybrid_no_fuzzy | hybrid_tuned | 零权重组件，结果不变 | 1/0/0 |
| 19 | hybrid_no_bayes | hybrid_tuned | 回退为 fuzzy 基线 | 0/1/0 |
| 19 | adaptive_no_rule | hybrid_adaptive | 零权重组件，结果不变 | 1/0/0 |
| 19 | adaptive_no_fuzzy | hybrid_adaptive | 零权重组件，结果不变 | 1/0/0 |
| 19 | adaptive_no_bayes | hybrid_adaptive | 回退为 fuzzy 基线 | 0/1/0 |
| 43 | hybrid_no_rule | hybrid_tuned | 零权重组件，结果不变 | 1/0/0 |
| 43 | hybrid_no_fuzzy | hybrid_tuned | 零权重组件，结果不变 | 1/0/0 |
| 43 | hybrid_no_bayes | hybrid_tuned | 回退为 fuzzy 基线 | 0/1/0 |
| 43 | adaptive_no_rule | hybrid_adaptive | 零权重组件，结果不变 | 1/0/0 |
| 43 | adaptive_no_fuzzy | hybrid_adaptive | 零权重组件，结果不变 | 1/0/0 |
| 43 | adaptive_no_bayes | hybrid_adaptive | 回退为 fuzzy 基线 | 0/1/0 |

## 模糊隶属度自动拟合

- seed 7: high_start=0.3，medium_weight=0.65；训练样本=1000，训练Brier=0.220994；自动融合权重={'bayes': 1.0, 'fuzzy': 0.0, 'rule': 0.0}。
- seed 19: high_start=0.3，medium_weight=0.65；训练样本=1000，训练Brier=0.221313；自动融合权重={'bayes': 1.0, 'fuzzy': 0.0, 'rule': 0.0}。
- seed 43: high_start=0.3，medium_weight=0.65；训练样本=1000，训练Brier=0.217723；自动融合权重={'bayes': 1.0, 'fuzzy': 0.0, 'rule': 0.0}。
仅用训练集选择15组预设参数；权重只用验证集。训练损失改善不保证测试改善，所有压力条件和组件消融均保留。

## D-S：来源B全部被反转的压力测试

| 方法 | Accuracy | F1 | Brier↓ | 平均无知质量 |
|---|---:|---:|---:|---:|
| ds_classic | 0.5567 | 0.4975 | 0.3472 | 0.0000 |
| ds_yager | 0.4856 | 0.4603 | 0.2391 | 0.6793 |
| ds_discount | 0.9622 | 0.9482 | 0.0717 | 0.0379 |
| ds_robust | 0.9622 | 0.9482 | 0.0944 | 0.1735 |

## 如何审计与解读

- samples.csv 保存实际输入和合成标签；predictions.csv 保存每条样本每个方法的评分，可独立重算指标。
- metrics.csv 为逐种子实测值，summary.csv 为跨种子均值/标准差；不使用人为延时伪造加速。
- models.json 保存仅由训练/验证集确定的参数；split_audit.json 检查样本ID不交叉。
- sensitivity.csv 是固定其余参数、单独扰动先验/CPT/隶属度后的测试结果，不用于择优报告。
- 严格消融只删除并归一化非零剩余权重；删除零权重组件保持结果不变。全零时预先指定的回退基线详见 ablations.csv，不据此归因组件边际贡献。
- Brier 对非概率评分是平方误差诊断，不能据此把模糊隶属度称为校准概率。
- 风险指标实验与结构推理效率实验不同：前者允许错误/缺测输入，后者验证同一知识库的精确闭包。
- 来源可靠性正确时折扣可能改善冲突融合；若可靠性标注本身错误，效果不受保证。
- 生成分布接近朴素贝叶斯假设；没有真实标签/跨分布验证，不能宣称可用于实际监管或优于所有方法。
