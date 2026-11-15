# 完整股权链路实验结果

本文件由真实运行结果生成，不是课程报告。全部标签与业务图均为合成数据。
分类指标使用最终复核行动；评分损失不把被锁定的行动1/0冒充概率。

## 清洁观测图

| 方法 | Accuracy | F1 | Recall | FPR | 已知约束违反率 | 原始评分Brier↓ |
|---|---:|---:|---:|---:|---:|---:|
| rules_only | 0.7778 | 0.6870 | 0.8066 | 0.2362 | 0.0000 | 0.2222 |
| bayes_only | 0.8139 | 0.7082 | 0.7483 | 0.1537 | 0.2105 | 0.1271 |
| fuzzy_only | 0.6833 | 0.6499 | 0.9568 | 0.4407 | 0.0325 | 0.2106 |
| hybrid_fixed_unlocked | 0.8167 | 0.7163 | 0.7712 | 0.1624 | 0.1847 | 0.1359 |
| hybrid_fixed_locked | 0.7778 | 0.6944 | 0.8341 | 0.2485 | 0.0000 | 0.1359 |
| hybrid_tuned_unlocked | 0.8139 | 0.7082 | 0.7483 | 0.1537 | 0.2105 | 0.1324 |
| hybrid_tuned_locked | 0.7806 | 0.6971 | 0.8341 | 0.2442 | 0.0000 | 0.1324 |
| fuzzy_learned | 0.6944 | 0.6584 | 0.9568 | 0.4233 | 0.0325 | 0.2074 |
| hybrid_adaptive_unlocked | 0.8139 | 0.7082 | 0.7483 | 0.1537 | 0.2105 | 0.1287 |
| hybrid_adaptive_locked | 0.7806 | 0.6971 | 0.8341 | 0.2442 | 0.0000 | 0.1287 |
| hybrid_adaptive_calibrated | 0.8000 | 0.7139 | 0.8186 | 0.2051 | 0.0686 | 0.1287 |

## 锁定相对不锁定：同图配对的固定权重对照

| 条件 | ΔTP | ΔFP | ΔRecall | ΔF1 | 去除的约束违反数 |
|---|---:|---:|---:|---:|---:|
| clean | 2.33 | 7.00 | 0.0629 | -0.0219 | 9.33 |
| value_noise | 2.67 | 7.00 | 0.0707 | -0.0180 | 9.67 |
| edge_missing_020 | 1.33 | 1.67 | 0.0354 | 0.0089 | 3.00 |
| false_loop_020 | 2.00 | 17.33 | 0.0553 | -0.0865 | 19.33 |
| combined | 1.00 | 0.33 | 0.0278 | 0.0124 | 1.33 |
| severe | 0.67 | 0.00 | 0.0199 | 0.0108 | 0.67 |

## 自动拟合与校准行动对照

模糊参数只读取训练图；融合权重只读取清洁验证图。行动门控仅读取预先固定的四种验证扰动。
代价为(FP+2FN)/N；硬锁定、不锁定、门控共用评分，门控不保证零结构违反或所有指标提升。

| 条件 | 方法 | F1 | 误报率 | 代价 | 已知约束违反率 |
|---|---|---:|---:|---:|---:|
| clean | fuzzy_learned | 0.6584 | 0.4233 | 0.3194 | 0.0325 |
| clean | hybrid_adaptive_unlocked | 0.7082 | 0.1537 | 0.2639 | 0.2105 |
| clean | hybrid_adaptive_locked | 0.6971 | 0.2442 | 0.2694 | 0.0000 |
| clean | hybrid_adaptive_calibrated | 0.7139 | 0.2051 | 0.2556 | 0.0686 |
| value_noise | fuzzy_learned | 0.6511 | 0.4220 | 0.3278 | 0.0203 |
| value_noise | hybrid_adaptive_unlocked | 0.7090 | 0.1350 | 0.2667 | 0.2305 |
| value_noise | hybrid_adaptive_locked | 0.7017 | 0.2232 | 0.2667 | 0.0000 |
| value_noise | hybrid_adaptive_calibrated | 0.7079 | 0.1928 | 0.2639 | 0.0669 |
| edge_missing_020 | fuzzy_learned | 0.6052 | 0.3006 | 0.3833 | 0.1231 |
| edge_missing_020 | hybrid_adaptive_unlocked | 0.6060 | 0.2361 | 0.3806 | 0.4913 |
| edge_missing_020 | hybrid_adaptive_locked | 0.6124 | 0.3347 | 0.3778 | 0.0000 |
| edge_missing_020 | hybrid_adaptive_calibrated | 0.6075 | 0.3261 | 0.3833 | 0.0494 |
| false_loop_020 | fuzzy_learned | 0.5922 | 0.5831 | 0.4222 | 0.0145 |
| false_loop_020 | hybrid_adaptive_unlocked | 0.6992 | 0.2101 | 0.2694 | 0.3140 |
| false_loop_020 | hybrid_adaptive_locked | 0.6144 | 0.4394 | 0.3750 | 0.0000 |
| false_loop_020 | hybrid_adaptive_calibrated | 0.6549 | 0.3528 | 0.3306 | 0.1034 |
| combined | fuzzy_learned | 0.5414 | 0.4237 | 0.4667 | 0.0346 |
| combined | hybrid_adaptive_unlocked | 0.6063 | 0.2736 | 0.3861 | 0.4876 |
| combined | hybrid_adaptive_locked | 0.5674 | 0.4412 | 0.4389 | 0.0000 |
| combined | hybrid_adaptive_calibrated | 0.5590 | 0.4412 | 0.4500 | 0.0185 |
| severe | fuzzy_learned | 0.4948 | 0.3844 | 0.5083 | 0.0374 |
| severe | hybrid_adaptive_unlocked | 0.5727 | 0.2668 | 0.4167 | 0.4857 |
| severe | hybrid_adaptive_locked | 0.5290 | 0.4056 | 0.4778 | 0.0000 |
| severe | hybrid_adaptive_calibrated | 0.5290 | 0.4056 | 0.4778 | 0.0000 |

## 验证集权重

- seed 7: 原模型{'bayes': 1.0, 'fuzzy': 0.0, 'rule': 0.0}；自动拟合{'bayes': 1.0, 'fuzzy': 0.0, 'rule': 0.0}；门控最低支持评分=0.25。
  评分器退化为单组件；是否仍有结构锁定作用，应看最终行动而非只看score。
- seed 19: 原模型{'bayes': 1.0, 'fuzzy': 0.0, 'rule': 0.0}；自动拟合{'bayes': 1.0, 'fuzzy': 0.0, 'rule': 0.0}；门控最低支持评分=0.15。
  评分器退化为单组件；是否仍有结构锁定作用，应看最终行动而非只看score。
- seed 43: 原模型{'bayes': 0.5, 'fuzzy': 0.5, 'rule': 0.0}；自动拟合{'bayes': 0.75, 'fuzzy': 0.25, 'rule': 0.0}；门控最低支持评分=0.0。

## 解读和复现边界

- 锁定与不锁定共享同一观测图、同一特征、同一模型参数和同一阈值，原始评分逐样本完全相同。
- 已知约束违反率分母是观测图推出高预警的样本数；为0是策略保证，不是对真实风险的完美预测。
- true_alert_retention 以干净图独立结构真值为分母；缺边造成未证实预警时，锁定也不保证保留它。
- 独立的外部模拟风险标签用于Precision/Recall/F1；存在结构线索的正常样本仍会计入误报，不能删除。
- reference_graphs.jsonl 保存标签和干净图；observed_graphs.jsonl 只保存观测输入及扰动审计信息，模型不读取标签或干净图。
- 源记录不完整时，未能确定的交易/担保比例和未证循环使用None，而不是假装为0。
- 错误闭环注入可以使硬规则保留错误线索。graph_audit.csv 与 unsupported_forced_actions 明确记录这类代价。
- 配对变化按三个种子汇总；没有作统计显著性宣称，也不保证融合在任何分布下优于单模型。
- examples 中展示的是首次出现的解释案例，全部总体预测均保留在 predictions.csv。
