# 原要求内的准确率研究（2026-11-25）

方案与结论：`experiments/lab04/REQUIREMENT_ACCURACY_PLAN.md`。

- `train_representation_diagnostic.json`：原图train分类、PCA与复制通道相关性，只是诊断。
- `cuda_probe.json`：只读核实RTX4050设备Code43；没有改变驱动、环境或设备状态。
- `learned_calibration/`：全部train特征训练的小校准头，两个12轮候选；未通过确认门槛。
- `local_reconstruction/`：FRN式冻结局部头，五组配置；未通过确认门槛。
- `channel_pilot/`：两分支各6144train图×2轮的CNN小试验，两边均选回epoch0；不能当作完整训练或正式成绩。
- `sources.json`/`source_audit.json`：实际联网来源与哈希。

根目录的外部网页/代码`.txt`只作为本地阅读缓存，不随项目包分发。子目录的JSON是原始方法×任务预测（含重复对照/复用任务，不当作独立任务数）。发布前模型、正式test和网站均未改变；没有新test图像/特征访问。
