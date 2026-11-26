> 历史目标记录：用户现已取消固定数字要求。当前工作以固定协议下的真实提升为准，见 [INCREMENTAL_STUDY.md](INCREMENTAL_STUDY.md)。

# 实验四下一阶段：95%目标与评测边界

用户目标：继续提升准确率，争取95%及以上。按原5-way 1-shot与5-shot、未见类别、每类15张查询、多任务平均准确率评价，不能将单个高分任务当作达标。

## 不能改变的边界

- 保持mini-ImageNet 64/16/20类别隔离。查询标签仅在预测完成后评分。
- 不筛选容易类、图片或高分种子，不丢弃误判，不改变分母，不用训练成绩替代测试成绩。
- 验证选型完成后才打开测试；每种shot保留600个配对任务。
- 不将ImageNet-1k监督预训练直接冒充未见类实验：mini-ImageNet来自ImageNet，测试类别可能在预训练中已被监督学习。
- 95%是目标，不是保证；未达到时明确写未达到。

## 算力核查

2026-11-24：机器列出RTX4050，但PnP Code43；CUDA驱动cuInit返回CUDA_ERROR_NO_DEVICE，现成CUDA版PyTorch也报告0设备。未擅自重装驱动、启停设备或重启。目前只能CPU训练。

## 计划

保留v2正式网站；候选写入.local/lab04-v3目录。检验空间池化、训练类白化、图标签传播；尝试ResNet12函数保持扩宽、按类别平衡的任务损失和独立图片增强。可中断恢复，只以验证类选型，最终真实评估。

## 方法出处核查

- RFS：https://github.com/WangYueFt/rfs ，表征质量。
- SimpleShot：https://github.com/mileyan/simple_shot ，均值中心化与L2归一化。
- PT-MAP：https://github.com/yhu01/PT-MAP ，官方README的WRN miniImageNet结果82.92±0.26%/88.82±0.13%（2026-11-24读取）。不是本项目成绩，也不是数学上限，但表明不能保证简单调参达到95%。


## 当前运行方式

```powershell
python -m experiments.lab04.train_v3 --data-root .local/mini-imagenet --output .local/lab04-v3-wide --epochs 24 --threads 4
python -m experiments.lab04.validate_v3 --candidate .local/lab04-v3-wide --data-root .local/mini-imagenet
python -m experiments.lab04.next_study_status
```

训练恢复使用相同命令和目录，计划或初始权重不一致时拒绝继续。最长24轮，可按验证停滞提前停止；模型选择不依赖测试成绩。辅助watcher在已核验训练进程终止后仅在状态complete时自动运行验证，不会自动修改网站或启动新的训练。

外部参考是Sha-Lab/FEAT公开权重的640维ResNet12编码器，仅作来源明确的强参考；不包含FEAT注意力适应，不声称精确复现论文分数。训练/验证/测试CSV类别与文件名均核对一致。权重只以torch.load(weights_only=True)读取，不执行下载脚本。

RTX4050的Code43是系统设备错误，不是仅安装CUDA版PyTorch即可解决；在用户明确授权硬件/驱动修复前，不变更显卡驱动或重启机器。
