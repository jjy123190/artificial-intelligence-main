# 实验四：小样本 ResNet12 特征校准

## 课程自训结果

最终汇报采用**增强模型＋图像局部匹配**（`regional`）。RTX 4050 Laptop GPU 上的 ResNet12 只用 mini-ImageNet 的 64 个基础类、共 38,400 张图训练。实验先按 16 个验证类里两种 shot 的正确数之和选择一套模型，再在 20 个测试类上评估。局部匹配复用增强组的骨干与投影权重，推理时将全局图像相似度与 3×3 区域的匹配分数按 80% 和 20% 融合。[推理代码](regional_kernel.py) · [验证选择记录](../../outputs/lab04_gpu_regional/validation_lock.json)

| 自训方案与测试记录 | 1-shot（正确数 / 45,000） | 5-shot（正确数 / 45,000） | 两项等权平均 |
| --- | ---: | ---: | ---: |
| [第六轮旋转辅助训练](../../outputs/lab04_gpu_rotation_views/results.json) | **74.56%**（33,550） | 82.05%（36,923） | 78.30% |
| [小角度旋转＋高斯模糊](../../outputs/lab04_gpu_image_augmentation/results.json) | 73.94%（33,271） | 82.18%（36,981） | 78.06% |
| [增强模型＋图像局部匹配](../../outputs/lab04_gpu_regional/results.json) | 74.37%（33,467） | **82.64%**（37,189） | **78.51%** |

每种 shot 固定 600 个五分类任务。推理联合使用同任务的无标签查询图，并利用每类 15 张查询图的均衡设定。相对同权重增强组，局部匹配在固定测试上分别多答对 196 和 208 次。5-shot 配对差为 +0.46 个百分点（95% 区间 +0.29 至 +0.64）。在同一批验证类重新抽取 600 个任务后，5-shot 配对差为 +0.10 个百分点（−0.08 至 +0.27）。各轮使用同一批公开测试任务，区间反映单个训练种子下的任务波动。[局部匹配对照](GPU_TRAINING.md#局部监督与区域匹配) · [复核记录](../../outputs/lab04_gpu_regional/confirmation.json)

在 `源码` 目录、装好 PyTorch 的环境中运行。前两条命令分别读取整组任务的归档成绩、用演示图像运行单个任务。后两条命令可查看 1-shot 最佳对照与增强消融。CPU 推理可使用 `--device cpu`。[环境安装](../../README.md#实验四推理环境)

```powershell
python main.py lab04 results --track regional
python main.py lab04 demo --track regional --shots 5 --device cuda
python main.py lab04 results --track rotation_views
python main.py lab04 results --track image_augmentation
```

## 各轮试验记录

**训练图像增强。** 同预算续训 6 轮，加入 ±12° 以内的随机旋转及轻度高斯模糊，5-shot 验证准确率从对照组的 83.28% 升至 83.57%。验证选中的增强模型在固定测试上为 **1-shot 73.94%、5-shot 82.18%**。5-shot 答对 36,981 次，比第四轮多尺度方案多 1 次。运行 `python main.py lab04 results --track image_augmentation` 可查看本次结果。[增强示例、对照与命令](GPU_TRAINING.md#训练图像增强对照) · [测试结果](../../outputs/lab04_gpu_image_augmentation/results.json)

文献方法试验固定第六轮权重，比较了任务内 PCA、支持标签传播以及两者的组合。四种新配置在两种 shot 的固定验证任务上均低于原配置。验证选回第六轮后，重放测试的 90,000 项预测与该轮归档逐项一致。可在源码目录执行 `python main.py lab04 results --track task_subspace` 查看成绩。[验证对照](../../outputs/lab04_gpu_task_subspace/shortlist.json)

第七轮在本机 RTX 4050 上比较了两种少样本度量训练、按裁剪差异校准和通道距离池化。空间特征占 25% 时，5-shot 在固定验证任务上从 83.43% 升至 83.56%，在同一验证类上重新抽取 600 个任务，结果为 83.60% → 83.72%。两次提升的配对 95% 区间均跨零。固定验证任务的 1-shot 从 76.57% 降到 76.34%。该轮选择第六轮权重，固定测试准确率为 **1-shot 74.56%、5-shot 82.05%**。用 `--track distance_covariance` 查看该轮选择并运行模型。[训练与复跑命令](GPU_TRAINING.md#第七轮特征度量与空间池化) · [5-shot 复核结果](../../outputs/lab04_gpu_distance_5shot/confirmation.json)

第六轮在本机 RTX 4050 上用四个旋转方向的类别预测继续训练轻量 ResNet12。按验证集选择第 24 轮权重后，固定测试任务上为 **1-shot 74.56%、5-shot 82.05%**，相对上一轮分别 +0.43、−0.09 个百分点。两项配对 95% 区间均跨零。用 `--track rotation_views` 查看成绩和运行模型。[训练与复跑命令](GPU_TRAINING.md#第六轮旋转辅助训练) · [固定测试结果](../../outputs/lab04_gpu_rotation_views/results.json)

第五轮按验证集选出的课程自训 ResNet12 在固定测试任务上为 **1-shot 74.12%、5-shot 82.14%**。该轮在同一台 RTX 4050 上比较了等步数的普通续训和自蒸馏。两者在固定验证任务上分别得到 74.88% / 82.14% 和 75.24% / 82.51%，上一轮权重为 75.59% / 82.96%，因此保留上一轮模型。使用 `--track self_distill` 可查看该轮选择及运行导出模型。[训练对照与复跑命令](GPU_TRAINING.md#第五轮训练对照) · [验证选型](../../outputs/lab04_gpu_self_distill/validation_lock.json)

2026-11-26，本机 RTX 4050 Laptop GPU 上的自训 ResNet12 在 mini-ImageNet 固定测试任务上达到 **1-shot 73.75%、5-shot 82.13%**。沿用 64 个基础类，从上轮权重续训 30 轮，在相同的五分类任务上分别提高 **2.89、2.19 个百分点**。验证选中的推理方式融合原图、中心裁剪及其翻转，再按每类 15 张查询图的均衡设定更新原型并分配标签。课程主结果采用自训权重。复跑命令与外部预训练参考见[笔记本 GPU 训练记录](GPU_TRAINING.md)。[验证对照](../../outputs/lab04_gpu_rotation/validation_lock.json) · [固定测试成绩](../../outputs/lab04_gpu_rotation/results.json)

后续固定同一自训骨干，单独选择裁剪视图得到 73.75% / 82.18%，再按验证集选择原型初始化方法得到 **74.12% / 82.14%**。最终方案相对上一轮测试提升 0.38 / 0.01 个百分点，两项配对 95% 区间均跨零。命令行使用 `--track restarts` 可运行该方案。[第四轮配置与对照](../../outputs/lab04_gpu_restarts/validation_lock.json) · [逐任务测试汇总](../../outputs/lab04_gpu_restarts/results.json)

保留课程要求的 **mini-ImageNet / 轻量 ResNet12 / Baseline、二阶 MAML、特征校准对比**。当前网站使用 v2；v1 的权重、逐任务记录及负结果完整保存在 `outputs/lab04/history/v1/`，原版报告/PPT/视频也保留原样。

## 原要求内的新提分路线（2026-11-25）

最新[约束核对与优先方案](REQUIREMENT_ACCURACY_PLAN.md)：重新核对原始课程要求，优先“不引入外部权重的独立初始化轻量ResNet12＋两阶段训练＋原校准”。发现当前扩宽复制通道相关均值0.995、未增强train准确率63.80%；只是诊断，不虚称已找到唯一原因。

11 月 25 日完成两种学习型校准各 12 轮、5 种局部重建组合及两分支同预算 CNN 小试验。候选方案均未达到该轮发布标准，当天发布的 `continued` 模型成绩为 56.47% / 68.81%。

11 月 25 日的 Code 43 和 CPU 版 PyTorch 是原实验机器的记录。以上各轮本机 GPU 训练已在当前笔记本的 RTX 4050 Laptop GPU 上完成，使用 PyTorch 2.7.1+cu118。

## 联网调研与新一轮准确率方案（2026-11-25）

已联网核验 RFS、TIM、PT-MAP、Distribution Calibration、FEAT、SimpleShot 和 CVPR2025 UNEM 的论文/官方实现，并完成冻结特征的验证研究。**本次不是修改历史分数，也不是继续UI微调。**

- [完整改进方案](LITERATURE_IMPROVEMENT_PLAN.md)：模型容量/训练配方诊断，严格自训与外部教师两条路线，课程要求/消融/算力/发布门槛。
- [实际验证与负结果](LITERATURE_VALIDATION_RESULTS.md)：同任务推断比较、线性教师迁移、独立外部参考诊断；所有新成绩明确为val。
- 现有自训轻量模型只约0.51M参数，官方参考encoder约12.42M；不能仅凭“都叫ResNet12”对标论文大骨干。
- TIM/PT-MAP候选与线性几何迁移未通过预先固定的确认门槛；保留负结果，**没有重新测试/自动发布/修改网页正式结果**。
- 新增 `train_feature_distillation.py`：默认仅登记，需显式`--run`才进行昂贵训练；教师冻结，学生保持0.51M，训练投影头不进入部署。已做合成前后向/安全测试，**尚未获得蒸馏学生的真实提分结果**。

## 后续增量优化（不设固定目标准确率）

用户已取消固定数字要求，当前以固定协议下可验证的改善为准。

| 策略 | 1-shot | 5-shot |
|---|---:|---:|
| 原完整校准 | 51.41% | 64.50% |
| 白化＋图传播 | 53.03% | 66.91% |
| 双视图稳健校准 | 54.49% | 67.16% |
| 较宽骨干＋度量训练（独立研究） | 55.16% | 67.62% |
| 较宽骨干＋双视图校准（2026-11-25） | 55.83% | 68.43% |
| **同容量续训＋双视图（额外16轮）** | **56.47%** | **68.81%** |

每种shot均600个相同配对任务。新双视图相对图传播的配对差为 **+1.45±0.52pp / +0.25±0.33pp**：1-shot区间高于零，5-shot区间跨零，只能说均值略高，不能确认改善。先前的完整方案和图传播均保留为对照。

新方案仍使用同一自训轻量骨干：对同图原图/水平翻转做特征拼接，使用**基础训练类内协方差**白化，然后原型与图传播。验证集没有选择硬容量分配，最终保持软分配。多视图增加编码开销，因此单视图图传播保留为默认，可在校准策略中选择“双视图稳健校准”。

较宽ResNet12的24轮训练、验证选型和固定测试已完成，已作为独立可选策略发布。宽度 `[16,32,64,128]`，509,856参数，不使用外部权重；它增加了模型容量和训练预算，因此不归类为同骨干校准消融。相对双视图的配对差为 **+0.67±0.67pp / +0.46±0.39pp**：1-shot精确区间仍跨零，不能确认改善；5-shot区间高于零。相对单视图图传播分别增加2.12±0.66pp/0.71±0.37pp。5-shot验证配置选择关闭图传播，不虚称该模块始终有益。

状态查看 `python -m experiments.lab04.next_study_status`。本轮计划见 [INCREMENTAL_STUDY.md](INCREMENTAL_STUDY.md)，实测详情见 `outputs/lab04/next_study/multiview/RESULTS.md`。旧的TARGET95.md仅为历史目标记录。

## 原完整方案结果（v2，保留的真实测试）

测试使用完整20类×600张图像池，5-way 1-shot / 5-shot，每类15张query，每种shot **600个配对episode**。所有方法使用相同支持与查询图片。下列 ± 为95%区间半宽，单位百分点。

| 方法 | 1-shot | 5-shot |
|---|---:|---:|
| 新版 Baseline | 45.87% ± 0.77 | 61.70% ± 0.72 |
| 二阶 MAML | 41.88% ± 0.75 | 54.89% ± 0.72 |
| 完整校准 | **51.41% ± 1.01** | **64.50% ± 0.75** |
| 仅原型更新（保留中心化） | 51.21% ± 1.02 | 64.38% ± 0.76 |
| 原版 Baseline，同任务复测 | 42.06% ± 0.76 | 56.56% ± 0.72 |
| 原版完整校准，同任务复测 | 43.38% ± 0.85 | 52.90% ± 0.72 |

完整校准相对同版Baseline的配对提升为 **+5.54 ± 0.58 pp / +2.81 ± 0.34 pp**；相对旧版完整方案的同任务配对提升为 **+8.04 ± 0.81 pp / +11.60 ± 0.65 pp**。原版100任务的公开成绩仍保留，不能直接与新协议均值相减当作配对提升。

**结果边界**：对齐训练在本次实验中几乎没有收益，完整方案与去对齐方案的均值相同；原型更新是主要有效因素。MAML增加预算与适应选型后仍低于同版Baseline。CI只表示固定一次训练权重后的任务抽样波动，不覆盖不同训练种子；不宣称所有模块均有效或方法普遍领先。

## 模型与方法

- **ResNet12-Lite**：4个残差块、每块3层3×3卷积，宽度 `[8,16,32,64]`，64维嵌入，GroupNorm，130,160参数。远小于标准论文骨干，不能对标640通道ResNet12成绩。
- **训练**：使用全部38,400张基础类图像，监督预训练40轮，只看验证类选择第39轮；标签平滑0.05，AdamW权重衰减0.0005。Baseline和对齐分支从相同权重出发，相同顺序与增强再训练3轮，仅对齐系数0/0.5不同。
- **MAML**：真正的全网络二阶元学习，4,000个任务。验证集选择检查点及测试适应参数；1-shot使用5步、学习率0.025，5-shot使用5步、学习率0.05。query标签不参与测试适应。
- **预处理**：基础训练类的L2特征中心用于固定中心化；不使用测试类标签或当前query均值。
- **均值与方差**：近邻基础类统计收缩，方差再与共享尺度混合，防止不可靠类条件先验主导新类距离。
- **原型更新**：无标签query软分配迭代；v2选择开启均衡类别质量先验。本评测每类query数量相同，真实不均衡任务不保证适用。

1-shot / 5-shot均值系数为0.1 / 0.05，方差混合为0.4 / 0.1；原型权重0.15、迭代4次、温度0.15。具体配置、所有候选及校验指纹保存在结果目录。

### 消融语义

`graph` 为单视图白化＋图传播，`multiview` 为双视图＋训练类内协方差白化＋图传播，`wide` 为独立较宽自训骨干研究。各方案结果单独记录，`wide`不与相同骨干的基础消融混为一谈。`wideview`复用较宽权重，原图/翻转均值与训练类内白化，没有新增训练，但每图编码两次。

`none`保留对齐骨干、不做中心化和推理校准。`mean`、`variance`、`prototype`、`full`保留锁定的训练类特征预处理；分别启用所示统计模块。评测另含“仅中心化”“去中心化”“去均衡先验”“去训练对齐”，将贡献分开。原型更新为转导方法；均值/方差策略不使用其他query图片。

## 公平性与可复核性

训练/验证/测试类别严格隔离。校准搜索使用验证类每类200张图片，候选复核用另外400张图片；两池图片互斥，但来自相同验证类别且复核结果用于选择，因此不作为独立最终成绩。测试打开前将权重与超参数写入锁文件。

原测试基准已有公开结果，本次属于后续优化，不声称是从未见过测试分布的外部验证。没有根据测试成绩修改参数，没有筛选高分演示种子。演示池从20个测试类中按固定种子随机抽取每类40张。

`episodes.json`与`evaluation_episodes.jsonl`包含样本ID、逐图预测、评分标签及逐方法结果；query标签仅在预测后用于评分。`verify`重新计算全部正确数、均值、95%区间、配对差与耗时，并核对文件指纹和支持/查询互斥。修改汇总数字但保留权重同样会核验失败。

## 启动和复现

Python 3.11+；依赖 `torch>=2.5`、`numpy`、`Pillow`。网站使用额外Web依赖，详见源码根目录README。

```powershell
# 在 源码 目录
python -m pip install -r experiments/lab04/requirements.txt
python main.py web --port 8765
# http://127.0.0.1:8765/#/labs/lab04
python main.py lab04 verify
python main.py lab04 demo --method calibrated --shots 1 --seed 20261124
python -m unittest experiments.lab04.test_lab04 experiments.lab04.test_live_artifacts tests.test_lab04_api -v
```

也可双击 `experiments/lab04/打开网站.cmd`。网页和必要权重、演示图片随源码保存；完整数据集不打包。

```powershell
python -m experiments.lab04.prepare_data --root .local/mini-imagenet
python -m experiments.lab04.improve --data-root .local/mini-imagenet --output .local/lab04-v2
python -m experiments.lab04.refine --stage cache --data-root .local/mini-imagenet --output .local/lab04-v2
python -m experiments.lab04.refine --stage maml --data-root .local/mini-imagenet --output .local/lab04-v2
python -m experiments.lab04.refine --stage tune --data-root .local/mini-imagenet --output .local/lab04-v2
python -m experiments.lab04.evaluate_v2 --data-root .local/mini-imagenet --output .local/lab04-v2 --previous outputs/lab04/history/v1 --episodes 600
python main.py lab04 verify --artifacts .local/lab04-v2
```

首次升级（历史目录尚未建立）时，`--previous`应指向原版`outputs/lab04`。原版训练脚本`experiment.py`保留，需输出到其他目录避免覆盖v2。

浏览器测试：在`源码/web`运行 `npm ci`、`npm run build`、`npm run test:lab04`；服务URL用`LAB04_TEST_URL`设置，浏览器路径可用`PLAYWRIGHT_EXECUTABLE_PATH`指定。

## UI与部署边界

新界面支持同任务三方法比较、误判与类别筛选、单图五类分数、混淆矩阵、PCA原型变化与记录导出。现场推理包含真实图像编码；单任务准确率不能代表基准平均。类别分数不是“预测正确概率”的保证。

评测表中的缓存特征分类耗时与MAML全流程不可直接排序；`edge_benchmark.json`另记录所有方法相同计时范围的CPU中位数与P95。这里是Windows本机CPU的端侧可行性示范，**没有冒充手机/ARM硬件实测**。网站图片与资料仅用于非商业教育研究。

新结果补充报告在`deliverables/lab4/优化补充/`。旧报告/PPT/视频明确属于v1；不把原版负结果删掉或改成新分数。更多实施记录见[OPTIMIZATION.md](OPTIMIZATION.md)。

## 方法出处

- Vinyals et al., *Matching Networks for One Shot Learning*, 2016，mini-ImageNet。
- Ravi & Larochelle, *Optimization as a Model for Few-Shot Learning*, 2017，类别划分。
- Finn et al., *Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks*, 2017。
- Snell et al., *Prototypical Networks for Few-shot Learning*, 2017。
- Yang et al., *Free Lunch for Few-shot Learning: Distribution Calibration*, ICLR 2021。当前实现是明确记录的轻量变体，不宣称复现原文成绩。


## 发布质量与可信展示

网页读数不再只读取results.json汇总：首次读取时独立核对文件指纹、逐任务预测、配置锁、配对样本、正确数、95%区间和配对差；任何相关文件修改都会使缓存失效。没有PyTorch也能核验和查看历史分数，缺少模型时明确禁止现场运行。

评测页支持“重点查看”方法，并显示固定预测的任务准确率分位、胜/平/负任务数及类别分数可靠性。分位区间不是均值置信区间，重复查询出现次数不是独立图片数。PCA图共同坐标与等比例轴，注明二维解释方差，不将二维距离当作真实分类距离。

质量计划与未完成项见[QUALITY_PLAN.md](QUALITY_PLAN.md)。较宽模型已经完成核验并单独发布；未来候选未完成测试前仍不得写入正式分数。


## 2026-11-25 第一轮接管：交互与数据边界修复（历史）

承接会话 `01a20b6f-0803-7000-a58a-c4bb2e138c66` 的“继续优化完善”。本轮不重训、不改锁定配置、不更改任何逐任务结果；较宽模型的多任务准确率仍为 **55.16% / 67.62%**。

- **取消等待**：现场推理与三方法对比均可直接取消；保留已完成的结果，不再发起后续方法。仅取消浏览器等待，服务器已经开始的当前计算可能仍会结束；迟到响应不能覆盖取消后的页面或下一次任务。取消后键盘焦点返回运行按钮。
- **不可信接口数据**：以 `unknown` 接收 JSON，先核查类别/样本/分数/版本指纹，再检查点与样本的一一对应、有限二维坐标、原型、解释步骤与可选 PCA 诊断。缺字段、空对象、非数值坐标等会被明确拒绝，保留上一次有效结果，不进入 React 或 SVG。
- **配对身份**：缺少样本池或发布版本指纹的两个结果不再因为字段都为空而被判为同一任务；同时核对类别顺序。
- 浏览器验收新增畸形接口、对比中途取消、取消后立即重试三组，总计 **22 组通过、零跳过**。接口错误数据由测试拦截构造，不是模型实验数据。

本机没有当前 Playwright 版本对应的浏览器下载缓存时，可使用已安装的 Chrome 执行，不必重新下载：

```powershell
$env:PLAYWRIGHT_EXECUTABLE_PATH = 'C:\Program Files\Google\Chrome\Application\chrome.exe'
$env:LAB04_TEST_URL = 'http://127.0.0.1:8765'
cd web
rtk npm run test:lab04-contract
rtk npm run test:lab04
```

该路径仅是本机示例。测试与接管记录见项目根目录 `制作资料/实验四/接管与增量验收_20261125.md`。此前报告与提交压缩包保留原样，不能把它们当作本轮前端补丁的新版本。


## 2026-11-25 第二轮：课程要求核对、独立页面与真实精度提升

- 要求逐项证据表：[REQUIREMENTS_AUDIT.md](REQUIREMENTS_AUDIT.md)。原任务的模型、学习范式、损失、校准、mini-ImageNet对照、消融及CPU本地部署都有对应证据；课程平台组员关联/项目/提交不由本地代码证明，指定手机/ARM硬件也未冒称已测。
- 结果独立URL：`evaluation`性能、`ablations`消融、`reliability`可靠性、`deployment`端侧性能、`history`来源；另有`inference`、`protocol`；课程要求索引改为本地文档。支持直接访问、刷新、前进后退、站内任务与shot保留。
- 可视化：可选择准确率区间条形图、配对差点区间、任务分位/改善持平下降、分箱可靠性图、中位数/P95耗时。精确表格可展开，不因可视化隐藏负结果。
- **较宽双视图55.8289% / 68.4267%**（每种shot600个原协议配对任务）。相对原较宽55.1556%/67.6222%，配对差 **+0.6733±0.4940pp / +0.8044±0.2558pp**，两个区间都高于零。原权重不变，不新增训练；成本是每图两次编码。默认仍为更轻的图传播。
- 新方案先比较18个验证候选，搜索160/复核400任务，锁定后才进行测试。mean/0.5白化指数/0.5类内混合两种shot相同，5-shot不启用图传播。此前特征融合验证未通过改善证据门槛的18个候选完整保留，未读取测试图像。
- 权重、选型、所有预测与评分指纹可重算；旧耗时归档在`history/pre-wideview-edge-20261125.json`，新全方法20次统一耗时记录不混入旧版本计时。
- 新补充材料：根目录`deliverables/lab4/优化补充/实验四_双视图宽骨干与分页面说明.md/.docx/.pdf`；原v1文档、PPT、视频保留，不冒充最新版。

```powershell
# 只读当前状态，显示所有已发布方法与候选
rtk proxy python -B -X utf8 -m experiments.lab04.next_study_status
# 全部证据重算（包括新扩展）
rtk proxy python -B -X utf8 main.py lab04 verify
# 新策略现场真实预测
rtk proxy python -B -X utf8 main.py lab04 demo --method calibrated --calibration wideview --shots 1
# 验证候选（原缓存/数据需完整；重复运行不改变已完成研究）
rtk proxy python -B -X utf8 -m experiments.lab04.study_cached_refinement
rtk proxy python -B -X utf8 -m experiments.lab04.study_wide_views
# 只能读取已完成锁定配置，不重新搜索
rtk proxy python -B -X utf8 -m experiments.lab04.evaluate_wide_views
```

没有宣称算法无错误、所有领域通用或达到了理论最优。只在本次实际验收范围报告结论，后续独立域/多训练种子验证仍是研究方向。


## 2026-11-25 第三轮：状态接口与选型证据加固

当前实验成绩与权重全部保持不变。本轮针对“逻辑无误”继续从反例复核，未重新打开候选搜索，也未按测试分数调参。

- 修复可复现的白屏：实验状态接口的`model.channels=null`等异常JSON之前会进入页面。现在首页与全部结果页以`unknown`接收状态，核查嵌套字段、表格身份、配对差与准确率、诊断分箱总数/ECE、任务分位、扩展可用性与耗时中位数/P95。
- 首次状态读取失败可重试；刷新异常保留上一份有效结果并明确提示“不是本次刷新成功”，不清空已完成推理记录。
- `selection_audit.py`使用标准库，从已有验证分数日志重算36个shot×候选记录和48条分数序列，核查搜索前四、复核区间下界选择、获胜候选及改善门槛。正研究和负研究均核对，不能只凭`promising=true`当作证明。
- 原验证锁与逐任务分数日志纳入网页证据审计及缓存失效范围。文件变化须重新校验；发布的算法、配置、得分、模型SHA256都未改变。
- 合约检查入口仍是`cd web; rtk npm run test:lab04-contract`，同时覆盖推理与状态；纯状态合约可单独运行`rtk proxy node scripts/check-lab04-snapshot.mjs`。

本轮不会重复生成宣称“精度继续上涨”的报告。有效分数仍为较宽双视图55.83%/68.43%，均衡查询、单训练种子、公开基准反复使用与CPU部署的边界不变。


## 2026-11-25 第四轮：完整材料与独立容量不变续训候选

当前报告、PPT、真实Demo与讲解已在根目录`deliverables/lab4/当前版本`整合，旧v1与补充不覆盖。“报告与演示”和“要求与交付”已按用户要求从网站移除，相关材料与证据保留在项目目录；旧网址进入推理工作台。完整交付命令：`rtk proxy python -B -X utf8 -m scripts.package_lab04_delivery`。

为检验24轮较宽训练是否充分收敛，启动额外16轮、同容量续训研究。计划见`CONTINUATION_STUDY.md`；从wide.pt开始，固定超参数，不改发布目录，第0轮也可被选中。下游600个验证配对任务/shot已提前登记，沿用较宽双视图固定配置。只有完成训练并通过验证门槛才考虑一次锁定测试，**运行中的分数不写入正式成绩**。

候选状态在`.local/lab04-continuation-20261125/training.json`，验证在`.local/lab04-continuation-validation-20261125`。只读状态入口`python -m experiments.lab04.next_study_status`；活跃进程必须另查PID/创建时间/命令。训练可从epoch边界优化器与RNG状态恢复，验证守护程序不会自动重启训练、自动读取测试或自动发布。


## 2026-11-25 第五轮：消融的参考对象必须明确

新增服务器从冻结逐任务分数重算的`module_contrasts`，六组对照×两种shot。消融页可切换“模块直接对照”与“相对Baseline”，不再将整体对基线提升误当作模块收益。每行标明参与比较的方法、信息条件、准确率、直接配对区间和改善/持平/下降任务数；导出也保留这些记录。

特别注意：`full − mean_variance`同时加入原型更新和均衡分配，归纳转为转导，不是原型单一因素；`full − prototype_only`同时改变均值和方差。区间为未作多重比較校正的逐项95%区间，不用边缘正值宣称全部模块显著有效。极小数值显示为0.00而非误导的−0.00。

当前报告/PPT/实际Demo与讲解同步说明，旧版本保存在deliverables/lab4/历史版本。未更改权重或正式测试产物。新增`continuation_audit.py`用于续训完成后的独立验证门槛：重算epoch选择、验证预测/配对区间，训练未完成时直接拒绝，不提前开放测试。


## 2026-11-25 续训研究完成并发布

固定16轮训练已全部完成，保持509,856参数，验证选择第12轮。固定600配对验证任务/shot相对旧较宽双视图改善 **+0.4378±0.2723pp / +0.3667±0.1558pp**，独立audit通过预登记门槛后才锁定一次原测试。

正式测试仍为原协议每种shot600任务，`continued`达到 **56.4733% / 68.8133%**。相对`wideview`的配对差 **+0.6444±0.3127pp / +0.3867±0.1739pp**，两个区间高于零。新模型单列`continued.pt`，旧模型/逐任务证据/分类设置不改写；计算容量不增加，但额外训练预算必须披露，不能说成新的校准算法贡献或同预算比较。

脚本：`train_continuation.py`（冻结预算）、`validate_continuation.py`（固定验证）、`continuation_audit.py`（训练与验证重算）、`evaluate_continuation.py`（一次锁定测试）、`continued_audit.py`（完整发布证据重算）、`continued_extension.py`（可移动现场推理）。预测API仍没有query标签输入。

```powershell
rtk proxy python -B -X utf8 main.py lab04 demo --method calibrated --calibration continued --shots 1
rtk proxy python -B -X utf8 main.py lab04 verify
rtk proxy python -B -X utf8 -m experiments.lab04.continued_audit --artifacts outputs/lab04 --reference outputs/lab04
```

所有8种方法重新在同一轮测量20次图像编码＋分类耗时，新旧计时档案独立保留。相同容量/结构的测量差仍受系统负载影响，不宣传继续训练本身使推理更快。默认保留轻量图传播，续训作为可选研究方案。

## 当前页面精简（2026-11-26）

网页提供“现场推理”“方法与协议”“对比与消融”三个入口。报告、PPT 和视频记录网页发布时的模型，可在仓库的课程材料目录查看。最新自训模型通过本页开头的 `--track regional` 命令演示。
