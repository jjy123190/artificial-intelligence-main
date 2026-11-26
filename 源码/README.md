# 人工智能课程项目源码

第33组 · 软件工程5班  
组长：梁镇贤；组员：陈宗波、温文辉、陈宇冲、刘亦达。

本目录包含实验一至实验四的完整代码、输入数据、已构建网页和实验结果。实验四按课程指定题目研究图像小样本学习，和前三个股权实验共用网站入口，数据与评价范围独立。报告与演示材料不参与网站运行。

## 1. 启动网站

### Windows双击启动

1. 将整个 `源码` 文件夹放在可写的普通目录中。
2. 安装Python 3.11或以上版本，双击 `打开网站.cmd`。
3. 在首页选择实验一至实验四。
4. 使用结束后双击 `停止网站.cmd`。

启动器会检查Python及网页依赖；首次安装缺失依赖需要联网。`web/dist` 已包含构建后的网页，正常浏览不需要Node.js。网站使用项目专属进程和本机状态，案例编辑不会覆盖原始CSV。

### 命令行启动

在本目录打开PowerShell：

```powershell
python -m pip install -r 运行说明/网页依赖锁定.txt
python -B -X utf8 main.py web --port 8876
```

浏览器访问 `http://127.0.0.1:8876/#/home`。也可执行 `powershell -NoProfile -ExecutionPolicy Bypass -File .\start_web.ps1 -Port 8876`。

## 2. 四个实验入口

| 实验 | 网站入口 | 实现与功能 |
| --- | --- | --- |
| 实验一 | `#/labs/lab01` | `experiments/lab01`；知识表示比较、知识增强、规则推理与关系查询 |
| 实验二 | `#/labs/lab02` | `experiments/lab02`；确定性、不确定性、混合推理、算法优化和实测对比 |
| 实验三 | `#/labs/lab03` | `experiments/lab03`；多跳搜索、GA与路径学习、LLM辅助、对比分析及搜索树回放 |
| 实验四 | `#/labs/lab04` | `experiments/lab04`；ResNet12、MAML、特征校准、真实图像推理与消融 |

四个实验共用 `backend` 和 `web`，前三个实验另共用图谱 `core`。实验四通过 `/api/labs/lab04/snapshot` 读取实测结果、通过 `/api/labs/lab04/run` 运行模型，额外计算依赖在请求时才加载。

## 3. 目录结构

```text
源码/
├─ main.py                 # 网站与实验命令入口
├─ backend/                # API、输入校验、隔离任务和网页服务
├─ core/                   # 图谱、逻辑、规则及证据公共模块
├─ experiments/
│  ├─ lab01/
│  ├─ lab02/
│  ├─ lab03/
│  └─ lab04/
├─ web/src/                # 前端源码
├─ web/dist/               # 已构建网页
├─ data/                   # 原始输入与固定配置
├─ outputs/lab01/          # 实验一结果
├─ outputs/lab02/          # 实验二结果
├─ outputs/lab03/          # 实验三结果与核验归档
├─ outputs/lab04/          # 实验四轻量权重、演示池和真实评测结果
├─ 资料/实验三/            # 报告图表、逐步轨迹和原始记录
├─ 资料/可选拓展/          # 等式证明与三积木规划样例
├─ tests/                  # 自动化测试
├─ scripts/                # 初始化、运行及辅助工具
├─ 运行说明/               # 环境与复现说明
├─ 核查提交文件.py         # 文件完整性核验
└─ 文件校验清单.json       # 文件大小与SHA-256
```

## 4. 实验与结果复核

基础命令：

```powershell
python -B -X utf8 main.py lab01
python -B -X utf8 main.py lab02 demo
python -B -X utf8 main.py lab03 demo --method bfs
python -B -X utf8 main.py lab03 demo --method hybrid
python -B -X utf8 main.py lab04 results
python -B -X utf8 main.py lab04 verify
python -B -X utf8 main.py lab04 demo --method calibrated --shots 1
```

文件及结果核验：

```powershell
python -B -X utf8 核查提交文件.py
python -B -X utf8 资料/可选拓展/verify_examples.py
python -B -X utf8 -m unittest discover -s tests -t . -v
python -B -X utf8 -m experiments.lab03.verification
```

文件核验与给定拓展样例不训练模型。完整测试和实验三证据复核需要计算依赖，安装方式见 `运行说明/环境与复现.md`；验证过程不下载大模型权重、不调用收费API。结果与运行环境摘要见 `运行说明/验证结果.json`。

## 5. LLM与实验边界

网站可回放我们保存的Qwen真实评分和隐藏表示分类结果。回放不等于现场重新调用LLM，不需要密钥、模型权重、PyTorch或Transformers；相关计算依赖只用于完整证据复核、重训练或真实本地推断。

数据包含教学合成图和虚构示例。搜索排序、路径有效性和证据合法性分别检查，预算耗尽不等于图中无解；LLM评分和解释不能替代独立证据核验。

`outputs/lab03` 中的基线ZIP用于对照证据核验，保留对应源码与结果指纹。它们是实验输入，不是额外的源码提交目录。

修改源码或重新生成结果后，文件校验清单不再代表修改后的版本。运行时产生的 `.venv`、`.state`、`outputs/runtime` 和缓存属于本机状态，不属于清单中的交付原件。

## 实验四推理环境

从本目录运行命令。GPU 自训与演示已在 Python 3.10、PyTorch 2.7.1+cu118、RTX 4050 Laptop GPU（6 GB）上验证。先在所用 Python 环境中安装与显卡驱动匹配的 CUDA 版 PyTorch，再安装实验依赖。SciPy 提供归档评测使用的均衡分配求解器。[PyTorch 安装指引](https://pytorch.org/get-started/locally/)

```powershell
python -m pip install -r experiments/lab04/requirements.txt scipy
python main.py lab04 results --track regional
python main.py lab04 demo --track regional --shots 5 --device cuda
```

使用 CPU 时可先执行 `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu` 安装 CPU 版 PyTorch，再在演示命令中使用 `--device cpu`。仓库提供推理权重、校准参数及样例图，完整 mini-ImageNet 数据集用于重训。主模型 1-shot / 5-shot 为 **74.37% / 82.64%**，可用 `--track rotation_views` 和 `--track image_augmentation` 查看另外两组对照。[三组自训结果](experiments/lab04/README.md#课程自训结果) · [训练与评估命令](experiments/lab04/GPU_TRAINING.md)


## 实验四 v2 更新

`#/labs/lab04` 是独立实验入口，`#/labs/lab04/inference` 可进行同任务三方法真实对照、误判分析和记录导出。v2 完整校准的 1-shot / 5-shot 为 51.41% / 64.50%（各 600 个配对任务），分数由逐任务记录重算核验。

v1原始权重、结果与演示池保存在 `outputs/lab04/history/v1`，旧报告/PPT/视频明确属于历史版。完整方法、限制、验证选型和复现命令见 `experiments/lab04/README.md`。直接入口可双击 `experiments/lab04/打开网站.cmd`。

```powershell
python main.py lab04 verify
python -m scripts.package_project --experiment lab04
```

打包包含轻量权重和800张演示池，不包含 `.local` 中完整数据集或训练临时状态。现场CPU部署是端侧可行性示范，不等于在手机/ARM硬件上验证过。


### 后续增量精度优化

在实验四“校准策略”中可选择“白化＋图传播校准”或“**双视图稳健校准**”。后者在相同600任务/shot测试为54.49%/67.16%，相对图传播53.03%/66.91%的配对差为+1.45±0.52/+0.25±0.33个百分点；仅1-shot可确认改善，5-shot差异不确定。双视图增加编码开销。

`outputs/lab04/multiview_results.json`、`multiview_episodes.json`与选择锁保留完整证据，`python main.py lab04 verify`同时检查全部方案。进一步训练的实时状态查看 `python -m experiments.lab04.next_study_status`；不保证数字目标，不用候选验证分数冒充测试成绩。


### 较宽自训模型研究已完成

实验四新增独立可选“较宽骨干＋度量训练”：55.16% / 67.62%（1/5-shot各600配对任务），509,856参数、额外24轮训练，无外部权重。相对双视图方案的1-shot差异区间跨零，5-shot小幅改善有区间支持。原轻量模型与所有旧结果保留；详细说明见实验四README。

网页首次载入会独立重算结果与配对区间并核验指纹，后续按文件版本缓存；“重点查看”方法驱动成绩摘要。原型图注明PCA解释方差，任务波动和概率分数可靠性可查看。


### 2026-11-25：独立结果页面与较宽双视图

实验四结果已按性能、消融、可靠性、端侧性能、历史来源分为独立页面；课程要求证据索引保留在本地文档中。新可选策略`wideview`复用较宽权重而不重新训练，固定配对实测 **55.83% / 68.43%**，相对较宽单视图分别改善0.67/0.80个百分点，两个配对区间均高于零；每图两次编码，代价单列。

打开`#/labs/lab04/evaluation`查看可交互准确率图；课程要求与适用边界见本地文档。完整审查见`experiments/lab04/REQUIREMENTS_AUDIT.md`。课程平台关联组员和提交动作需人工确认，不能由源码宣称已代办。


### 网页版最高准确率方案（同容量续训双视图）

新增`continued`：**56.47% / 68.81%**，每种shot600个原配对任务。保持509,856参数，从旧较宽模型追加16轮训练并验证选第12轮，不改变双视图分类配置。相对旧较宽双视图+0.64/+0.39个百分点，配对区间均高于零；增加训练预算单独披露，旧权重与结果保留。

当前报告与PPT、真实演示/合成语音讲解保留在根目录`deliverables/lab4/当前版本/`；网站不再提供材料入口。课程平台提交状态仍需人工确认。

## 当前UI（2026-11-26）

实验四网页顶部为“现场推理 / 方法与协议 / 对比与消融”，旧的材料链接进入推理工作台。课程材料可从仓库根目录的 `deliverables/lab4/当前版本/` 直接访问，网页支持核验数据及导出推理记录。
