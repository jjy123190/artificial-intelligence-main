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

## 6. 实验四依赖与范围

网站基础安装保留原有轻量依赖。实验四现场推理和重训练另安装：

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r experiments/lab04/requirements.txt
```

实验四随代码携带约0.7 MB的三个权重和14 MB演示池，不需要下载大模型或配置API key。CPU窄ResNet12为50,352参数，完整二阶MAML使用500个元训练任务，不能直接对比标准宽度模型的论文成绩。校准方法使用无标签查询集更新原型，属于转导推理；所有查询标签只进入最后的评分。完整数据下载、训练复现、真实指标和限制见 [实验四说明](experiments/lab04/README.md)。
