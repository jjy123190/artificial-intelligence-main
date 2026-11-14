# 实验3_源码

第31组 · 实验三：大模型辅助的企业股权知识图谱多跳路径搜索

本目录是**完整源码、必要输入、已构建网页和经过核验的实验运行数据**，不是纯源码克隆，也不是另一套报告/PPT提交材料。仅需把本目录整体压缩提交，不必再把同级报告、PPT、视频塞入其中。

## 1. 最快查看网站（Windows）

1. 将本文件夹整体解压到可写的普通目录，保持内部目录结构；不要在ZIP预览窗口内运行。
2. 电脑需要可用的 Python。基础网页启动脚本会检查环境；依赖缺少时需要联网安装，不承诺完全免安装或完全离线。
3. 双击 **`experiments/lab03/打开网站.cmd`**，直接进入实验三。也可双击根目录 `打开网站.cmd` 后选择实验三。
4. 查看路径搜索、启发式优化、LLM辅助、实验评估四个模块；现成训练参数和评分回放已随包保留。
5. 结束后双击根目录 `停止网站.cmd`。默认端口被其他项目占用时，可在PowerShell执行 `powershell -NoProfile -ExecutionPolicy Bypass -File .\start_web.ps1 -Entry lab03 -Port 8876`，不要误用另一项目的页面。

`web/dist` 已构建，普通查看**无需安装Node.js或重新构建前端**。修改前端并重新构建时才需要Node.js和npm，步骤见第4节。启动时新生成 `.venv`、`.state` 和 `outputs/runtime` 属于本机运行状态，它们不在交付副本中预装。

## 2. 环境与命令行启动

本次验收所用版本见 `运行说明/交付核验.json`；依赖要求见根目录 `requirements*.txt`。原项目建议Python 3.11及以上，实际兼容性以依赖能成功安装为准。首次安装需要联网。`运行说明/网页依赖锁定.txt` 保存本次干净环境实测的网页依赖版本；仅用于网页与评分回放，不包含完整实验重训依赖。

在本文件夹根目录打开PowerShell：

```powershell
python -m pip install -r 运行说明/网页依赖锁定.txt
python -B -X utf8 main.py web --port 8876
```

随后浏览器访问 `http://127.0.0.1:8876/#/labs/lab03`。当前文件夹的服务必须单独启动，不能借用原项目已打开的服务。

现成数据的基础演示：

```powershell
python -B -X utf8 main.py lab03 demo --method bfs
python -B -X utf8 main.py lab03 demo --method hybrid
```

## 3. LLM回放与真实推断必须区分

- 网站默认可重放本项目保存的真实Qwen评分和证据分类头结果，**本次点击不等于再次调用LLM**。
- 查看回放不需要API密钥、PyTorch或下载模型权重；随包的是评分记录和轻量训练参数，不包含大模型权重。
- 若需要现场重新运行真实本地模型，先复制本目录作实验副本，再执行：

```powershell
python -m pip install -r requirements-lab03.txt
python -B -X utf8 scripts/prepare_lab03_model.py
python -B -X utf8 scripts/prepare_lab03.py --force --llm
```

`--force` 明确要求重新计算；不加时若现有结果有效，程序会复用结果而不重新推断。这会联网下载固定模型、消耗时间和磁盘，并改写实验结果。重跑后不能把新结果当作原报告冻结结果。远程模式仅在使用者自行配置 `LAB03_LLM_*` 环境变量后可用；本包不含密钥，也不保证远程服务可用。

## 4. 源码与实验复现

| 目录 | 用途 |
|---|---|
| `experiments/lab03` | 状态空间、BFS/A*/MCTS、GA、路径学习、LLM、评估与核验 |
| `backend`、`core` | 共用API、知识图谱、规则和证据能力 |
| `web/src`、`web/scripts` | React/TypeScript源码与前端测试 |
| `web/dist` | 与源码配套的已构建网页 |
| `data` | 原始CSV、图谱、固定配置与模型身份 |
| `outputs/lab01`、`outputs/lab02`、`outputs/lab03` | 统一网站和实验核验使用的运行数据，不是制作素材 |
| `tests`、`scripts` | 自动化测试、初始化、复现实验及原项目工具 |
| `outputs/lab03_materials/可选拓展_20261113` | 两个自包含样例的核验程序及冻结结果，只有这两份必要文件 |

实验一、二的源码和运行数据为统一网站的配套模块，保留是为了避免切换页面或共享接口缺项，本次提交重点仍是实验三。

只读复核（建议先在副本中执行耗时测试）：

```powershell
python -B -X utf8 核查提交文件.py
python -B -X utf8 outputs/lab03_materials/可选拓展_20261113/verify_examples.py
python -B -X utf8 -m unittest discover -s tests -t . -v
python -B -X utf8 -m experiments.lab03.verification
```

单元测试会使用临时目录；最后一条是完整实验证据复核，耗时更长，不重新调用真实LLM；其中重算证据分类头需要 `requirements-lab03.txt` 中的PyTorch等计算依赖，但不需要下载大模型权重。全套单元测试也建议在安装该依赖组后执行；普通网页展示不需要这一组。请以命令返回结果为准，不把历史 `acceptance.json` 当作本轮自动通过证明。

重新训练路径分类器、优化GA并生成基准（**会覆盖运行数据，先复制整个目录**）：

```powershell
python -B -X utf8 scripts/prepare_lab03.py --force
```

修改前端后重建：

```powershell
cd web
npm ci
npm run build
cd ..
```

## 5. 数据、旧归档和材料边界

- 主基准及扩展评估使用教学合成图；共享场景来自项目CSV，不宣称这是实际企业业务评测。
- `outputs/lab03` 的五个既有基线ZIP是证据核验依赖，不是新制作的提交压缩包。核验程序校验其完整SHA256；其中部分包含历史报告/演示，**均非当前提交版本**。不能删改ZIP内部文件后继续声称原核验有效。
- 当前正式报告、PPT、Demo和演讲视频在本文件夹之外单独提交，本包不在顶层重复放置这些材料。
- 报告附录A.1中的 `verify_materials.py` 和A.3中的高清 `figures/evidence` 索引属于原制作工作区，不随代码包交付。包内可以运行上列源码/证据核验命令和两个拓展样例；不要在这里执行缺失的制作资料核验命令。
- `scripts/materials`、`experiments/lab03/report.py` 等原制作工具源码保留以保持完整性，但不附带其全部历史输入；不要运行 `lab03 report/all` 来覆盖当前正式报告。
- `scripts/check_source.py` 只检查“纯源码目录”。本包有意包含运行数据与已构建网页，不适用该纯源码边界检查。请使用根目录 `核查提交文件.py`。
- `运行说明/原项目运行说明.md` 解释GitHub纯源码的初始化方式；本提交包的运行方式以本文件为准。

## 6. 文件核对与提交

`文件校验清单.json` 列出交付文件的SHA256和大小，`核查提交文件.py` 校验交付文件是否缺失或被改变，并报告额外文件。运行产生的缓存、虚拟环境和本机状态不算交付原件；自行修改源码后原清单不再匹配是正常现象。

源码基线：GitHub仓库 `jjy123190/artificial-intelligence-main`，`main` 分支，提交 `ed5e93d74d82dbf88de9ab8796761b167db2c854`。

本目录不包含 `.git`、已安装依赖、本机案例数据库、服务日志、凭证或模型权重。课程平台的创建项目、关联成员和管理员提交仍由小组完成。本次只准备文件夹，不生成外层ZIP。
