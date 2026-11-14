# 人工智能课程项目 · 源码与完整成果

本仓库完整保存实验一、实验二、实验三的当前源码、测试、运行脚本、输入数据、实验结果、正式报告、PPT、视频和制作核验资料。
2026年11月14日起，GitHub不再仅维护源码；`deliverables/`、实验运行数据和 `web/dist/` 均随项目版本保存。
仅排除密钥、模型权重、已安装依赖、缓存、本机案例数据库和服务运行状态。课程提交材料与制作资料仍分开存放，完整上传不表示全部文件都必须交给老师。

## 最新成果入口

- 实验三正式材料：`deliverables/lab3/`，包含30页报告Word/PDF、17页PPT、1分钟Demo、4分钟配音字幕演讲视频。
- 已核查的可运行代码目录：`deliverables/lab3/实验3_源码/`；用户自行压缩的提交文件：`deliverables/lab3/实验3_源码.zip`。两者原样保存，不因仓库说明调整而改写内部文件或校验清单。
- 实验一、二正式材料：`deliverables/lab1/`、`deliverables/lab2/`。
- 网站运行数据：`outputs/lab01/`、`outputs/lab02/`、`outputs/lab03/`。
- 实验三制作与核验资料：`outputs/lab03_materials/`，包含冻结证据、图表、视频制作源及历史版本；不是另一套正式提交材料。

本README是项目运行说明，不是实验报告。提交目录内部的README和Git基线描述属于交付当时的冻结说明，以各目录用途区分，不把旧材料快照当作当前正式成果。

## 文件结构

```text
artificial-intelligence-main/
├─ main.py                  # 三次实验 / 网页 / 测试的统一入口
├─ backend/                 # 本机 API、输入校验、限时任务与服务
├─ core/kb/                 # 公共知识图谱、逻辑表示、规则与证据能力
├─ experiments/
│  ├─ lab01/                # 知识表示、产生式与图推理
│  ├─ lab02/                # 确定性 / 不确定性推理、优化和评估
│  └─ lab03/                # BFS / A* / MCTS、GA、学习评估器、LLM 与核验
├─ data/
│  ├─ raw/                  # 原始 CSV 输入，不是实验输出
│  └─ lab03/                # 固定问答图谱、课程配置、GA 初始配置及模型身份
├─ web/
│  ├─ src/                  # React / TypeScript / CSS 源码
│  ├─ public/               # 源 SVG 图标
│  ├─ scripts/              # 前端和浏览器检查源码
│  └─ package*.json         # 前端依赖与锁定版本
├─ scripts/
│  ├─ prepare_lab02.py      # 从源码初始化实验二
│  ├─ prepare_lab03.py      # 从源码初始化实验三；默认不调用 LLM
│  ├─ prepare_lab03_model.py# 下载并验证固定本地模型
│  ├─ check_source.py       # 检查纯源码边界，不生成结果
│  ├─ export_source.py      # 导出三次实验纯源码到外部新目录
│  ├─ materials/           # 可选材料生成工具的代码，不包含生成材料
│  └─ check_*.py            # API / 网页 / 知识库检查工具
└─ tests/                   # 单元测试与需要额外产物的集成核验
```

根目录的 `.cmd`、`.ps1` 是 Windows 启动入口；`requirements*.txt` 为 Python 依赖。
没有为了整理结构而改变算法包的导入路径、图谱输入或网页 CSS。

## 基础运行

建议 Python 3.11+；修改源码后重新构建网页需要 Node.js 18+。仓库已带 `web/dist/`，普通启动无需先安装前端依赖。

```powershell
python -m pip install -r requirements.txt
python -B -X utf8 main.py lab02 demo
python -B -X utf8 main.py lab03 demo
```

实验三默认 demo 明确使用 `astar_fixed`，不冒充已经训练的混合模型。
基础 BFS、下界 A*、固定 A*、MCTS 不依赖历史实验结果。

直接运行随仓库保存的已构建网页：

```powershell
python -B -X utf8 main.py web --port 8765
```

修改前端源码后再构建并运行：

```powershell
cd web
npm ci
npm run build
cd ..
python -B -X utf8 main.py web --port 8765
```

或者双击根目录 `打开网站.cmd`；首次会自动安装网页依赖并构建前端。
三次实验也可以使用 `experiments/lab01/打开网站.cmd`、`experiments/lab02/打开网站.cmd`、
`experiments/lab03/打开网站.cmd` 直达。关闭网站使用 `停止网站.cmd`。

完整克隆包含当前实验结果，可直接查看统计和真实LLM评分回放。若自行导出纯源码或删去运行数据，页面会如实显示“未生成 / 不可用”，不会用固定数字冒充实验记录。启动脚本不隐式执行完整实验或生成报告。

LLM评分回放不等于现场再次调用模型；普通网页使用基础依赖即可，不要求下载大模型权重。

## 从源码初始化实验结果

以下命令用于主动重新计算结果，可能改写 `outputs/`。仓库已有冻结结果，普通展示无需重跑；完整重算前请复制项目，以免新结果与正式报告不一致。已有有效结果时初始化脚本会复用，明确重算需要添加 `--force`。

```powershell
# 实验一基准
python -B -X utf8 main.py lab01 benchmark

# 实验二：重新计算、绘图并做完整核验
python -B -X utf8 scripts/prepare_lab02.py

# 实验三：训练路径分类器、仅用验证集优化 GA、运行基准
# 不生成报告、PPT、录屏，也不调用真实 LLM
python -B -X utf8 scripts/prepare_lab03.py
```

实验三初始化后可运行：

```powershell
python -B -X utf8 main.py lab03 demo --method hybrid
```

本地真实 LLM 是可选的独立步骤，使用固定的 `Qwen/Qwen2.5-0.5B-Instruct`。
模型权重存于用户缓存，不存入项目源码。需要额外依赖、下载空间和推断时间：

```powershell
python -m pip install -r requirements-lab03.txt
python -B -X utf8 scripts/prepare_lab03_model.py
python -B -X utf8 scripts/prepare_lab03.py --llm
```

远程模式仅读取应用 `LAB03_LLM_*` 环境变量；不要将真实密钥提交到目录。
历史 before/after 比较、确认集冻结和旧录制核验依赖相应归档原始输入；
它们与“从源码重新运行当前算法”是不同任务。缺少归档时不编造历史比较。

## 测试与源码导出

```powershell
# 验证源码和已有实验证据；完整验证需要安装实验计算依赖
python -B -X utf8 -m unittest discover -s tests -t . -v

# 本仓库包含完整成果，不适用“纯源码目录”检查；该检查仅用于下面导出的目录

# 导出所有三次实验；目标必须是项目外、尚不存在的新目录
python -B -X utf8 scripts/export_source.py --destination C:\path\to\new-source
```

普通逻辑测试使用临时目录或明确标注的内存测试样例。
真实 LLM 重放和历史确认集测试保留，但没有原始证据时不声称通过；
恢复相应产物后重新启动测试进程，会重新启用这些核验。

前端安装依赖后可运行 `npm run test:routes`、`test:notation`、`test:api`、`test:lab03`。
浏览器检查还需要实际运行的网页与 Playwright 浏览器。

## 完整版本保存与课程提交边界

- `deliverables/`、`outputs/lab01/`、`outputs/lab02/`、`outputs/lab03/`、`outputs/lab03_materials/` 和 `web/dist/` 现已纳入版本管理。
- `.gitignore` 只排除不适合提交的安装依赖、缓存、凭证、本机 `.state/` 与 `outputs/runtime/` 等运行状态；不会再整体忽略课程成果。
- 代码交付目录中的原始 `.gitignore` 按冻结文件保留，相关运行数据已经显式加入Git；克隆仍会获得已跟踪文件。若以后往该目录新增原规则忽略的文件，应先核查再显式添加，不要误认为普通 `git add` 已包括全部内容。
- `outputs/lab03/runtime/` 是实验运行时优化的冻结证据，**不是**本机服务目录 `outputs/runtime/`，前者需要保留。
- 原始基线ZIP是证据核验的输入，内含历史材料也不能随意删改；它们不替代当前正式报告和视频。
- 制作源、日志和历史版本用于完整保存与追溯，不代表老师要求把它们全部作为课程附件提交。
- 纯源码导出功能保留：使用 `scripts/export_source.py` 到外部新目录，不要手工挪动或改写原始数据。
- 课程平台的项目创建、成员关联与管理员提交仍由小组自行完成，GitHub上传不能代替课程平台提交。

## 实验三材料位置（当前）

- 正式提交文件：`deliverables/lab3/`（30页报告、17页PPT、1分钟Demo及4分钟配音字幕演讲视频）。
- 制作与核验资料：`outputs/lab03_materials/`（图表、冻结证据、制作源和历史版本；不是另一套提交材料）。该目录随GitHub完整版本保存，使用方法见其中的 `README.md`。
- 网站运行数据仍在 `outputs/lab03/`，不依赖上述制作资料目录。
- 报告、图片及验收记录的历史快照不作为当前正式交付版本；当前核验入口是 `python -B -X utf8 outputs/lab03_materials/verify_materials.py`。
