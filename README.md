# 人工智能课程项目 · 源码工作区

GitHub 仓库维护实验一、实验二、实验三的源码、测试、依赖配置、运行脚本和必要输入数据。
本地工作区还可包含报告、PPT、视频、运行结果和制作资料；这些生成文件按 `.gitignore` 排除，不随源码提交。
`README.md` 是运行说明，不是实验报告。`.git/` 仅用于版本管理，不参与源码交付。

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

建议 Python 3.11+；首次构建网页需要 Node.js 18+。

```powershell
python -m pip install -r requirements.txt
python -B -X utf8 main.py lab02 demo
python -B -X utf8 main.py lab03 demo
```

实验三默认 demo 明确使用 `astar_fixed`，不冒充已经训练的混合模型。
基础 BFS、下界 A*、固定 A*、MCTS 不依赖历史实验结果。

构建并运行网页：

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

**首次运行时没有历史结果是正常状态。** 页面会显示“未生成 / 不可用”及操作说明，
不会用零、固定参数或中性分数冒充真实实验记录。启动脚本不隐式执行完整实验或生成报告。

## 从源码初始化实验结果

以下命令仅在主动运行时创建 `outputs/`；纯源码整理完成时不预先保留它。

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
# 不生成实验结果；缺少历史实测文件的核验会明确 skipped
python -B -X utf8 -m unittest discover -s tests -t . -v

# 检查目录只有列入白名单的源码/输入/配置；不写任何结果
python -B -X utf8 scripts/check_source.py

# 导出所有三次实验；目标必须是项目外、尚不存在的新目录
python -B -X utf8 scripts/export_source.py --destination C:\path\to\new-source
```

普通逻辑测试使用临时目录或明确标注的内存测试样例。
真实 LLM 重放和历史确认集测试保留，但没有原始证据时不声称通过；
恢复相应产物后重新启动测试进程，会重新启用这些核验。

前端安装依赖后可运行 `npm run test:routes`、`test:notation`、`test:api`、`test:lab03`。
浏览器检查还需要实际运行的网页与 Playwright 浏览器。

## 生成文件与提交边界

主动安装/构建/实验会产生 `web/node_modules/`、`web/dist/`、`outputs/`、虚拟环境和运行状态；
这些目录已在 `.gitignore` 中排除，**不是纯源码的一部分**。
导出源码时使用 `scripts/export_source.py`，不要把生成结果改名放入 `data/`。

GitHub 源码与可运行课程提交目录用途不同：

- GitHub 不跟踪 `outputs/`、`deliverables/`、`web/dist/` 或本机运行状态；已有本地文件不因取消跟踪而被删除。
- 全新克隆可按前述命令构建前端并生成当前实验结果。没有历史结果时，网页会如实显示缺项。
- 课程提交目录可额外携带已构建前端和经核验的运行数据，便于直接查看现有实验、GA/路径学习结果和真实 LLM 评分回放。
- 回放不等于现场调用模型；实时模型推断另需安装可选依赖并准备固定模型权重。
- 历史实验核验存在相互校验的基线归档，恢复时应保留完整依赖，不改写记录指纹，也不以重跑结果冒充原报告数据。
- 课程平台的项目创建、成员关联与管理员正式提交仍由小组自行确认，GitHub 推送不能代替课程提交。

## 实验三材料位置（当前）

- 正式提交文件：`deliverables/lab3/`（30页报告、17页PPT、1分钟Demo及4分钟配音字幕演讲视频）。
- 制作与核验资料：`outputs/lab03_materials/`（图表、冻结证据、制作源和历史版本；不是另一套提交材料）。该目录仅在本地保存，使用方法见其中的 `README.md`。
- 网站运行数据仍在 `outputs/lab03/`，不依赖上述制作资料目录。
- 报告、图片及验收记录的历史快照不作为当前正式交付版本；当前核验入口是 `python -B -X utf8 outputs/lab03_materials/verify_materials.py`。
