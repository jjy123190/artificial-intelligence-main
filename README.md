# 企业股权知识表示与风险推理

包含实验一、实验二、实验三的完整算法、后端、网页、测试及辅助工具。各次最终材料分别位于 `deliverables/lab01`、`deliverables/lab02`、`deliverables/lab03`。

## 运行网站

- 双击根目录 **打开网站.cmd**，在页面中选择实验一、实验二或实验三。
- 也可双击 `experiments/lab01/打开网站.cmd`、`experiments/lab02/打开网站.cmd`、`experiments/lab03/打开网站.cmd` 直达对应实验。
- 关闭服务：双击 **停止网站.cmd**。
- 默认网址：`http://127.0.0.1:8765`；端口被占用时启动脚本自动选择后续可用端口。

已保留 `web/dist` 构建文件，普通运行不依赖 `web/node_modules`。Python要求3.11及以上；缺少依赖时启动脚本会提示或自动准备。

```powershell
python -m pip install -r requirements.txt
python main.py web --port 8765
```

## 实验一、二最终材料（9份，保留原件）

### 实验一：`deliverables/lab01`

- 企业股权知识表示与风险推理.pptx
- 课堂报告.docx
- 实验一_PPT汇报.mp4
- 实验一_Demo.mp4

### 实验二：`deliverables/lab02`

- 第31组_创新点说明.docx
- 第31组_人工智能推理实验报告.docx
- 推理实验汇报.pptx
- 实验二_PPT汇报.mp4
- 实验二_Demo.mp4

下载目录中的指定文件已复制到上述目录，保持原文件字节内容；下载目录原件未移动或删除。

## 代码与必要数据

| 路径 | 内容 |
| --- | --- |
| `main.py`、根目录启动脚本 | 统一命令行与网站入口 |
| `core/`、`backend/` | 共享知识表示、推理能力与API |
| `experiments/lab01/` | 图谱、产生式、框架、穿透查询等完整实现 |
| `experiments/lab02/` | 确定性、不确定性、混合推理、优化与评估实现 |
| `experiments/lab03/` | BFS / A* / MCTS、GA、路径学习、LLM辅助与评估 |
| `web/src/`、`web/public/`、`web/dist/` | 网页源码、资源与可直接运行的构建结果 |
| `tests/`、`scripts/` | 测试、运行、打包和录制等工具代码 |
| `scripts/retained_tools/` | 从临时目录中提取并去重的辅助/历史实现；只保留源码，不保留录屏帧和核验报告 |
| `data/` | 教学CSV、输入样例和数据指纹 |
| `outputs/lab01/benchmark.*` | 实验一原始查询基准，网页读取使用 |
| `outputs/lab02/` | 网站读取的模型、数值结果、图表及其封存配套文件；不是中间核验目录 |
| `requirements*.txt`、`web/package*.json` | Python/前端依赖与锁定配置 |

旧版材料已清理；`outputs/qa` 是可删除的临时验收/录制目录，`outputs/submissions` 用于导出完整提交包。运行测试、重新录制、重新打包会再次生成相应产物；`outputs/runtime` 和 `.state` 由网站运行时自动生成，不需要提交。

## 测试、复现与开发

```powershell
# 不在项目中写入核验报告的单元测试
python -B -X utf8 -m unittest discover -s tests -t .

# 实验与结果生成
python -B -X utf8 main.py lab01
python -B -X utf8 main.py lab02 demo
python -B -X utf8 scripts/prepare_lab02.py --check
python -B -X utf8 scripts/prepare_lab02.py --force
```

保存的实验结果保留原始运行指纹，不能把整理文件视为重新实验；源码变更后需要重新生成结果，才能通过严格源码指纹回放核验。

修改前端后需安装开发依赖并构建：

```powershell
cd web
npm ci
npm run build
```

`retained_tools` 中的历史辅助代码按原内容保留，可能引用已按要求删除的旧材料或运行产物；它们不参与网站启动。如需复用，需另行提供输入或调整输出路径。已有核验/录制脚本同样可能重新生成临时材料。

## 使用边界

所有实体、数据和风险标签均为教学构造。控制关系、经济权益、风险评分与复核建议含义不同，不构成法律或监管认定。实验一、实验二各自使用独立方法版本，不用实验二结果替换实验一。


## 实验三：多跳路径搜索与智能优化

已接入实验三，双击 `experiments/lab03/打开网站.cmd`，或在统一首页选择实验三。

### 实验三工作台更新

- 石墨黑图形工作台：窄工具栏、中心搜索画布、右侧参数检查器和底部回放时间轴。桌面分隔条支持拖拽或方向键调宽；手机切换为纵向布局，窄平板保留紧凑双栏。
- 2 / 3 / 4 跳示例快捷选择，直接展示起点、关系序列、终点属性，以及状态、动作、转移与终止条件。问题配置不提供答案或标准路径。
- 搜索事件时间线支持跳转；搜索树节点支持鼠标、Enter / 空格回看对应事件，空白画布支持拖拽平移、缩放和定位当前节点。橙色标记返回分支，节点显示深度与启发评分；完整名称与过程可在步骤列表中阅读。
- 同题对比 BFS、固定 A*、GA A*、MCTS，使用同一问题与扩展预算依次调用真实搜索接口。总工作量包含 MCTS 模拟步数，避免将扩展节点数误认为相同计算成本。单题结果用于演示，整体性能结论仍以独立测试集统计为准。
- 对比可取消、重试和导出；切换问题、预算或离开模块会丢弃过期对比响应。取消仅停止网页等待及后续调用，正在执行的后端任务可能仍在结束。
- “对比实验与失败分析”可按场景、预算导出当前指标与配对收益区间。导出格式为 JSON，保留原始数值；缺失值和失败结果不以零替代。

布局参考 [21st.dev](https://21st.dev/) 上的 [Interactive Workflow Blueprint](https://21st.dev/@automate_flows/components/interactive-workflow-blueprint)、[Rail Panel Sidebar](https://21st.dev/@dhxwhy/components/rail-panel-sidebar) 和 [Resizable](https://21st.dev/@wensity/components/resizable)：借鉴节点画布、窄工具轨道和可调整分区，在本项目中独立实现。配色采用石墨黑与橙色，圆角与留白按专业图形工具处理；没有复制组件源码或引入额外运行依赖。

本次界面更新沿用仓库中的基准与 LLM 实测记录，没有重新训练或伪造新基准结果；远程 LLM 仍需服务端配置，默认重放已有评分。

新增交互回归（先启动本地服务，可直接使用已安装的 Edge）：

```powershell
cd web
$env:PLAYWRIGHT_CHANNEL = "msedge"
$env:LAB03_TEST_URL = "http://127.0.0.1:8766"
npm run test:browser
npm run test:workbench
```

需要下载依赖时可使用本地代理，端口按实际配置调整；以下只影响当前终端：

```powershell
$env:HTTP_PROXY = "http://127.0.0.1:7890"
$env:HTTPS_PROXY = "http://127.0.0.1:7890"
$env:NO_PROXY = "localhost,127.0.0.1"
python -m pip install --proxy http://127.0.0.1:7890 -r requirements.txt
# 在 web 目录运行
npm ci --proxy=http://127.0.0.1:7890 --https-proxy=http://127.0.0.1:7890
```

### 算法与已有实验数据

- 四个模块：多跳搜索工作台、遗传优化与路径学习、LLM辅助与证据约束、对比实验与失败分析。
- 实现：`experiments/lab03/`；数据：`data/lab03/`；真实运行结果：`outputs/lab03/`；报告和PPT：`deliverables/lab03/`。
- 搜索：BFS、A*、UCT/MCTS；遗传算法只在验证图选择启发式参数；逻辑回归只使用训练图路径前缀。
- 基准：3种子、45张互不重叠图、405道2—4跳题、3,888次算法运行。
- LLM：真实Qwen2.5-0.5B-Instruct小模型，9题、72条候选边双任务评估。网页默认重放实测评分，明确标注本次没有再次调用模型。
- 模型解释存在失真风险，界面标注实体核对结果；答案始终由图中的有效证据路径产生。

```powershell
python -B -X utf8 main.py lab03 demo
python -B -X utf8 main.py lab03 benchmark
python -m pip install -r requirements-lab03.txt
python -B -X utf8 scripts/prepare_lab03_model.py
python -B -X utf8 main.py lab03 llm
python -B -X utf8 main.py lab03 report
```

网页展示和实测重放不需要下载模型；重跑LLM才需要约0.99GB模型，下载到用户缓存。远程Responses接口可用 `LAB03_LLM_BASE_URL`、`LAB03_LLM_API_KEY`、`LAB03_LLM_MODEL` 配置，不读取或导出开发工具凭据。网页实际处理时间包括服务开销，实验表的核心搜索计时不包含模型加载或LLM前向推断。

GA/学习启发式不保证可采纳性；MCTS需计入额外模拟；所有扩展标签均为教学构造。评分建议和要求映射见 `deliverables/lab03/实验三_实施计划与验收.md`，不能将建议自评当成教师成绩。

## 实验三交互与可靠性检查

- 搜索完成后可逐步前进、后退、调整速度；搜索树可缩放、定位当前节点，也可切换为完整名称的步骤列表。
- 预算耗尽、候选穷尽、证据不合格是三种不同状态。无专家消融可能返回无效路径，界面不会把它当成答案。
- GA/路径模型缺失、参数无效或种子不匹配时明确报错，不会用默认值冒充已训练模型。BFS、固定A*、下界A*和MCTS不依赖这些可选模型。
- LLM实测重放须匹配当前图和问题；尚未配置远程接口时，页面会在运行前提示。实时远程模式只评分起点前6条边，离线实验为前两层最多8条边，二者不作直接性能比较。
- “停止等待”取消网页等待和旧响应写回；后台隔离任务可能仍在结束，不能将其视为撤销已发出的远程请求。
- 图与学习结果是教学数据，不是对真实企业的法律判断。重跑基准后应重新生成报告并核对提交材料。

前端回归（位于 `web/`）：

```powershell
npm ci
npm run build
npm run test:routes
npm run test:notation
npm run test:api
npm run test:lab03
# 网站服务运行后，安装浏览器并执行完整页面/无障碍/竞态检查
npx playwright install chromium
npm run test:browser
# 已安装 Chrome 时可使用：$env:PLAYWRIGHT_CHANNEL='chrome'
# 自定义测试网址：$env:LAB03_TEST_URL='http://127.0.0.1:8766'
```

浏览器检查覆盖四个实验三模块的手机、平板、笔记本布局，错误重试、停止等待、取消后的旧响应隔离、模型缺失状态与键盘操作。验收截图和新版视频的临时录制帧写入系统临时目录，不进入提交包。依赖目录不是运行网站所必需，修改网页后需重新构建 `web/dist`。
