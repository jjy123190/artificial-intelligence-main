# 企业股权知识表示与风险推理

包含实验一、实验二的完整算法、后端、网页、测试及辅助工具。最终提交材料仅保留在 `deliverables/lab01`、`deliverables/lab02`。

## 运行网站

- 双击根目录 **打开网站.cmd**，在页面中选择实验一或实验二。
- 也可双击 `experiments/lab01/打开网站.cmd`、`experiments/lab02/打开网站.cmd` 直达对应实验。
- 关闭服务：双击 **停止网站.cmd**。
- 默认网址：`http://127.0.0.1:8765`；端口被占用时启动脚本自动选择后续可用端口。

已保留 `web/dist` 构建文件，普通运行不依赖 `web/node_modules`。Python要求3.11及以上；缺少依赖时启动脚本会提示或自动准备。

```powershell
python -m pip install -r requirements.txt
python main.py web --port 8765
```

## 最终提交材料（9份）

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
| `web/src/`、`web/public/`、`web/dist/` | 网页源码、资源与可直接运行的构建结果 |
| `tests/`、`scripts/` | 测试、运行、打包和录制等工具代码 |
| `scripts/retained_tools/` | 从临时目录中提取并去重的辅助/历史实现；只保留源码，不保留录屏帧和核验报告 |
| `data/` | 教学CSV、输入样例和数据指纹 |
| `outputs/lab01/benchmark.*` | 实验一原始查询基准，网页读取使用 |
| `outputs/lab02/` | 网站读取的模型、数值结果、图表及其封存配套文件；不是中间核验目录 |
| `requirements*.txt`、`web/package*.json` | Python/前端依赖与锁定配置 |

`outputs/qa`、`outputs/submissions`、缓存和旧版材料已清理。运行测试、重新录制、重新打包会再次生成相应产物；`outputs/runtime` 和 `.state` 由网站运行时自动生成，不需要提交。

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
