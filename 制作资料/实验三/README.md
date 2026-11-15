# 实验三制作与核验资料

**用途：保留报告、PPT、演示材料的制作源、图表、冻结证据和核验记录。此处不是正式提交材料目录，也不是网站运行数据目录。**

- 本目录：`outputs/lab03_materials`
- 正式报告/PPT/后续Demo：[`deliverables/lab3`](../../deliverables/lab3)
- 网站实际读取的实验结果：`outputs/lab03`，与本目录相互独立，勿混淆。

## 文件分类

| 目录/文件 | 用途 |
| --- | --- |
| `evidence/` | 报告使用的冻结数据副本、搜索轨迹、统计与核验记录 |
| `figures/` | 高清PNG、可编辑SVG及完整搜索树 |
| `reproduce/` | 早期报告/图表制作工具；对应历史版本，不用于覆盖当前报告 |
| `可选拓展_20261113/` | 数学证明和规划的两个样例、可运行验证程序及历史增补记录 |
| `新版PPT_20261113/` | 18页PPT的可编辑制作源、预览及检查工具 |
| `图片与案例索引.html` | 本地资料预览入口，不是项目网站页面 |
| 本目录根部的报告DOCX/PDF/MD | 历史26页快照；正式30页报告只以 `deliverables/lab3` 为准 |
| `_maintenance/relocation_20261114/` | 本次目录调整的迁移清单、改动前副本和报告检查资料 |
| `verify_materials.py` | 当前资料与正式报告的只读检查入口 |

## 当前核验命令

在项目根目录运行：

```powershell
python -B -X utf8 outputs/lab03_materials/verify_materials.py
python -B -X utf8 outputs/lab03_materials/可选拓展_20261113/verify_examples.py
python -B -X utf8 outputs/lab03_materials/新版PPT_20261113/verify_ppt.py
```

前两项默认不写文件，PPT检查仅另存新的检查记录。检查不训练模型、不重建基准、不启动网站。报告检查需要 `python-docx` 和 `PyMuPDF`；PPT检查另需 `python-pptx`。

历史验收JSON和快照原样保留，里面的旧路径、旧文件哈希属于制作当时的记录，不表示还需保留旧目录。不要修改历史清单来掩盖版本变化；当前清单在 `_maintenance/relocation_20261114/current_manifest.json`。主动修改材料后出现指纹提示属于正常版本变化，需要重新审核。

## 是否提交或删除

老师要求代码、报告、搜索树样例和失败分析，没有要求整包提交本目录。报告已包含代表性图表与分析，是否另附高清图按实际提交安排决定。

本目录不被项目网站读取。将来不再需要制作/复核时可自行清理，但建议先把 `可选拓展_20261113/verify_examples.py` 及 `example_verification.json` 整理进正式代码包。不要连同 `outputs/lab03` 一起删除。
