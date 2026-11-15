# 实验三可选迁移拓展资料

正式报告：[Word](../../../deliverables/lab3/第31组_实验三_实验报告.docx) · [PDF](../../../deliverables/lab3/第31组_实验三_实验报告.pdf)。

报告第14章（第24—27页）包含数学证明与规划的迁移设计、两个逐步核验样例、指标与局限。本目录为制作与核验资料，不是另一套提交文件。

## 当前可运行入口

在项目根目录运行：

```powershell
python -B -X utf8 outputs/lab03_materials/可选拓展_20261113/verify_examples.py
python -B -X utf8 outputs/lab03_materials/verify_materials.py
```

- `verify_examples.py`：仅用Python标准库，默认只读；1份等式证书与1条四步积木计划通过，4个证明错误及4个规划错误被拒。
- `example_verification.json`：上述核验的完整中间状态与拒绝原因。
- `../verify_materials.py`：检查当前文件指纹、报告DOCX/PDF、嵌入图片与表格、目录引用，并重放这两个样例。

没有运行Lean等通用证明内核，没有实现完整跨领域搜索器；这两个样例不是泛化准确率实验。

## 历史制作记录

`before/`、`draft/`、`extension_content.json`、`第14章_可选迁移拓展.md`、`reference_checks.json`、`qa/` 和 `release_manifest.json` 保留增补时的内容与指纹。历史记录中出现旧目录名，是当时位置的记录，不是当前路径或另外的提交目录。

`build_extension.py`、`export_staged.py`、`publish_report.py`、`verify_report.py` 对应2026-11-13增补版本与当时基线；不要用它们覆盖当前报告，也不要把旧清单当作迁移后的验收。当前统一使用上面的 `verify_materials.py`，不为消除版本差异重写历史清单。

本次迁移只整理资料位置与正式报告中的路径说明，实验数值、30张表、14张图片、成员分工和拓展结论均保持不变。
