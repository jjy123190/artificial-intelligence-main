# 实验三PPT制作资料

正式文件：[第31组_实验三_汇报.pptx](../../../deliverables/lab3/第31组_实验三_汇报.pptx)。

18页、16:9，第1—16页主讲，第17—18页答辩备用，逐页讲解提示在备注中。文字、图表、流程图和表格为原生可编辑对象；仅网站入口使用截图。

本目录不是另一个提交目录。2026-11-14随制作资料迁移到 `outputs/lab03_materials`，正式PPT未被改写。

## 制作与检查

- `build_ppt.py`：在本目录 `draft/` 生成PPT，不改写正式PPT；已改为相对资料目录读取证据及拓展样例。
- `export_ppt.py`：通过PowerPoint导出 `draft/` 预览。
- `verify_ppt.py`：核对18页、备注、4张可编辑图表、真实数据、现有Office预览与当前资料清单；新结果写入 `qa/ppt_validation_relocated.json`，不覆盖历史验收。
- `source/`：制作当时的报告和内容快照。当前正式报告仅更新了资料路径/说明，不回写历史快照。
- `qa/`：逐页预览、缩略图、Office文字边界及历史核验记录。

在项目根目录运行：

```powershell
python -B -X utf8 outputs/lab03_materials/新版PPT_20261113/verify_ppt.py
```

原 `release_manifest.json` 和 `qa/ppt_validation.json` 保留制作时的SHA-256与时间，不重新封存。迁移前脚本版本在 `../_maintenance/relocation_20261114/before_files/`；当前目录核验入口为 `../verify_materials.py`。
