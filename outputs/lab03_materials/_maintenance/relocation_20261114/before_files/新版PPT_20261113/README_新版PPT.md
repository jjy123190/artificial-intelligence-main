# 实验三新版PPT

正式文件：`../../lab3/第31组_实验三_汇报.pptx`。

内容依据当前30页实验报告，18页、16:9。第1—16页主讲，第17—18页为四项指标和成员职责答辩备用。逐页讲解提示已写入PowerPoint的“备注”，可在演示者视图查看。字体使用本机常见的Microsoft YaHei、Segoe UI和Consolas。

覆盖状态建模、BFS/A*/MCTS、GA真实验证曲线、路径分类器和专家规则、实验协议与对比、搜索轨迹和失败分析、LLM真实配对、迁移方案及两个核验样例。图表为可编辑的原生图表，内嵌Excel数据；流程图、文字和表格也是原生对象。

唯一截图为本次从当前工作区网站抓取的实验三入口，已核对workspace_id和网页构建文件。PPT没有采用此前版本的PPT或Demo录屏。

## 核对与预览

- `draft/第31组_实验三_汇报.pptx`：制作完成的暂存副本。
- `draft/第31组_实验三_汇报.pdf`：通过本机PowerPoint导出的18页预览。
- `qa/slide_01.png`至`slide_18.png`：逐页实际渲染预览。
- `qa/contact_01.png`、`contact_02.png`：全套缩略图。
- `qa/ppt_validation.json`：PPTX结构、图表数值、原始CSV、PDF文字一致性及Office文字边界核对。
- `qa/visual_review.json`：最终版人工视觉检查范围。
- `source/deck_spec.json`、`outline.md`：逐页内容、来源、讲解提示和设计说明。

生成脚本`build_ppt.py`只生成本目录暂存PPT。`export_ppt.py`只导出暂存PPT，`verify_ppt.py`检查暂存稿及报告来源。本轮没有执行训练、模型推断或实验基准。

## 制作期间的报告保存

制作期间检测到正式报告DOCX被独立保存，两个文字变化为封面班级行、封面附件说明。正文实验数值、30张表和14张图均未变化。制作源已采用最新保存的DOCX，最初快照另存为`source/第31组_实验三_实验报告-制作开始快照.docx`，差异记录见`qa/user_report_saved_during_build.json`。本轮PPT脚本没有改写正式报告或其PDF。

正式目录仅增加PPTX；所有制作源、版本副本、PDF预览和核验记录放在本备份子目录。
