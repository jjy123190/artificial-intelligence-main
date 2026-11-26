# 实验四材料版本

本目录原有 `build_report.py`、`build_slides.mjs`、`record_demo.mjs`、`build_videos.py` 描述v1（32维、100任务）的历史交付。已有报告、PPT、90秒Demo和3分钟讲解保留原样。

v2优化改变了模型、评测协议与网页布局，不能把v2数据直接套进v1讲稿。报告/PPT生成器默认优先使用 `源码/outputs/lab04/history/v1`；v1录制脚本仅供重现旧界面，不作为新UI测试工具。

v2补充报告由 `源码/experiments/lab04/study_report.py` 从独立校验通过的逐任务记录生成Markdown、DOCX、PDF。新UI回归脚本是 `源码/web/scripts/check-lab04-browser.mjs`。旧视频明确属于原版实验，不代表优化后分数或界面。
