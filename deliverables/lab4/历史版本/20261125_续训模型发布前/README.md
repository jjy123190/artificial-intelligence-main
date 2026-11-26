# 实验四当前完整材料

本目录对应已发布v2与graph/multiview/wide/wideview研究。当前较宽双视图55.83%/68.43%，只包括已锁定并独立核验结果；正在进行的续训候选不属于正式成绩。

- 当前整合报告：Markdown、可编辑Word、8页PDF，完整方法/公式/训练/全部比较/直接模块配对差/负结果/部署/限制/参考文献。
- 当前汇报：12页可编辑PPT，全部数据从已核验证据生成。
- 当前页面与真实推理Demo：约80秒、1600×900，实际浏览器记录，4次真实推理，无音轨。WebVTT/SRT为操作字幕。
- 当前版本PPT讲解：约5分28秒，配本机Microsoft Huihui标准合成中文语音，非真人录音、非声音克隆；不依赖上传材料到语音服务。WebVTT为句级近似时间字幕，并非逐字强制对齐。
- *_evidence.json：来源指纹、数值诊断、模型版本、实际演示任务、音视频/图像校验和。

旧v1报告、PPT和视频保留在上一级，历次补充保留在“优化补充”。不要将旧材料的100任务与当前600任务、32维与64/128维骨干混在一起解释。

网页入口：#/labs/lab04/materials。播放器不自动播放、不预先下载整段视频，提供预览与字幕；材料缺失时明确提示而不伪造下载成功。

重新制作入口在制作资料/实验四：build_integrated_report.py、record_current_demo.mjs、build_current_slides.py、render_current_slides.py、narrate_current_slides.py、build_current_captions.py。独立源码包不含本目录，可用python -m scripts.package_lab04_delivery一次打包源码与材料，保持源码/和deliverables/相邻。

课程平台组员关联、项目与管理员提交仍须实际核对；本目录不是提交成功证明。已完成部署范围为Windows CPU，不冒称指定手机/ARM硬件实测。


消融页面现提供模块直接对照／相对Baseline两种口径，报告与讲解同步更新。原型更新和均衡分配的组合变化不冒充单模块贡献，逐项区间未作多重比较校正。更新前实际录像与PPT另存`../历史版本/20261125_消融双口径更新前`，保留制作历史。
