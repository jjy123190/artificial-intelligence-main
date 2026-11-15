# 实验三PPT演讲视频：最新PPT重录版

- 制作日期：2026年11月14日。
- 唯一页面来源：`C:\Users\Administrator\Desktop\artificial-intelligence-main\deliverables\lab3\第4组_企业股权知识图谱多跳智能设计.pptx`。
- 源PPT保存时间：`2026-11-14T20:38:50.932099+08:00`。
- 源PPT SHA256：`dad564d9403346f9ad6861824fb3d4e04c3dd31bebd795175e3cc5451e88505f`。
- 正式成片目标：`C:\Users\Administrator\Desktop\artificial-intelligence-main\deliverables\lab3\第4组_实验三_PPT演讲.mp4`。

## 本轮范围

只重新制作PPT演讲视频。PPT原文件、Demo、实验报告、项目源码和提交源码目录不修改；不创建、不恢复外层源码ZIP。

使用Microsoft PowerPoint以只读方式打开源PPT的字节一致快照，重新导出全部17页，包含用户修改后的封面、总结页及结束页。没有使用旧页面截图，也没有在旧视频上覆盖局部区域。

逐页核对新版内容后，原已校对中文配音仍与各页内容对应，因此复用原合成配音音轨与词级时间信息，不假称本轮重新合成声音。重新生成完整时间轴、逐句字幕并与最新17页画面合成。

## 成片规格

- 时长240秒（4分00秒），30fps，共7200帧。
- 2560×1440，16:9；H.264视频、AAC普通话配音。
- 85条中文字幕直接烧录在PPT画面底部，深浅页使用对应高对比字色。
- 字幕不另加黑边或字幕条；逐页检查不遮住正文、姓名或页码。
- 保留17页全部内容，最后5秒显示最新结束页。

## 本目录用途

这是制作和核验工作区，不是另一套必须提交的材料。直接提交 `deliverables/lab3/第4组_实验三_PPT演讲.mp4` 即可；播放不依赖本目录里的字幕或音频文件。

- `source/`：最新PPT快照、逐页提取文本，以及替换前的演讲视频备份 `previous-ppt-video.mp4`。
- `slides/`：本轮原生PowerPoint导出的17页高清图及渲染记录。
- `speech/`：经逐页核对后复用的中文配音、词级时间戳、240秒配音母带。
- `timeline.json`、`subtitles.*`、`演讲稿.md`：新成片逐页时间轴与字幕文本。
- `preview/`：字幕预览；`qa/`：布局、完整解码、页面匹配、音画同步及保护文件检查。
- `delivery.json`：只有核验通过并替换正式视频后才生成。

原演讲视频备份SHA256：`d848576083b9555ab15c67915f15055f91228425604990b3d24470a5991f4970`。

## 制作复核

`render_latest.py` 只读导出源PPT；`prepare_video.py` 组装时间轴；`check_subtitle_layout.py` 检查字幕布局；`encode_video.py` 合成；`verify_video.py` 检查成片；`publish_video.py` 在核验通过且用户源文件未被改动时替换正式演讲视频。

这些脚本会拒绝覆盖已有渲染成片，不要在此工作区重复执行来覆盖历史记录。需要再次录制时新建工作区并重新读取用户最新PPT。
