"""Create a captioned edit of the existing live capture without changing its timing.

Requires FFmpeg/ffprobe on PATH and Pillow. The original MP4 is retained.
"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/qa/lab01-recording/system_live_recording.mp4"
OUTPUT = ROOT / "outputs/qa/lab01-recording/system_live_recording_字幕版.mp4"
SRT = OUTPUT.with_suffix(".srt")
QA = ROOT / "outputs/qa/lab01-caption"
FPS = 15
# Boundaries were inspected in the original capture; units are source frames.
GRAPH_FRAME = 931
CUES = [
    (0, 57, "启动系统，分步演示企业股权关系分析与风险推理。"),
    (57, 113, "程序依次构建图谱、执行规则，并回写推理结果。"),
    (113, 188, "首先构建知识图谱，描述自然人、企业及各类关系。"),
    (188, 263, "本例包含16个实体、25条关系，涵盖持股、担保和交易。"),
    (263, 313, "图谱自动转换为一阶谓词，形成53条初始事实。"),
    (313, 361, "结构化事实为后续产生式规则提供推理输入。"),
    (361, 411, "企业档案采用框架表示，支持属性继承与缺省值。"),
    (411, 460, "推理前，实际控制人和风险信息保留默认状态。"),
    (460, 511, "15条规则通过正向链推理，将事实由53条扩充至135条。"),
    (511, 559, "系统输出实际控制关系、关联交易及风险预警线索。"),
    (559, 609, "针对关联交易待核查结论，系统展开完整推理解释。"),
    (609, 657, "沿规则编号逐层追溯，可查看结论依赖的关系和事实。"),
    (657, 706, "股权穿透按路径计算持股比例，再汇总多条路径。"),
    (706, 755, "恒通科技示例中，两条路径合计持股比例为34.20%。"),
    (755, 805, "图算法继续识别循环持股、担保圈和企业关联路径。"),
    (805, 854, "这些连接关系用于揭示单一企业档案中不易发现的线索。"),
    (854, 893, "随后将实际控制人、风险等级和风险提示回写企业框架。"),
    (893, 931, "图谱新增9条控制关系，并为6家企业写入风险标签。"),
    (931, 982, "最后展示推理增强后的知识图谱，集中呈现关系与风险。"),
    (982, 1033, "推理结果与原始关系共同展示，便于核查和追溯。"),
    (1033, 1084, "风险输出用于辅助分析与核查，不等同于法律认定。"),
]


def run(args, **kwargs):
    return subprocess.run(args, check=True, capture_output=True, **kwargs)


def probe(path):
    return json.loads(run(["ffprobe", "-v", "error", "-show_format", "-show_streams",
                           "-of", "json", str(path)]).stdout)


def timestamp(frame, ass=False):
    rate = 100 if ass else 1000
    ticks = round(frame * rate / FPS)
    seconds, fraction = divmod(ticks, rate)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if ass:
        return f"{hours}:{minutes:02}:{seconds:02}.{fraction:02}"
    return f"{hours:02}:{minutes:02}:{seconds:02},{fraction:03}"


def main():
    QA.mkdir(parents=True, exist_ok=True)
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("FFmpeg and ffprobe must be on PATH")
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    source_probe = probe(SOURCE)
    stream = next(s for s in source_probe["streams"] if s["codec_type"] == "video")
    if (stream["width"], stream["height"], stream["avg_frame_rate"], stream["nb_frames"]) != (1280, 720, "15/1", "1084"):
        raise ValueError("Source capture changed; inspect its timing before rebuilding captions")
    for index, (start, end, text) in enumerate(CUES):
        assert start < end and (index == 0 or start == CUES[index-1][1])
        assert "虚拟教学数据" not in text and "虚构教学数据" not in text

    # Replace only the static status row. The recorded program output is untouched.
    patch = Image.new("RGB", (1280, 34), (16, 26, 44))
    draw = ImageDraw.Draw(patch)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 16)
    draw.text((640, 17), "独立 Python 进程 · 自动分步操作", anchor="mm", font=font, fill="#a7b5ca")
    patch.save(QA / "header-clean.png")

    SRT.write_text("\n\n".join(f"{i}\n{timestamp(a)} --> {timestamp(b)}\n{text}"
                               for i, (a, b, text) in enumerate(CUES, 1)) + "\n", encoding="utf-8-sig")
    ass = """[Script Info]
ScriptType: v4.00+
PlayResX: 1600
PlayResY: 900
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,Microsoft YaHei,32,&H00F4F6F8,&H00FFFFFF,&H001F1309,&H001F1309,0,0,0,0,100,100,0,0,1,0.5,0,2,80,80,25,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ass += "\n".join(f"Dialogue: 0,{timestamp(a, True)},{timestamp(b, True)},Caption,,0,0,0,,{text}"
                     for a, b, text in CUES) + "\n"
    (QA / "captions.ass").write_text(ass, encoding="utf-8-sig")
    filters = (
        f"[0:v][1:v]overlay=0:64:enable='lt(n,{GRAPH_FRAME})':eof_action=repeat[clean];"
        "[clean]scale=1440:810:flags=lanczos,pad=1600:900:80:0:color=0x101a2c,"
        "drawbox=x=0:y=810:w=1600:h=90:color=0x09131f:t=fill,"
        "drawbox=x=80:y=810:w=1440:h=2:color=0x315b61:t=fill,"
        "subtitles=captions.ass,setsar=1,format=yuv420p[out]"
    )
    print("Rendering captioned video...", flush=True)
    rendered = run([
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
        "-i", str(SOURCE), "-i", "header-clean.png", "-filter_complex", filters,
        "-map", "[out]", "-map", "0:a?", "-c:v", "libx264", "-preset", "medium",
        "-crf", "18", "-r", "15", "-c:a", "copy", "-movflags", "+faststart",
        "-metadata", "title=企业股权知识表示与风险推理｜系统运行字幕版",
        str(OUTPUT),
    ], cwd=QA)
    (QA / "render.log").write_bytes(rendered.stderr)
    print("Checking full-video decoding and output timing...", flush=True)
    run(["ffmpeg", "-v", "error", "-xerror", "-i", str(OUTPUT), "-f", "null", "-"])
    result = probe(OUTPUT)
    output_stream = next(s for s in result["streams"] if s["codec_type"] == "video")
    assert output_stream["nb_frames"] == stream["nb_frames"]
    assert abs(float(result["format"]["duration"]) - float(source_probe["format"]["duration"])) < 0.01
    assert source_hash == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    report = {
        "source": str(SOURCE), "source_sha256": source_hash,
        "output": str(OUTPUT), "output_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "subtitle_file": str(SRT), "subtitle_count": len(CUES),
        "source_preserved": True, "full_decode_passed": True,
        "edit": "Existing live capture: replace the initial status row, retain program output, add burnt-in Chinese captions in a separate bottom band.",
        "new_live_capture": False, "duration_seconds": float(result["format"]["duration"]),
        "resolution": [output_stream["width"], output_stream["height"]],
        "frames": int(output_stream["nb_frames"]),
    }
    (QA / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
