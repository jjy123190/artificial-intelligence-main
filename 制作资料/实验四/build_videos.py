"""Narrate actual browser recordings and verified PowerPoint renders.

Uses Microsoft Edge's standard Mandarin voice, not a cloned personal voice.
Temporary narration and renders belong in the caller's --tmp directory.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import wave

import edge_tts
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
ROOT = Path(__file__).resolve().parents[2]
VOICE = "zh-CN-YunxiNeural"


def ffmpeg(args, cwd=None):
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *map(str, args)],
                   cwd=cwd, check=True)


def wav_duration(file):
    with wave.open(str(file), "rb") as source:
        return source.getnframes() / source.getframerate()


async def voice(text, file, proxy):
    key = hashlib.sha256((VOICE + "+5%" + text).encode()).hexdigest()
    marker = file.with_suffix(".sha256")
    if not file.is_file() or not marker.is_file() or marker.read_text() != key:
        await edge_tts.Communicate(text, VOICE, rate="+5%", proxy=proxy or None).save(str(file))
        marker.write_text(key, encoding="ascii")
    wav = file.with_suffix(".wav")
    ffmpeg(["-i", file, "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", wav])
    return wav, wav_duration(wav)


def phrases(text, limit=27):
    result = []
    for piece in re.split(r"(?<=[，。；：])", text):
        piece = piece.strip()
        while len(piece) > limit:
            at = piece.rfind("，", 9, limit + 1)
            if at < 9:
                at = limit
            else:
                at += 1
            result.append(piece[:at])
            piece = piece[at:]
        if piece:
            result.append(piece)
    return result


def ass_time(seconds):
    centiseconds = round(seconds * 100)
    minutes, cs = divmod(centiseconds, 6000)
    hours, minutes = divmod(minutes, 60)
    sec, cs = divmod(cs, 100)
    return f"{hours}:{minutes:02d}:{sec:02d}.{cs:02d}"


def subtitle_file(file, cues):
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Microsoft YaHei,34,&H00FFFFFF,&H00FFFFFF,&H80302A24,&H80302A24,0,0,0,0,100,100,0,0,3,8,0,2,90,90,20,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    for cue in cues:
        text = cue["text"].replace("\\", " ").replace("{", "(").replace("}", ")").replace("\n", "\\N")
        lines.append(f"Dialogue: 0,{ass_time(cue['start'])},{ass_time(cue['end'])},Default,,0,0,0,,{text}")
    file.write_text(header + "\n".join(lines) + "\n", encoding="utf-8-sig")


def add_cues(cues, text, start, spoken_seconds):
    parts = phrases(text)
    total = sum(len(p) for p in parts)
    cursor = start
    for part in parts:
        end = cursor + spoken_seconds * len(part) / total
        cues.append({"text": part, "start": cursor, "end": end})
        cursor = end


def audio_segment(source, output, slot, tempo=1.0, lead=0.3):
    ffmpeg(["-i", source, "-af", f"atempo={tempo:.8f},asetpts=PTS-STARTPTS,adelay={round(lead * 1000)},apad,atrim=duration={slot:.8f},aresample=async=1:first_pts=0",
            "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", output])
    if abs(wav_duration(output) - slot) > 1 / 48000 + 1e-6:
        raise ValueError(f"Narration segment duration does not match its slot: {output.name}")


def concatenate_audio(files, output, work):
    listing = work / "audio-concat.txt"
    listing.write_text("\n".join(f"file '{file.name}'" for file in files), encoding="utf-8")
    ffmpeg(["-f", "concat", "-safe", "0", "-i", listing.name, "-c:a", "pcm_s16le", output], cwd=work)


async def build_demo(temporary, proxy):
    take = json.loads((temporary / "demo_take.json").read_text(encoding="utf-8"))
    work = temporary / "demo-audio"
    work.mkdir(parents=True, exist_ok=True)
    script = [
        (0, 9, "实验四已接入统一平台。下面用真实模型完成小样本图像分类。"),
        (9, 21, "先运行普通基线，每类只给一张支持图片，并固定随机种子。每次点击都会实际执行图像编码和分类。"),
        (21, 33, "这里是五个类别的支持图片。查询图片与支持图片分开，页面显示预测类别、真实类别和模型分数。"),
        (33, 44, "保持同一任务，切换完整特征校准。它借用基础类统计，联合无标签查询图片更新原型。"),
        (44, 53, "两幅图使用相同投影。此时特征点不移动，变化来自类别原型和距离度量。"),
        (53, 64, "接着运行五样本MAML。模型用支持集适应全网络和分类头，再计算查询图片的预测。"),
        (64, 74, "多任务评测比较三种方法，报告均值和置信区间。计时口径分别标注，不能直接排速度名次。"),
        (74, 83, "五样本完整方案有所下降。消融中，方差校准是当前的薄弱环节。"),
        (83, 90, "最后核对类别隔离和实验边界。所有结果均可用固定种子复现。"),
    ]
    files, cues = [], []
    for index, (start, end, text) in enumerate(script):
        source, duration = await voice(text, work / f"voice-{index:02d}.mp3", proxy)
        tempo = max(1.0, duration / (end - start - 0.5))
        if tempo > 1.25:
            raise ValueError(f"Demo narration {index} too long: {duration:.2f}s. Shorten it instead of clipping speech.")
        output = work / f"part-{index:02d}.wav"
        audio_segment(source, output, end - start, tempo, lead=0.25)
        files.append(output)
        add_cues(cues, text, start + 0.25, duration / tempo)
        print(f"Demo narration {index + 1}/{len(script)}", flush=True)
    joined = work / "narration.wav"
    concatenate_audio(files, joined, work)
    subtitle_file(work / "captions.ass", cues)
    destination = ROOT / "deliverables/lab4/实验四_Demo.mp4"
    destination.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg(["-ss", take["offset_seconds"], "-i", take["raw"], "-i", joined,
            "-vf", "scale=1760:990:flags=lanczos,pad=1920:1080:80:0:color=0xF8F3E9,subtitles=captions.ass",
            "-t", "90", "-r", "30", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
            "-map", "0:v:0", "-map", "1:a:0", "-movflags", "+faststart", destination], cwd=work)
    (temporary / "demo_timeline.json").write_text(json.dumps({"duration_seconds": 90, "voice": VOICE, "cues": cues}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(destination, flush=True)


async def build_presentation(temporary, proxy):
    # This mode is deliberately run only after the user-requested slide QA gate.
    content = json.loads((temporary / "ppt_narration.json").read_text(encoding="utf-8"))
    work = temporary / "ppt-audio"
    work.mkdir(parents=True, exist_ok=True)
    sources = []
    for item in content["slides"]:
        source, duration = await voice(item["text"], work / f"voice-{item['slide']:02d}.mp3", proxy)
        sources.append((item, source, duration))
        print(f"PPT narration {item['slide']}/{len(content['slides'])}", flush=True)
    total_speech = sum(d for _, _, d in sources)
    tempo = max(1.0, total_speech / 169.0)
    if tempo > 1.20:
        raise ValueError(f"Presentation narration too long ({total_speech:.2f}s). Shorten the manuscript.")
    padding = (180.0 - total_speech / tempo) / len(sources)
    files, cues, schedule = [], [], []
    cursor = 0.0
    for item, source, duration in sources:
        seconds = duration / tempo + padding
        output = work / f"part-{item['slide']:02d}.wav"
        audio_segment(source, output, seconds, tempo, lead=0.35)
        files.append(output)
        add_cues(cues, item["text"], cursor + 0.35, duration / tempo)
        schedule.append({"slide": item["slide"], "start": cursor, "duration": seconds, "speech_seconds": duration / tempo})
        cursor += seconds
    joined = work / "narration.wav"
    concatenate_audio(files, joined, work)
    subtitle_file(work / "captions.ass", cues)
    image_listing = []
    for item in schedule:
        original = temporary / "slides" / f"slide-{item['slide']:02d}.png"
        if not original.is_file():
            raise FileNotFoundError(f"Verified PowerPoint render missing: {original}")
        image_listing.extend([f"file '{original.as_posix()}'", f"duration {item['duration']:.8f}"])
    image_listing.append(image_listing[-2])
    (work / "slides-concat.txt").write_text("\n".join(image_listing), encoding="utf-8")
    destination = ROOT / "deliverables/lab4/实验四_PPT汇报.mp4"
    # First materialize a continuous video from sparse still-image timestamps.
    # Subtitles must run in a second pass so every video frame receives its cue.
    continuous = work / "slides.mp4"
    ffmpeg(["-f", "concat", "-safe", "0", "-i", "slides-concat.txt",
            "-vf", "scale=1760:990:flags=lanczos,pad=1920:1080:80:0:color=0xF8F3E9",
            "-an", "-t", "180", "-r", "30", "-c:v", "libx264", "-preset", "fast",
            "-crf", "12", "-pix_fmt", "yuv420p", continuous], cwd=work)
    ffmpeg(["-i", continuous, "-i", joined, "-vf", "subtitles=captions.ass",
            "-t", "180", "-r", "30", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
            "-map", "0:v:0", "-map", "1:a:0", "-movflags", "+faststart", destination], cwd=work)
    (temporary / "ppt_timeline.json").write_text(json.dumps({"duration_seconds": 180, "voice": VOICE,
        "speed_factor": tempo, "slides": schedule, "cues": cues}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(destination, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("demo", "presentation"))
    parser.add_argument("--tmp", type=Path, required=True)
    parser.add_argument("--proxy", default=os.environ.get("HTTPS_PROXY", "http://127.0.0.1:7890"))
    args = parser.parse_args()
    operation = build_demo if args.mode == "demo" else build_presentation
    asyncio.run(operation(args.tmp.resolve(), args.proxy))


if __name__ == "__main__":
    main()
