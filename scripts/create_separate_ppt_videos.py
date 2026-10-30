"""Record two independent, narrated PowerPoint presentations under two minutes.

Each video uses all slides from its original deck, a standalone introduction
and conclusion, synthesized Mandarin speech, and burned-in Chinese subtitles.
The existing joint presentation and source deliverables are hash-protected.
Run --stage init, prepare, record, render, verify, deliver in order, or --stage
all. Use --lab lab01/lab02 to resume one presentation independently.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import create_lab02_ppt_video as media

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "outputs/qa/lab01-lab02-separate-20261027"
JOINT = ROOT / "outputs/qa/lab01-lab02-joint-20261027"
SUBMISSION = ROOT.parent / "人工智能实验2"
NARRATION = ROOT / "scripts/narration"
LABS = {
    "lab01": {"label": "实验一", "count": 15, "reference_start": 1},
    "lab02": {"label": "实验二", "count": 9, "reference_start": 16},
}
BASE_SUBTITLES = media.write_subtitles


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def protect_sources(create=False):
    path = WORK / "source-baseline.json"
    if create and not path.exists():
        sources = read_json(JOINT / "timeline.json")["source_decks"]
        protected = {Path(s["pptx"]) for s in sources}
        protected.update(p for p in (ROOT / "deliverables").rglob("*") if p.is_file())
        protected.update(p for p in SUBMISSION.iterdir() if p.is_file())
        write_json(path, {str(p): {"sha256": digest(p), "bytes": p.stat().st_size}
                          for p in sorted(protected)})
    if not path.exists():
        raise RuntimeError("Run --stage init before creating the independent videos")
    for filename, expected in read_json(path).items():
        source = Path(filename)
        if not source.is_file() or digest(source) != expected["sha256"]:
            raise RuntimeError(f"An existing source or deliverable changed: {source}")


def initialize():
    WORK.mkdir(parents=True, exist_ok=True)
    protect_sources(create=True)
    sources = {s["lab"]: s for s in read_json(JOINT / "timeline.json")["source_decks"]}
    for lab, specification in LABS.items():
        folder = WORK / lab
        references = folder / "slide-renders"
        references.mkdir(parents=True, exist_ok=True)
        source = sources[lab]
        original = Path(source["pptx"])
        if digest(original) != source["sha256"] or source["slide_count"] != specification["count"]:
            raise RuntimeError(f"Original deck no longer matches the verified slide renders: {lab}")
        shutil.copy2(original, folder / "source.pptx")
        write_json(folder / "source.json", source)
        for number in range(1, specification["count"] + 1):
            reference = specification["reference_start"] + number - 1
            shutil.copy2(JOINT / f"slide-renders/slide-{reference:02}.png", references / f"slide-{number:02}.png")
        (ROOT / "deliverables" / lab).mkdir(parents=True, exist_ok=True)
        media.log(f"Initialized {lab}: {specification['count']} original slides")
    protect_sources()


def write_subtitles(timeline):
    BASE_SUBTITLES(timeline)
    path = media.WORK / "subtitles.ass"
    content = path.read_text(encoding="utf-8-sig")
    # Fit the empty top strip of the lab01 deck without covering its header.
    content = content.replace("Narration,Microsoft YaHei,42,", "Narration,Microsoft YaHei,38,")
    content = content.replace(",110,110,24,1", ",110,110,8,1")
    path.write_text(content, encoding="utf-8-sig")


def configure(lab):
    media.WORK = WORK / lab
    media.PPTX = media.WORK / "source.pptx"
    media.REFERENCES = media.WORK / "slide-renders"
    media.DELIVERY = ROOT / "deliverables" / lab
    media.STEM = f"第31组_{LABS[lab]['label']}_PPT独立汇报_2分钟内"
    media.VIDEO = media.DELIVERY / f"{media.STEM}.mp4"
    media.SRT = media.DELIVERY / f"{media.STEM}.srt"
    media.write_subtitles = write_subtitles
    media.write_editing_guide = lambda *_: None


def write_guide(lab, timeline, settings):
    from docx import Document
    from docx.oxml.ns import qn
    from docx.shared import Pt

    label = LABS[lab]["label"]
    title = f"{label}两分钟内独立汇报逐页讲稿"
    intro = (f"成片118秒，1920×1080，30帧/秒。按原PPT的{LABS[lab]['count']}页顺序独立讲解，"
             "包含选题介绍和结束语。普通话合成旁白与中文字幕同步，字幕已内嵌；播放时无需再加载SRT。")
    source = read_json(media.WORK / "source.json")
    document = Document()
    normal = document.styles["Normal"]
    normal.font.name = "Microsoft YaHei"
    normal.font.size = Pt(11)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    document.add_heading(title, 0)
    document.add_paragraph(intro)
    document.add_paragraph(f"PPT来源：{source['pptx']}")
    lines = [f"# {title}", "", intro, "", f"PPT来源：{source['pptx']}", "",
             "## 逐页时间表", "", "| 页 | 开始 | 结束 | 主题 |", "|---|---|---|---|"]
    rows = []
    for slide in timeline["slides"]:
        start, end = media.timestamp(slide["start"]), media.timestamp(slide["end"])
        lines.append(f"| {slide['slide']} | {start} | {end} | {slide['title']} |")
        rows.append([slide["slide"], slide["start"], slide["end"], start, end, slide["title"]])
    for slide, narration in zip(timeline["slides"], settings["slides"]):
        heading = f"第{slide['slide']}页 · {slide['title']}"
        times = f"{media.timestamp(slide['start'])} — {media.timestamp(slide['end'])}"
        speech = "".join(cue["text"] for cue in narration["cues"])
        lines.extend(["", f"## {heading}", "", times, "", speech])
        document.add_heading(heading, 2)
        document.add_paragraph(times)
        document.add_paragraph(speech)
    note = "实验数据和测试数量引用原PPT已有记录；本次工作重新制作汇报视频，未重新运行项目实验。原联合视频及原PPT保留。"
    lines.extend(["", "## 说明", "", note, ""])
    document.add_paragraph(note)
    document.save(media.DELIVERY / f"{media.STEM}_逐页讲稿.docx")
    (media.DELIVERY / f"{media.STEM}_逐页讲稿.md").write_text("\n".join(lines), encoding="utf-8-sig")
    with (media.DELIVERY / f"{media.STEM}_剪辑时间表.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["slide", "start_seconds", "end_seconds", "start_timecode", "end_timecode", "title"])
        writer.writerows(rows)


def prepare(lab):
    settings = read_json(NARRATION / f"{lab}-standalone.json")
    if len(settings["slides"]) != LABS[lab]["count"]:
        raise RuntimeError("Narration must cover every original slide")
    if settings["target_duration"] != 118 or settings["max_duration"] != 120:
        raise RuntimeError("Each independent presentation must stay within two minutes")
    write_json(media.WORK / "narration.json", settings)
    # Identical sentences can reuse the earlier verified voice cache. The
    # synthesizer checks the complete speech/voice/rate fingerprint before reuse.
    for slide in settings["slides"]:
        old_number = LABS[lab]["reference_start"] + slide["slide"] - 1
        for suffix in ["mp3", "json"]:
            target = media.WORK / f"voice-{slide['slide']:02}.{suffix}"
            if not target.exists():
                shutil.copy2(JOINT / f"voice-{old_number:02}.{suffix}", target)
    asyncio.run(media.prepare())
    timeline = media.load_timeline()
    timeline["original_source"] = read_json(media.WORK / "source.json")
    timeline["standalone"] = True
    timeline["lab"] = lab
    write_json(media.WORK / "timeline.json", timeline)
    write_guide(lab, timeline, settings)


def measure_capture():
    import cv2
    import numpy as np

    timeline = media.load_timeline()
    events = read_json(media.WORK / "capture-events.json")
    cap = cv2.VideoCapture(str(media.WORK / "slideshow-raw.mkv"))
    references = []
    for slide in timeline["slides"]:
        picture = cv2.imread(str(media.REFERENCES / f"slide-{slide['slide']:02}.png"))
        if picture is None:
            raise RuntimeError("Missing original-slide reference")
        references.append(cv2.resize(picture, (160, 90)).astype(np.float32))
    transitions = []
    try:
        for index, slide in enumerate(timeline["slides"][1:], 1):
            begin = max(0, slide["start"] + events["warmup"] - 1.3)
            cap.set(cv2.CAP_PROP_POS_MSEC, begin * 1000)
            found = None
            seen_previous = False
            for _ in range(90):
                ok, frame = cap.read()
                if not ok:
                    break
                second = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                small = cv2.resize(frame, (160, 90)).astype(np.float32)
                errors = [float(np.mean((small - reference) ** 2)) for reference in references]
                best = int(np.argmin(errors))
                if best == index - 1 and errors[best] < 1200:
                    seen_previous = True
                if best == index and errors[best] < 500 and seen_previous:
                    found = second
                    break
            if found is None:
                raise RuntimeError(f"Cannot locate recorded transition to slide {slide['slide']}")
            transitions.append({"slide": slide["slide"], "raw_transition": found,
                                "planned_transition": slide["start"], "offset": found - slide["start"]})
    finally:
        cap.release()
    offset = float(np.median([item["offset"] for item in transitions]))
    frames = round(offset * media.FPS)
    error = max(abs(item["offset"] - frames / media.FPS) for item in transitions)
    if error > .25 or not 0 <= offset <= 2:
        raise RuntimeError(f"Capture timing out of bounds: offset={offset}, maximum error={error}")
    events.update({"measured_trim": frames / media.FPS, "trim_frames": frames,
                   "maximum_transition_error": error, "measured_transitions": transitions})
    write_json(media.WORK / "capture-events.json", events)
    media.log(f"Measured capture alignment: trim={frames / media.FPS:.3f}s, maximum error={error:.3f}s")


def verify_audio(timeline):
    import numpy as np

    decoded = media.WORK / "final-audio-verify.wav"
    media.run_ffmpeg(["-i", media.VIDEO, "-map", "0:a:0", "-vn", "-ac", 1, "-ar", media.RATE,
                      "-c:a", "pcm_s16le", decoded], "audio-verification-decode.log")
    expected = np.frombuffer(media.read_pcm(media.WORK / "narration.wav"), dtype="<i2").astype(np.float64) / 32768
    actual = np.frombuffer(media.read_pcm(decoded), dtype="<i2").astype(np.float64) / 32768
    if abs(len(actual) - len(expected)) / media.RATE > .05:
        raise RuntimeError("Final audio duration differs from the narration")
    checks = []
    for slide in timeline["slides"]:
        start = round(slide["speech_start"] * media.RATE)
        end = round((slide["speech_start"] + slide["speech_duration"]) * media.RATE)
        source, sound = expected[start:end], actual[start:end]
        correlation = float(np.corrcoef(source, sound)[0, 1])
        rms = float(20 * np.log10(np.sqrt(np.mean(sound ** 2)) + 1e-12))
        if not np.isfinite(correlation) or correlation < .94 or rms < -32:
            raise RuntimeError(f"Missing or misaligned audio at slide {slide['slide']}")
        checks.append({"slide": slide["slide"], "correlation": correlation, "rms_dbfs": rms})
    peak = float(np.max(np.abs(actual)))
    if peak >= .999:
        raise RuntimeError("Final audio clips")
    report = {"all_speech_segments_present_and_aligned": True,
              "minimum_correlation": min(c["correlation"] for c in checks),
              "peak_dbfs": float(20 * np.log10(peak + 1e-12)), "checks": checks}
    write_json(media.WORK / "audio-verification.json", report)
    return report


def verify(lab):
    timeline = media.load_timeline()
    settings = read_json(media.WORK / "narration.json")
    if settings != read_json(NARRATION / f"{lab}-standalone.json"):
        raise RuntimeError("Prepared narration is stale")
    if len(timeline["slides"]) != LABS[lab]["count"] or timeline["duration"] != 118:
        raise RuntimeError("Unexpected slide count or video duration")
    last_end = -1
    for slide, source in zip(timeline["slides"], settings["slides"]):
        if [c["text"] for c in slide["cues"]] != [c["text"] for c in source["cues"]]:
            raise RuntimeError("Subtitles do not match the independent narration")
        speech = "".join(c.get("speech", c["text"]) for c in source["cues"])
        expected = hashlib.sha256(json.dumps([speech, settings["voice"], settings["rate"], "WordBoundary"],
                                            ensure_ascii=False).encode()).hexdigest()
        cached = read_json(media.WORK / f"voice-{slide['slide']:02}.json")
        if expected != cached["fingerprint"]:
            raise RuntimeError("Voice cache does not match current narration")
        for cue in slide["cues"]:
            if not slide["start"] <= cue["start"] < cue["end"] <= slide["end"] or cue["start"] <= last_end:
                raise RuntimeError("Invalid subtitle timing")
            last_end = cue["end"]
    media.verify()
    audio = verify_audio(timeline)
    report = read_json(media.WORK / "verification.json")
    report.update({"standalone": True, "original_source": timeline["original_source"],
                   "video_sha256": digest(media.VIDEO), "all_existing_deliverables_unchanged": True,
                   "maximum_transition_error": read_json(media.WORK / "capture-events.json")["maximum_transition_error"],
                   "audio_minimum_correlation": audio["minimum_correlation"]})
    protect_sources()
    write_json(media.WORK / "verification.json", report)
    media.log(f"Verified {lab}: all slides, subtitles, narration, and original-file hashes")


def deliver(lab):
    report = read_json(media.WORK / "verification.json")
    if digest(media.VIDEO) != report["video_sha256"]:
        raise RuntimeError("Only the verified video can be delivered")
    destination = SUBMISSION / media.VIDEO.name
    if destination.exists() and digest(destination) != report["video_sha256"]:
        raise RuntimeError(f"A different file already exists: {destination}")
    shutil.copy2(media.VIDEO, destination)
    if digest(destination) != report["video_sha256"]:
        raise RuntimeError("Submission copy verification failed")
    protect_sources()
    write_json(media.WORK / "delivery.json", {"lab": lab, "video": str(destination),
                                             "duration_seconds": report["duration_seconds"],
                                             "sha256": report["video_sha256"], "joint_video_unchanged": True})
    media.log(f"Delivered {destination}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=["all", "init", "prepare", "probe", "record", "render", "verify", "deliver"])
    parser.add_argument("--lab", choices=["lab01", "lab02", "both"], default="both")
    args = parser.parse_args()
    if args.stage in ["init", "all"]:
        initialize()
    if args.stage == "init":
        return
    protect_sources()
    for lab in LABS if args.lab == "both" else [args.lab]:
        configure(lab)
        if args.stage in ["prepare", "all"]:
            prepare(lab)
        if args.stage == "probe":
            media.record(probe=True)
        if args.stage in ["record", "all"]:
            media.record()
        if args.stage in ["render", "all"]:
            measure_capture()
            media.render()
        if args.stage in ["verify", "all"]:
            verify(lab)
        if args.stage in ["deliver", "all"]:
            deliver(lab)
        protect_sources()


if __name__ == "__main__":
    main()
