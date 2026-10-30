"""Remove greetings and thanks from the two verified presentation masters."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import wave

import cv2
import numpy as np

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[2]
SOURCE = ROOT / "outputs/qa/lab01-lab02-separate-20261027"
sys.path.insert(0, str(ROOT / "scripts"))
import create_lab02_ppt_video as media


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def revise(lab):
    source = SOURCE / lab
    folder = WORK / lab
    folder.mkdir(exist_ok=True)
    original = read(source / "timeline.json")
    capture = read(source / "capture-events.json")
    first = original["slides"][0]["cues"]
    last = original["slides"][-1]["cues"]
    assert first[0]["text"] == "各位老师好，我们是第31组。"
    assert last[-1]["text"] == "谢谢大家。"
    start_frame = round((first[0]["end"] + first[1]["start"]) / 2 * media.FPS)
    end_frame = round((last[-2]["end"] + last[-1]["start"]) / 2 * media.FPS)
    cut_start, cut_end = start_frame / media.FPS, end_frame / media.FPS
    frames = end_frame - start_frame
    duration = frames / media.FPS
    timeline = copy.deepcopy(original)
    timeline.update(duration=duration, target_duration=duration, edited=True,
                    original_duration=original["duration"], cut_start=cut_start, cut_end=cut_end)
    timeline["slides"][0]["cues"].pop(0)
    timeline["slides"][-1]["cues"].pop()
    if lab == "lab01":
        closing = timeline["slides"].pop()
        timeline["slides"][-1]["cues"].extend(closing["cues"])
        timeline["slides"][-1]["end"] = closing["end"]
        timeline["closing_visual"] = "Hold the last substantive slide during the retained conclusion; omit the thank-you slide."
    for slide in timeline["slides"]:
        slide["start"] = max(0, slide["start"] - cut_start)
        slide["end"] = min(duration, slide["end"] - cut_start)
        slide["frame_count"] = round((slide["end"] - slide["start"]) * media.FPS)
        for cue in slide["cues"]:
            cue["start"] = round(cue["start"] - cut_start, 6)
            cue["end"] = round(cue["end"] - cut_start, 6)
    assert all(0 <= c["start"] < c["end"] <= duration for s in timeline["slides"] for c in s["cues"])
    assert not any("谢谢" in c["text"] or "老师好" in c["text"] or "第31组" in c["text"]
                   for s in timeline["slides"] for c in s["cues"])

    label = "实验一" if lab == "lab01" else "实验二"
    stem = f"第31组_{label}_PPT独立汇报_2分钟内_精剪版"
    media.WORK = folder
    media.DELIVERY = folder
    media.PPTX = source / "source.pptx"
    media.REFERENCES = source / "slide-renders"
    media.STEM = stem
    media.VIDEO = folder / f"{stem}.mp4"
    media.SRT = folder / f"{stem}.srt"
    write(folder / "timeline.json", timeline)
    media.write_subtitles(timeline)
    ass = folder / "subtitles.ass"
    ass.write_text(ass.read_text(encoding="utf-8-sig").replace("Narration,Microsoft YaHei,42,", "Narration,Microsoft YaHei,38,").replace(",110,110,24,1", ",110,110,8,1"), encoding="utf-8-sig")
    pcm = media.read_pcm(source / "narration.wav")
    samples_per_frame = media.RATE // media.FPS
    edited_pcm = pcm[start_frame * samples_per_frame * 2:end_frame * samples_per_frame * 2]
    with wave.open(str(folder / "narration.wav"), "wb") as out:
        out.setparams((1, 2, media.RATE, 0, "NONE", "not compressed"))
        out.writeframes(edited_pcm)

    raw_start = capture["measured_trim"] + cut_start
    video_filter = f"fps={media.FPS},trim=end_frame={frames},setpts=PTS-STARTPTS"
    extra_inputs = []
    video_map = ["-map", "0:v:0"]
    if lab == "lab01":
        transition = capture["measured_transitions"][-1]["raw_transition"]
        freeze_frame = math.floor((transition - raw_start - .12) * media.FPS)
        cap = cv2.VideoCapture(str(source / "slideshow-raw.mkv"))
        cap.set(cv2.CAP_PROP_POS_MSEC, (transition - .3) * 1000)
        ok, frozen = cap.read()
        cap.release()
        assert ok
        freeze_image = folder / "closing-content.png"
        cv2.imencode(".png", frozen)[1].tofile(str(freeze_image))
        extra_inputs = ["-loop", "1", "-framerate", media.FPS, "-i", freeze_image]
        video_filter = (f"[0:v]{video_filter}[base];[base][2:v]"
                        f"overlay=0:0:enable='gte(t,{freeze_frame / media.FPS:.9f})':shortest=1")
        video_map = ["-map", "[edited]"]
        timeline["closing_freeze_start"] = freeze_frame / media.FPS
    video_filter += f",ass=filename='{ass.relative_to(ROOT).as_posix()}'"
    filters = ["-vf", video_filter]
    if lab == "lab01":
        filters = ["-filter_complex", video_filter + "[edited]"]
    media.run_ffmpeg(["-ss", raw_start, "-i", source / "slideshow-raw.mkv",
                      "-i", folder / "narration.wav", *extra_inputs, "-t", duration,
                      *video_map, "-map", "1:a:0", *filters,
                      "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                      "-pix_fmt", "yuv420p", "-r", media.FPS,
                      "-af", "loudnorm=I=-16:TP=-1.5:LRA=7", "-ar", 48000, "-ac", 2,
                      "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                      "-metadata", f"title={stem}", "-metadata", "comment=已删除问候、自我介绍和致谢；中文旁白与字幕同步",
                      media.VIDEO], "render.log")
    write(folder / "timeline.json", timeline)
    media.verify()
    media.run_ffmpeg(["-i", media.VIDEO, "-map", "0:a:0", "-vn", "-ac", 1,
                      "-ar", media.RATE, "-c:a", "pcm_s16le", folder / "verified-audio.wav"], "audio-verify.log")
    actual = np.frombuffer(media.read_pcm(folder / "verified-audio.wav"), dtype="<i2").astype(float)
    expected = np.frombuffer(edited_pcm, dtype="<i2").astype(float)
    checks = []
    for slide in timeline["slides"]:
        for cue in slide["cues"]:
            begin, end = (round(cue[k] * media.RATE) for k in ["start", "end"])
            correlation = float(np.corrcoef(expected[begin:end], actual[begin:end])[0, 1])
            if not np.isfinite(correlation) or correlation < .93:
                raise RuntimeError(f"Audio not aligned: {cue['text']}, {correlation}")
            checks.append(correlation)
    # Inspect both edit boundaries as well as the retained conclusion.
    video = cv2.VideoCapture(str(media.VIDEO))
    for label, second in [("opening", .45), ("ending", duration - .35)]:
        video.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
        ok, picture = video.read()
        assert ok
        cv2.imencode(".png", picture)[1].tofile(str(folder / f"{label}.png"))
    video.release()
    report = read(folder / "verification.json")
    report.update({"greeting_and_thanks_removed": True, "cut_start_seconds": cut_start,
                   "cut_end_seconds": cut_end, "audio_minimum_correlation": min(checks),
                   "video_sha256": sha(media.VIDEO), "source_master_unchanged": True,
                   "original_source": original["original_source"], "source_lab": lab})
    write(folder / "verification.json", report)
    guide = [f"# {label}精剪版逐页讲稿", "", f"时长：{duration:.3f} 秒；1080p，30 fps。已去掉开场问候、自我介绍和结尾致谢，字幕已内嵌。", ""]
    for slide in timeline["slides"]:
        guide.extend([f"## {slide['title']}", "", f"{media.timestamp(slide['start'])} — {media.timestamp(slide['end'])}", "",
                      "".join(c["text"] for c in slide["cues"]), ""])
    (folder / f"{stem}_逐页讲稿.md").write_text("\n".join(guide), encoding="utf-8-sig")
    print(json.dumps({"lab": lab, "duration": duration, "frames": frames, "audio_minimum_correlation": min(checks), "video": str(media.VIDEO)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import os
    sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(ROOT)
    for lab in ["lab01", "lab02"]:
        revise(lab)
