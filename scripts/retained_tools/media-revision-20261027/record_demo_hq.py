"""Capture real browser operations as native 1440p PNG frames, then encode once.

The 90-second operation sequence follows the existing demo. Its final speed is
1.5x, matching the user's current one-minute, silent, caption-free video.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import cv2
import imageio_ffmpeg
import numpy as np
from playwright.sync_api import sync_playwright, expect

WORK = Path(__file__).resolve().parent / "demo-native"
ROOT = WORK.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import record_lab02_demo as b

URL = "http://127.0.0.1:8772"
SIZE = {"width": 2560, "height": 1440}
ZOOM = 1.6
b.VIEWPORT = SIZE
b.OUTPUT = WORK

PRESENTATION = r"""() => {
    // Emulate the same 1600x900 layout at 160% zoom on a native 2560x1440 surface.
    // Viewport-unit corrections keep the page's existing header and graph sizes.
    // This style exists only in the recording tab; no application files change.
    const css = [...document.styleSheets].map(s => [...s.cssRules].map(r => r.cssText).join('\n')).join('\n');
    const style = document.createElement('style');
    style.textContent = css.replace(/(-?\d*\.?\d+)(vw|dvh|vh)\b/g, (_, number, unit) => (Number(number) * (unit === 'vw' ? 16 : 9)) + 'px');
    document.head.append(style);
    document.body.style.zoom = '1.6';
    document.querySelector('.course-app-header').style.setProperty('padding-inline', '28px', 'important');
    // Canvas charts otherwise keep a 1x backing store beneath the browser zoom.
    // Re-render their existing data at native output density in this tab only.
    window.demoCanvasAdjustments = 0;
    setInterval(() => {
        document.querySelectorAll('.graph-canvas').forEach(element => {
            const cy = element._cyreg?.cy;
            const renderer = cy?.renderer();
            if (renderer && renderer.forcedPixelRatio !== 1.6) {
                renderer.forcedPixelRatio = 1.6;
                cy.resize();
                cy.forceRender();
                window.demoCanvasAdjustments++;
            }
        });
    }, 150);
    const pointerStyle = document.createElement('style');
    pointerStyle.textContent = '*{cursor:none!important} #demo-pointer{position:fixed;left:0;top:0;width:43.2px;height:51.2px;z-index:2147483647;pointer-events:none;filter:drop-shadow(1.6px 3.2px 3.2px #0006)} .demo-click{position:fixed;width:67.2px;height:67.2px;margin:-33.6px;border:4.8px solid #ab4d26;border-radius:50%;pointer-events:none;z-index:2147483646;animation:demo-ring .6s ease-out forwards} @keyframes demo-ring{from{transform:scale(.35);opacity:.95}to{transform:scale(1.2);opacity:0}}';
    document.head.append(pointerStyle);
    const pointer = document.createElement('div'); pointer.id = 'demo-pointer';
    pointer.innerHTML = '<svg viewBox="0 0 27 32"><path d="M3 2 L3 25 L9 19 L14 29 L18 27 L13 18 L23 18 Z" fill="white" stroke="#3b2c24" stroke-width="1.6" stroke-linejoin="round"/></svg>';
    document.documentElement.append(pointer);
    document.addEventListener('mousemove', e => {window.demoPointerPosition={x:e.clientX,y:e.clientY};pointer.style.transform='translate('+e.clientX+'px,'+e.clientY+'px)'});
    document.addEventListener('mousedown', e => {const ring=document.createElement('div');ring.className='demo-click';ring.style.left=e.clientX+'px';ring.style.top=e.clientY+'px';document.documentElement.append(ring);setTimeout(()=>ring.remove(),650)});
    const marker = document.createElement('div'); marker.id='demo-sync';marker.style.cssText='position:fixed;left:0;top:0;width:24px;height:24px;background:rgb(255,0,255);z-index:2147483647';document.documentElement.append(marker);
}"""


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def scroll(page, locator, top=100):
    b.scroll_to(page, locator, round(top * ZOOM))


def choose(page, locator, value=None, label=None):
    b.move_to(page, locator)
    locator.focus()
    page.wait_for_timeout(160)
    locator.select_option(value=value) if value is not None else locator.select_option(label=label)
    page.wait_for_timeout(200)


class Capture:
    def __init__(self, context, page):
        self.folder = WORK / ("frames-" + time.strftime("%H%M%S"))
        self.folder.mkdir(parents=True)
        self.frames = []
        self.cdp = context.new_cdp_session(page)

    def receive(self, event):
        data = base64.b64decode(event["data"])
        filename = self.folder / f"frame-{len(self.frames):06}.png"
        filename.write_bytes(data)
        self.frames.append({"file": str(filename.relative_to(WORK)), "timestamp": event["metadata"]["timestamp"], "bytes": len(data)})
        self.cdp.send("Page.screencastFrameAck", {"sessionId": event["sessionId"]})

    def start(self):
        self.cdp.on("Page.screencastFrame", self.receive)
        self.cdp.send("Page.startScreencast", {"format": "png", "maxWidth": SIZE["width"], "maxHeight": SIZE["height"], "everyNthFrame": 1})

    def stop(self):
        self.cdp.send("Page.stopScreencast")
        write(WORK / "capture-frames.json", self.frames)
        print(f"Saved {len(self.frames)} native 1440p PNG frames", flush=True)


def record(probe=False):
    WORK.mkdir(parents=True, exist_ok=True)
    audit = {"url": URL, "live_results": {}, "runs": [], "shots": [], "browser_errors": [], "http_errors": [],
             "capture": "Chrome DevTools screencast; native 2560x1440; PNG lossless intermediates",
             "composition": {"website_fills_frame": True, "audio": False, "subtitles": False, "added_titles": False, "visible_pointer": True},
             "viewport": SIZE, "display_zoom": ZOOM, "target_seconds": 60, "playback_speed": 1.5}
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        context = browser.new_context(viewport=SIZE, locale="zh-CN", device_scale_factor=1)
        page = context.new_page()
        page.set_default_timeout(18000)
        page.on("pageerror", lambda e: audit["browser_errors"].append(str(e)))
        page.on("response", lambda r: audit["http_errors"].append({"status": r.status, "url": r.url}) if r.status >= 400 else None)
        page.goto(URL + "/#/labs/lab02/optimization")
        expect(page.get_by_role("button", name="确定性优化", exact=True)).to_be_visible()
        expect(page.locator(".chart-grid")).to_be_visible()
        page.locator('.course-navigation > a[href="#/labs/lab02"]').click()
        expect(page.locator(".module-row")).to_have_count(5)
        expect(page.locator(".header-status")).to_contain_text("已就绪")
        page.evaluate(PRESENTATION)
        page.mouse.move(2288, 232)
        page.wait_for_timeout(400)
        capture = Capture(context, page)
        capture.start()
        try:
            page.wait_for_timeout(900)
            page.evaluate("document.getElementById('demo-sync').remove()")
            started = time.perf_counter()
            audit["started_epoch"] = time.time()

            def at(second):
                lag = time.perf_counter() - started - second
                if lag > 6:
                    raise RuntimeError(f"Operation timeline is {lag:.2f}s late at {second:.1f}s")
                if lag < 0:
                    page.wait_for_timeout(-lag * 1000)

            def shot(name):
                b.capture(page, name, started, audit)

            def run(module, label):
                b.run_task(page, module, audit)
                audit["runs"].append({"label": label, "at_seconds": round(time.perf_counter() - started, 3), "result": audit["live_results"][module]})

            shot("01-overview")
            at(2.5); b.enter(page, "知识表示与建模")
            at(4.0); run("modeling", "谓词事实与真实来源")
            at(5.0); scroll(page, page.get_by_text("一阶谓词与来源（47 条）", exact=True), 120)
            shot("02-predicate-facts")
            at(7.0)
            b.click(page, page.get_by_text("一阶谓词与来源（47 条）", exact=True))
            b.click(page, page.get_by_role("button", name="展开一阶规则知识库", exact=True))
            scroll(page, page.get_by_role("heading", name="一阶规则知识库", exact=True), 95)
            shot("03-rule-library")
            if probe:
                page.wait_for_timeout(500)
            else:
                at(11.0); b.directory(page); b.enter(page, "确定性推理")
                at(12.5); run("deterministic", "正向推理 Controls(P1,C3)")
                scroll(page, page.locator(".task-result"), 100); shot("04-forward-proof")
                at(17.0); scroll(page, page.locator(".lab-task-form"), 135)
                choose(page, page.get_by_label("实验二推理方法"), "backward")
                run("deterministic", "表格化反向推理 Controls(P1,C3)")
                scroll(page, page.locator(".task-result"), 100); shot("05-backward-proof")
                at(22.5)
                b.click(page, page.get_by_role("link", name="推导关系与证明演示", exact=True))
                expect(page.get_by_role("button", name="播放当前结论的证明")).to_be_enabled()
                scroll(page, page.locator(".graph-panel"), 90)
                at(25.3); b.click(page, page.get_by_role("button", name="播放当前结论的证明"))
                at(28.0); shot("06-proof-graph-inputs")
                audit["graph_canvas_density"] = page.locator(".graph-canvas canvas").evaluate_all("es=>es.map(e=>({backing_width:e.width,display_width:e.getBoundingClientRect().width,backing_height:e.height,display_height:e.getBoundingClientRect().height}))")
                assert audit["graph_canvas_density"] and all(abs(c["backing_width"] - c["display_width"]) < 2 for c in audit["graph_canvas_density"])
                at(32.5); shot("07-proof-graph-rules")
                at(37.2)
                expect(page.locator(".step-counter")).to_have_text("10 / 10")
                shot("08-proof-graph-complete")
                at(38.0); b.directory(page); b.enter(page, "不确定性推理")
                at(40.0); run("uncertain", "教学预设 Bayes 与 Mamdani")
                scroll(page, page.locator(".task-result"), 90); shot("09-manual-scores")
                at(44.0); scroll(page, page.locator(".lab-task-form"), 130)
                choose(page, page.get_by_label("实验二参数来源"), label="图级训练 · 7 · 自动模糊拟合")
                run("uncertain", "训练参数 Bayes 与自动拟合 Mamdani")
                assert audit["live_results"]["uncertain"]["fuzzy"]["parameters"]["training_count"] > 0
                scroll(page, page.locator(".task-result"), 90); shot("10-trained-scores")
                at(49.5); b.directory(page); b.enter(page, "算法优化与对比实验")
                expect(page.locator(".chart-grid")).to_be_visible()
                at(51.0); scroll(page, page.locator(".experiment-toolbar"), 90); shot("11-clean-comparison")
                at(54.0); choose(page, page.get_by_label("观测条件", exact=True), "value_noise"); shot("12-noise-comparison")
                at(56.5); b.click(page, page.get_by_role("button", name="确定性优化", exact=True))
                scroll(page, page.locator(".performance-section"), 90)
                expect(page.get_by_role("region", name="确定性优化实测").locator("tbody tr")).to_have_count(3)
                shot("13-runtime-and-search-cost")
                at(61.0); b.click(page, page.get_by_role("button", name="评分与参数优化", exact=True))
                scroll(page, page.get_by_role("heading", name="评分算法优化与消融"), 90); shot("14-scoring-ablation")
                at(65.0); choose(page, page.get_by_label("评分实验观测条件"), "noise_010"); shot("15-noisy-scoring-ablation")
                at(68.0); b.directory(page); b.enter(page, "混合推理与创新验证")
                at(70.0); choose(page, page.get_by_label("实验二参数来源"), "manual")
                run("hybrid", "同分数的三种结构约束与校准策略")
                scroll(page, page.locator(".task-result"), 110); shot("16-three-hybrid-strategies")
                assert len({x["score"] for x in audit["live_results"]["hybrid"]["decisions"]}) == 1
                at(74.0); b.move_to(page, page.get_by_role("table", name="实验二混合策略对照").get_by_text("保留结构约束", exact=True))
                at(78.0); b.click(page, page.get_by_role("link", name="查看目标企业分析", exact=True))
                expect(page.get_by_role("heading", name="企业分析", exact=True)).to_be_visible()
                scroll(page, page.get_by_role("heading", name="恒通科技股份有限公司", exact=True), 140)
                at(81.0); shot("17-company-analysis")
                at(84.0); b.move_to(page, page.get_by_text("结构约束对照", exact=True))
                at(87.0); shot("18-final-evidence-and-action")
                at(91.0)
            audit["recorded_seconds"] = time.perf_counter() - started
        finally:
            capture.stop()
            write(WORK / "recording-audit.json", audit)
            context.close()
            browser.close()
    if audit["browser_errors"] or audit["http_errors"]:
        raise RuntimeError("Browser or HTTP errors were recorded")
    return audit


def render():
    # Chrome may deliver a delayed compositor frame after a screenshot-triggered
    # frame. Presentation timestamps, rather than callback order, define the cut.
    frames = sorted(read(WORK / "capture-frames.json"), key=lambda item: item["timestamp"])
    frames = list({item["timestamp"]: item for item in frames}.values())
    seen_marker = False
    begin = None
    for i, frame in enumerate(frames):
        picture = cv2.imread(str(WORK / frame["file"]))
        assert picture.shape[:2] == (1440, 2560)
        blue, green, red = map(int, picture[10, 10])
        marked = red > 240 and blue > 240 and green < 20
        if seen_marker and not marked:
            begin = i
            break
        seen_marker |= marked
    if begin is None:
        raise RuntimeError("Recording start marker was not located")
    origin = frames[begin]["timestamp"]
    selected = [f for f in frames[begin:] if f["timestamp"] < origin + 90]
    audit = read(WORK / "recording-audit.json")
    assert audit["started_epoch"] + audit["recorded_seconds"] >= origin + 90, "Recording did not cover all 90 seconds"
    lines = ["ffconcat version 1.0"]
    for i, frame in enumerate(selected):
        next_stamp = selected[i + 1]["timestamp"] if i + 1 < len(selected) else origin + 90
        duration = next_stamp - frame["timestamp"]
        assert duration > 0
        lines += [f"file '{frame['file'].replace(chr(92), '/')}'", "option framerate 1000", f"duration {duration:.9f}"]
    lines += [f"file '{selected[-1]['file'].replace(chr(92), '/')}'", "option framerate 1000"]
    (WORK / "frames.ffconcat").write_text("\n".join(lines) + "\n", encoding="utf-8")
    destination = WORK / "第31组_实验二_Demo_高清版.mp4"
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-y", "-safe", "0", "-i", "frames.ffconcat",
               "-vf", "setpts=(PTS-STARTPTS)/1.5,fps=30", "-t", "60", "-frames:v", "1800", "-an",
               "-c:v", "libx264", "-preset", "medium", "-crf", "14", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", "-metadata", "title=实验二网站演示 · 1440p高清版",
               "-metadata", "comment=原生2560×1440网页录制；无损PNG采集后单次编码；无声无字幕", str(destination)]
    with (WORK / "render.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=WORK, stdout=subprocess.DEVNULL, stderr=log)
    result.check_returncode()
    audit = read(WORK / "recording-audit.json")
    audit.update(capture_start_epoch=origin, source_frames=len(selected), final_video=str(destination), sha256=sha(destination))
    write(WORK / "recording-audit.json", audit)
    print(f"Rendered {destination}", flush=True)


def verify():
    audit = read(WORK / "recording-audit.json")
    destination = Path(audit["final_video"])
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-v", "error", "-i", str(destination), "-f", "null", "-"]
    result = subprocess.run(command, capture_output=True, encoding="utf-8", errors="replace")
    (WORK / "decode.log").write_text(result.stderr, encoding="utf-8")
    result.check_returncode()
    assert not result.stderr.strip(), result.stderr
    video = cv2.VideoCapture(str(destination))
    width, height, fps, count = [video.get(k) for k in [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT]]
    assert (width, height, fps, count) == (2560, 1440, 30, 1800), (width, height, fps, count)
    thumbs = []
    samples = []
    for shot in audit["shots"]:
        second = (audit["started_epoch"] + shot["at_seconds"] - audit["capture_start_epoch"]) / 1.5 + .15
        second = min(59.9, max(0, second))
        video.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
        ok, picture = video.read()
        assert ok
        blue, green, red = map(int, picture[10, 10])
        assert not (red > 200 and blue > 200 and green < 70), "Start marker leaked into the final video"
        filename = WORK / f"final-{shot['scene']}.png"
        cv2.imencode(".png", picture)[1].tofile(str(filename))
        thumb = cv2.resize(picture, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.putText(thumb, f"{second:05.2f}s", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 2)
        thumbs.append(thumb)
        samples.append({"scene": shot["scene"], "second": second, "frame": str(filename)})
    video.release()
    while len(thumbs) % 3:
        thumbs.append(np.full((360, 640, 3), 245, dtype=np.uint8))
    sheet = np.vstack([np.hstack(thumbs[i:i + 3]) for i in range(0, len(thumbs), 3)])
    cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(str(WORK / "contact-sheet.jpg"))
    probe = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(destination)], capture_output=True, encoding="utf-8", errors="replace")
    assert "Audio:" not in probe.stderr and "Subtitle:" not in probe.stderr
    report = {"video": str(destination), "width": int(width), "height": int(height), "fps": fps, "frame_count": int(count),
              "duration_seconds": count / fps, "audio_streams": 0, "subtitle_streams": 0, "full_decode": "passed",
              "bytes": destination.stat().st_size, "sha256": sha(destination), "samples": samples,
              "browser_errors": audit["browser_errors"], "http_errors": audit["http_errors"],
              "live_computations": len(audit["runs"]), "capture_source": audit["capture"]}
    assert len(samples) == 18 and report["live_computations"] == 6
    assert not audit["browser_errors"] and not audit["http_errors"]
    write(WORK / "verification.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "probe", "record", "render", "verify"], default="all")
    args = parser.parse_args()
    if args.stage == "probe":
        record(probe=True)
    if args.stage in ["all", "record"]:
        record()
    if args.stage in ["all", "render"]:
        render()
    if args.stage in ["all", "verify"]:
        verify()
