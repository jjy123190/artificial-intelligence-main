"""Record the current lab-one website as a native 1440p, one-minute demo.

Only the recording tab receives display zoom and a visible cursor. No app data,
UI source, or previous deliverables are changed. PNG compositor frames preserve
real operation timing; the final MP4 is encoded once, without audio or captions.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SIZE = {"width": 2560, "height": 1440}
DURATION = 60.0
FPS = 30
ZOOM = 1.6

PRESENTATION = r"""() => {
    const css = [...document.styleSheets].map(sheet => {
        try { return [...sheet.cssRules].map(rule => rule.cssText).join('\n'); }
        catch { return ''; }
    }).join('\n');
    const style = document.createElement('style');
    style.textContent = css.replace(/(-?\d*\.?\d+)(vw|dvh|vh)\b/g,
        (_, number, unit) => (Number(number) * (unit === 'vw' ? 16 : 9)) + 'px');
    document.head.append(style);
    document.body.style.zoom = '1.6';
    document.querySelector('.course-app-header')?.style.setProperty('padding-inline', '28px', 'important');
    // Preserve the graph's real data and layout; match canvas backing resolution
    // to its enlarged physical display instead of enlarging a low-res bitmap.
    setInterval(() => {
        document.querySelectorAll('.knowledge-canvas').forEach(element => {
            const cy = element._cyreg?.cy;
            const renderer = cy?.renderer();
            if (renderer && renderer.forcedPixelRatio !== 1.6) {
                renderer.forcedPixelRatio = 1.6;
                cy.resize(); cy.forceRender();
            }
        });
    }, 150);
    const cursorStyle = document.createElement('style');
    cursorStyle.textContent = `
        * { cursor: none !important; }
        #recording-pointer { position:fixed; left:0; top:0; width:36px; height:43px;
            pointer-events:none; z-index:2147483647; filter:drop-shadow(1px 2px 2px #0005); }
        .recording-click { position:fixed; width:56px; height:56px; margin:-28px;
            border:3px solid #ab4d26; border-radius:50%; pointer-events:none;
            z-index:2147483646; animation:recording-ring .5s ease-out forwards; }
        @keyframes recording-ring { from { transform:scale(.4); opacity:.9; }
            to { transform:scale(1.15); opacity:0; } }
    `;
    document.head.append(cursorStyle);
    const pointer = document.createElement('div'); pointer.id = 'recording-pointer';
    pointer.innerHTML = '<svg viewBox="0 0 27 32"><path d="M3 2 L3 25 L9 19 L14 29 L18 27 L13 18 L23 18 Z" fill="white" stroke="#3b2c24" stroke-width="1.6" stroke-linejoin="round"/></svg>';
    document.documentElement.append(pointer);
    document.addEventListener('mousemove', event => {
        window.recordingPointer = {x:event.clientX, y:event.clientY};
        pointer.style.transform = `translate(${event.clientX}px,${event.clientY}px)`;
    });
    document.addEventListener('mousedown', event => {
        const ring = document.createElement('div'); ring.className='recording-click';
        ring.style.left=event.clientX+'px'; ring.style.top=event.clientY+'px';
        document.documentElement.append(ring); setTimeout(()=>ring.remove(),550);
    });
    const marker = document.createElement('div'); marker.id='recording-sync';
    marker.style.cssText='position:fixed;left:0;top:0;width:24px;height:24px;background:rgb(255,0,255);z-index:2147483647';
    document.documentElement.append(marker);
}"""


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_hashes() -> dict:
    files = sorted((ROOT / "data" / "raw").glob("*.csv"))
    files += sorted((ROOT / "web" / "dist").rglob("*.js"))
    files += sorted((ROOT / "web" / "dist").rglob("*.css"))
    index = ROOT / "web" / "dist" / "index.html"
    if index.exists():
        files.append(index)
    return {str(path.relative_to(ROOT)): sha(path) for path in files}


def scroll(page, locator, top=180) -> None:
    locator.evaluate("(el, top) => window.scrollTo({top:Math.max(0,el.getBoundingClientRect().top+scrollY-top),behavior:'smooth'})", top)
    page.wait_for_timeout(650)


def move(page, locator) -> None:
    expect(locator).to_be_visible()
    box = locator.bounding_box()
    if box is None:
        raise RuntimeError("Target has no visible bounds")
    if box["y"] < 135 or box["y"] + box["height"] > SIZE["height"] - 20:
        scroll(page, locator, 200)
        box = locator.bounding_box()
    assert box is not None
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    origin = page.evaluate("window.recordingPointer || {x:2240,y:170}")
    for step in range(1, 11):
        f = step / 10
        f = f * f * (3 - 2 * f)
        page.mouse.move(origin["x"] + (x-origin["x"])*f, origin["y"] + (y-origin["y"])*f)
        page.wait_for_timeout(6)


def click(page, locator) -> None:
    move(page, locator)
    page.wait_for_timeout(75)
    locator.click()
    page.wait_for_timeout(100)


def choose(page, locator, value: str) -> None:
    move(page, locator)
    locator.focus()
    page.wait_for_timeout(140)
    locator.select_option(value)
    page.wait_for_timeout(160)


class Capture:
    def __init__(self, context, page, work: Path):
        self.work = work
        self.folder = work / "frames"
        self.folder.mkdir(exist_ok=False)
        self.frames = []
        self.cdp = context.new_cdp_session(page)

    def receive(self, event):
        data = base64.b64decode(event["data"])
        filename = self.folder / f"frame-{len(self.frames):06}.png"
        filename.write_bytes(data)
        self.frames.append({"file": filename.relative_to(self.work).as_posix(),
                            "timestamp": event["metadata"]["timestamp"], "bytes": len(data)})
        self.cdp.send("Page.screencastFrameAck", {"sessionId": event["sessionId"]})

    def start(self):
        self.cdp.on("Page.screencastFrame", self.receive)
        self.cdp.send("Page.startScreencast", {"format": "png", "maxWidth": SIZE["width"],
                                              "maxHeight": SIZE["height"], "everyNthFrame": 1})

    def stop(self):
        self.cdp.send("Page.stopScreencast")
        write(self.work / "capture-frames.json", self.frames)
        print(f"Captured {len(self.frames)} native 2560x1440 PNG frames", flush=True)


def record(url: str, work: Path) -> None:
    work.mkdir(parents=True, exist_ok=True)
    audit = {"url": url, "viewport": SIZE, "display_zoom": ZOOM, "target_seconds": DURATION,
             "playback_speed": 1, "capture": "Chrome DevTools PNG screencast of the real lab-one website",
             "audio": False, "subtitles": False, "added_titles": False,
             "browser_errors": [], "http_errors": [], "runs": [], "shots": [],
             "source_hashes_before": source_hashes()}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(viewport=SIZE, device_scale_factor=1, locale="zh-CN")
        page = context.new_page()
        page.set_default_timeout(15000)
        page.on("pageerror", lambda error: audit["browser_errors"].append(str(error)))
        page.on("response", lambda response: audit["http_errors"].append(
            {"url": response.url, "status": response.status}) if response.status >= 400 else None)
        # Load the lazy graph and fonts before the clock starts, without saving data.
        page.goto(url + "/#/labs/lab01/representation")
        expect(page.locator(".knowledge-canvas canvas").first).to_be_visible()
        expect(page.locator(".header-status")).to_contain_text("已就绪")
        page.evaluate("document.fonts.ready")
        page.evaluate(PRESENTATION)
        page.locator('.course-navigation > a[href="#/labs/lab01"]').click()
        expect(page.locator(".module-row")).to_have_count(4)
        page.mouse.move(2250, 240)
        page.wait_for_timeout(500)
        capture = Capture(context, page, work)
        capture.start()
        try:
            page.wait_for_timeout(900)
            page.evaluate("document.getElementById('recording-sync').remove()")
            started = time.perf_counter()
            audit["started_epoch"] = time.time()

            def at(second):
                lag = time.perf_counter() - started - second
                if lag > 5:
                    raise RuntimeError(f"Timeline is {lag:.2f}s late at {second}s")
                if lag < 0:
                    page.wait_for_timeout(-lag * 1000)

            def shot(name):
                epoch = time.time()
                page.screenshot(path=str(work / f"source-{name}.png"))
                (work / f"source-{name}.txt").write_text(page.locator("body").inner_text(), encoding="utf-8")
                audit["shots"].append({"scene": name, "epoch": epoch,
                                       "at_seconds": round(epoch-audit["started_epoch"], 3), "url": page.url})
                print(f"Scene {name}: {epoch-audit['started_epoch']:.2f}s", flush=True)

            def directory():
                click(page, page.locator('.course-navigation > a[href="#/labs/lab01"]'))
                expect(page.locator(".module-row")).to_have_count(4)

            def enter(title):
                click(page, page.get_by_role("link", name=title, exact=True))
                expect(page.get_by_role("heading", name=title, exact=True)).to_be_visible()

            def run(task):
                expect(page.get_by_label("实验一演示任务", exact=True)).to_have_value(task)
                with page.expect_response(lambda r: r.url.endswith("/api/labs/lab01/run")
                                          and r.request.method == "POST") as pending:
                    click(page, page.get_by_role("button", name="运行实验一任务", exact=True))
                response = pending.value
                if not response.ok:
                    raise RuntimeError(response.text())
                result = response.json()
                if task == "rules":
                    assert result["proved"] is True
                    assert result["stats"]["input_facts"] == 53
                    assert result["stats"]["total_facts"] == 135
                    assert result["stats"]["added_relations"] == 9
                    assert result["proof"]
                elif task == "penetration":
                    owner = next(o for o in result["owners"] if o["owner"] == "P1")
                    assert math.isclose(owner["ratio"], .342)
                    assert len(owner["paths"]) == 2
                else:
                    assert [(x[0], x[2]) for x in result["path"]] == [("P1", "C1"), ("C1", "C3")]
                audit["runs"].append({"task": task, "at_seconds": round(time.perf_counter()-started, 3),
                                      "result": result})
                write(work / f"live-{task}.json", result)
                expect(page.locator(".task-result")).to_be_visible()

            shot("01-overview")
            at(2.0); enter("知识表示与方法比较")
            expect(page.locator(".knowledge-canvas canvas").first).to_be_visible()
            scroll(page, page.locator(".knowledge-diagram"), 180)
            shot("02-holdings-graph")
            at(5.7); choose(page, page.get_by_label("实验一图谱关系", exact=True), "all")
            page.wait_for_timeout(350)
            audit["graph_density"] = page.locator(".knowledge-canvas canvas").evaluate_all(
                "es=>es.map(e=>({width:e.width,height:e.height,display_width:e.getBoundingClientRect().width,display_height:e.getBoundingClientRect().height}))")
            assert audit["graph_density"] and all(abs(c["width"]-c["display_width"]) < 3 for c in audit["graph_density"])
            shot("03-all-relations")
            at(9.0); scroll(page, page.locator(".method-tabs"), 155)
            click(page, page.get_by_role("button", name="产生式规则", exact=True))
            expect(page.get_by_role("table", name="实验一产生式规则")).to_be_visible()
            shot("04-rule-library")
            at(13.0); click(page, page.get_by_role("button", name="框架表示", exact=True))
            expect(page.get_by_role("table", name="实验一框架槽值")).to_be_visible()
            shot("05-company-frame")
            at(16.8); directory(); enter("知识增强与推理演示")
            expect(page.get_by_role("button", name="运行实验一任务", exact=True)).to_be_enabled()
            shot("06-task-input")
            at(19.5); run("rules")
            scroll(page, page.locator(".task-result"), 180)
            shot("07-control-proof")
            at(28.3); scroll(page, page.locator(".lab-task-form"), 180)
            choose(page, page.get_by_label("实验一演示任务", exact=True), "penetration")
            at(30.0); run("penetration")
            scroll(page, page.locator(".task-result"), 180)
            shot("08-equity-penetration")
            at(39.0); scroll(page, page.locator(".lab-task-form"), 180)
            choose(page, page.get_by_label("实验一演示任务", exact=True), "path")
            at(40.8); run("path")
            scroll(page, page.locator(".task-result"), 180)
            shot("09-shortest-path")
            at(48.5); directory(); enter("局限与发展方向")
            scroll(page, page.locator(".limits-list"), 185)
            shot("10-boundaries")
            at(54.5); directory()
            shot("11-overview-finish")
            at(60.7)
            page.screenshot(path=str(work / "source-last.png"))
            page.wait_for_timeout(200)
            audit["recorded_seconds"] = round(time.perf_counter()-started, 3)
            audit["source_hashes_after"] = source_hashes()
            assert audit["source_hashes_before"] == audit["source_hashes_after"], "App assets or CSV changed during recording"
            assert not audit["browser_errors"] and not audit["http_errors"]
            audit["recording_complete"] = True
        except Exception as error:
            audit["failure"] = str(error)
            page.screenshot(path=str(work / "failure.png"))
            raise
        finally:
            capture.stop()
            write(work / "recording-audit.json", audit)
            browser.close()


def render(work: Path) -> Path:
    audit = read(work / "recording-audit.json")
    assert audit.get("recording_complete"), "Cannot render an incomplete recording"
    frames = sorted(read(work / "capture-frames.json"), key=lambda item: item["timestamp"])
    frames = list({f["timestamp"]: f for f in frames}.values())
    seen_marker = False
    begin = None
    for i, frame in enumerate(frames):
        picture = cv2.imdecode(np.fromfile(work / frame["file"], dtype=np.uint8), cv2.IMREAD_COLOR)
        assert picture.shape[:2] == (1440, 2560)
        b, g, r = map(int, picture[10, 10])
        marked = r > 240 and b > 240 and g < 20
        if seen_marker and not marked:
            begin = i
            break
        seen_marker |= marked
    if begin is None:
        raise RuntimeError("No clean recording start after sync marker")
    origin = frames[begin]["timestamp"]
    assert audit["started_epoch"] + audit["recorded_seconds"] >= origin + DURATION
    selected = [f for f in frames[begin:] if f["timestamp"] < origin + DURATION]
    lines = ["ffconcat version 1.0"]
    for i, frame in enumerate(selected):
        end = selected[i+1]["timestamp"] if i+1 < len(selected) else origin+DURATION
        assert end > frame["timestamp"]
        lines += [f"file '{frame['file']}'", "option framerate 1000", f"duration {end-frame['timestamp']:.9f}"]
    lines += [f"file '{selected[-1]['file']}'", "option framerate 1000"]
    (work / "frames.ffconcat").write_text("\n".join(lines)+"\n", encoding="utf-8")
    destination = work / "第31组_实验一_Demo_网页版_1分钟.mp4"
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-y", "-safe", "0", "-i", "frames.ffconcat",
               "-vf", "setpts=PTS-STARTPTS,fps=30", "-t", "60", "-frames:v", "1800", "-an",
               "-c:v", "libx264", "-preset", "medium", "-crf", "14", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", "-metadata", "title=实验一网页版 Demo · 1分钟",
               "-metadata", "comment=原生2560x1440网页录制，原速，无声无字幕，未改写网站或实验数据",
               str(destination)]
    print("Encoding one-minute 1440p MP4...", flush=True)
    with (work / "render.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=work, stdout=subprocess.DEVNULL, stderr=log, check=True)
    audit.update(capture_start_epoch=origin, source_frames=len(selected),
                 final_video=str(destination), sha256=sha(destination))
    write(work / "recording-audit.json", audit)
    print(f"Rendered {destination}", flush=True)
    return destination


def verify(work: Path) -> None:
    audit = read(work / "recording-audit.json")
    destination = Path(audit["final_video"])
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run([ffmpeg, "-hide_banner", "-v", "error", "-i", str(destination),
                             "-f", "null", "-"], capture_output=True, encoding="utf-8", errors="replace")
    (work / "decode.log").write_text(result.stderr, encoding="utf-8")
    result.check_returncode()
    assert not result.stderr.strip(), result.stderr
    video = cv2.VideoCapture(str(destination))
    width, height, fps, count = [video.get(k) for k in [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT,
                                                       cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT]]
    assert (width, height, fps, count) == (2560, 1440, 30, 1800), (width, height, fps, count)
    thumbs, samples = [], []
    for shot in audit["shots"]:
        second = min(59.85, max(0, shot["epoch"] - audit["capture_start_epoch"] + .25))
        video.set(cv2.CAP_PROP_POS_MSEC, second*1000)
        ok, image = video.read()
        assert ok
        b, g, r = map(int, image[10, 10])
        assert not (r > 200 and b > 200 and g < 70), "Sync marker leaked into video"
        assert image.std() > 15, "Blank sample frame"
        image_path = work / f"final-{shot['scene']}.png"
        cv2.imencode(".png", image)[1].tofile(image_path)
        thumb = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.rectangle(thumb, (0, 0), (210, 26), (250, 246, 240), -1)
        cv2.putText(thumb, f"{second:05.2f}s {shot['scene'][:16]}", (8, 19), cv2.FONT_HERSHEY_SIMPLEX,
                    .45, (40, 40, 40), 1, cv2.LINE_AA)
        thumbs.append(thumb)
        samples.append({"scene": shot["scene"], "second": round(second, 3), "file": str(image_path)})
    for second in [0, 59.9]:
        video.set(cv2.CAP_PROP_POS_MSEC, second*1000)
        ok, image = video.read()
        assert ok
        b, g, r = map(int, image[10, 10])
        assert not (r > 200 and b > 200 and g < 70)
    video.release()
    while len(thumbs) % 3:
        thumbs.append(np.full((360, 640, 3), 245, dtype=np.uint8))
    sheet = np.vstack([np.hstack(thumbs[i:i+3]) for i in range(0, len(thumbs), 3)])
    cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(work / "contact-sheet.jpg")
    probe = subprocess.run([ffmpeg, "-hide_banner", "-i", str(destination)],
                           capture_output=True, encoding="utf-8", errors="replace")
    (work / "stream-info.txt").write_text(probe.stderr, encoding="utf-8")
    assert "Audio:" not in probe.stderr and "Subtitle:" not in probe.stderr
    assert len(audit["runs"]) == 3 and len(samples) == 11
    assert not audit["browser_errors"] and not audit["http_errors"]
    report = {"video": str(destination), "width": int(width), "height": int(height), "fps": fps,
              "frame_count": int(count), "duration_seconds": count/fps, "bytes": destination.stat().st_size,
              "sha256": sha(destination), "full_decode": "passed", "audio_streams": 0, "subtitle_streams": 0,
              "real_computations": [run["task"] for run in audit["runs"]],
              "source_data_unchanged": audit["source_hashes_before"] == audit["source_hashes_after"],
              "browser_errors": audit["browser_errors"], "http_errors": audit["http_errors"], "samples": samples}
    write(work / "verification.json", report)
    print(json.dumps({k:v for k,v in report.items() if k != "samples"}, ensure_ascii=False), flush=True)


def deliver(work: Path) -> None:
    report = read(work / "verification.json")
    source = Path(report["video"])
    assert report["full_decode"] == "passed" and sha(source) == report["sha256"]
    destinations = [ROOT / "deliverables" / "lab01" / source.name,
                    ROOT.parent / "人工智能实验2" / "实验一_Demo.mp4"]
    records = []
    for target in destinations:
        target.parent.mkdir(parents=True, exist_ok=True)
        backup = None
        if target.exists() and sha(target) != report["sha256"]:
            backup_dir = work / "previous-deliverables"
            backup_dir.mkdir(exist_ok=True)
            backup = backup_dir / (target.stem + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + target.suffix)
            shutil.copy2(target, backup)
        shutil.copy2(source, target)
        assert sha(target) == report["sha256"]
        records.append({"path": str(target), "sha256": sha(target), "backup": str(backup) if backup else None})
    write(work / "delivery.json", records)
    print(json.dumps(records, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["record", "render", "verify", "deliver", "all"])
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    work = args.work.resolve()
    if not work.is_relative_to(ROOT / "outputs" / "qa"):
        raise ValueError("Recording work must be within the project's outputs/qa directory")
    if args.action in ("record", "all"):
        record(args.url.rstrip("/"), work)
    if args.action in ("render", "all"):
        render(work)
    if args.action in ("verify", "all"):
        verify(work)
    if args.action == "deliver":
        deliver(work)


if __name__ == "__main__":
    main()