"""Record the real local lab-two UI and render a silent demo with optional captions."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import cv2
import imageio_ffmpeg
from playwright.sync_api import expect, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "qa" / "lab02-demo"
DESTINATION = ROOT / "deliverables" / "lab02"
NO_CAPTIONS = False
VIEWPORT = {"width": 1600, "height": 900}
MODULES = [
    ("modeling", "知识表示与建模"),
    ("deterministic", "确定性推理"),
    ("uncertain", "不确定性推理"),
    ("optimization", "算法优化与对比实验"),
    ("hybrid", "混合推理与创新验证"),
]


def inspect(url: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome")
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=1)
        for module, title in MODULES:
            page.goto(f"{url}/#/labs/lab02/{module}")
            expect(page.get_by_role("heading", name=title, exact=True)).to_be_visible()
            expect(page.locator(".header-status")).to_contain_text("已就绪")
            if module != "optimization":
                button_name = "构建当前案例的谓词事实" if module == "modeling" else "运行本节实验"
                with page.expect_response(lambda response: response.url.endswith("/api/labs/lab02/run") and response.request.method == "POST") as pending:
                    page.get_by_role("button", name=button_name, exact=True).click()
                response = pending.value
                if not response.ok:
                    raise RuntimeError(response.text())
                result = response.json()
                (OUTPUT / f"{module}-preflight.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                expect(page.get_by_role("button", name=button_name, exact=True)).to_be_enabled()
            else:
                expect(page.get_by_role("button", name="确定性优化", exact=True)).to_be_visible()
            page.wait_for_timeout(350)
            (OUTPUT / f"{module}-dom.txt").write_text(page.locator("body").inner_text(), encoding="utf-8")
            page.screenshot(path=str(OUTPUT / f"{module}-inspect.png"), full_page=True)
            controls = page.locator("h2, h3, summary, [role=tab], .task-result").evaluate_all("elements => elements.map(element => ({tag: element.tagName, text: element.innerText.slice(0, 450), y: Math.round(element.getBoundingClientRect().top + scrollY)}))")
            print(json.dumps({"module": module, "controls": controls}, ensure_ascii=False), flush=True)
        browser.close()


def wait_until(page, started: float, second: float) -> None:
    remaining = second - (time.perf_counter() - started)
    if remaining < -4.0:
        raise RuntimeError(f"Recording fell behind at {second}s by {-remaining:.2f}s")
    if remaining > 0:
        page.wait_for_timeout(remaining * 1000)


def move_to(page, locator) -> None:
    expect(locator).to_be_visible()
    bounds = locator.bounding_box()
    if bounds is None:
        raise RuntimeError("Cannot locate the target control")
    if bounds["y"] < 80 or bounds["y"] + bounds["height"] > VIEWPORT["height"] - 25:
        scroll_to(page, locator, 150)
        bounds = locator.bounding_box()
    destination = (bounds["x"] + bounds["width"] / 2, bounds["y"] + bounds["height"] / 2)
    origin = page.evaluate("window.demoPointerPosition || {x: 1200, y: 100}")
    for step in range(1, 17):
        fraction = step / 16
        eased = fraction * fraction * (3 - 2 * fraction)
        page.mouse.move(origin["x"] + (destination[0] - origin["x"]) * eased, origin["y"] + (destination[1] - origin["y"]) * eased)
        page.wait_for_timeout(8)


def click(page, locator) -> None:
    move_to(page, locator)
    page.wait_for_timeout(110)
    locator.click()
    page.wait_for_timeout(120)


def scroll_to(page, locator, top: int = 100) -> None:
    locator.evaluate("(element, top) => window.scrollTo({top: Math.max(0, element.getBoundingClientRect().top + scrollY - top), behavior: 'smooth'})", top)
    page.wait_for_timeout(650)


def directory(page) -> None:
    click(page, page.locator('.course-navigation > a[href="#/labs/lab02"]'))
    expect(page.locator(".module-row")).to_have_count(5)


def enter(page, title: str) -> None:
    click(page, page.get_by_role("link", name=title, exact=True))
    expect(page.get_by_role("heading", name=title, exact=True)).to_be_visible()


def run_task(page, module: str, audit: dict) -> None:
    name = "构建当前案例的谓词事实" if module == "modeling" else "运行本节实验"
    with page.expect_response(lambda response: response.url.endswith("/api/labs/lab02/run") and response.request.method == "POST") as pending:
        click(page, page.get_by_role("button", name=name, exact=True))
    response = pending.value
    if not response.ok:
        raise RuntimeError(response.text())
    result = response.json()
    audit["live_results"][module] = result
    expect(page.get_by_role("button", name=name, exact=True)).to_be_enabled()
    if module == "deterministic":
        assert result["proved"] and result["equivalent"] and result["forward_proof"]
    if module == "hybrid":
        assert len(result["decisions"]) == 3
        assert len({decision["score"] for decision in result["decisions"]}) == 1


def capture(page, name: str, started: float, audit: dict) -> None:
    page.screenshot(path=str(OUTPUT / f"recorded-{name}.png"))
    audit["shots"].append({"scene": name, "at_seconds": round(time.perf_counter() - started, 3), "url": page.url})
    print(f"Recorded {name}", flush=True)


def record(url: str) -> tuple[Path, dict]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw_directory = OUTPUT / "raw"
    audit = {"url": url, "live_results": {}, "shots": [], "browser_errors": [], "http_errors": []}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome")
        context = browser.new_context(viewport=VIEWPORT, device_scale_factor=1, locale="zh-CN", record_video_dir=str(raw_directory), record_video_size=VIEWPORT)
        page = context.new_page()
        page.set_default_timeout(15000)
        page.on("pageerror", lambda error: audit["browser_errors"].append(str(error)))
        page.on("response", lambda response: audit["http_errors"].append({"status": response.status, "url": response.url}) if response.status >= 400 else None)
        page.goto(f"{url}/#/labs/lab02/optimization")
        expect(page.get_by_role("button", name="确定性优化", exact=True)).to_be_visible()
        page.locator('.course-navigation > a[href="#/labs/lab02"]').click()
        expect(page.locator(".module-row")).to_have_count(5)
        expect(page.locator(".header-status")).to_contain_text("已就绪")
        page.evaluate(r"""() => {
            const style = document.createElement('style');
            style.textContent = '* {cursor: none !important;} #demo-pointer {position:fixed; left:0; top:0; width:27px; height:32px; z-index:2147483647; pointer-events:none; filter:drop-shadow(1px 2px 2px #0006);} .demo-click {position:fixed; width:42px; height:42px; margin:-21px; border:3px solid #ab4d26; border-radius:50%; pointer-events:none; z-index:2147483646; animation:demo-ring .6s ease-out forwards;} @keyframes demo-ring {from{transform:scale(.35);opacity:.95} to{transform:scale(1.2);opacity:0}}';
            document.head.append(style);
            const pointer = document.createElement('div'); pointer.id = 'demo-pointer';
            pointer.innerHTML = '<svg viewBox="0 0 27 32"><path d="M3 2 L3 25 L9 19 L14 29 L18 27 L13 18 L23 18 Z" fill="white" stroke="#3b2c24" stroke-width="1.6" stroke-linejoin="round"/></svg>';
            document.body.append(pointer);
            document.addEventListener('mousemove', event => {window.demoPointerPosition = {x:event.clientX,y:event.clientY}; pointer.style.transform = 'translate('+event.clientX+'px,'+event.clientY+'px)';});
            document.addEventListener('mousedown', event => {const ring = document.createElement('div'); ring.className='demo-click'; ring.style.left=event.clientX+'px'; ring.style.top=event.clientY+'px'; document.body.append(ring); setTimeout(()=>ring.remove(),650);});
            const marker = document.createElement('div'); marker.id='demo-sync'; marker.style.cssText='position:fixed;left:0;top:0;width:24px;height:24px;background:rgb(255,0,255);z-index:2147483647'; document.body.append(marker);
        }""")
        page.mouse.move(1360, 145)
        page.wait_for_timeout(1000)
        page.evaluate("document.getElementById('demo-sync').remove()")
        started = time.perf_counter()
        capture(page, "overview", started, audit)
        wait_until(page, started, 3.5)
        enter(page, "知识表示与建模")
        wait_until(page, started, 5.1)
        run_task(page, "modeling", audit)
        wait_until(page, started, 6.6)
        scroll_to(page, page.get_by_text("一阶谓词与来源（47 条）", exact=True), 255)
        capture(page, "modeling", started, audit)
        wait_until(page, started, 10.9)
        directory(page)
        wait_until(page, started, 12.0)
        enter(page, "确定性推理")
        wait_until(page, started, 13.5)
        run_task(page, "deterministic", audit)
        wait_until(page, started, 15.0)
        scroll_to(page, page.locator(".task-result"), 105)
        capture(page, "deterministic", started, audit)
        wait_until(page, started, 21.9)
        directory(page)
        wait_until(page, started, 23.0)
        enter(page, "不确定性推理")
        wait_until(page, started, 24.5)
        run_task(page, "uncertain", audit)
        wait_until(page, started, 25.9)
        scroll_to(page, page.locator(".task-result"), 100)
        capture(page, "uncertain", started, audit)
        wait_until(page, started, 33.3)
        directory(page)
        wait_until(page, started, 34.4)
        enter(page, "算法优化与对比实验")
        expect(page.get_by_role("button", name="确定性优化", exact=True)).to_be_visible()
        wait_until(page, started, 36.0)
        scroll_to(page, page.locator(".chart-grid"), 110)
        capture(page, "comparison-charts", started, audit)
        wait_until(page, started, 38.9)
        scroll_to(page, page.locator(".method-tabs"), 110)
        click(page, page.get_by_role("button", name="确定性优化", exact=True))
        expect(page.get_by_role("region", name="确定性优化实测").locator("tbody tr")).to_have_count(3)
        scroll_to(page, page.get_by_role("region", name="确定性优化实测"), 170)
        capture(page, "optimization", started, audit)
        wait_until(page, started, 44.4)
        directory(page)
        wait_until(page, started, 45.5)
        enter(page, "混合推理与创新验证")
        wait_until(page, started, 47.3)
        run_task(page, "hybrid", audit)
        wait_until(page, started, 48.8)
        scroll_to(page, page.locator(".task-result"), 115)
        capture(page, "hybrid", started, audit)
        wait_until(page, started, 54.0)
        move_to(page, page.get_by_role("table", name="实验二混合策略对照").get_by_text("保留结构约束", exact=True))
        wait_until(page, started, 60.5)
        video = page.video
        context.close()
        raw_path = Path(video.path())
        browser.close()
    if audit["browser_errors"] or audit["http_errors"]:
        raise RuntimeError(json.dumps(audit, ensure_ascii=False))
    audit["raw_video"] = str(raw_path.relative_to(ROOT))
    (OUTPUT / "recording-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return raw_path, audit


CAPTIONS = [
    (0, 3.5, "实验总览", "实验二：确定性、不确定性与混合推理"),
    (3.5, 6.8, "01 / 知识建模", "从同一个虚构股权案例出发，构建一阶谓词事实。"),
    (6.8, 10.9, "01 / 知识建模", "每条事实保留数据来源，并由27条结构规则组织推理。"),
    (10.9, 15.6, "02 / 确定性推理", "查询 P1 是否控制 C3，点击运行确定性推理。"),
    (15.6, 21.9, "02 / 确定性推理", "正反向结论一致；证明路径逐层展示规则和输入事实。"),
    (21.9, 26.5, "03 / 不确定性推理", "再对同一企业运行 Bayes 与 Mamdani 两种方法。"),
    (26.5, 30.5, "03 / 不确定性推理", "同一输入：Bayes {bayes}，Mamdani {fuzzy}。"),
    (30.5, 33.3, "03 / 不确定性推理", "模糊评分不是概率，单个案例也不能代表准确率。"),
    (33.3, 38.9, "04 / 优化与对比", "对比正式实验结果，查看复核效果与结构约束。"),
    (38.9, 44.4, "04 / 优化与对比", "切换“确定性优化”，核对结论一致后的实测耗时。"),
    (44.4, 49.5, "05 / 混合推理", "运行混合推理，对照不锁定、结构锁定和校准复核。"),
    (49.5, 53.5, "05 / 混合推理", "原始分数相同，结构锁定改变复核行动而非分数。"),
    (53.5, 57.0, "05 / 混合推理", "校准保留证明，也可能暂缓复核；收益与误报都需实测。"),
    (57.0, 60.0, "演示结束", "用可追溯证据连接证明、评分与行动。"),
]


def timestamp(seconds: float, ass: bool = False) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    whole, fraction = divmod(remainder, 1000)
    if ass:
        return f"{hours}:{minutes:02}:{whole:02}.{fraction // 10:02}"
    return f"{hours:02}:{minutes:02}:{whole:02},{fraction:03}"


def subtitles(audit: dict) -> Path:
    values = audit["live_results"]["uncertain"]
    scores = {"bayes": f'{values["bayes"]["score"]:.4f}', "fuzzy": f'{values["fuzzy"]["score"]:.4f}'}
    ass_path = OUTPUT / "captions.ass"
    lines = [
        "[Script Info]", "ScriptType: v4.00+", "PlayResX: 1920", "PlayResY: 1080", "WrapStyle: 2", "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Caption,Microsoft YaHei,34,&H00FFFFFF,&H00FFFFFF,&H00251F1B,&H80251F1B,0,0,0,0,100,100,0,0,1,2,1,2,60,60,48,1", "",
        "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    srt = []
    for index, (start, end, _chapter, text) in enumerate(CAPTIONS, 1):
        caption = text.format(**scores)
        lines.append(f"Dialogue: 0,{timestamp(start, True)},{timestamp(end, True)},Caption,,0,0,0,,{caption}")
        srt.append(f"{index}\n{timestamp(start)} --> {timestamp(end)}\n{caption}\n")
    ass_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    (DESTINATION / "第31组_实验二_1分钟Demo.srt").write_text("\n".join(srt), encoding="utf-8-sig")
    return ass_path


def find_start(raw_path: Path) -> float:
    capture = cv2.VideoCapture(str(raw_path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    seen_marker = False
    for frame_number in range(round(fps * 15)):
        valid, frame = capture.read()
        if not valid:
            break
        blue, green, red = frame[10, 10]
        marker = int(red) > 200 and int(blue) > 200 and int(green) < 70
        if seen_marker and not marker:
            capture.release()
            return frame_number / fps
        seen_marker = seen_marker or marker
    capture.release()
    raise RuntimeError("Cannot find the recording synchronization marker")


def render(raw_path: Path, audit: dict) -> Path:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    source = cv2.VideoCapture(str(raw_path))
    source_size = (int(source.get(cv2.CAP_PROP_FRAME_WIDTH)), int(source.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    source.release()
    if source_size != (VIEWPORT["width"], VIEWPORT["height"]):
        raise RuntimeError("Re-record the 16:9 website viewport without --render-only; the old footage reserves space for a footer.")
    if not NO_CAPTIONS:
        subtitles(audit)
    start = find_start(raw_path)
    destination = DESTINATION / "第31组_实验二_1分钟Demo.mp4"
    staging = OUTPUT / "rendered-demo.mp4"
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-y", "-ss", f"{start:.3f}", "-i", str(raw_path),
        "-vf", "fps=30,scale=1920:1080:flags=lanczos" + ("" if NO_CAPTIONS else ",ass=captions.ass"),
        "-t", "60", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-metadata", "title=实验二 1分钟网站演示", str(staging),
    ]
    completed = subprocess.run(command, cwd=OUTPUT, capture_output=True, encoding="utf-8", errors="replace")
    (OUTPUT / "render.log").write_text(completed.stderr, encoding="utf-8")
    completed.check_returncode()
    audit["trim_start_seconds"] = start
    audit["output"] = str(destination.relative_to(ROOT))
    validate(staging, audit)
    staging.replace(destination)
    audit["composition"] = {"website_fills_frame": True, "subtitles": not NO_CAPTIONS,
                            "subtitle_position": None if NO_CAPTIONS else "overlaid on webpage above lower edge",
                            "footer_bar": False, "chapter_labels": False, "group_labels": False}
    (OUTPUT / "recording-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def validate(destination: Path, audit: dict) -> None:
    capture = cv2.VideoCapture(str(destination))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    assert (width, height, count) == (1920, 1080, 1800), (width, height, count)
    assert abs(fps - 30) < 0.01
    samples = []
    for second in (2, 9, 18, 29, 37.5, 42.5, 51.5, 58.5):
        capture.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
        valid, frame = capture.read()
        assert valid, second
        assert frame[-60:, :120].mean() > 180 and frame[-60:, -120:].mean() > 180, "Unexpected footer strip"
        name = OUTPUT / f"final-{second:04.1f}s.png"
        encoded, image = cv2.imencode(".png", frame)
        assert encoded
        image.tofile(str(name))
        thumbnail = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.putText(thumbnail, f"{second:04.1f}s", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        samples.append(thumbnail)
    capture.release()
    if not NO_CAPTIONS:
        subtitle_text = (OUTPUT / "captions.ass").read_text(encoding="utf-8-sig")
        assert "第31组" not in subtitle_text and "Chapter" not in subtitle_text
        assert subtitle_text.count("Dialogue:") == len(CAPTIONS)
    import numpy as np
    sheet = np.vstack([np.hstack(samples[offset:offset + 2]) for offset in range(0, len(samples), 2)])
    encoded, image = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])
    assert encoded
    image.tofile(str(OUTPUT / "contact-sheet.jpg"))
    verification = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(destination), "-f", "null", "-"], capture_output=True, encoding="utf-8", errors="replace")
    verification.check_returncode()
    assert "Audio:" not in verification.stderr
    assert "Subtitle:" not in verification.stderr
    (OUTPUT / "decode-check.log").write_text(verification.stderr, encoding="utf-8")
    audit["validation"] = {"width": width, "height": height, "fps": fps, "frames": count, "duration_seconds": count / fps, "audio_streams": 0, "full_decode_passed": True, "size_bytes": destination.stat().st_size, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
    print(json.dumps(audit["validation"], ensure_ascii=False), flush=True)
def main() -> None:
    global OUTPUT, DESTINATION, NO_CAPTIONS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--work-dir", type=Path, default=OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DESTINATION)
    parser.add_argument("--no-captions", action="store_true")
    args = parser.parse_args()
    OUTPUT, DESTINATION = args.work_dir.resolve(), args.output_dir.resolve()
    OUTPUT.relative_to(ROOT)
    DESTINATION.relative_to(ROOT)
    NO_CAPTIONS = args.no_captions
    if urlparse(args.url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("The recording target must be a local loopback server.")
    if args.inspect:
        inspect(args.url.rstrip("/"))
    elif args.render_only:
        audit = json.loads((OUTPUT / "recording-audit.json").read_text(encoding="utf-8"))
        print(render(ROOT / audit["raw_video"], audit))
    else:
        raw_path, audit = record(args.url.rstrip("/"))
        print(render(raw_path, audit))


if __name__ == "__main__":
    main()
