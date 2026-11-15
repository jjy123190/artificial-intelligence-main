"""Task-local input flow: no cross-module navigation or implicit inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from playwright.sync_api import sync_playwright, expect
from scripts.check_web import isolated_server

OUTPUT = ROOT / "outputs/qa/task-input-flow"


def source_hashes():
    files = list((ROOT / "data/raw").glob("*.csv"))
    files += [file for file in (ROOT / "deliverables").rglob("*") if file.is_file()]
    return {str(file.relative_to(ROOT)): hashlib.sha256(file.read_bytes()).hexdigest() for file in files}


def run(url):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    checks, errors, images = [], [], []

    def passed(name):
        checks.append(name)
        print("PASS " + name, flush=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)

        def start(viewport, module="deterministic"):
            context = browser.new_context(viewport=viewport, reduced_motion="reduce")
            page = context.new_page()
            posts = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("request", lambda request: posts.append((request.url, request.post_data_json)) if request.method == "POST" else None)
            fragment = "#/labs/lab02/" + module
            page.goto(url + "/" + fragment)
            expect(page.get_by_role("button", name="运行本节实验", exact=True)).to_be_enabled(timeout=30000)
            return context, page, posts, fragment

        def stable(page, fragment):
            expect(page).to_have_url(url + "/" + fragment)
            assert page.locator("#main-content").count() == 1
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1"), page.viewport_size
            expect(page.locator("#main-content .task-links a")).to_have_count(0)

        def open_panel(page, label):
            button = page.get_by_role("button", name=label, exact=True)
            button.focus()
            page.keyboard.press("Enter")
            panel = page.get_by_role("region", name=label, exact=True)
            expect(panel).to_be_visible()
            expect(panel.get_by_role("heading", name=label, exact=True)).to_be_focused()
            expect(button).to_have_attribute("aria-expanded", "true")
            expect(page.get_by_role("button", name="运行本节实验", exact=True)).to_be_disabled()
            return panel

        def cancel(page, panel, label):
            panel.get_by_role("button", name="取消", exact=True).click()
            expect(page.locator(".task-input-panel")).to_have_count(0)
            expect(page.get_by_role("button", name=label, exact=True)).to_be_focused()

        def execute(page):
            with page.expect_response(lambda response: response.url.endswith("/api/labs/lab02/run") and response.request.method == "POST", timeout=30000) as response:
                page.get_by_role("button", name="运行本节实验", exact=True).click()
            assert response.value.ok, response.value.text()
            expect(page.locator(".task-result")).to_be_visible()
            return response.value.request.post_data_json

        layouts = {
            "laptop": {"width": 1508, "height": 768},
            "tablet": {"width": 1024, "height": 900},
            "mobile": {"width": 390, "height": 844},
            "narrow": {"width": 320, "height": 900},
            "zoom200": {"width": 754, "height": 768},
        }
        for layout, viewport in layouts.items():
            context, page, posts, fragment = start(viewport)
            page.get_by_label("实验二推理方法", exact=True).select_option("backward")
            page.get_by_label("实验二查询主体", exact=True).select_option("P3")
            page.get_by_label("实验二目标企业", exact=True).select_option("C4")
            original = execute(page)
            panel = open_panel(page, "选择案例")
            panel.get_by_label("本次实验案例", exact=True).select_option("extended")
            cancel(page, panel, "选择案例")
            stable(page, fragment)
            panel = open_panel(page, "选择案例")
            expect(panel.get_by_label("本次实验案例", exact=True)).to_have_value("base")
            panel.get_by_label("本次实验案例", exact=True).select_option("extended")
            panel.get_by_role("button", name="使用此案例，继续实验", exact=True).click()
            expect(page.locator(".task-input-panel")).to_have_count(0)
            expect(page.get_by_role("button", name="选择案例", exact=True)).to_be_focused()
            expect(page.locator(".workspace-change")).to_be_visible()
            expect(page.get_by_label("实验二推理方法", exact=True)).to_have_value("backward")
            expect(page.get_by_label("实验二查询主体", exact=True)).to_have_value("P3")
            expect(page.get_by_label("实验二目标企业", exact=True)).to_have_value("C4")
            stable(page, fragment)
            assert len(posts) == 1, posts
            extended = execute(page)
            assert len(extended["case"]["entities"]) == 29 and len(original["case"]["entities"]) == 16
            panel = open_panel(page, "调整目标企业输入")
            expect(panel.locator(".task-input-target")).to_contain_text("C4")
            original_value = panel.locator("#holding-value").input_value()
            edge = int(panel.locator("#task-holding-edge").input_value())
            panel.locator("#holding-value").fill("")
            expect(panel.get_by_role("button", name="应用并返回实验", exact=True)).to_be_disabled()
            panel.locator("#holding-value").fill("101")
            expect(panel.get_by_role("alert")).to_be_visible()
            expect(panel.get_by_role("button", name="应用并返回实验", exact=True)).to_be_disabled()
            panel.locator("#holding-value").fill("10")
            cancel(page, panel, "调整目标企业输入")
            panel = open_panel(page, "调整目标企业输入")
            expect(panel.locator("#holding-value")).to_have_value(original_value)
            maximum = float(panel.get_by_role("slider", name="持股比例滑块", exact=True).get_attribute("max"))
            new_value = round(maximum / 2, 2)
            panel.locator("#holding-value").fill(str(new_value))
            panel.get_by_role("button", name="应用并返回实验", exact=True).click()
            expect(page.locator(".task-input-panel")).to_have_count(0)
            expect(page.get_by_role("button", name="调整目标企业输入", exact=True)).to_be_focused()
            expect(page.locator(".workspace-change")).to_be_visible()
            stable(page, fragment)
            assert len(posts) == 2, posts
            panel = open_panel(page, "资料完整性设置")
            expect(panel.get_by_role("combobox")).to_have_count(0)
            panel.get_by_role("switch", name="交易记录", exact=True).uncheck()
            cancel(page, panel, "资料完整性设置")
            panel = open_panel(page, "资料完整性设置")
            expect(panel.get_by_role("switch", name="交易记录", exact=True)).to_be_checked()
            panel.get_by_role("switch", name="交易记录", exact=True).uncheck()
            stable(page, fragment)
            if layout in ("laptop", "mobile"):
                page.evaluate("scrollTo(0, document.querySelector('.lab-task-form').getBoundingClientRect().top + scrollY - document.querySelector('.course-app-header').offsetHeight - 16)")
                image = OUTPUT / ("task-completeness-" + layout + ".png")
                page.screenshot(path=str(image))
                images.append(image)
            panel.get_by_role("button", name="应用并返回实验", exact=True).click()
            expect(page.get_by_role("button", name="资料完整性设置", exact=True)).to_be_focused()
            stable(page, fragment)
            assert len(posts) == 2, posts
            changed = execute(page)
            assert changed["task"] == "deterministic" and changed["source"] == "P3" and changed["target"] == "C4"
            assert abs(changed["case"]["holdings"][edge]["ratio"] - new_value / 100) < 1e-9
            assert changed["completeness"]["trades"] is False
            expect(page.locator(".workspace-change")).to_have_count(0)
            page.evaluate("location.hash = '#/labs/lab02'")
            expect(page.locator(".module-row")).to_have_count(5)
            page.evaluate("location.hash = " + json.dumps(fragment))
            expect(page.get_by_label("实验二推理方法", exact=True)).to_have_value("backward")
            expect(page.locator(".task-result")).to_be_visible()
            expect(page.locator(".workspace-change")).to_have_count(0)
            assert all(address.endswith("/api/labs/lab02/run") for address, payload in posts), posts
            passed(layout + ": all three inputs stay in task; keyboard/focus, cancel, validation, manual rerun and retained query/result work")
            context.close()

        context, page, posts, fragment = start(layouts["laptop"])
        panel = open_panel(page, "调整目标企业输入")
        panel.locator("#holding-value").fill("20")
        panel.get_by_role("button", name="应用并返回实验", exact=True).click()
        panel = open_panel(page, "选择案例")
        panel.get_by_label("本次实验案例", exact=True).select_option("extended")
        panel.get_by_role("button", name="使用此案例，继续实验", exact=True).click()
        dialog = page.get_by_role("dialog", name="切换案例？", exact=True)
        expect(dialog).to_be_visible()
        expect(dialog.get_by_role("button", name="保留草稿", exact=True)).to_be_focused()
        page.keyboard.press("Escape")
        expect(dialog).not_to_be_visible()
        expect(panel).to_be_visible()
        panel.get_by_role("button", name="取消", exact=True).click()
        panel = open_panel(page, "调整目标企业输入")
        expect(panel.locator("#holding-value")).to_have_value("20")
        cancel(page, panel, "调整目标企业输入")
        panel = open_panel(page, "选择案例")
        panel.get_by_label("本次实验案例", exact=True).select_option("extended")
        panel.get_by_role("button", name="使用此案例，继续实验", exact=True).click()
        dialog.get_by_role("button", name="切换案例", exact=True).click()
        expect(page.locator(".task-input-panel")).to_have_count(0)
        stable(page, fragment)
        assert not posts, posts
        passed("case replacement keeps unsaved-edit confirmation; Escape preserves edited inputs and accepting resumes current task")
        context.close()

        context, page, posts, fragment = start(layouts["laptop"])
        page.route("**/api/cases/extended", lambda route: route.fulfill(status=503, content_type="application/json", body=json.dumps({"detail": "模拟案例读取失败"})))
        panel = open_panel(page, "选择案例")
        panel.get_by_label("本次实验案例", exact=True).select_option("extended")
        panel.get_by_role("button", name="使用此案例，继续实验", exact=True).click()
        expect(panel.get_by_role("alert")).to_be_visible()
        expect(panel.get_by_role("button", name="使用此案例，继续实验", exact=True)).to_be_enabled()
        stable(page, fragment)
        page.unroute("**/api/cases/extended")
        panel.get_by_role("button", name="使用此案例，继续实验", exact=True).click()
        expect(page.locator(".task-input-panel")).to_have_count(0)
        stable(page, fragment)
        assert not posts, posts
        passed("case load errors remain in current task and retry succeeds without automatic inference")
        context.close()

        for module in ("uncertain", "hybrid"):
            context, page, posts, fragment = start(layouts["mobile"], module)
            for label in ("选择案例", "调整目标企业输入", "资料完整性设置"):
                panel = open_panel(page, label)
                stable(page, fragment)
                cancel(page, panel, label)
            panel = open_panel(page, "资料完整性设置")
            panel.get_by_role("switch", name="交易记录", exact=True).uncheck()
            panel.get_by_role("button", name="应用并返回实验", exact=True).click()
            assert not posts, posts
            payload = execute(page)
            assert payload["task"] == module and payload["completeness"]["trades"] is False
            stable(page, fragment)
            assert len(posts) == 1 and posts[0][0].endswith("/api/labs/lab02/run"), posts
            passed(module + ": task-local input controls do not redirect to another module or run /api/analyze")
            context.close()
        browser.close()

    assert not errors, errors
    assert hashes == source_hashes(), "Original CSV or deliverables changed"
    passed("original data and deliverables remain byte-for-byte unchanged")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "url": url, "checks": checks, "browser_errors": errors, "source_sha256": hashes, "screenshots": {image.name: hashlib.sha256(image.read_bytes()).hexdigest() for image in images}}
    (OUTPUT / "browser-checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(len(checks)) + " task input browser groups passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--isolated", action="store_true")
    args = parser.parse_args()
    if not args.url.startswith(("http://127.0.0.1:", "http://localhost:")):
        parser.error("Only local loopback servers are supported")
    if args.isolated:
        with isolated_server() as url:
            run(url)
    else:
        run(args.url.rstrip("/"))
