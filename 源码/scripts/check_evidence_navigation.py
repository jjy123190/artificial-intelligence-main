"""Proof-graph and evidence navigation stay inside deterministic tasks."""
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

OUTPUT = ROOT / "outputs/qa/evidence-navigation"


def run(url):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    checks, errors, images = [], [], []

    def passed(name):
        checks.append(name)
        print("PASS " + name, flush=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        for name, width, height in (("laptop", 1508, 768), ("tablet", 1024, 900), ("mobile", 390, 844), ("narrow", 320, 900), ("zoom200", 754, 768)):
            context = browser.new_context(viewport={"width": width, "height": height}, reduced_motion="reduce")
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            task_posts = []
            page.on("request", lambda request: task_posts.append(request.post_data_json) if request.method == "POST" and request.url.endswith("/api/labs/lab02/run") else None)
            page.goto(url + "/#/labs/lab02/deterministic")
            expect(page.get_by_role("button", name="运行本节实验", exact=True)).to_be_enabled(timeout=30000)
            page.get_by_label("实验二推理方法", exact=True).select_option("backward")
            with page.expect_response(lambda response: response.url.endswith("/api/labs/lab02/run") and response.request.method == "POST", timeout=30000) as response:
                page.get_by_role("button", name="运行本节实验", exact=True).click()
            assert response.value.ok and response.value.json()["proved"], response.value.text()
            expect(page.locator(".task-result")).to_be_visible()
            page.get_by_role("link", name="推导关系与证明演示", exact=True).click()
            expect(page).to_have_url(url + "/#/labs/lab02/deterministic/graph/C3")
            expect(page.locator(".graph-canvas")).to_be_visible(timeout=30000)
            graph_entry = page.get_by_role("link", name="查看所选实体推理证据", exact=True)
            expect(graph_entry).to_have_attribute("href", "#/labs/lab02/deterministic/entity/C3/evidence")
            expect(page.get_by_role("link", name="查看所选实体分析", exact=True)).to_have_count(0)
            graph_entry.click()
            expect(page).to_have_url(url + "/#/labs/lab02/deterministic/entity/C3/evidence")
            expect(page.get_by_role("heading", name="推理证据", exact=True)).to_be_visible()
            expect(page.locator(".decision-status,.score-row,.decision-compare")).to_have_count(0)
            expect(page.locator(".proof-tree")).to_be_visible(timeout=30000)
            expect(page.locator(".page-breadcrumb")).to_contain_text("确定性推理")
            footer = page.get_by_role("navigation", name="相关页面", exact=True)

            def evidence_footer(entity):
                expect(footer.get_by_role("link")).to_have_count(2)
                expect(footer.get_by_role("link", name="返回确定性推理", exact=True)).to_have_attribute("href", "#/labs/lab02/deterministic")
                expect(page.get_by_role("link", name="在关系图谱中查看证明", exact=True)).to_have_attribute("href", "#/labs/lab02/deterministic/graph/" + entity)
                expect(footer.get_by_role("link", name="验证与推理记录", exact=True)).to_have_attribute("href", "#/labs/lab02/deterministic/records")
                expect(footer.get_by_role("link", name="返回实验二全案例分析", exact=True)).to_have_count(0)
                expect(footer.get_by_role("link", name="分析设置", exact=True)).to_have_count(0)
                expect(footer.locator('a[href^="#/labs/lab02/hybrid/"]')).to_have_count(0)
                assert page.locator("#main-content").count() == 1
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1"), page.viewport_size
                for link in footer.get_by_role("link").all():
                    box = link.bounding_box()
                    assert box and box["height"] >= 44 and box["x"] >= -1 and box["x"] + box["width"] <= width + 1, box

            evidence_footer("C3")
            footer.scroll_into_view_if_needed()
            if name in ("laptop", "mobile"):
                image = OUTPUT / ("evidence-footer-" + name + ".png")
                page.screenshot(path=str(image))
                images.append(image)
            footer.get_by_role("link", name="返回确定性推理", exact=True).click()
            expect(page).to_have_url(url + "/#/labs/lab02/deterministic")
            expect(page.get_by_label("实验二推理方法", exact=True)).to_have_value("backward")
            expect(page.get_by_label("实验二查询主体", exact=True)).to_have_value("P1")
            expect(page.get_by_label("实验二目标企业", exact=True)).to_have_value("C3")
            expect(page.locator(".task-result")).to_be_visible()
            expect(page.locator(".workspace-change")).to_have_count(0)
            assert len(task_posts) == 1, task_posts
            page.get_by_role("link", name="查看目标企业推理证据", exact=True).click()
            evidence_footer("C3")
            page.get_by_role("link", name="在关系图谱中查看证明", exact=True).click()
            expect(page).to_have_url(url + "/#/labs/lab02/deterministic/graph/C3")
            expect(page.locator(".graph-canvas")).to_be_visible(timeout=30000)
            expect(page.locator(".page-breadcrumb")).to_contain_text("确定性推理")
            page.get_by_label("选择实体", exact=True).select_option("C6")
            expect(page).to_have_url(url + "/#/labs/lab02/deterministic/graph/C6")
            graph_entry = page.get_by_role("link", name="查看所选实体推理证据", exact=True)
            expect(graph_entry).to_have_attribute("href", "#/labs/lab02/deterministic/entity/C6/evidence")
            graph_entry.click()
            expect(page).to_have_url(url + "/#/labs/lab02/deterministic/entity/C6/evidence")
            evidence_footer("C6")
            expect(page.locator(".decision-status,.score-row,.decision-compare")).to_have_count(0)
            page.go_back()
            expect(page).to_have_url(url + "/#/labs/lab02/deterministic/graph/C6")
            page.go_back()
            expect(page).to_have_url(url + "/#/labs/lab02/deterministic/graph/C3")
            page.go_back()
            evidence_footer("C3")
            footer.get_by_role("link", name="验证与推理记录", exact=True).click()
            expect(page).to_have_url(url + "/#/labs/lab02/deterministic/records")
            expect(page.get_by_role("heading", name="验证与推理记录", exact=True)).to_be_visible()
            expect(page.locator(".page-breadcrumb")).to_contain_text("确定性推理")
            page.go_back()
            page.get_by_label("选择实体", exact=True).select_option("C6")
            expect(page).to_have_url(url + "/#/labs/lab02/deterministic/entity/C6/evidence")
            evidence_footer("C6")
            page.evaluate("location.hash = '#/entity/C3/evidence'")
            expect(page).to_have_url(url + "/#/labs/lab02/deterministic/entity/C3/evidence")
            evidence_footer("C3")
            assert len(task_posts) == 1, task_posts
            passed(name + ": task to proof graph to selected evidence stays deterministic; return preserves query/result, records and legacy URLs work")
            context.close()

        context = browser.new_context(viewport={"width": 1508, "height": 768}, reduced_motion="reduce")
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        inference = []
        page.on("request", lambda request: inference.append(request.url) if request.method == "POST" and request.url.endswith("/api/analyze") else None)
        page.goto(url + "/#/graph/C3")
        expect(page.locator(".graph-canvas")).to_be_visible(timeout=30000)
        expect(page.get_by_role("link", name="查看所选实体资料", exact=True)).to_have_attribute("href", "#/case/C3")
        expect(page.get_by_role("link", name="查看所选实体推理证据", exact=True)).to_have_count(0)
        page.get_by_role("button", name="整个案例", exact=True).click()
        canvas = page.locator(".graph-canvas")
        page.wait_for_function("document.querySelector('.graph-canvas')._cyreg.cy.nodes().length === 16")
        position = canvas.evaluate("el => el._cyreg.cy.getElementById('entity:C4').renderedPosition()")
        canvas.click(position=position)
        expect(page.get_by_label("选择实体", exact=True)).to_have_value("C4")
        page.get_by_role("link", name="查看所选实体资料", exact=True).click()
        expect(page).to_have_url(url + "/#/case/C4")
        expect(page.get_by_role("heading", name="实体资料", exact=True)).to_be_visible()
        assert not inference, inference
        passed("public graph node selection still opens raw entity data without running analysis")
        page.goto(url + "/#/labs/lab02/deterministic/graph/P1")
        expect(page.locator(".graph-canvas")).to_be_visible(timeout=30000)
        page.get_by_role("link", name="查看所选实体推理证据", exact=True).click()
        expect(page).to_have_url(url + "/#/labs/lab02/deterministic/entity/P1/evidence")
        page.locator(".person-holdings button").filter(has_text="华鑫控股集团").click()
        expect(page).to_have_url(url + "/#/labs/lab02/deterministic/entity/C1/evidence")
        expect(page.get_by_role("heading", name="推理证据", exact=True)).to_be_visible()
        expect(page.locator(".page-breadcrumb")).to_contain_text("确定性推理")
        expect(page.locator(".decision-status,.score-row,.decision-compare")).to_have_count(0)
        passed("natural person to invested company follows the same deterministic evidence context")
        context.close()

        context = browser.new_context()
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url + "/#/labs/lab02/hybrid/entity/C3")
        expect(page.locator(".decision-status")).to_be_visible(timeout=30000)
        footer = page.get_by_role("navigation", name="相关页面", exact=True)
        expect(footer.get_by_role("link", name="返回实验二全案例分析", exact=True)).to_have_attribute("href", "#/labs/lab02/hybrid/overview")
        expect(footer.get_by_role("link", name="分析设置", exact=True)).to_have_attribute("href", "#/labs/lab02/hybrid/settings")
        footer.get_by_role("link", name="分析设置", exact=True).click()
        expect(page.get_by_role("heading", name="分析设置", exact=True)).to_be_visible()
        expect(page.locator(".page-breadcrumb")).to_contain_text("混合推理与创新验证")
        page.evaluate("location.hash = '#/settings'")
        expect(page).to_have_url(url + "/#/labs/lab02/hybrid/settings")
        expect(page.get_by_label("推理方式", exact=True)).to_be_visible()
        page.evaluate("location.hash = '#/entity/P1'")
        expect(page).to_have_url(url + "/#/labs/lab02/hybrid/entity/P1")
        page.locator(".person-holdings button").filter(has_text="华鑫控股集团").click()
        expect(page).to_have_url(url + "/#/labs/lab02/hybrid/entity/C1")
        expect(page.get_by_role("heading", name="企业分析", exact=True)).to_be_visible()
        passed("hybrid settings, enterprise-analysis footer, person links and legacy analysis URLs keep their original meaning")
        context.close()
        browser.close()

    assert not errors, errors
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "url": url, "checks": checks, "browser_errors": errors, "screenshots": {image.name: hashlib.sha256(image.read_bytes()).hexdigest() for image in images}}
    (OUTPUT / "browser-checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(len(checks)) + " evidence navigation browser groups passed", flush=True)


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
