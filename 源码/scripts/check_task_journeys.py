"""Regressions for task-to-detail continuity and requirement-oriented navigation.

Uses a temporary database. Shared CSVs and formal deliverables are read-only.
"""
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

from playwright.sync_api import expect, sync_playwright
from scripts.check_web import isolated_server

OUTPUT = ROOT / "outputs/qa/course-review/after"


def protected_hashes():
    return {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for directory in (ROOT / "data/raw", ROOT / "deliverables")
        for path in directory.rglob("*") if path.is_file()
    }


def run(url):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    checks, errors, comparisons = [], [], []
    before = protected_hashes()

    def passed(name):
        checks.append(name)
        print("PASS " + name, flush=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(viewport={"width": 1508, "height": 768}, reduced_motion="reduce")
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        posts = []
        page.on("request", lambda request: posts.append(request.url) if request.method == "POST" else None)

        def go(fragment, title=None):
            page.evaluate("fragment => location.hash = fragment", fragment)
            expect(page).to_have_url(url + "/" + fragment)
            if title:
                expect(page.get_by_role("heading", name=title, exact=True)).to_be_visible(timeout=30000)

        def task():
            with page.expect_response(lambda r: r.url.endswith("/api/labs/lab02/run") and r.request.method == "POST", timeout=30000) as pending:
                page.get_by_role("button", name="运行本节实验", exact=True).click()
            assert pending.value.ok, pending.value.text()
            expect(page.locator(".task-result")).to_be_visible()
            return pending.value.json(), pending.value.request.post_data_json

        def detail():
            with page.expect_response(lambda r: r.url.endswith("/api/analyze") and r.request.method == "POST", timeout=30000) as pending:
                page.get_by_role("link", name="查看目标企业分析", exact=True).click()
            assert pending.value.ok, pending.value.text()
            expect(page.locator(".workspace-context")).to_contain_text("实验二结果", timeout=30000)
            return pending.value.json(), pending.value.request.post_data_json

        def compare(task_value, task_request, detail_value, detail_request):
            expected = next(d for d in task_value["decisions"] if d["method"] == "hybrid_locked")
            actual = detail_value["decisions"][task_request["target"]]
            for key in ("case", "completeness", "model_source", "threshold"):
                assert task_request[key] == detail_request[key], key
            assert abs(expected["score"] - actual["score"]) < 1e-12
            assert expected["review_required"] == actual["review_required"]
            comparisons.append({"threshold": task_request["threshold"], "model": task_request["model_source"],
                                "task_score": expected["score"], "detail_score": actual["score"],
                                "input_matches": True, "review_matches": True})

        page.goto(url + "/#/labs/lab02/hybrid")
        expect(page.get_by_role("button", name="运行本节实验", exact=True)).to_be_enabled(timeout=30000)
        model_options = page.get_by_label("实验二参数来源", exact=True).locator("option").evaluate_all("els => els.map(el => el.value)")
        trained = next((value for value in model_options if value != "manual"), "manual")
        page.get_by_label("实验二参数来源", exact=True).select_option(trained)
        page.get_by_label("实验二复核阈值", exact=True).select_option("0.7")
        value, request = task()
        expect(page.locator(".result-parameters")).to_contain_text("0.70")
        detailed, detail_request = detail()
        compare(value, request, detailed, detail_request)
        passed("hybrid task and company detail use identical model, threshold, case, scores and review action")

        page.get_by_role("link", name="分析设置", exact=True).click()
        expect(page.get_by_role("heading", name="分析设置", exact=True)).to_be_visible()
        page.get_by_label("参数来源", exact=True).select_option("manual")
        slider = page.locator("#threshold-range")
        slider.focus()
        slider.press("Home")
        for _ in range(16):
            slider.press("ArrowRight")
        expect(slider).to_have_value("0.85")
        page.get_by_role("link", name="返回混合推理任务", exact=True).click()
        expect(page.get_by_label("实验二复核阈值", exact=True)).to_have_value("0.85")
        expect(page.get_by_label("实验二参数来源", exact=True)).to_have_value("manual")
        expect(page.locator(".workspace-change")).to_be_visible()
        value, request = task()
        detailed, detail_request = detail()
        compare(value, request, detailed, detail_request)
        passed("settings changes return to the task, including thresholds outside the former 0.30–0.70 menu")

        go("#/labs/lab02/hybrid", "混合推理与创新验证")
        page.get_by_role("button", name="调整目标企业输入", exact=True).click()
        panel = page.locator(".task-input-panel")
        panel.locator("#holding-value").fill("10")
        panel.get_by_role("button", name="应用并返回实验", exact=True).click()
        expect(panel).to_have_count(0)
        value, request = task()
        detailed, detail_request = detail()
        compare(value, request, detailed, detail_request)
        passed("opening details after changing task inputs refreshes the old analysis snapshot")

        def change_parameters_before_reply(route):
            response = route.fetch()
            go("#/labs/lab02/hybrid", "混合推理与创新验证")
            page.get_by_label("实验二复核阈值", exact=True).select_option("0.4")
            route.fulfill(response=response)

        page.route("**/api/analyze", change_parameters_before_reply)
        with page.expect_response(lambda r:r.url.endswith("/api/analyze"), timeout=30000) as late:
            page.get_by_role("button", name="运行推理", exact=False).click()
        assert late.value.ok
        page.unroute("**/api/analyze")
        expect(page.get_by_label("实验二复核阈值", exact=True)).to_have_value("0.4")
        expect(page.locator(".workspace-change")).to_be_visible()
        passed("a delayed detail response cannot overwrite a newer task parameter selection")

        go("#/labs/lab02/deterministic", "确定性推理")
        page.get_by_label("实验二查询主体", exact=True).select_option("P3")
        value, _ = task()
        assert not value["proved"]
        page.get_by_role("link", name="查看目标企业推理证据", exact=True).click()
        expect(page.locator(".detail-query-context")).to_contain_text("Controls(P3, C3)")
        expect(page.locator(".detail-query-context")).to_contain_text("未证明")
        expect(page.locator("#claim-select")).to_have_value("")
        expect(page.locator(".entity-page-nav")).to_have_count(0)
        expect(page.locator("#main-content a[href*='/hybrid/']")).to_have_count(0)
        expect(page.locator("#main-content a[href$='/graph/C3']")).to_have_count(1)
        passed("an unproved task goal stays unproved in evidence and does not select an unrelated positive claim")

        page.get_by_role("link", name="返回确定性推理", exact=True).click()
        expect(page.get_by_label("实验二查询主体", exact=True)).to_have_value("P3")
        page.get_by_label("实验二查询主体", exact=True).select_option("P1")
        page.get_by_label("实验二目标企业", exact=True).select_option("C1")
        value, _ = task()
        assert value["proved"]
        page.get_by_role("link", name="查看目标企业推理证据", exact=True).click()
        expect(page.locator(".detail-query-context")).to_contain_text("Controls(P1, C1)")
        expect(page.locator(".proof-tree")).to_contain_text("王建国")
        page.get_by_role("link", name="验证与推理记录", exact=True).click()
        expect(page.get_by_role("combobox", name="控制主体", exact=True)).to_have_value("P1")
        expect(page.get_by_role("combobox", name="目标企业", exact=True)).to_have_value("C1")
        with page.expect_response(lambda r: r.url.endswith("/api/query"), timeout=30000) as pending:
            page.get_by_role("button", name="验证结论", exact=True).click()
        assert pending.value.ok and pending.value.json()["proved"]
        passed("positive proof and reverse verification retain the selected source and nondefault target")

        go("#/labs/lab02/uncertain", "不确定性推理")
        value, _ = task()
        assert value["model_source"] == "manual" and "单次预测" in value["timing_scope"]
        assert value["bayes"]["elapsed_ms"] >= 0 and value["fuzzy"]["elapsed_ms"] >= 0
        passed("both uncertainty timings disclose the prediction-call scope and retain the selected parameter source")

        page.close()
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        direct_posts = []
        page.on("request", lambda request: direct_posts.append(request.url) if request.method == "POST" else None)
        page.goto(url + "/#/labs/lab02/modeling")
        expect(page.get_by_label("实验二建模编辑对象", exact=True)).to_be_visible(timeout=30000)
        page.wait_for_load_state("networkidle")
        assert not direct_posts
        page.get_by_label("实验二建模编辑对象", exact=True).select_option("C7")
        page.get_by_role("button", name="调整目标企业输入", exact=True).click()
        expect(page.locator(".task-input-target")).to_contain_text("C7")
        page.get_by_role("button", name="取消", exact=True).click()
        expect(page).to_have_url(url + "/#/labs/lab02/modeling")
        expect(page.get_by_role("button", name="调整目标企业输入", exact=True)).to_be_focused()
        expect(page.locator(".task-bottom-nav a")).to_have_count(1)
        expect(page.locator("#main-content a[href*='/entity/']")).to_have_count(0)
        passed("modeling lets users select and edit an entity locally, with cancel and focus return")

        for fragment, title in (("#/labs/lab02/modeling/entity/C3/edit", "调整企业数据"),
                                ("#/labs/lab02/hybrid/settings", "分析设置")):
            page.goto(url + "/" + fragment)
            expect(page.get_by_role("heading", name=title, exact=True)).to_be_visible(timeout=30000)
            page.wait_for_load_state("networkidle")
        assert not direct_posts
        go("#/labs/lab02/modeling/entity/C3/edit", "调整企业数据")
        expect(page.locator(".entity-page-nav,.settings-shortcut")).to_have_count(0)
        expect(page.locator(".related-pages a")).to_have_count(1)
        expect(page.get_by_role("link", name="返回知识表示与建模", exact=True)).to_be_visible()
        passed("direct input editing and settings do not auto-run an unrelated analysis; editor returns to modeling")

        go("#/labs/lab02/deterministic", "确定性推理")
        page.get_by_label("实验二目标企业", exact=True).select_option("C7")
        page.get_by_label("实验二推理方法", exact=True).select_option("resolution")
        expect(page.get_by_role("button", name="资料完整性设置", exact=True)).to_have_count(0)
        expect(page.locator(".task-related")).to_have_count(0)
        page.get_by_role("button", name="调整 C3 持股输入", exact=True).click()
        expect(page.locator(".task-input-target")).to_contain_text("C3")
        page.get_by_role("button", name="取消", exact=True).click()
        value, _ = task()
        assert value["proved"] and value["goal"].replace(" ", "") == "Controls(P1,C3)", (value["proved"], value["goal"])
        passed("the independent resolution example edits C3 and hides unrelated full-case detail and completeness controls")

        go("#/labs/lab02/optimization", "算法优化与对比实验")
        expect(page.get_by_role("link", name="查看完整保存结果", exact=True)).to_have_count(0)
        go("#/labs/lab02/hybrid", "混合推理与创新验证")
        expect(page.locator("#main-content a[href$='/hybrid/settings']")).to_have_count(0)
        expect(page.locator(".task-detail-links a")).to_have_count(2)
        passed("optimization has one result surface and hybrid removes the duplicate settings shortcut")

        contrasts = page.evaluate("""() => {
          const style = getComputedStyle(document.documentElement);
          const ctx = document.createElement('canvas').getContext('2d', {willReadFrequently:true});
          function luminance(token) {
            ctx.fillStyle = style.getPropertyValue(token).trim(); ctx.fillRect(0,0,1,1);
            const rgb = [...ctx.getImageData(0,0,1,1).data].slice(0,3).map(v => {
              const c=v/255; return c <= .04045 ? c/12.92 : ((c+.055)/1.055)**2.4;
            });
            return .2126*rgb[0]+.7152*rgb[1]+.0722*rgb[2];
          }
          return [['--ink','--bg'],['--muted','--bg'],['--muted','--paper'],
                  ['--muted','--surface'],['--primary','--paper'],['--on-primary','--primary']]
            .map(([fg,bg])=>{const a=luminance(fg),b=luminance(bg);return {fg,bg,ratio:(Math.max(a,b)+.05)/(Math.min(a,b)+.05)};});
        }""")
        assert all(pair["ratio"] >= 4.5 for pair in contrasts), contrasts
        passed("body, secondary text, controls and primary-action theme pairs meet 4.5:1 contrast")

        for width, height, label in ((1508, 768, "laptop"), (1024, 900, "tablet"), (390, 844, "mobile"), (320, 740, "narrow"), (754, 768, "zoom200")):
            page.set_viewport_size({"width": width, "height": height})
            for module, title in (("modeling", "知识表示与建模"), ("deterministic", "确定性推理"), ("hybrid", "混合推理与创新验证")):
                page.goto(url + "/#/labs/lab02/" + module)
                # Hash-only navigation retains task state; reload for comparable initial screenshots.
                page.reload()
                expect(page.get_by_role("heading", name=title, exact=True)).to_be_visible(timeout=30000)
                page.wait_for_load_state("networkidle")
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
                assert page.locator("#main-content").count() == 1
                expect(page.locator(".course-header .requirement-label")).to_have_count(0)
                for button in page.locator(".task-links button").all():
                    assert button.bounding_box()["height"] >= 44
                if label == "laptop":
                    bounds = page.locator(".lab-task-form button.primary").bounding_box()
                    assert bounds and bounds["y"] + bounds["height"] < height, bounds
                page.screenshot(path=str(OUTPUT / (module + "-" + label + ".png")), full_page=True)
            passed(label + " has one main region, usable controls and no horizontal overflow" +
                   ("; primary task actions fit the initial viewport" if label == "laptop" else ""))

        assert protected_hashes() == before
        assert not errors, errors
        passed("source CSVs and retained formal deliverables remain byte-for-byte unchanged")
        browser.close()

    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "checks": checks,
              "unexpected_browser_errors": errors, "parameter_comparisons": comparisons,
              "contrast_checks": contrasts, "protected_files_sha256": before,
              "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (ROOT / "web/src").rglob("*") if p.is_file()}}
    (OUTPUT / "browser-checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(len(checks)) + " task journey groups passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--isolated", action="store_true")
    args = parser.parse_args()
    if not args.url.startswith(("http://127.0.0.1:", "http://localhost:")):
        parser.error("Only local loopback servers are supported")
    if args.isolated:
        with isolated_server() as local_url:
            run(local_url)
    else:
        run(args.url.rstrip("/"))
