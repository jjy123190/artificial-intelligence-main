"""Browser regression for shared variable guides and complete executable rule conditions."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from playwright.sync_api import sync_playwright, expect
from scripts.check_web import isolated_server

OUTPUT = ROOT / "outputs" / "qa" / "rule-notation"
VARIABLE = re.compile(r"\?[A-Za-z][A-Za-z0-9_]*")


def original_hashes():
    paths = list((ROOT / "data" / "raw").glob("*.csv"))
    paths += [path for path in (ROOT / "deliverables").rglob("*") if path.is_file()]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def run(url):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    checks, errors, images = [], [], []
    source_hashes = original_hashes()

    def passed(name):
        checks.append(name)
        print("PASS " + name, flush=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        api_context = browser.new_context()
        rules = api_context.request.get(url + "/api/rules").json()["items"]
        snapshot = api_context.request.get(url + "/api/labs/lab01/snapshot").json()
        base = api_context.request.get(url + "/api/cases/base").json()
        assert len(rules) == 27 and all(rule["guard_complete"] for rule in rules)
        assert len(snapshot["rules"]) == 15 and all(rule["guard_complete"] for rule in snapshot["rules"])
        by_name = {rule["name"]: rule for rule in rules}
        assert by_name["R01"]["guard_conditions"] == ["?r > 0.5", "?x ≠ ?y"]
        assert by_name["R02"]["guard_conditions"] == ['?rel = "配偶"']
        assert by_name["R04"]["guard_conditions"] == ["?r1 ≥ ?r2", "?r1 + ?r2 > 0.5", "?p ≠ ?q"]
        passed("both rule APIs expose complete conditions directly from Python definitions")
        api_context.close()

        def watch(page, requests=None):
            page.on("pageerror", lambda error: errors.append(str(error)))
            if requests is not None:
                page.on("request", lambda request: requests.append((request.method, request.url)))

        def go(page, fragment):
            page.goto(url + "/" + fragment, wait_until="domcontentloaded")
            expect(page.locator("#main-content")).to_be_visible(timeout=30000)
            page.wait_for_load_state("networkidle", timeout=30000)

        def expand_modeling_rules(page):
            toggle = page.get_by_role("button", name="展开一阶规则知识库", exact=True)
            expect(toggle).to_have_attribute("aria-expanded", "false")
            expect(page.locator(".rules-content")).to_be_hidden()
            expect(page.get_by_role("heading", name="一阶规则知识库", exact=True)).to_be_visible()
            toggle.click()
            expect(page.locator(".rules-content")).to_be_visible()
            expect(page.get_by_role("button", name="收起一阶规则知识库", exact=True)).to_have_attribute("aria-expanded", "true")

        def no_overflow(page):
            assert page.locator("#main-content").count() == 1
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1"), page.viewport_size

        def capture(page, stem):
            destination = OUTPUT / (stem + ".png")
            page.screenshot(path=str(destination), full_page=False)
            images.append(destination)

        def guide(page, expected_record):
            aside = page.locator("#main-content .rule-notation")
            expect(aside).to_be_visible()
            expect(aside).to_contain_text("不是数据缺失")
            expect(aside).to_contain_text("同名变量必须代入同一个值")
            summary = aside.locator("summary")
            geometry = summary.bounding_box()
            assert geometry and geometry["height"] >= 44, geometry
            summary.focus()
            page.keyboard.press("Enter")
            expect(aside.locator("details")).to_have_attribute("open", "")
            example = aside.get_by_label("当前案例变量代入示例", exact=True)
            expect(example).to_be_visible()
            expect(example).to_contain_text(expected_record)
            expect(example).to_contain_text("不是已运行的推理结果")
            expect(example.locator(".rule-example-disclosure")).to_be_visible()
            assert not VARIABLE.search(example.locator(".rule-expression").inner_text())
            assert len(example.locator(".rule-bindings li").all_text_contents()) == 3
            no_overflow(page)
            return aside

        viewports = {
            "laptop": {"width": 1508, "height": 768},
            "tablet": {"width": 1024, "height": 900},
            "mobile": {"width": 390, "height": 844},
            "narrow": {"width": 320, "height": 900},
            "zoom200": {"width": 754, "height": 768},
        }
        for viewport_name, viewport in viewports.items():
            context = browser.new_context(viewport=viewport, reduced_motion="reduce")
            page = context.new_page()
            requests = []
            watch(page, requests)
            go(page, "#/labs/lab01/representation")
            page.get_by_role("button", name="产生式规则", exact=True).click()
            table = page.get_by_role("table", name="实验一产生式规则")
            expect(table).to_be_visible()
            expect(table.locator("tbody tr")).to_have_count(15)
            for rule in snapshot["rules"]:
                row = table.locator("tbody tr").filter(has=page.get_by_role("rowheader", name=rule["name"], exact=True))
                expect(row).to_have_count(1)
                for condition in rule["guard_conditions"]:
                    expect(row.locator(".rule-guard").filter(has_text=condition)).to_have_count(1)
            fact = next(row for row in snapshot["facts"] if row[0] == "Holds")
            concrete = "Holds(" + ",".join(map(str, fact[1:])) + ")"
            guide(page, concrete)
            if viewport_name in ("laptop", "mobile"):
                capture(page, "implemented-lab01-" + viewport_name)
            passed(viewport_name + " lab01 shared guide, original guards, real binding and keyboard disclosure")

            for name, fragment in (("modeling", "#/labs/lab02/modeling"), ("rules", "#/labs/lab02/modeling/rules")):
                go(page, fragment)
                if name == "modeling":
                    expand_modeling_rules(page)
                expect(page.locator("#main-content .rule-row")).to_have_count(27)
                expect(page.get_by_text("这些是有限知识库中的教学规则。", exact=False)).to_have_count(0)
                if name == "rules":
                    expect(page.locator(".rules-toggle")).to_have_count(0)
                no_overflow(page)
                if viewport_name in ("laptop", "mobile") and name == "rules":
                    capture(page, "implemented-rules-collapsed-" + viewport_name)
                for rule in rules:
                    row = page.locator("#main-content .rule-row").filter(has=page.locator(".rule-row-code", has_text=re.compile("^" + rule["name"] + "$")))
                    expect(row).to_have_count(1)
                    for condition in rule["guard_conditions"]:
                        expect(row.locator(".rule-guard").filter(has_text=condition)).to_have_count(1)
                expect(page.locator("#main-content .rule-guard-unavailable")).to_have_count(0)
                guide(page, "Holds(P1,C1,0.8)")
                example = page.locator("#main-content .rule-binding-example")
                expect(example).to_contain_text("0.8 > 0.5")
                expect(example).to_contain_text("P1 ≠ C1")
                if viewport_name in ("laptop", "mobile") and name == "rules":
                    capture(page, "implemented-rules-expanded-" + viewport_name)
                    guide_image = OUTPUT / ("implemented-guide-" + viewport_name + ".png")
                    page.locator("#main-content .rule-notation").screenshot(path=str(guide_image))
                    images.append(guide_image)
                if name == "modeling":
                    search = page.get_by_role("textbox", name="搜索规则编号、说明或谓词", exact=True)
                    search.fill("R01")
                    category = page.get_by_role("combobox", name="规则类别", exact=True)
                    category.select_option("控制关系")
                    expect(page.locator("#main-content .rule-row")).to_have_count(1)
                    toggle = page.get_by_role("button", name="收起一阶规则知识库", exact=True)
                    content_id = toggle.get_attribute("aria-controls")
                    assert content_id and page.locator(".rules-content").get_attribute("id") == content_id
                    toggle.focus(); page.keyboard.press("Space")
                    expect(page.get_by_role("button", name="展开一阶规则知识库", exact=True)).to_have_attribute("aria-expanded", "false")
                    expect(page.locator(".rules-content")).to_be_hidden()
                    expect(search).to_be_hidden()
                    expect(page.locator("#main-content .rule-row").first).to_be_hidden()
                    expect(page.get_by_role("heading", name="一阶规则知识库", exact=True)).to_be_visible()
                    expect(page.locator(".rules-header-actions .data-badge")).to_have_text("27 条结构规则")
                    page.keyboard.press("Tab")
                    assert page.evaluate("!document.querySelector('.rules-content').contains(document.activeElement)")
                    no_overflow(page)
                    if viewport_name in ("laptop", "mobile"):
                        capture(page, "implemented-modeling-collapsed-" + viewport_name)
                    page.get_by_role("button", name="展开一阶规则知识库", exact=True).focus()
                    page.keyboard.press("Enter")
                    expect(search).to_have_value("R01")
                    expect(category).to_have_value("控制关系")
                    expect(page.locator("#main-content .rule-row")).to_have_count(1)
                    expect(page.locator("#main-content .rule-row").first).to_be_visible()
                    expect(page.locator("#main-content .rule-binding-example")).to_be_visible()
                    search.fill("")
                    category.select_option("全部")
                    expect(page.locator("#main-content .rule-row")).to_have_count(27)
                    if viewport_name in ("laptop", "mobile"):
                        capture(page, "implemented-modeling-expanded-" + viewport_name)
                    passed(viewport_name + " modeling rule module collapses by keyboard and restores filters and example state")
                passed(viewport_name + " lab02 " + name + " shows all 27 rules, eight complete guards and a current-record example")
            assert not any(method == "POST" and (address.endswith("/api/analyze") or address.endswith("/api/labs/lab02/run")) for method, address in requests)
            context.close()
        passed("reading rule guides never runs experiment-two inference")

        context = browser.new_context(viewport={"width": 1508, "height": 768}, reduced_motion="reduce")
        page = context.new_page()
        watch(page)
        go(page, "#/labs/lab02/modeling/rules")
        expect(page.locator(".rule-row")).to_have_count(27)
        search = page.get_by_role("textbox", name="搜索规则编号、说明或谓词", exact=True)
        search.fill('abs(?r - 0.5)')
        expect(page.locator(".rule-row-code")).to_have_text("R06")
        search.fill('R04')
        expect(page.locator(".rule-row-code")).to_have_text("R04")
        expect(page.locator(".rule-row .rule-guard")).to_have_count(3)
        search.fill('不存在的条件')
        expect(page.get_by_role("heading", name="没有匹配规则", exact=True)).to_be_visible()
        page.get_by_role("button", name="清除筛选", exact=True).click()
        expect(page.locator(".rule-row")).to_have_count(27)
        passed("rule search includes actual guard expressions and retains empty-state recovery")

        go(page, "#/workbench")
        changed = copy.deepcopy(base["case"])
        changed["label"] = "变量说明更新检查"
        changed["holdings"][0]["ratio"] = 0.2
        with page.expect_response(lambda response: response.url.endswith("/api/cases/validate")) as response:
            page.get_by_label("导入股权图JSON", exact=True).set_input_files({
                "name": "notation-check.json", "mimeType": "application/json",
                "buffer": json.dumps({"case": changed}, ensure_ascii=False).encode("utf-8"),
            })
        assert response.value.ok, response.value.text()
        expect(page.locator(".case-summary h2")).to_have_text(changed["label"])
        page.evaluate("location.hash = '#/labs/lab02/modeling'")
        expand_modeling_rules(page)
        expect(page.locator("#main-content .rule-row")).to_have_count(27)
        guide(page, "Holds(P1,C1,0.2)")
        expect(page.locator(".rule-binding-example")).to_contain_text("0.2 > 0.5")
        expect(page.locator(".rule-binding-example")).to_contain_text("20% 股权")
        assert "proved" not in page.locator(".rule-binding-example").inner_text()
        page.evaluate("location.hash = '#/labs/lab02/modeling/rules'")
        expect(page.locator("#main-content .rule-row")).to_have_count(27)
        guide(page, "Holds(P1,C1,0.2)")
        passed("imported case updates embedded and standalone examples without inventing a control conclusion")
        context.close()

        for scenario in ("half-share", "empty-holdings", "opaque-guard", "rules-error"):
            context = browser.new_context(viewport={"width": 320, "height": 900}, reduced_motion="reduce")
            page = context.new_page()
            watch(page)
            response_case = copy.deepcopy(base)
            if scenario == "half-share":
                response_case["case"]["holdings"][0]["ratio"] = 0.5
            if scenario == "empty-holdings":
                response_case["case"]["holdings"] = []
            page.route("**/api/cases/base", lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps(response_case, ensure_ascii=False)))
            if scenario == "opaque-guard":
                incomplete = copy.deepcopy(rules)
                incomplete[0]["guard_conditions"] = []
                incomplete[0]["guard_complete"] = False
                page.route("**/api/rules", lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps({"items": incomplete}, ensure_ascii=False)))
            if scenario == "rules-error":
                page.route("**/api/rules", lambda route: route.fulfill(status=500, content_type="application/json", body=json.dumps({"error": {"message": "规则读取检查失败", "code": "test_failure"}}, ensure_ascii=False)))
            go(page, "#/labs/lab02/modeling/rules")
            if scenario == "half-share":
                guide(page, "Holds(P1,C1,0.5)")
                expect(page.locator(".rule-binding-example")).to_contain_text("0.5 > 0.5")
                expect(page.locator(".rule-binding-example")).to_contain_text("不是已运行的推理结果")
            else:
                page.locator("#main-content .rule-notation summary").click()
                expect(page.locator(".rule-example-empty")).to_be_visible()
                expect(page.locator(".rule-binding-example")).to_have_count(0)
                if scenario == "opaque-guard":
                    expect(page.locator(".rule-guard-unavailable")).to_contain_text("附加条件尚未展开")
                    expect(page.locator(".guard-note")).to_contain_text("不能将上式理解为无条件规则")
                if scenario == "rules-error":
                    expect(page.get_by_role("alert")).to_contain_text("规则读取检查失败")
            no_overflow(page)
            passed("320px " + scenario + " has a truthful explanation without fabricated records or guard results")
            context.close()

        context = browser.new_context(viewport={"width": 390, "height": 844}, reduced_motion="reduce")
        page = context.new_page()
        watch(page)
        go(page, "#/labs/lab02/modeling")
        page.get_by_role("button", name="构建当前案例的谓词事实", exact=True).click()
        table = page.get_by_role("table", name="实验二一阶谓词事实")
        expect(table).to_be_visible(timeout=30000)
        assert not VARIABLE.search(table.inner_text())
        expect(page.get_by_text("这里是已代入实体编号和数值的具体事实，与下方带变量的规则模板不同。", exact=True)).to_be_visible()
        no_overflow(page)
        go(page, "#/labs/lab02/deterministic")
        page.get_by_label("实验二推理方法", exact=True).select_option("resolution")
        page.get_by_role("button", name="运行本节实验", exact=True).click()
        expect(page.locator(".task-result")).to_be_visible(timeout=30000)
        expect(page.get_by_text("归结子句和中间公式中的 ? 开头符号仍是逻辑变量，不是缺失数据；最终待证目标使用具体实体。", exact=True)).to_be_visible()
        expect(page.locator(".task-verdict")).to_contain_text("Controls(P1,C3)")
        assert not VARIABLE.search(page.locator(".task-verdict").inner_text())
        no_overflow(page)
        passed("concrete facts and resolution intermediate formulas have separate, accurate explanations")
        context.close()
        browser.close()

    assert not errors, errors
    assert original_hashes() == source_hashes, "Raw data or retained deliverables changed"
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "url": url,
        "checks": checks, "unexpected_browser_errors": errors,
        "viewports": viewports, "raw_and_retained_files_unchanged": True,
        "source_file_sha256": source_hashes,
        "screenshots": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in images},
    }
    (OUTPUT / "implementation-checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(len(checks)) + " rule-notation browser groups passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--isolated", action="store_true")
    args = parser.parse_args()
    if not args.url.startswith(("http://127.0.0.1:", "http://localhost:")):
        parser.error("Only local loopback servers are supported")
    if args.isolated:
        with isolated_server() as isolated_url:
            run(isolated_url)
    else:
        run(args.url.rstrip("/"))
