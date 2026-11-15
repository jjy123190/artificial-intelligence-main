"""Additional browser regressions for UI refinement; isolated database only."""
from __future__ import annotations
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.check_web import isolated_server, canonical_fragment
from playwright.sync_api import sync_playwright, expect
OUT = ROOT / "outputs" / "qa" / "hardening"

def run(url):
    OUT.mkdir(parents=True, exist_ok=True)
    checks, errors = [], []
    def passed(name):
        checks.append(name)
        print("PASS " + name, flush=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(viewport={"width":1440,"height":900}, accept_downloads=True, reduced_motion="reduce")
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        def ready():
            expect(page.locator(".directory")).to_be_visible(timeout=30000)
            expect(page.locator(".header-status")).to_contain_text("已就绪", timeout=30000)
        def go(hash):
            hash=canonical_fragment(hash)
            page.evaluate("hash => location.hash = hash", hash)
            expect(page).to_have_url(url + "/" + hash)
        def menu():
            if not page.locator("#case-manager").evaluate("el => el.open"):
                page.locator("#case-manager > summary").click()
        def compute(action):
            with page.expect_response(lambda r: r.url.endswith("/api/analyze") and r.request.method == "POST", timeout=30000) as response:
                action()
            assert response.value.ok, response.value.text()
            expect(page.locator(".header-status")).to_contain_text("已就绪", timeout=30000)
            return response.value.json()
        page.goto(url+"/#/workbench")
        ready()
        assert page.locator(".toast").count() == 0
        initial_scripts = page.evaluate("performance.getEntriesByType('resource').filter(r=>r.initiatorType==='script').map(r=>r.name)")
        assert not any("GraphView-" in name for name in initial_scripts),initial_scripts
        passed("initial overview has no blocking success toast and does not load graph engine")
        assert page.locator(".directory-filter,.directory-status").count()==0
        go("#/labs/lab02/hybrid/overview")
        expect(page.locator(".workspace-context")).to_contain_text("实验二结果",timeout=30000)
        filter_button=page.get_by_role("button", name="只看优先复核", exact=True)
        filter_button.click()
        expect(page.locator(".directory-row")).to_have_count(7)
        page.get_by_label("搜索企业或自然人", exact=True).fill("恒通")
        expect(page.locator(".directory-row")).to_have_count(1)
        page.get_by_role("button", name="清空目录搜索",exact=True).click()
        expect(filter_button).to_have_attribute("aria-pressed","true")
        expect(page.locator(".directory-row")).to_have_count(7)
        page.get_by_label("搜索企业或自然人", exact=True).fill("NO_MATCH")
        page.get_by_role("button", name="清除搜索", exact=True).click()
        expect(filter_button).to_have_attribute("aria-pressed","false")
        expect(page.locator(".directory-row")).to_have_count(12)
        passed("review filter, text search and clear actions keep their promised semantics")
        go("#/entity/C3/edit")
        page.locator("#holding-value").fill("25")
        expect(page.locator(".workspace-change")).to_contain_text("结果尚未更新")
        menu()
        page.get_by_role("button",name="重置案例",exact=True).click()
        dialog=page.get_by_role("dialog")
        expect(dialog).to_be_visible()
        expect(dialog.get_by_role("button",name="保留草稿",exact=True)).to_be_focused()
        for _ in range(8):
            page.keyboard.press("Tab")
            assert page.evaluate("document.activeElement.closest('dialog') !== null")
        page.keyboard.press("Escape")
        expect(dialog).not_to_be_visible()
        expect(page.locator("#holding-value")).to_have_value("25")
        menu()
        page.get_by_label("选择案例",exact=True).select_option("extended")
        expect(dialog).to_be_visible()
        dialog.get_by_role("button",name="保留草稿",exact=True).click()
        expect(page.get_by_label("选择案例",exact=True)).to_have_value("base")
        expect(page.locator("#holding-value")).to_have_value("25")
        passed("replacement confirmation defaults to safe cancel, traps focus and Escape preserves input")
        menu()
        with page.expect_download() as download:
            page.get_by_role("button",name="导出案例",exact=True).click()
        download.value.save_as(OUT / "protected-draft.json")
        assert json.loads((OUT / "protected-draft.json").read_text(encoding="utf8"))["case"]["holdings"][0]["ratio"] > 0
        expect(page.locator(".draft-state")).not_to_be_visible()
        expect(page.locator(".workspace-change")).to_be_visible()
        menu()
        restored=compute(lambda: page.get_by_role("button",name="重置案例",exact=True).click())
        assert restored["case"]["entities"][0]["eid"]
        expect(page.locator(".workspace-change")).not_to_be_visible()
        page.locator("#holding-value").fill("24")
        compute(lambda:page.get_by_role("button",name="运行推理",exact=False).click())
        expect(page.locator(".workspace-change")).not_to_be_visible()
        expect(page.locator(".draft-state")).to_be_visible()
        menu()
        page.get_by_role("button",name="重置案例",exact=True).click()
        expect(dialog).to_be_visible()
        compute(lambda:dialog.get_by_role("button",name="恢复初始输入",exact=True).click())
        passed("export marks input backed up without applying it; computation does not mark draft saved")
        page.locator("#holding-value").fill("999")
        expect(page.get_by_role("button",name="运行推理",exact=False)).to_be_disabled()
        menu()
        page.get_by_role("button",name="重置案例",exact=True).click()
        expect(dialog).to_be_visible()
        compute(lambda:dialog.get_by_role("button",name="恢复初始输入",exact=True).click())
        expect(page.locator("#holding-value")).not_to_have_value("999")
        expect(page.get_by_role("button",name="运行推理",exact=False)).to_be_enabled()
        passed("confirmed reset clears invalid local numeric text instead of leaving compute blocked")
        go("#/graph")
        expect(page.locator(".graph-canvas")).to_be_visible(timeout=20000)
        page.get_by_label("选择实体",exact=True).select_option("C6")
        expect(page).to_have_url(url+"/#/graph")
        assert page.locator(".graph-canvas").evaluate("el=>el._cyreg.cy.nodes().length")==16
        passed("full graph selector changes highlight without silently narrowing to a subgraph")
        go("#/entity/C3")
        requests=[]
        def delayed(route):
            requests.append(route.request.url)
            result=route.fetch()
            page.get_by_label("选择实体",exact=True).select_option("C6")
            route.fulfill(response=result)
        page.route("**/api/analyze",delayed)
        compute(lambda:page.get_by_role("button",name="运行推理",exact=False).evaluate("el=>{el.click();el.click();}"))
        page.unroute("**/api/analyze")
        assert len(requests)==1,requests
        expect(page).to_have_url(url+"/"+canonical_fragment("#/entity/C6"))
        expect(page.locator(".entity-kicker")).to_contain_text("C6")
        passed("rapid duplicate run submits once and newer entity navigation wins over delayed response")
        for hash in ["#/unknown","#/entity","#/entity/C3/missing","#/graph/C3/extra","#/entity/%INVALID"]:
            go(hash)
            expect(page.get_by_role("heading",name="页面未找到",exact=True)).to_be_visible()
        assert "页面未找到" in page.title()
        page.get_by_role("link",name="返回案例概览",exact=True).last.click()
        ready()
        passed("malformed and unknown routes have a recovery page rather than silently showing the wrong task")
        go("#/governance")
        expect(page.locator(".governance-controls")).to_be_visible()
        page.get_by_label("案例名称",exact=True).fill("防丢失核查测试")
        save_requests=[]
        page.on("request", lambda request: save_requests.append(request.url) if request.url.endswith("/api/governance/save") else None)
        with page.expect_response(lambda r:r.url.endswith("/api/governance/save"),timeout=30000) as saved:
            page.get_by_role("button",name="保存新版本",exact=True).evaluate("el=>{el.click();el.click();}")
        assert saved.value.ok
        expect(page.get_by_role("button",name="保存新版本",exact=True)).to_be_enabled()
        assert len(save_requests)==1,save_requests
        passed("rapid governance save does not create duplicate revisions")
        page.get_by_role("tab",name="凭证与身份",exact=True).click()
        page.get_by_role("button",name="添加文本凭证",exact=True).click()
        page.get_by_label("凭证文本 1",exact=True).fill("未经保存的虚构核查记录")
        page.get_by_role("tab",name="历史与实验",exact=True).click()
        page.get_by_role("button",name="加载版本 1",exact=True).click()
        expect(dialog).to_be_visible()
        page.keyboard.press("Escape")
        page.get_by_role("tab",name="凭证与身份",exact=True).click()
        expect(page.get_by_label("凭证文本 1",exact=True)).to_have_value("未经保存的虚构核查记录")
        page.get_by_role("tab",name="历史与实验",exact=True).click()
        page.get_by_role("button",name="加载版本 1",exact=True).click()
        with page.expect_response(lambda r:r.url.endswith("/api/governance/load"),timeout=30000) as loaded:
            dialog.get_by_role("button",name="加载历史版本",exact=True).click()
        assert loaded.value.ok
        expect(page.get_by_role("button",name="保存新版本",exact=True)).to_be_enabled()
        page.get_by_role("tab",name="凭证与身份",exact=True).click()
        expect(page.get_by_label("凭证文本 1",exact=True)).not_to_be_visible()
        passed("loading history asks before discarding unsaved governance edits and cancel keeps all evidence")
        page.locator(".research-menu > summary").click()
        page.get_by_role("link",name="案例概览",exact=True).click()
        ready()
        assert page.locator("#main-content").count()==1
        assert page.locator("#main-content").is_visible()
        page.locator(".skip-link").focus();page.keyboard.press("Enter")
        expect(page.locator("#main-content")).to_be_focused()
        passed("visiting governance leaves only one visible main-content focus target")
        go("#/entity/C3/edit")
        page.locator("#holding-value").fill("23")
        unload=[]
        def handle(dialog):
            unload.append(dialog.type)
            dialog.dismiss()
        page.on("dialog",handle)
        page.evaluate("() => { setTimeout(() => location.reload(), 0); }")
        page.wait_for_timeout(500)
        assert unload==["beforeunload"],unload
        expect(page.locator("#holding-value")).to_have_value("23")
        page.remove_listener("dialog",handle)
        passed("browser reload warns for unexported edits and cancelling keeps the current draft")
        for width in [1440,900,390,320]:
            page.set_viewport_size({"width":width,"height":900})
            menu();page.get_by_role("button",name="重置案例",exact=True).click()
            expect(dialog).to_be_visible()
            assert page.evaluate("document.documentElement.scrollWidth<=innerWidth+1")
            page.screenshot(path=str(OUT/('discard-'+str(width)+'.png')),full_page=True)
            page.keyboard.press("Escape")
        passed("draft dialog remains readable and keyboard-operable at desktop, tablet and 320px")
        assert not errors,errors
        browser.close()
    report={"created_utc":datetime.now(timezone.utc).isoformat(),"checks":checks,"browser_errors":errors,"isolated":True}
    (OUT/"refinement-checks.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8")
    print(str(len(checks))+" additional browser groups passed",flush=True)

if __name__=="__main__":
    with isolated_server() as url:run(url)
