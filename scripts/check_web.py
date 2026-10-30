"""Real-browser acceptance checks; run against a local server, never external services.

Optional dependency: python -m pip install playwright
Uses installed Chrome when available, otherwise Playwright's Chromium.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import multiprocessing
import socket
import tempfile
import time
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "qa" / "legacy-workbench"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _serve_isolated(port, database):
    import uvicorn
    from backend.api import create_app
    uvicorn.run(create_app(governance_database=Path(database)), host="127.0.0.1", port=port, log_level="warning")


@contextmanager
def isolated_server():
    temp = tempfile.TemporaryDirectory(prefix="equity_browser_tests_")
    base = Path(temp.name).resolve()
    if base.parent != Path(tempfile.gettempdir()).resolve() or not base.name.startswith("equity_browser_tests_"):
        raise AssertionError("浏览器测试临时目录越界")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    process = multiprocessing.get_context("spawn").Process(target=_serve_isolated, args=(port, str(base / "history.sqlite3")))
    process.start()
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                with urllib.request.urlopen(url + "/api/health", timeout=1) as response:
                    if response.status == 200:
                        break
            except OSError:
                if not process.is_alive():
                    raise RuntimeError("隔离浏览器测试服务启动失败")
                time.sleep(.2)
        else:
            raise RuntimeError("隔离浏览器测试服务启动超时")
        yield url
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=10)
        if process.is_alive():
            process.kill(); process.join(timeout=5)
        assert base.parent == Path(tempfile.gettempdir()).resolve() and base.name.startswith("equity_browser_tests_")
        temp.cleanup()


def canonical_fragment(fragment):
    """Resolve previous UI links to their experiment-scoped replacement."""
    aliases={
        '#/settings':'#/labs/lab02/hybrid/settings',
        '#/verification':'#/labs/lab02/deterministic/records',
        '#/experiments':'#/labs/lab02/optimization/results',
        '#/rules':'#/labs/lab02/modeling/rules',
        '#/governance':'#/labs/lab02/modeling/governance',
    }
    if fragment in aliases:return aliases[fragment]
    parts=fragment.split('/')
    if len(parts) in (3,4) and parts[0]=='#' and parts[1]=='entity' and parts[2]:
        if len(parts)==3:return '#/labs/lab02/hybrid/entity/'+parts[2]
        if parts[3]=='evidence':return '#/labs/lab02/deterministic/entity/'+parts[2]+'/evidence'
        if parts[3]=='edit':return '#/labs/lab02/modeling/entity/'+parts[2]+'/edit'
    return fragment

def run(url: str):
    from playwright.sync_api import sync_playwright, expect
    OUTPUT.mkdir(parents=True, exist_ok=True)
    checks, errors = [], []

    def passed(name):
        checks.append(name)
        print("PASS", name, flush=True)

    def wait_ready(page):
        expect(page.locator(".header-status")).to_contain_text("已就绪", timeout=20000)
        if page.locator(".workspace-context").count():
            expect(page.locator(".workspace-context")).to_contain_text("当前输入数据" if page.locator(".source-directory,.source-entity-details").count() or "/#/graph" in page.url else "实验二结果",timeout=20000)

    def go(page, fragment, heading=None):
        fragment=canonical_fragment(fragment)
        page.evaluate("value => { location.hash = value; }", fragment)
        if heading:
            expect(page.get_by_role("heading", name=heading, exact=True)).to_be_visible(timeout=20000)

    def open_menu(page, identifier):
        menu = page.locator("#" + identifier)
        if not menu.evaluate("el=>el.open"):
            menu.locator(":scope > summary").click()

    def governance_tab(page, name):
        page.get_by_role("tab", name=name, exact=True).click()

    def navigate(page, name):
        entries={"实验对比":("#/experiments","实验对比"),"规则库":("#/rules","规则知识库"),"证据与更新":("#/governance","证据与更新")}
        if name in entries:
            fragment,heading=entries[name];go(page,fragment,heading);return
        if name=="案例概览":
            menu=page.locator(".research-menu")
            if not menu.evaluate("el=>el.open"):menu.locator(":scope > summary").click()
        page.get_by_role("link", name=name, exact=True).click()

    def compute(page, action):
        with page.expect_response(lambda r: r.url.endswith("/api/analyze") and r.request.method == "POST", timeout=20000) as pending:
            action()
        response = pending.value
        assert response.ok, response.text()
        wait_ready(page)
        return response.json()

    def query(page, source, expected):
        go(page, "#/verification", "验证与推理记录")
        page.get_by_label("控制主体").select_option(source)
        with page.expect_response(lambda r: r.url.endswith("/api/query"), timeout=20000) as pending:
            page.get_by_role("button", name="验证结论", exact=True).click()
        assert pending.value.json()["proved"] is expected
        expect(page.locator(".query-result strong")).to_have_text("证明成立" if expected else "暂不能证明")

    def governance_compute(page, action, endpoint="preview"):
        with page.expect_response(lambda r: r.url.endswith("/api/governance/" + endpoint) and r.request.method == "POST", timeout=20000) as pending:
            action()
        response = pending.value
        assert response.ok, response.text()
        expect(page.get_by_role("button", name="重新核查有效证据", exact=True)).to_be_enabled()
        return response.json()

    def no_overflow(page):
        valid = page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
        if not valid:
            print("OVERFLOW", page.evaluate("() => [...document.querySelectorAll('body *')].map(el=>({tag:el.tagName,cls:el.className?.baseVal ?? el.className,right:el.getBoundingClientRect().right,width:el.getBoundingClientRect().width})).filter(el=>el.right>innerWidth+1).slice(0,30)"), flush=True)
            page.screenshot(path=str(OUTPUT / "overflow-debug.png"), full_page=True)
        assert valid, page.viewport_size

    def graph_dimensions(page):
        dimensions = page.locator(".graph-canvas").evaluate("el=>({width:el.clientWidth,height:el.clientHeight,nodes:el._cyreg.cy.nodes().length,zoom:el._cyreg.cy.zoom(),canvas:[...el.querySelectorAll('canvas')].some(c=>c.width>0&&c.height>0)})")
        assert dimensions["height"] > 200 and dimensions["width"] > 250 and dimensions["nodes"] > 0 and dimensions["canvas"], dimensions
        return dimensions

    with sync_playwright() as playwright:
        chrome = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
        browser = playwright.chromium.launch(headless=True, **({"executable_path":str(chrome)} if chrome.exists() else {}))
        context = browser.new_context(viewport={"width":1440,"height":1000}, accept_downloads=True, reduced_motion="reduce")
        page = context.new_page()
        page.on("dialog", lambda dialog: dialog.accept() if dialog.type == "beforeunload" else dialog.dismiss())
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
        page.goto(url+"/#/workbench")
        wait_ready(page)
        expect(page.get_by_role("heading", name="案例概览", exact=True)).to_be_visible()
        expect(page.locator(".directory-row")).to_have_count(12)
        assert page.locator(".graph-canvas, .standalone-inspector, .query-section").count() == 0
        page.get_by_label("搜索企业或自然人", exact=True).fill("恒通")
        expect(page.locator(".directory-row")).to_have_count(1)
        page.get_by_label("搜索企业或自然人", exact=True).fill("NOT_FOUND")
        expect(page.get_by_role("heading",name="没有匹配的实体",exact=True)).to_be_visible()
        page.get_by_role("button",name="清除搜索",exact=True).click()
        page.get_by_role("button",name="自然人",exact=False).click()
        expect(page.locator(".directory-row")).to_have_count(4)
        page.get_by_role("button",name="企业目录",exact=False).click()
        expect(page.locator(".toast")).not_to_be_visible(timeout=5000)
        no_overflow(page)
        page.screenshot(path=str(OUTPUT / "overview-desktop.png"),full_page=True)
        passed("overview contains only searchable directory, not graph, scoring panels or query tools")

        page.locator("#case-manager > summary").focus()
        page.keyboard.press("ArrowDown")
        expect(page.get_by_label("选择案例",exact=True)).to_be_focused()
        page.keyboard.press("Escape")
        expect(page.locator("#case-manager > summary")).to_be_focused()
        page.locator(".research-menu > summary").click()
        expect(page.get_by_role("link",name="关系图谱",exact=True)).to_be_visible()
        expect(page.locator(".research-menu .menu-link")).to_have_count(2)
        assert page.locator(".research-menu a[href*=lab02]").count()==0
        page.get_by_role("heading",name="案例概览",exact=True).click()
        assert not page.locator(".research-menu").evaluate("el=>el.open")
        passed("case and research menus retain keyboard entry, Escape and outside dismissal")

        go(page,"#/labs/lab02/hybrid/overview","全案例分析")
        wait_ready(page)
        page.get_by_role("link",name="查看C3分析",exact=True).click()
        expect(page).to_have_url(url+"/"+canonical_fragment("#/entity/C3"))
        expect(page.locator(".entity-kicker")).to_contain_text("C3")
        expect(page.locator(".decision-status h3")).to_have_text("优先复核")
        assert page.locator(".decision-compare small").all_text_contents() == ["0.088","0.088"]
        assert page.locator(".graph-canvas, .directory").count() == 0
        no_overflow(page)
        page.screenshot(path=str(OUTPUT / "entity-desktop.png"),full_page=True)
        page.get_by_role("link",name="查看确定性推理证据",exact=True).click()
        expect(page).to_have_url(url+"/"+canonical_fragment("#/entity/C3/evidence"))
        expect(page.locator(".proof-tree")).to_be_visible()
        assert page.locator(".graph-canvas, .score-row").count() == 0
        page.locator(".proof-node-label").first.click()
        page.screenshot(path=str(OUTPUT / "evidence-desktop.png"),full_page=True)
        page.get_by_role("link",name="在关系图谱中查看证明",exact=True).click()
        expect(page).to_have_url(url+"/#/labs/lab02/deterministic/graph/C3")
        expect(page.locator(".graph-canvas")).to_be_visible()
        assert page.locator(".inspector, .directory, .score-row").count() == 0
        assert page.locator(".graph-canvas").evaluate("el=>el._cyreg.cy.nodes('.evidence').length") > 0
        dimensions = graph_dimensions(page)
        assert dimensions["width"] > 1100 and dimensions["zoom"] > .65, dimensions
        page.screenshot(path=str(OUTPUT / "graph-desktop.png"),full_page=True)
        passed("entity conclusion, evidence and graph use separate URLs and preserve selected proof")

        page.get_by_role("button",name="整个案例",exact=True).click()
        assert graph_dimensions(page)["nodes"] == 16
        count = page.locator(".graph-canvas").evaluate("el=>el._cyreg.cy.edges().length")
        page.get_by_role("button",name="持股",exact=True).click()
        assert page.locator(".graph-canvas").evaluate("el=>el._cyreg.cy.edges().length") < count
        page.get_by_role("button",name="持股",exact=True).click()
        page.get_by_role("button",name="当前企业关系",exact=True).click()
        old_zoom = page.locator(".graph-canvas").evaluate("el=>el._cyreg.cy.zoom()")
        page.locator(".graph-canvas").focus(); page.keyboard.press("+")
        assert page.locator(".graph-canvas").evaluate("el=>el._cyreg.cy.zoom()") > old_zoom
        page.get_by_role("button",name="适配画布",exact=True).click()
        page.get_by_role("button",name="下一步证明",exact=True).click()
        expect(page.locator(".step-counter")).to_contain_text("1 /")
        page.get_by_role("button",name="播放当前结论的证明",exact=True).click()
        expect(page.get_by_role("button",name="暂停推理演示",exact=True)).to_be_visible()
        page.wait_for_timeout(1400)
        page.get_by_role("button",name="暂停推理演示",exact=True).click()
        with page.expect_download() as pending:
            page.get_by_role("button",name="导出当前图谱PNG",exact=True).click()
        pending.value.save_as(OUTPUT / "graph-export.png")
        assert (OUTPUT / "graph-export.png").stat().st_size > 10000
        go(page,"#/labs/lab02/deterministic/graph","推理关系图")
        expect(page.get_by_role("button",name="整个案例",exact=True)).to_have_attribute("aria-pressed","true")
        full = graph_dimensions(page)
        assert full["nodes"] == 16 and full["zoom"] > .5, full
        location = page.locator(".graph-canvas").evaluate("el=>el._cyreg.cy.getElementById('entity:C6').renderedPosition()")
        page.locator(".graph-canvas").click(position=location)
        expect(page.get_by_label("选择实体",exact=True)).to_have_value("C6")
        assert graph_dimensions(page)["nodes"] == 16
        page.get_by_role("link",name="查看所选实体推理证据",exact=True).click()
        expect(page).to_have_url(url+"/"+canonical_fragment("#/entity/C6/evidence"))
        expect(page.get_by_role("heading",name="推理证据",exact=True)).to_be_visible()
        expect(page.locator(".page-breadcrumb")).to_contain_text("确定性推理")
        expect(page.locator(".decision-status,.score-row,.decision-compare")).to_have_count(0)
        expect(page.locator(".entity-kicker")).to_contain_text("C6")
        passed("full graph keeps all entities on selection, filtering, zoom, proof playback and PNG export work")

        go(page,"#/entity/C3","企业分析")
        go(page,"#/entity/C3/edit","调整企业数据")
        options=page.locator("#holding-edge option").evaluate_all("els=>els.map(e=>({label:e.textContent,value:e.value}))")
        edge=next(o for o in options if "华鑫实业" in o["label"])
        page.locator("#holding-edge").select_option(edge["value"])
        expect(page.locator("#holding-value")).to_have_value("35")
        page.locator("#holding-value").fill("")
        expect(page.locator("#holding-error")).to_be_visible()
        expect(page.get_by_role("button",name="运行推理",exact=False)).to_be_disabled()
        page.keyboard.press("Control+Enter")
        expect(page.locator("#holding-value")).to_have_value("")
        page.locator("#holding-value").fill("101")
        expect(page.locator("#holding-value")).to_have_attribute("aria-invalid","true")
        page.locator("#holding-value").fill("25")
        go(page,"#/entity/C3","企业分析")
        expect(page.locator(".workspace-change")).to_be_visible()
        go(page,"#/entity/C3/edit","调整企业数据")
        page.locator("#holding-edge").select_option(edge["value"])
        expect(page.locator("#holding-value")).to_have_value("25")
        changed=compute(page,lambda:page.get_by_role("button",name="运行推理",exact=False).click())
        assert not any(e["source"]=="P1" and e["target"]=="C3" for e in changed["derived_edges"])
        query(page,"P1",False)
        open_menu(page,"case-manager")
        page.get_by_role("button",name="重置案例",exact=True).click()
        expect(page.get_by_role("dialog")).to_be_visible()
        compute(page,lambda:page.get_by_role("button",name="恢复初始输入",exact=True).click())
        query(page,"P1",True);query(page,"P3",False)
        page.screenshot(path=str(OUTPUT / "verification-desktop.png"),full_page=True)
        passed("editing blocks invalid compute, keeps accepted drafts across pages, and changes real control proof")

        go(page,"#/entity/C3","企业分析")
        page.get_by_role("link",name="分析设置",exact=True).click()
        expect(page).to_have_url(url+"/"+canonical_fragment("#/settings"))
        assert page.locator(".graph-canvas, .inspector, .directory").count()==0
        page.get_by_label("推理方式",exact=True).select_option("hybrid_unlocked")
        unlocked=compute(page,lambda:page.keyboard.press("Control+Enter"))
        assert not unlocked["decisions"]["C3"]["review_required"]
        page.get_by_label("推理方式",exact=True).select_option("hybrid_locked")
        page.get_by_label("登记与测度资料",exact=True).uncheck()
        page.locator("#threshold-range").focus();page.keyboard.press("ArrowRight");page.keyboard.press("ArrowRight")
        models=page.get_by_label("参数来源",exact=True).locator("option").evaluate_all("els=>els.map(e=>e.value)")
        adaptive=next(value for value in models if value.endswith(":adaptive"))
        page.get_by_label("参数来源",exact=True).select_option(adaptive)
        page.get_by_label("推理方式",exact=True).select_option("hybrid_calibrated")
        page.get_by_role("link",name="返回企业分析",exact=True).click()
        expect(page.locator(".workspace-change")).to_be_visible()
        go(page,"#/settings","分析设置")
        expect(page.get_by_label("参数来源",exact=True)).to_have_value(adaptive)
        calibrated=compute(page,lambda:page.keyboard.press("Control+Enter"))
        assert calibrated["parameters"]["threshold"]==.6 and calibrated["completeness"]["registry"] is False
        assert "fuzzy" in calibrated["model"] and "review_policy" in calibrated["model"] and calibrated["provenance"]
        page.screenshot(path=str(OUTPUT / "settings-desktop.png"),full_page=True)
        passed("standalone settings preserve algorithm, learned model, threshold and completeness across navigation")

        open_menu(page,"case-manager")
        with page.expect_download() as pending:
            page.get_by_role("button",name="导出案例",exact=True).click()
        pending.value.save_as(OUTPUT / "roundtrip-case.json")
        exported=json.loads((OUTPUT / "roundtrip-case.json").read_text(encoding="utf-8"))
        upload=page.get_by_label("导入股权图JSON",exact=True)
        upload.set_input_files({"name":"bad.json","mimeType":"application/json","buffer":b"{"})
        expect(page.locator(".error-banner")).to_be_visible()
        page.get_by_role("button",name="关闭错误提示",exact=True).click()
        with page.expect_response(lambda r:r.url.endswith("/api/cases/validate"),timeout=20000) as imported:
            upload.set_input_files(OUTPUT / "roundtrip-case.json")
        assert imported.value.ok,imported.value.text()
        restored=compute(page,lambda:page.get_by_role("button",name="运行推理",exact=False).click())
        assert restored["case"]==exported["case"]
        go(page,"#/verification","验证与推理记录")
        with page.expect_download() as pending:
            page.get_by_role("button",name="下载分析JSON",exact=True).click()
        pending.value.save_as(OUTPUT / "analysis-export.json")
        assert json.loads((OUTPUT / "analysis-export.json").read_text(encoding="utf-8"))["provenance"]
        page.locator(".all-steps > summary").click()
        expect(page.locator(".steps-list")).to_be_visible()
        passed("case JSON roundtrip, safe invalid import, complete analysis export and dedicated records page")

        navigate(page,"实验对比")
        expect(page.locator('[aria-label="各方法的实测指标"] tbody tr')).to_have_count(11)
        page.get_by_label("观测条件").select_option("severe")
        assert page.locator(".recharts-bar-rectangle").count()>0
        page.get_by_role("button",name="确定性优化",exact=True).click()
        page.get_by_label("核对网络规模").select_option("4")
        expect(page.locator('[aria-label="确定性优化实测"] tbody tr')).to_have_count(3)
        no_overflow(page)
        page.screenshot(path=str(OUTPUT / "experiments-desktop.png"),full_page=True)
        navigate(page,"规则库")
        expect(page.locator(".rule-row")).to_have_count(27)
        page.get_by_placeholder("搜索规则编号、说明或谓词").fill("THIS_DOES_NOT_EXIST")
        expect(page.get_by_role("heading",name="没有匹配规则",exact=True)).to_be_visible()
        page.get_by_role("button",name="清除筛选",exact=True).click()
        expect(page.locator(".rule-row")).to_have_count(27)
        page.screenshot(path=str(OUTPUT / "rules-desktop.png"),full_page=True)
        go(page,"#/settings","分析设置")
        page.get_by_label("推理方式",exact=True).select_option("hybrid_locked")
        page.get_by_label("参数来源",exact=True).select_option("manual")
        page.get_by_label("登记与测度资料",exact=True).check()
        page.locator("#threshold-range").fill("0.5")
        compute(page,lambda:page.get_by_role("button",name="运行推理",exact=False).click())
        passed("research pages still read actual experiments and Python rules through the shared entry")

        navigate(page,"证据与更新")
        expect(page.locator(".governance-case-tools")).to_be_visible()
        original = governance_compute(page, lambda: page.get_by_role("button", name="重新核查有效证据", exact=True).click())
        assert ["Controls", "P1", "C3"] in original["claims"]
        governance_tab(page, "核查结果")
        expect(page.get_by_role("table", name="经济权益计算结果")).to_contain_text("34.2000%")
        governance_tab(page, "历史与实验")
        expect(page.get_by_role("table", name="循环权益扩容实测结果").locator("tbody tr")).to_have_count(4)
        governance_tab(page, "关系与协议")
        page.get_by_label("选择有效期关系").select_option("holdings:C2:C3")
        governance_tab(page, "关系与协议")
        page.get_by_label("关系失效日", exact=True).fill("2026-10-23")
        governance_tab(page, "关系与协议")
        page.get_by_role("button", name="更新关系设置", exact=True).click()
        page.get_by_label("核查日期", exact=True).fill("2026-10-22")
        saved = governance_compute(page, lambda: page.get_by_role("button", name="保存新版本", exact=True).click(), "save")
        assert saved["revision"] == 1
        page.get_by_label("核查日期", exact=True).fill("2026-10-23")
        withdrawn = governance_compute(page, lambda: page.get_by_role("button", name="保存新版本", exact=True).click(), "save")
        assert ["Controls", "P1", "C3"] in withdrawn["changes"]["withdrawn"]
        governance_tab(page, "核查结果")
        expect(page.get_by_label("撤回结论列表")).to_contain_text("Controls(P1, C3)")
        governance_tab(page, "历史与实验")
        expect(page.get_by_role("table", name="案例版本历史").locator("tbody tr")).to_have_count(2)
        governance_tab(page, "历史与实验")
        loaded = governance_compute(page, lambda: page.get_by_role("button", name="加载版本 1", exact=True).click(), "load")
        assert loaded["revision"] == 1 and loaded["payload"]["as_of"] == "2026-10-23"
        passed("governance real snapshot, date expiry withdrawal, persisted versions and historical re-evaluation")

        open_menu(page, "governance-case-menu")
        with page.expect_download() as pending:
            page.get_by_role("button", name="导出核查案例", exact=True).click()
        pending.value.save_as(OUTPUT / "governance-case.json")
        exported = json.loads((OUTPUT / "governance-case.json").read_text(encoding="utf-8"))
        assert exported["schema_version"] == 1 and exported["payload"]["relations"]
        exported["snapshot"] = {"claims": [["Controls", "FAKE", "C3"]]}
        imported = governance_compute(page, lambda: page.get_by_label("导入核查案例JSON", exact=True).set_input_files({"name": "roundtrip.json", "mimeType": "application/json", "buffer": json.dumps(exported, ensure_ascii=False).encode("utf-8")}), "import")
        page.get_by_role("button", name="确认替换草稿", exact=True).click()
        assert not imported["persisted"] and ["Controls", "FAKE", "C3"] not in imported["snapshot"]["claims"]
        governance_tab(page, "历史与实验")
        expect(page.get_by_role("table", name="案例版本历史")).not_to_be_visible()
        page.get_by_label("导入核查案例JSON", exact=True).set_input_files({"name": "bad.json", "mimeType": "application/json", "buffer": b"bad json"})
        expect(page.locator(".governance-page .error-banner")).to_be_visible()
        governance_tab(page, "核查结果")
        expect(page.get_by_role("table", name="经济权益计算结果")).to_be_visible()
        passed("governance JSON roundtrip recomputes untrusted claims and rejects invalid input without data loss")

        open_menu(page, "governance-case-menu")
        page.get_by_role("button", name="接入当前工作台案例", exact=True).click()
        page.get_by_role("button", name="保留当前草稿", exact=True).click()
        governance_tab(page, "核查结果")
        expect(page.get_by_role("table", name="经济权益计算结果")).to_be_visible()
        open_menu(page, "governance-case-menu")
        page.get_by_role("button", name="接入当前工作台案例", exact=True).click()
        page.get_by_role("button", name="确认替换草稿", exact=True).click()
        governance_tab(page, "凭证与身份")
        page.get_by_role("button", name="添加文本凭证", exact=True).click()
        governance_tab(page, "凭证与身份")
        page.get_by_label("凭证文本 1", exact=True).fill("虚构表决权核验材料，仅课堂教学")
        governance_tab(page, "凭证与身份")
        page.get_by_label("凭证核验人 1", exact=True).fill("教学核验人")
        governance_tab(page, "凭证与身份")
        page.get_by_label("凭证状态 1", exact=True).select_option("verified")
        governance_compute(page, lambda: page.get_by_role("button", name="重新核查有效证据", exact=True).click())
        governance_tab(page, "关系与协议")
        page.get_by_label("权利主体", exact=True).select_option("P3")
        governance_tab(page, "关系与协议")
        evidence_id = page.get_by_label("权利凭证", exact=True).locator("option").nth(1).get_attribute("value")
        governance_tab(page, "关系与协议")
        page.get_by_label("权利凭证", exact=True).select_option(evidence_id)
        governance_tab(page, "关系与协议")
        page.get_by_role("button", name="添加权利记录", exact=True).click()
        voting = governance_compute(page, lambda: page.get_by_role("button", name="重新核查有效证据", exact=True).click())
        assert ["Controls", "P3", "C3"] in voting["claims"] and ["Controls", "P1", "C3"] not in voting["claims"]
        assert abs(voting["equity"]["owners"][0]["share"] - .342) < 1e-10
        governance_tab(page, "凭证与身份")
        page.get_by_label("凭证状态 1", exact=True).select_option("revoked")
        revoked = governance_compute(page, lambda: page.get_by_role("button", name="重新核查有效证据", exact=True).click())
        assert ["Controls", "P3", "C3"] not in revoked["claims"]
        governance_tab(page, "核查结果")
        expect(page.get_by_label("撤回结论列表")).to_contain_text("Controls(P3, C3)")
        governance_tab(page, "凭证与身份")
        page.get_by_label("凭证文本 1", exact=True).fill("已修改的虚构核验材料")
        governance_tab(page, "凭证与身份")
        expect(page.get_by_label("凭证状态 1", exact=True)).to_have_value("pending")
        expect(page.locator(".governance-hash")).to_contain_text("重新核查时生成")
        navigate(page,"案例概览")
        wait_ready(page)
        navigate(page,"证据与更新")
        governance_tab(page, "凭证与身份")
        expect(page.get_by_label("凭证文本 1", exact=True)).to_have_value("已修改的虚构核验材料")
        passed("manual evidence voting control, revocation, economic-share separation and draft persistence across routes")

        open_menu(page, "governance-case-menu")
        page.get_by_role("button", name="加载循环持股教学案例", exact=True).click()
        expect(page.get_by_role("button", name="确认替换草稿", exact=True)).to_be_visible()
        page.get_by_role("button", name="确认替换草稿", exact=True).click()
        cycle = governance_compute(page, lambda: page.get_by_role("button", name="重新核查有效证据", exact=True).click())
        assert abs(cycle["equity"]["owners"][0]["share"] - .4/.9) < 1e-10
        governance_tab(page, "核查结果")
        expect(page.get_by_role("table", name="经济权益计算结果")).to_contain_text("44.4444%")
        no_overflow(page)
        page.screenshot(path=str(OUTPUT / "governance-desktop.png"), full_page=True)
        open_menu(page, "governance-case-menu")
        page.get_by_role("button", name="接入当前工作台案例", exact=True).click()
        page.get_by_role("button", name="确认替换草稿", exact=True).click()
        governance_compute(page, lambda: page.get_by_role("button", name="重新核查有效证据", exact=True).click())
        governance_tab(page, "核查结果")
        expect(page.get_by_role("table", name="经济权益计算结果")).to_contain_text("34.2000%")
        passed("cyclic equity example solves infinite paths and safely resets to workbench case")

        # Cross-day re-evaluation must use the accepted draft, not an older saved revision.
        open_menu(page, "governance-options")
        page.get_by_label("跟随今天", exact=False).check()
        base_saved = governance_compute(page, lambda: page.get_by_role("button", name="保存新版本", exact=True).click(), "save")
        current_day = base_saved["payload"]["as_of"]
        from datetime import date, timedelta
        next_day = (date.fromisoformat(current_day) + timedelta(days=1)).isoformat()
        governance_tab(page, "关系与协议")
        page.get_by_label("选择有效期关系").select_option("holdings:C2:C3")
        governance_tab(page, "关系与协议")
        page.get_by_label("关系失效日", exact=True).fill(next_day)
        governance_tab(page, "关系与协议")
        page.get_by_role("button", name="更新关系设置", exact=True).click()
        governance_tab(page, "凭证与身份")
        page.get_by_role("button", name="添加文本凭证", exact=True).click()
        governance_tab(page, "凭证与身份")
        page.get_by_label("凭证文本 1", exact=True).fill("跨日核查不得覆盖的当前草稿")
        governance_compute(page, lambda: page.get_by_role("button", name="重新核查有效证据", exact=True).click())
        page.evaluate("""() => { const Original = window.Date; window.__originalDate = Original; window.Date = class extends Original { constructor(...args) { if (args.length) super(...args); else super(Original.now()+86400000); } }; }""")
        with page.expect_response(lambda r: r.url.endswith("/api/governance/preview"), timeout=45000) as pending:
            expect(page.get_by_label("核查日期", exact=True)).to_have_value(next_day, timeout=45000)
        automatic = pending.value.json()
        assert ["Controls", "P1", "C3"] not in automatic["claims"]
        governance_tab(page, "核查结果")
        expect(page.get_by_label("撤回结论列表")).to_contain_text("Controls(P1, C3)")
        governance_tab(page, "凭证与身份")
        expect(page.get_by_label("凭证文本 1", exact=True)).to_have_value("跨日核查不得覆盖的当前草稿")
        governance_tab(page, "历史与实验")
        expect(page.get_by_role("table", name="案例版本历史").locator("tbody tr")).to_have_count(1)
        page.evaluate("() => { window.Date = window.__originalDate; delete window.__originalDate; }")
        passed("cross-day active evidence auto-withdrawal preserves unsaved accepted draft and original saved revision")


        go(page,"#/entity/C3","企业分析")
        previous=len(errors)
        page.route("**/api/analyze",lambda route:route.abort("failed"))
        page.get_by_role("button",name="运行推理",exact=False).click()
        expect(page.locator(".error-banner")).to_contain_text("无法连接本地引擎")
        page.unroute("**/api/analyze")
        assert all("ERR_FAILED" in e for e in errors[previous:]),errors[previous:]
        del errors[previous:]
        compute(page,lambda:page.get_by_role("button",name="运行推理",exact=False).click())
        passed("network failure preserves current page and accepted draft, supports retry")

        page.get_by_role("link",name="查看确定性推理证据",exact=True).click()
        expect(page).to_have_url(url+"/"+canonical_fragment("#/entity/C3/evidence"))
        page.get_by_role("link",name="在关系图谱中查看证明",exact=True).click()
        expect(page).to_have_url(url+"/#/labs/lab02/deterministic/graph/C3")
        page.go_back();expect(page).to_have_url(url+"/"+canonical_fragment("#/entity/C3/evidence"))
        expect(page.locator(".proof-tree")).to_be_visible()
        page.go_forward();expect(page).to_have_url(url+"/#/labs/lab02/deterministic/graph/C3")
        page.goto(url+"/"+canonical_fragment("#/entity/C6/evidence"))
        wait_ready(page)
        expect(page.locator(".entity-kicker")).to_contain_text("C6")
        expect(page.locator(".proof-tree")).to_be_visible()
        page.goto(url+"/#/entity/NO_SUCH_ENTITY")
        expect(page.get_by_role("heading",name="当前案例没有这个实体",exact=True)).to_be_visible(timeout=20000)
        page.goto(url+"/"+canonical_fragment("#/entity/P1"))
        wait_ready(page)
        expect(page.locator(".person-holdings")).to_be_visible()
        assert page.locator(".score-row").count()==0
        passed("browser Back/Forward, direct entity links, unknown IDs and natural-person pages behave correctly")

        long_case=context.request.get(url+"/api/cases/base").json()["case"]
        long_case=json.loads(json.dumps(long_case).replace('"P1"','"holding-0"'))
        for entity in long_case["entities"]:
            if entity["eid"]=="C3":entity["name"]="超长企业名称显示验证"*8
        payload=json.dumps({"case":long_case,"target_id":"C3"},ensure_ascii=False).encode("utf-8")
        page.get_by_label("导入股权图JSON",exact=True).set_input_files({"name":"long.json","mimeType":"application/json","buffer":payload})
        expect(page.get_by_role("dialog")).to_be_visible()
        compute(page,lambda:page.get_by_role("dialog").get_by_role("button",name="导入案例",exact=True).click())
        go(page,"#/workbench","案例概览")
        go(page,"#/labs/lab02/hybrid/overview","全案例分析")
        wait_ready(page)
        page.get_by_role("link",name="查看C3分析",exact=True).click()
        no_overflow(page)
        go(page,"#/graph","关系图谱")
        expect(page.locator(".graph-canvas")).to_be_visible()
        assert page.locator(".graph-canvas").evaluate("el=>el._cyreg.cy.getElementById('entity:holding-0').length")==1
        go(page,"#/workbench","案例概览")
        open_menu(page,"case-manager")
        with page.expect_response(lambda response:response.url.endswith("/api/cases/extended")) as loaded:
            page.get_by_label("选择案例",exact=True).select_option("extended")
        assert len(loaded.value.json()["case"]["entities"])==29
        wait_ready(page)
        expect(page.locator(".directory-row")).to_have_count(20)
        open_menu(page,"case-manager")
        with page.expect_response(lambda response:response.url.endswith("/api/cases/base")):
            page.get_by_label("选择案例",exact=True).select_option("base")
        wait_ready(page)
        passed("long entity names, isolated imported IDs and case changes work across independent pages")

        for width,height,label in ((1508,768,"laptop"),(900,1000,"tablet"),(390,844,"mobile"),(320,740,"narrow"),(720,500,"zoom200-equivalent")):
            page.set_viewport_size({"width":width,"height":height})
            page.goto(url+"/#/workbench")
            wait_ready(page)
            no_overflow(page)
            page.screenshot(path=str(OUTPUT / ("overview-"+label+".png")),full_page=True)
            page.get_by_role("link",name="查看C3资料",exact=True).click()
            expect(page.get_by_role("heading",name="实体资料",exact=True)).to_be_visible()
            no_overflow(page)
            page.screenshot(path=str(OUTPUT / ("entity-"+label+".png")),full_page=True)
            for fragment,heading in (("#/entity/C3/evidence","推理证据"),("#/entity/C3/edit","调整企业数据"),("#/settings","分析设置"),("#/verification","验证与推理记录"),("#/graph","关系图谱"),("#/experiments","实验对比"),("#/rules","规则知识库"),("#/governance","证据与更新")):
                go(page,fragment,heading)
                if fragment=="#/graph":graph_dimensions(page)
                no_overflow(page)
                if fragment=="#/governance":
                    for identifier in ("governance-options","governance-case-menu"):
                        open_menu(page,identifier);no_overflow(page);page.keyboard.press("Escape")
                page.screenshot(path=str(OUTPUT / (fragment.split('/')[1]+"-"+label+".png")),full_page=True)
            passed(label+" overview and task pages, graph and menus without horizontal overflow")

        # Fresh document tests first Tab; hash navigation intentionally focuses the new main content.
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url+"/#/workbench")
        wait_ready(page)
        page.keyboard.press("Tab")
        expect(page.locator(".skip-link")).to_be_focused()
        page.keyboard.press("Enter")
        expect(page.locator("#main-content")).to_be_focused()
        go(page,"#/labs/lab02/hybrid/overview","全案例分析")
        wait_ready(page)
        page.get_by_role("link",name="查看C3分析",exact=True).click()
        page.get_by_role("link",name="查看确定性推理证据",exact=True).focus()
        page.keyboard.press("Enter")
        expect(page).to_have_url(url+"/"+canonical_fragment("#/entity/C3/evidence"))
        assert page.evaluate("matchMedia('(prefers-reduced-motion: reduce)').matches")
        passed("keyboard skip link and real page links preserve reduced-motion behavior")
        contrast=page.evaluate("""() => {
          const ctx=document.createElement('canvas').getContext('2d',{willReadFrequently:true});
          const rgb=name=>{ctx.fillStyle=getComputedStyle(document.documentElement).getPropertyValue(name).trim();ctx.fillRect(0,0,1,1);return [...ctx.getImageData(0,0,1,1).data].slice(0,3);};
          const lum=rgb=>rgb.map(v=>{v/=255;return v<=.04045?v/12.92:((v+.055)/1.055)**2.4;}).reduce((s,v,i)=>s+v*[.2126,.7152,.0722][i],0);
          return [['--ink','--bg'],['--muted','--paper'],['--muted','--surface'],['--primary','--primary-soft'],['--accent-ink','--accent-soft'],['--danger','--danger-soft'],['--on-primary','--primary']].map(([fg,bg])=>{const a=lum(rgb(fg)),b=lum(rgb(bg));return {fg,bg,ratio:(Math.max(a,b)+.05)/(Math.min(a,b)+.05)};});
        }""")
        assert all(row["ratio"]>=4.5 for row in contrast),contrast
        passed("warm palette text pairs meet WCAG 4.5:1 contrast")
        assert not errors,errors
        browser.close()
    report={"created_utc":datetime.now(timezone.utc).isoformat(),"url":url,"layout":"task-specific-pages","checks":checks,"unexpected_browser_errors":errors,"contrast":contrast,"artifacts":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUTPUT.glob("*.png") if p.name.startswith(("overview-","entity-","evidence-","graph-","settings-","verification-","experiments-","rules-","governance-"))}}
    (OUTPUT / "browser-checks.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"{len(checks)} browser check groups passed. Artifacts: {OUTPUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--isolated", action="store_true", help="使用临时版本数据库和独立本机服务，不写入用户案例历史")
    arguments = parser.parse_args()
    if not arguments.url.startswith(("http://127.0.0.1:", "http://localhost:")):
        parser.error("Only a local loopback test server is allowed")
    if arguments.isolated:
        with isolated_server() as url:
            run(url)
    else:
        run(arguments.url.rstrip("/"))
