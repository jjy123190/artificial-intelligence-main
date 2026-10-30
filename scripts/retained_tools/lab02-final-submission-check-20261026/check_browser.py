"""Exercise the five experiment-two pages on the isolated server."""
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
from playwright.sync_api import sync_playwright, expect

root = Path(sys.argv[1]).resolve()
url = sys.argv[2].rstrip("/")
qa = Path(__file__).resolve().parent
with urllib.request.urlopen(url + "/api/health") as response:
    health = json.load(response)
assert health["workspace_id"] == hashlib.sha256(str(root).casefold().encode()).hexdigest()[:16]
checks, errors = [], []

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(channel="chrome", headless=True)
    context = browser.new_context(viewport={"width": 1440, "height": 900}, reduced_motion="reduce")
    page = context.new_page()
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(url + "/#/labs/lab02")
    expect(page.locator(".module-row")).to_have_count(5, timeout=30000)

    def go(task, title):
        page.evaluate("hash => location.hash = hash", "#/labs/lab02/" + task)
        expect(page.get_by_role("heading", name=title, exact=True)).to_be_visible(timeout=30000)

    def run(button="运行本节实验"):
        with page.expect_response(lambda r: r.url.endswith("/api/labs/lab02/run") and
                                  r.request.method == "POST", timeout=30000) as pending:
            page.get_by_role("button", name=button, exact=True).click()
        response = pending.value
        assert response.ok, response.text()
        return response.json()

    go("modeling", "知识表示与建模")
    expect(page.locator(".rule-row")).to_have_count(27)
    facts = run("构建当前案例的谓词事实")
    assert facts["facts"] and facts["derived_facts"]
    checks.append("27 rules and first-order facts")

    go("deterministic", "确定性推理")
    result = run()
    assert result["proved"] and result["equivalent"]
    page.get_by_label("实验二推理方法", exact=True).select_option("backward")
    assert run()["backward_proof"]
    page.get_by_label("实验二推理方法", exact=True).select_option("resolution")
    assert run()["status"] == "proved"
    checks.append("forward, backward and resolution proofs")

    go("uncertain", "不确定性推理")
    result = run()
    assert 0 < result["bayes"]["score"] < 1 and 0 < result["fuzzy"]["score"] < 1
    page.get_by_label("实验二参数来源", exact=True).select_option("indicator:7:adaptive")
    trained = run()
    assert trained["model_source"] == "indicator:7:adaptive"
    checks.append("Bayes/Mamdani and current trained model selection")

    go("hybrid", "混合推理与创新验证")
    decisions = run()["decisions"]
    assert len(decisions) == 3 and len({d["score"] for d in decisions}) == 1
    assert all(d["proofs"] for d in decisions)
    checks.append("three hybrid actions retain shared scores and proofs")

    go("optimization", "算法优化与对比实验")
    expect(page.locator('[aria-label="各方法的实测指标"] tbody tr')).to_have_count(11, timeout=30000)
    page.get_by_role("button", name="确定性优化", exact=True).click()
    expect(page.locator('[aria-label="确定性优化实测"] tbody tr')).to_have_count(3)
    page.get_by_role("button", name="评分与参数优化", exact=True).click()
    expect(page.get_by_role("table", name="评分优化与消融实测").locator("tbody tr")).to_have_count(14)
    page.get_by_text("查看先验、CPT与模糊参数敏感性原始实测", exact=True).click()
    expect(page.get_by_role("table", name="参数敏感性实测").locator("tbody tr")).to_have_count(45)
    page.get_by_role("button", name="扩展：证据冲突", exact=True).click()
    expect(page.get_by_role("table", name="证据冲突扩展实测").locator("tbody tr")).to_have_count(24)
    checks.append("fresh optimization, sensitivity, ablation and conflict tables")
    page.screenshot(path=str(qa / "standalone-desktop.png"), full_page=True)
    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    page.screenshot(path=str(qa / "standalone-mobile.png"), full_page=True)
    checks.append("desktop and mobile navigation without page overflow")
    browser.close()

assert not errors, errors
report = {"url": url, "source_root": str(root), "passed": True, "checks": checks, "page_errors": errors}
(qa / "browser-checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
