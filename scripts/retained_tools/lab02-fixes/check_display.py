"""Check that the real scoring page labels fallback baselines from saved metadata."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.check_web import isolated_server
from experiments.lab02.eval.artifacts import fingerprint
from playwright.sync_api import sync_playwright, expect


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    out = Path(__file__).resolve().parent
    errors, checks, screenshots = [], [], []
    with isolated_server() as url, sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900}, reduced_motion="reduce")
        response = context.request.get(url + "/api/experiments")
        assert response.ok
        rows = [r for r in response.json()["indicator_summary"] if r["scenario"] == "clean"]
        expected_fallbacks = [r for r in rows if (r.get("ablation_fallback_runs") or 0) > 0]
        assert len(rows) == 14 and expected_fallbacks
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url + "/#/labs/lab02/optimization", wait_until="domcontentloaded")
        tab = page.get_by_role("button", name="评分与参数优化", exact=True)
        expect(tab).to_be_visible(timeout=30000)
        tab.click()
        table = page.get_by_role("table", name="评分优化与消融实测")
        expect(table.locator("tbody tr")).to_have_count(len(rows))
        components = {"bayes": "Bayes", "fuzzy": "Mamdani", "rule": "指标规则"}
        for index, row in enumerate(rows):
            displayed = table.locator("tbody tr").nth(index)
            count = int(row.get("ablation_fallback_runs") or 0)
            if count:
                baseline = "／".join(components[c] for c in row["ablation_fallback_component"].split(";"))
                runs = int(row["ablation_runs"])
                assert count == row["ablation_fallback_runs"] and runs == row["ablation_runs"]
                expected = f"{count}/{runs} 个种子回退为 {baseline} 基线"
                expect(displayed).to_contain_text(expected)
                checks.append({"model": row["model"], "fallback_note": expected})
            else:
                assert "个种子回退为" not in displayed.inner_text()
        expect(page.get_by_text("不能解释为组件边际贡献", exact=False)).to_be_visible()
        for name, width in (("desktop", 1440), ("mobile", 390)):
            page.set_viewport_size({"width": width, "height": 900})
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"), name
            screenshot = out / ("fallback-" + name + ".png")
            page.screenshot(path=str(screenshot), full_page=True)
            screenshots.append(screenshot.relative_to(ROOT).as_posix())
            checks.append({"viewport": name, "width": width, "no_page_horizontal_overflow": True})
        browser.close()
    assert not errors, errors
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "successful": True,
              "checks": checks, "browser_errors": errors, "screenshots": screenshots,
              "source_sha256": {"web/src/components/AdditionalExperiments.tsx": fingerprint(ROOT / "web/src/components/AdditionalExperiments.tsx")},
              "built_index_sha256": fingerprint(ROOT / "web/dist/index.html")}
    (out / "browser-checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
