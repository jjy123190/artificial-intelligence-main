"""Read-only notation audit: rule variables versus concrete teaching data."""
from __future__ import annotations
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from backend.worker import analyze_payload, preset_case, query_payload
from core.kb.rules import build_rules
from core.kb.rules_dsl import eval_guard, guard_vars
from experiments.lab01.service import run_task as lab_one
from experiments.lab02.service import run_task as lab_two
from playwright.sync_api import sync_playwright, expect
from scripts.check_web import isolated_server
OUTPUT = Path(__file__).resolve().parent
VARIABLE = re.compile(r"\?[A-Za-z][A-Za-z0-9_]*")

def hits(value, path="root"):
    found = []
    if isinstance(value, str):
        if VARIABLE.search(value) or "\ufffd" in value or "??" in value:
            found.append({"path": path, "value": value[:300]})
    elif isinstance(value, (list, tuple)):
        for index, entry in enumerate(value):
            found.extend(hits(entry, f"{path}[{index}]"))
    elif isinstance(value, dict):
        for key, entry in value.items():
            found.extend(hits(entry, f"{path}.{key}"))
    return found

def check_clean(value, label):
    found = hits(value)
    assert not found, (label, found[:5])
    return {"label": label, "unexpected_variable_or_encoding_hits": 0}

def source_and_result_checks():
    report = {"csv": [], "templates": {}, "lab01": [], "cases": [], "guards": []}
    for filename in ("persons", "companies", "holdings", "guarantees", "trades", "kinships"):
        source = ROOT / "data" / "raw" / (filename + ".csv")
        with source.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        check_clean(rows, str(source.relative_to(ROOT)))
        report["csv"].append({"file": str(source.relative_to(ROOT)), "rows": len(rows), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    for task in ("snapshot", "rules", "penetration", "path"):
        result = lab_one({"task": task, "source": "P1", "target": "C3"})
        templates = result.pop("rules")
        check_clean(result, "lab01/" + task)
        for rule in templates:
            body = {arg for row in rule["conditions"] for arg in row[1:] if isinstance(arg, str) and arg.startswith("?")}
            head = {arg for row in rule["conclusions"] for arg in row[1:] if isinstance(arg, str) and arg.startswith("?")}
            assert head <= body, rule["name"]
        report["templates"]["lab01"] = {"rules": len(templates), "all_head_variables_bound_by_premises": True}
        report["lab01"].append({"task": task, "input_facts": len(result["facts"]), "unexpected_variable_or_encoding_hits": 0, "total_facts": result["stats"].get("total_facts")})
    rules = build_rules()
    for rule in rules:
        body = {value for pattern in rule.antecedents for value in pattern.variables()}
        assert set(rule.consequent.variables()) | set(guard_vars(rule.guard)) <= body, rule.name
    report["templates"]["lab02"] = {"rules": len(rules), "guarded_rules": sum(rule.guard is not None for rule in rules), "all_head_and_guard_variables_bound_by_premises": True}
    by_id = {rule.name: rule for rule in rules}
    guard_cases = [
        ("R01", {"?x": "P1", "?y": "C1", "?r": 0.8}, True),
        ("R01", {"?x": "P1", "?y": "C1", "?r": 0.5}, False),
        ("R01", {"?x": "P1", "?y": "C1", "?r": 0.2}, False),
        ("R01", {"?x": "C1", "?y": "C1", "?r": 0.8}, False),
        ("R02", {"?rel": "配偶"}, True),
        ("R02", {"?rel": "朋友"}, False),
        ("R04", {"?p": "P1", "?q": "P2", "?r1": 0.3, "?r2": 0.25}, True),
        ("R04", {"?p": "P1", "?q": "P2", "?r1": 0.2, "?r2": 0.25}, False),
        ("R04", {"?p": "P1", "?q": "P2", "?r1": 0.25, "?r2": 0.25}, False),
        ("R04", {"?p": "P1", "?q": "P1", "?r1": 0.3, "?r2": 0.3}, False),
    ]
    for rule_id, binding, wanted in guard_cases:
        actual = eval_guard(by_id[rule_id].guard, binding)
        assert actual is wanted, (rule_id, binding, actual, wanted)
        report["guards"].append({"rule": rule_id, "binding": binding, "passes": actual})
    for case_id in ("base", "extended"):
        case = preset_case(case_id)
        check_clean(case, case_id)
        payload = {"case": case, "completeness": {key: True for key in ("holdings", "guarantees", "trades", "registry")}, "model_source": "manual", "source": "P1", "target": "C3", "threshold": 0.6}
        case_report = {"id": case_id, "entities": len(case["entities"]), "tasks": [], "analysis_methods": [], "queries": []}
        for task in ("modeling", "deterministic", "uncertain", "hybrid", "resolution"):
            result = lab_two({**payload, "task": task})
            if task == "resolution":
                proof = result.pop("proof")
                check_clean(result, case_id + "/resolution-goal-and-result")
                assert result["proved"] and result["status"] == "proved", result
                assert any("□" in line or "⊥" in line for line in proof), proof[-1]
                case_report["tasks"].append({"task": task, "goal": result["goal"], "proved": True, "legitimate_intermediate_formula_variable_lines": len(hits(proof)), "examples": [entry["value"] for entry in hits(proof)[:2]], "last_proof_line": proof[-1]})
            else:
                check_clean(result, case_id + "/" + task)
                case_report["tasks"].append({"task": task, "unexpected_variable_or_encoding_hits": 0, "input_facts": result["input_facts"]})
        for method in ("hybrid_locked", "hybrid_unlocked", "hybrid_calibrated", "bayes_only", "fuzzy_only", "rules_only"):
            analysis = analyze_payload({**payload, "method": method})
            check_clean(analysis, case_id + "/analysis/" + method)
            case_report["analysis_methods"].append({"method": method, "provenance_facts": len(analysis["provenance"]), "companies": len(analysis["decisions"]), "unexpected_variable_or_encoding_hits": 0})
        for source, target in (("P1", "C1"), ("P1", "C3"), ("P4", "C12"), ("C6", "C6")):
            result = query_payload({**payload, "predicate": "Controls", "arguments": [source, target]})
            check_clean(result, case_id + "/query/" + source + "/" + target)
            case_report["queries"].append({"source": source, "target": target, "proved": result["proved"], "equivalent": result["equivalent"], "unexpected_variable_or_encoding_hits": 0})
        try:
            query_payload({**payload, "predicate": "Controls", "arguments": ["?x", "C3"]})
        except ValueError as error:
            case_report["web_variable_query_rejected"] = str(error)
        else:
            raise AssertionError("Web concrete query accepted a free variable")
        report["cases"].append(case_report)
        print("PASS concrete facts, scoring, proof outputs and query boundaries: " + case_id, flush=True)
    return report

SCAN = r"""() => {
  const main = document.querySelector('#main-content');
  const walker = document.createTreeWalker(main, NodeFilter.SHOW_TEXT);
  const variables = [], encoding = [];
  while (walker.nextNode()) {
    const node = walker.currentNode, parent = node.parentElement;
    if (!parent || !parent.checkVisibility({checkVisibilityCSS: true})) continue;
    const value = node.textContent.trim();
    if (/\?[A-Za-z][A-Za-z0-9_]*/.test(value)) variables.push({text: value.slice(0, 360), container: parent.closest('article,table,pre,p')?.className || parent.tagName});
    if (/\uFFFD|\?{2,}/.test(value)) encoding.push(value.slice(0, 200));
  }
  return {title: main.querySelector('h1')?.textContent || '', variables, encoding,
    mainCount: document.querySelectorAll('#main-content').length,
    overflow: document.documentElement.scrollWidth > innerWidth + 1};
}"""

ROUTES = [
    ("home", "#/home"), ("lab01", "#/labs/lab01"), ("lab02", "#/labs/lab02"),
    ("public-overview", "#/workbench"), ("public-person", "#/case/P1"), ("public-company", "#/case/C3"),
    ("public-graph", "#/graph"), ("public-focused-graph", "#/graph/C3"),
    ("lab01-representation", "#/labs/lab01/representation"), ("lab01-industry", "#/labs/lab01/industry"),
    ("lab01-demo", "#/labs/lab01/demo"), ("lab01-limitations", "#/labs/lab01/limitations"),
    ("lab02-modeling", "#/labs/lab02/modeling"), ("lab02-rules", "#/labs/lab02/modeling/rules"),
    ("lab02-input", "#/labs/lab02/modeling/entity/C3/edit"), ("lab02-governance", "#/labs/lab02/modeling/governance"),
    ("lab02-deterministic", "#/labs/lab02/deterministic"), ("lab02-evidence", "#/labs/lab02/deterministic/entity/C3/evidence"),
    ("lab02-reasoning-graph", "#/labs/lab02/deterministic/graph/C3"), ("lab02-records", "#/labs/lab02/deterministic/records"),
    ("lab02-uncertain", "#/labs/lab02/uncertain"), ("lab02-optimization", "#/labs/lab02/optimization"),
    ("lab02-results", "#/labs/lab02/optimization/results"), ("lab02-hybrid", "#/labs/lab02/hybrid"),
    ("lab02-analysis", "#/labs/lab02/hybrid/overview"), ("lab02-company-analysis", "#/labs/lab02/hybrid/entity/C3"),
    ("lab02-settings", "#/labs/lab02/hybrid/settings"),
]

def browser_checks(url):
    report = {"routes": [], "page_errors": [], "screenshots": []}
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        for viewport in ({"width": 1508, "height": 768}, {"width": 390, "height": 844}, {"width": 320, "height": 900}):
            context = browser.new_context(viewport=viewport, reduced_motion="reduce")
            page = context.new_page()
            page.on("pageerror", lambda error: report["page_errors"].append(str(error)))
            for name, fragment in ROUTES:
                page.goto(url + "/" + fragment, wait_until="domcontentloaded")
                expect(page.locator("#main-content")).to_be_visible(timeout=30000)
                page.wait_for_load_state("networkidle", timeout=30000)
                expect(page.get_by_role("heading", name="页面未找到", exact=True)).to_have_count(0)
                if name in ("lab02-rules", "lab02-modeling"):
                    expect(page.locator(".rule-row")).to_have_count(27)
                if name == "lab01-representation":
                    page.get_by_role("button", name="产生式规则", exact=True).click()
                    expect(page.get_by_role("table", name="实验一产生式规则")).to_be_visible()
                if name == "lab02-modeling":
                    page.get_by_role("button", name="构建当前案例的谓词事实", exact=True).click()
                    expect(page.get_by_role("table", name="实验二一阶谓词事实")).to_be_visible(timeout=30000)
                    assert not VARIABLE.search(page.get_by_role("table", name="实验二一阶谓词事实").inner_text())
                scan = page.evaluate(SCAN)
                assert not scan["encoding"], (fragment, scan["encoding"])
                assert scan["mainCount"] == 1, fragment
                assert not scan["overflow"], (fragment, viewport)
                permitted = name in ("lab01-representation", "lab02-modeling", "lab02-rules")
                assert permitted or not scan["variables"], (fragment, scan["variables"][:3])
                report["routes"].append({"name": name, "route": fragment, "viewport": viewport, "legitimate_rule_template_variable_text_nodes": len(scan["variables"]), "examples": scan["variables"][:3], "title": scan["title"], "horizontal_overflow": False, "encoding_errors": []})
                if name in ("lab01-representation", "lab02-rules"):
                    filename = name + "-" + str(viewport["width"]) + ".png"
                    page.screenshot(path=str(OUTPUT / filename))
                    report["screenshots"].append(filename)
                if name == "lab02-deterministic":
                    for mode in ("forward", "backward", "resolution"):
                        page.get_by_label("实验二推理方法", exact=True).select_option(mode)
                        page.get_by_role("button", name="运行本节实验", exact=True).click()
                        expect(page.locator(".task-result")).to_be_visible(timeout=30000)
                        page.wait_for_load_state("networkidle", timeout=30000)
                        result_text = page.locator(".task-result").inner_text()
                        variable_count = len(VARIABLE.findall(result_text))
                        assert mode == "resolution" or variable_count == 0, (mode, result_text)
                        assert "\ufffd" not in result_text and "??" not in result_text
                        report["routes"].append({"name": name + "-" + mode, "route": fragment, "viewport": viewport, "legitimate_intermediate_formula_variables": variable_count, "proved_goal_has_free_variables": False})
            context.close()
            print("PASS rendered notation, concrete facts and overflow checks at width " + str(viewport["width"]), flush=True)
        browser.close()
    assert not report["page_errors"], report["page_errors"]
    return report

def main():
    source_checks = source_and_result_checks()
    with isolated_server() as url:
        browser = browser_checks(url)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "app_source_modified": False, "scope": "read-only audit of source teaching CSVs, both presets, both rule libraries, five lab02 task adapters, six analysis methods, and rendered canonical pages", "source_checks": source_checks, "browser": browser, "findings": [
        {"severity": "P2", "name": "Unexplained rule-variable notation", "locations": ["web/src/components/RulesPage.tsx", "web/src/components/LabOne.tsx"], "impact": "Question-prefixed logical variables look like missing data to learners.", "recommendation": "Keep engine notation and add a shared Chinese variable legend plus a concrete example."},
        {"severity": "P2", "name": "Guarded formulas omit explicit guard expressions", "locations": ["web/src/components/RulesPage.tsx", "web/src/components/LabOne.tsx", "backend/api.py"], "impact": "Displayed premise-to-conclusion arrows can be read as unconditional; numeric and comparison guards are enforced in the engines but only summarized in prose or flagged.", "recommendation": "Expose complete trigger conditions from the authoritative rule definition without changing inference semantics."},
    ]}
    (OUTPUT / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PASS notation audit: " + str(len(browser["routes"])) + " rendered states; no unexpected concrete-data variables or replacement characters", flush=True)

if __name__ == "__main__":
    main()
