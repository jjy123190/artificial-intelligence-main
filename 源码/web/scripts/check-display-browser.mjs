/** Real-browser text-only regression: all modules, live search, evidence exports. */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { chromium } from "playwright";

const base = process.env.LAB03_TEST_URL || "http://127.0.0.1:8765";
const browser = await chromium.launch({
  headless: true,
  channel: process.env.PLAYWRIGHT_CHANNEL || undefined,
});
const context = await browser.newContext({ viewport: { width: 1366, height: 900 } });
const page = await context.newPage();
const errors = [];
page.on("pageerror", (error) => errors.push(error.message));
const forbidden = /教学|合成数据|合成标签|合成设定|合成指标样本|课堂演示/;
const checks = [];
async function clearText(label) {
  await page.locator("details").evaluateAll((nodes) => nodes.forEach((node) => { node.open = true; }));
  assert.doesNotMatch(await page.locator("body").textContent(), forbidden, label);
  assert.doesNotMatch(await page.title(), forbidden, label + " title");
  const attributes = await page.locator("[title],[aria-label],[aria-description],input,textarea,option").evaluateAll(
    (nodes) => nodes.flatMap((node) => [
      node.getAttribute("title"), node.getAttribute("aria-label"), node.getAttribute("aria-description"),
      node.tagName === "OPTION" ? node.textContent : node.value,
    ]).filter((value) => typeof value === "string"),
  );
  for (const text of attributes) assert.doesNotMatch(text, forbidden, label + " attribute");
  assert.equal(await page.getByText("部分实验数据不可用", { exact: true }).count(), 0, label);
  checks.push(label);
}
async function visit(route) {
  await page.goto(base + "/#/" + route);
  await page.locator("main").filter({ visible: true }).first().waitFor();
  if (route.startsWith("labs/lab03/")) await page.locator(".lab3-page .task-section").first().waitFor();
  else await page.waitForLoadState("networkidle");
}
try {
  const snapshot = await (await context.request.get(base + "/api/labs/lab03/snapshot")).json();
  assert.deepEqual(snapshot.warnings, []);
  // Raw measurements keep their identity and provenance; only rendered labels differ.
  assert.ok(snapshot.queries.some((query) => query.start_name.includes("教学")));
  for (const route of [
    "home", "labs/lab01/representation", "labs/lab01/industry", "labs/lab01/demo", "labs/lab01/limitations",
    "labs/lab02/modeling", "labs/lab02/deterministic", "labs/lab02/uncertain", "labs/lab02/optimization", "labs/lab02/hybrid",
    "labs/lab03/search", "labs/lab03/optimization", "labs/lab03/assistant", "labs/lab03/evaluation",
    "labs/lab02/modeling/rules", "labs/lab02/modeling/governance", "labs/lab02/hybrid/settings",
    "labs/lab02/hybrid/overview", "labs/lab02/deterministic/records", "graph", "case/C3",
  ]) {
    await visit(route);
    await clearText(route);
  }
  for (const queryId of ["test-7-0-Q0", "test-7-0-Q1", "test-7-0-Q2"]) {
    await visit("labs/lab03/search");
    await page.getByLabel("实验三问题", { exact: true }).selectOption(queryId);
    await page.getByLabel("实验三搜索方法", { exact: true }).selectOption("hybrid_llm");
    const responseReady = page.waitForResponse((response) => response.url().endsWith("/api/labs/lab03/run") && response.request().method() === "POST");
    await page.getByRole("button", { name: "运行路径搜索", exact: true }).click();
    const response = await responseReady;
    assert.equal(response.ok(), true);
    const raw = await response.json();
    await page.getByRole("button", { name: "导出本次结果" }).waitFor();
    assert.equal(raw.query.id, queryId);
    assert.equal(raw.llm.mode, "replay");
    assert.ok(raw.query.text.includes("教学投资人"));
    await clearText(queryId + " live tree and path");
    const downloadReady = page.waitForEvent("download");
    await page.getByRole("button", { name: "导出本次结果" }).click();
    const download = await downloadReady;
    const exported = JSON.parse(await readFile(await download.path(), "utf8"));
    assert.deepEqual(exported, raw, "UI wording must not rewrite exported measurements or evidence");
    checks.push(queryId + " raw JSON export unchanged");
  }
  await visit("labs/lab01/demo");
  await page.getByLabel("实验一演示任务", { exact: true }).selectOption("penetration");
  await page.getByRole("button", { name: "运行实验一任务", exact: true }).click();
  await page.getByRole("button", { name: "导出结果", exact: true }).waitFor();
  await clearText("lab01 penetration result and note");
  for (const module of ["deterministic", "uncertain", "hybrid"]) {
    await visit("labs/lab02/" + module);
    await page.getByRole("button", { name: "运行本节实验", exact: true }).click();
    await page.getByRole("button", { name: "导出结果", exact: true }).waitFor();
    await clearText("lab02 " + module + " result and JSON preview");
  }
  await visit("labs/lab02/modeling/governance");
  await page.getByText("核查案例", { exact: true }).click();
  await page.getByRole("button", { name: "加载循环持股案例", exact: true }).click();
  await page.getByText(/当前案例：循环持股权益案例/).waitFor();
  const preview = page.waitForResponse((response) => response.url().endsWith("/api/governance/preview") && response.request().method() === "POST");
  await page.getByRole("button", { name: "重新核查有效证据", exact: true }).click();
  assert.equal((await preview).ok(), true);
  await page.getByText("循环持股经济权益", { exact: true }).waitFor();
  await clearText("governance example names and verified evidence");
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ passed: checks.length, checks, browserErrors: errors, rawExportsUnchanged: true }, null, 2));
} finally {
  await browser.close();
}
