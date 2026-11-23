import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const { chromium } = await import(pathToFileURL(path.join(root, "源码/web/node_modules/playwright/index.mjs")));
const arg = (key, fallback) => { const at = process.argv.indexOf(key); return at < 0 ? fallback : process.argv[at + 1]; };
const base = arg("--base-url", "http://127.0.0.1:18876");
const temporary = arg("--tmp", "");
const rehearsal = process.argv.includes("--rehearse");
if (!temporary) throw new Error("Pass a temporary --tmp directory.");
await fs.mkdir(path.join(temporary, "raw-video"), { recursive: true });
await fs.mkdir(path.join(temporary, "assets"), { recursive: true });
const browser = await chromium.launch({ channel: "chrome", headless: true });
const context = await browser.newContext({ viewport: { width: 1600, height: 900 },
  deviceScaleFactor: 1, ...(rehearsal ? {} : { recordVideo: { dir: path.join(temporary, "raw-video"), size: { width: 1600, height: 900 } } }) });
const page = await context.newPage();
const video = page.video();
const pageCreatedAt = Date.now();
const errors = [];
page.on("pageerror", error => errors.push(error.message));
const runResponses = [];
async function click(locator) {
  const box = await locator.boundingBox();
  if (box) await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2, { steps: 18 });
  await locator.click();
}
async function run() {
  const runButton = page.getByRole("button", { name: "运行分类任务", exact: true });
  if (!(await runButton.isEnabled())) {
    const seed = await page.getByRole("spinbutton", { name: "抽样种子", exact: true }).inputValue();
    throw new Error(`Run button is disabled; check model readiness and seed (expected 0-9999, got ${seed}).`);
  }
  const [value] = await Promise.all([
    page.waitForResponse(r => r.url().endsWith("/api/labs/lab04/run") && r.request().method() === "POST", { timeout: 120000 }),
    click(runButton),
  ]);
  if (!value.ok()) throw new Error(`Live inference failed: ${value.status()}`);
  const data = await value.json();
  runResponses.push({ method: data.method, seed: data.seed, shots: data.shots, correct: data.correct,
    total: data.total, accuracy: data.accuracy, compute: data.compute, checkpoint_sha256: data.checkpoint_sha256 });
  await page.getByRole("heading", { name: "本次分类结果", exact: true }).waitFor();
}
async function top() { await page.evaluate(() => window.scrollTo({ top: 0, behavior: "smooth" })); }
async function showSection(locator) {
  await locator.evaluate(element => window.scrollTo({
    top: window.scrollY + element.getBoundingClientRect().top - 102,
    behavior: "smooth",
  }));
}
try {
  const state = await page.request.get(`${base}/api/labs/lab04/snapshot`);
  if (!state.ok() || !(await state.json()).available) throw new Error("Only a verified, ready model can be recorded.");
  await page.goto(`${base}/#/home`, { waitUntil: "networkidle" });
  await page.getByRole("heading", { name: "股权知识与智能推理实验平台", exact: true }).waitFor();
  await page.waitForTimeout(750);
  const start = Date.now();
  const offset = (start - pageCreatedAt) / 1000;
  const at = async seconds => { const wait = start + seconds * 1000 - Date.now(); if (!rehearsal && wait > 0) await page.waitForTimeout(wait); };
  await at(4);
  await click(page.getByRole("link", { name: "实验四", exact: true }));
  await at(7);
  await click(page.locator('a[href="#/labs/lab04/inference"]'));
  await page.getByRole("combobox", { name: "推理方法", exact: true }).waitFor();
  await at(10);
  await page.getByRole("combobox", { name: "推理方法", exact: true }).selectOption("baseline");
  await page.getByRole("combobox", { name: "每类标注样本", exact: true }).selectOption("1");
  await page.getByRole("spinbutton", { name: "抽样种子", exact: true }).fill("7");
  await at(14);
  await run();
  await at(21);
  await showSection(page.locator(".lab4-support-grid"));
  await page.waitForTimeout(500);
  await page.locator(".lab4-support-grid").screenshot({ path: path.join(temporary, "assets/support.png") });
  await at(27);
  await showSection(page.getByRole("heading", { name: "查询集预测", exact: true }));
  await at(32);
  await top();
  await page.getByRole("combobox", { name: "推理方法", exact: true }).selectOption("calibrated");
  await page.getByRole("combobox", { name: "校准策略", exact: true }).selectOption("full");
  await at(36);
  await run();
  await top();
  await page.waitForTimeout(700);
  await page.screenshot({ path: path.join(temporary, "assets/platform.png") });
  await at(44);
  await showSection(page.locator("#lab4-features-title"));
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(temporary, "assets/features.png") });
  await at(52);
  await top();
  await page.getByRole("combobox", { name: "推理方法", exact: true }).selectOption("maml");
  await page.getByRole("combobox", { name: "每类标注样本", exact: true }).selectOption("5");
  await at(56);
  await run();
  await at(64);
  await click(page.locator(".lab4-section-nav").getByRole("link", { name: "对比与消融", exact: true }));
  await click(page.getByRole("button", { name: "5-shot", exact: true }));
  await showSection(page.locator("#lab4-comparison-title"));
  await at(70);
  await showSection(page.locator('[aria-labelledby="lab4-comparison-title"] .table-scroll'));
  await at(74);
  await showSection(page.locator("#lab4-ablation-title"));
  await at(82);
  await top();
  await click(page.locator(".lab4-section-nav").getByRole("link", { name: "方法与协议", exact: true }));
  await at(90.4);
  if (errors.length) throw new Error(errors.join("\n"));
  if (runResponses.length !== 3 || runResponses.some(r => r.compute !== "live")) throw new Error("Expected three real model inference runs.");
  if (!rehearsal && Date.now() - start > 93000) throw new Error("Recording exceeded the narration timeline; record again without clipping actions.");
  await context.close();
  if (rehearsal) {
    console.log(JSON.stringify({ rehearsal: "passed", runs: runResponses }));
  } else {
    const raw = await video.path();
    await fs.writeFile(path.join(temporary, "demo_take.json"), JSON.stringify({ raw, offset_seconds: offset,
      duration_seconds: 90, viewport: { width: 1600, height: 900 }, runs: runResponses }, null, 2), "utf8");
    console.log(JSON.stringify({ raw, offset_seconds: offset, recorded_runs: runResponses }));
  }
} finally {
  await browser.close();
}
