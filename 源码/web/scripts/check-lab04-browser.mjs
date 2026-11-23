import assert from "node:assert/strict";
import { chromium } from "playwright";

// Against the running application. No mocked data or screenshots are retained.
const base = process.env.LAB04_BASE_URL || "http://127.0.0.1:18876";
const live = process.argv.includes("--live");
const browser = await chromium.launch({ channel: "chrome", headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
const errors = [];
let runs = 0;
page.on("pageerror", (error) => errors.push(error.message));
page.on("request", (request) => {
  if (
    request.method() === "POST" &&
    new URL(request.url()).pathname === "/api/labs/lab04/run"
  )
    runs += 1;
});
try {
  const snapshotResponse = await page.request.get(
    `${base}/api/labs/lab04/snapshot`,
  );
  assert.equal(
    snapshotResponse.status(),
    200,
    "experiment-four snapshot must be readable",
  );
  const snapshot = await snapshotResponse.json();
  await page.goto(`${base}/#/labs/lab04/inference`, {
    waitUntil: "domcontentloaded",
  });
  await page
    .getByRole("heading", { name: "小样本图像分类与特征校准" })
    .waitFor();
  const nav = page.locator(".lab4-section-nav");
  await page.getByRole("combobox", { name: "推理方法", exact: true }).waitFor();
  assert.equal(await page.locator("#main-content").count(), 1);

  if (!snapshot.available) {
    await page
      .getByRole("heading", { name: "模型与实验数据尚未就绪" })
      .waitFor();
    assert.equal(
      await page.getByRole("button", { name: "运行分类任务" }).isDisabled(),
      true,
    );
    assert.equal(
      await page.locator(".lab4-metrics").count(),
      0,
      "unavailable state must not display invented metrics",
    );
    assert.equal(live, false, "--live requires a real trained model and data");
  } else if (live) {
    await page
      .getByRole("combobox", { name: "推理方法", exact: true })
      .selectOption("calibrated");
    await page
      .getByRole("combobox", { name: "每类标注样本", exact: true })
      .selectOption("1");
    await page
      .getByRole("spinbutton", { name: "抽样种子", exact: true })
      .fill("73");
    const responsePromise = page.waitForResponse(
      (response) =>
        new URL(response.url()).pathname === "/api/labs/lab04/run" &&
        response.request().method() === "POST",
      { timeout: 120_000 },
    );
    await page.getByRole("button", { name: "运行分类任务" }).click();
    const response = await responsePromise;
    assert.equal(response.status(), 200);
    const result = await response.json();
    assert.equal(result.compute, "live");
    assert.equal(result.seed, 73);
    assert.equal(result.shots, 1);
    assert.equal(result.classes.length, 5);
    assert.equal(result.support.length, 5);
    assert.ok(result.queries.length > 0);
    assert.equal(result.total, result.queries.length);
    assert.ok(result.accuracy >= 0 && result.accuracy <= 1);
    await page.getByRole("heading", { name: "本次分类结果" }).waitFor();
    assert.equal(
      await page.locator(".lab4-query").count(),
      Math.min(12, result.queries.length),
    );
    const brokenImages = await page
      .locator(".lab4-run-result img")
      .evaluateAll(async (images) => {
        await Promise.all(
          images.map((image) => image.decode().catch(() => {})),
        );
        return images.filter((image) => image.naturalWidth === 0).length;
      });
    assert.equal(brokenImages, 0, "support and query images should decode");
    await nav.getByRole("link", { name: "方法与协议" }).click();
    await page.getByRole("heading", { name: "数据与任务划分" }).waitFor();
    await nav.getByRole("link", { name: "现场推理" }).click();
    await page.getByRole("heading", { name: "本次分类结果" }).waitFor();
    assert.equal(
      runs,
      1,
      "changing sections must preserve the episode, without new inference",
    );
    await page
      .getByRole("spinbutton", { name: "抽样种子", exact: true })
      .fill("74");
    await page.locator(".lab4-stale").waitFor();
    assert.equal(runs, 1, "editing parameters alone must not run a task");
    await page
      .getByRole("spinbutton", { name: "抽样种子", exact: true })
      .fill("73");
    await page.locator(".lab4-stale").waitFor({ state: "hidden" });
  }

  await nav.getByRole("link", { name: "对比与消融" }).click();
  await page.getByRole("heading", { name: "多任务评测" }).waitFor();
  await page.getByRole("button", { name: "5-shot", exact: true }).click();
  assert.equal(
    await page
      .getByRole("button", { name: "5-shot", exact: true })
      .getAttribute("aria-pressed"),
    "true",
  );
  for (const width of [1440, 960, 390, 320]) {
    await page.setViewportSize({ width, height: 1000 });
    await page.evaluate(
      () =>
        new Promise((resolve) =>
          requestAnimationFrame(() => requestAnimationFrame(resolve)),
        ),
    );
    const geometry = await page.evaluate(() => ({
      content: document.documentElement.scrollWidth,
      viewport: innerWidth,
    }));
    assert.ok(
      geometry.content <= geometry.viewport + 1,
      `page overflow at ${width}px: ${JSON.stringify(geometry)}`,
    );
  }
  assert.deepEqual(errors, [], "no uncaught browser errors");
  console.log(
    `Lab 4 browser checks passed: readiness, navigation, responsive layout${live ? ", real inference, images, state preservation and stale-result marking" : ""}.`,
  );
} finally {
  await browser.close();
}
