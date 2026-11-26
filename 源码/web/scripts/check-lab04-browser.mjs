/** Real API + browser acceptance suite. Does not train or change saved results. */
import assert from "node:assert/strict";
import { mkdir, readFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { tmpdir } from "node:os";
import { chromium } from "playwright";
import AxeBuilder from "@axe-core/playwright";

const base =
  process.env.LAB04_TEST_URL ||
  process.env.LAB04_BASE_URL ||
  "http://127.0.0.1:8766";
const output =
  process.env.LAB04_SCREENSHOTS || join(tmpdir(), "lab04-ui-acceptance");
await mkdir(output, { recursive: true });
const browser = await chromium.launch({
  headless: true,
  executablePath: process.env.PLAYWRIGHT_EXECUTABLE_PATH || undefined,
});
const context = await browser.newContext({
  viewport: { width: 1366, height: 900 },
});
const page = await context.newPage();
const errors = [];
let runCount = 0;
page.on("request", (r) => {
  if (r.method() === "POST" && r.url().endsWith("/api/labs/lab04/run"))
    runCount++;
});
page.on("pageerror", (e) => errors.push(e.message));
const checks = [];
const skipped = [];
async function check(name, fn) {
  try {
    const completed = await fn();
    if (completed === false) {
      skipped.push(name);
      console.log("SKIP", name);
      return;
    }
    checks.push(name);
    console.log("PASS", name);
  } catch (e) {
    console.error("FAIL", name);
    throw e;
  }
}
async function audit(label) {
  const result = await new AxeBuilder({ page }).include(".lab4-page").analyze();
  assert.deepEqual(
    result.violations.map((v) => ({
      id: v.id,
      nodes: v.nodes.map((n) => ({
        target: n.target,
        summary: n.failureSummary,
        html: n.html,
      })),
    })),
    [],
    label,
  );
}
async function screenshot(name) {
  await page.screenshot({ path: join(output, name + ".png"), fullPage: false });
}
try {
  await check("loads independent live workbench", async () => {
    await page.goto(base + "/#/labs/lab04/inference");
    await page.locator(".lab4-setup").waitFor();
    assert.equal(await page.locator(".lab4-quickstart").isVisible(), true);
    await audit("initial");
    await screenshot("desktop-initial");
  });
  await check(
    "rejects invalid and out-of-bound seeds without request",
    async () => {
      const seed = page.getByRole("textbox", { name: "任务种子", exact: true });
      for (const invalid of ["", "-1", "2.5", "2147483648", "abc"]) {
        await seed.fill(invalid);
        assert.equal(
          await page
            .getByRole("button", { name: "运行分类任务", exact: true })
            .isDisabled(),
          true,
        );
      }
      await seed.fill("20261124");
    },
  );
  await check("compares actual methods on identical images", async () => {
    const responses = [];
    const capture = async (response) => {
      if (response.url().endsWith("/api/labs/lab04/run") && response.ok())
        responses.push(await response.json());
    };
    page.on("response", capture);
    await page.getByRole("button", { name: "同任务对比三种方法" }).click();
    await page
      .locator(".lab4-run-tabs button")
      .nth(2)
      .waitFor({ timeout: 90000 });
    await page.waitForFunction(
      () => !document.querySelector(".lab4-run").disabled,
      null,
      { timeout: 90000 },
    );
    page.off("response", capture);
    assert.equal(responses.length, 3);
    assert.deepEqual(
      new Set(responses.map((r) => r.method)),
      new Set(["baseline", "maml", "calibrated"]),
    );
    for (const result of responses) {
      assert.equal(result.compute, "live");
      assert.equal(result.total, 75);
      assert.equal(
        result.correct,
        result.queries.filter((q) => q.predicted_id === q.target_id).length,
      );
      assert.deepEqual(
        result.queries.map((q) => q.id),
        responses[0].queries.map((q) => q.id),
      );
      assert.deepEqual(
        result.support.map((q) => q.id),
        responses[0].support.map((q) => q.id),
      );
      assert.equal(result.artifact_revision, responses[0].artifact_revision);
    }
    assert.equal(await page.locator("#main-content").count(), 1);
    const broken = await page
      .locator(".lab4-support-images img,.lab4-query-tile img")
      .evaluateAll(async (imgs) => {
        await Promise.all(imgs.map((i) => i.decode().catch(() => {})));
        return imgs.filter((i) => !i.naturalWidth).length;
      });
    assert.equal(broken, 0);
    await audit("desktop prediction");
    await page.locator(".lab4-result-summary").scrollIntoViewIfNeeded();
    await screenshot("desktop-result");
  });
  await check(
    "inference steps remain without the removed fingerprint footer",
    async () => {
      const before = runCount;
      const details = page.locator(".lab4-process-details");
      assert.equal(
        await details.locator("summary").innerText(),
        "本次推理过程",
      );
      await details.locator("summary").click();
      assert.ok((await details.locator("ol > li").count()) > 0);
      assert.equal(
        await details.locator(":scope > p, :scope > code").count(),
        0,
      );
      assert.ok(!(await details.innerText()).includes("SHA-256"));
      await details.scrollIntoViewIfNeeded();
      await screenshot("inference-without-fingerprint-footer");
      await details.locator("summary").click();
      assert.equal(runCount, before);
    },
  );
  await check("error filters and probability inspection work", async () => {
    await page.getByRole("button", { name: /^误判/ }).click();
    const tiles = page.locator(".lab4-query-tile");
    assert.ok((await tiles.count()) > 0);
    assert.equal(await page.locator(".lab4-query-tile.correct").count(), 0);
    await tiles.nth(1).click();
    assert.equal(await page.locator(".lab4-probabilities li").count(), 5);
    await page.getByRole("button", { name: /^全部/ }).click();
    await page.locator(".lab4-confusion summary").click();
    const sum = await page
      .locator(".lab4-confusion tbody td")
      .evaluateAll((els) => els.reduce((s, e) => s + Number(e.textContent), 0));
    assert.equal(sum, 75);
    await page.locator(".lab4-query-toolbar").scrollIntoViewIfNeeded();
    await screenshot("desktop-query");
  });
  await check(
    "exports self-contained task records, no bitmap bloat",
    async () => {
      const downloadPromise = page.waitForEvent("download");
      await page.getByRole("button", { name: "导出记录", exact: true }).click();
      const download = await downloadPromise;
      const path = await download.path();
      const record = JSON.parse(await readFile(path, "utf8"));
      assert.equal(record.runs.length, 3);
      assert.equal(record.runs[0].result.queries.length, 75);
      assert.equal("image" in record.runs[0].result.queries[0], false);
      assert.match(record.runs[0].result.checkpoint_sha256, /^[a-f0-9]{64}$/);
      assert.equal(typeof record.runs[0].result.scope, "string");
      assert.ok(record.runs[0].result.scope.length > 0);
    },
  );
  await check(
    "preserves task when reading protocol and evaluation",
    async () => {
      const before = runCount;
      await page.getByRole("link", { name: "方法与协议", exact: true }).click();
      await page.getByRole("heading", { name: "数据与任务划分" }).waitFor();
      assert.equal(await page.locator(".lab4-continuity").count(), 0);
      assert.equal(
        await page
          .getByRole("heading", { name: "与原平台的衔接", exact: true })
          .count(),
        0,
      );
      assert.ok((await page.locator(".lab4-training-curve").count()) > 0);
      await page
        .getByRole("heading", { name: "本轮模型与训练", exact: true })
        .scrollIntoViewIfNeeded();
      await screenshot("protocol-without-platform-footer");
      await audit("protocol");
      await page.getByRole("link", { name: "对比与消融", exact: true }).click();
      await page.locator(".lab4-evaluation").waitFor();
      await audit("evaluation 1shot");
      await screenshot("desktop-evaluation");
      await page.getByRole("button", { name: "5-shot", exact: true }).click();
      assert.ok(
        (await page.locator(".lab4-evaluation-facts").innerText()).includes(
          "5-shot",
        ),
      );
      await page.getByRole("link", { name: "现场推理", exact: true }).click();
      assert.equal(await page.locator(".lab4-run-tabs button").count(), 3);
      assert.equal(
        runCount,
        before,
        "Reading explanation tabs must not trigger inference",
      );
    },
  );
  await check(
    "new shot invalidates previous task and runs 25 supports",
    async () => {
      await page.locator(".lab4-shot-cards button").nth(1).click();
      assert.equal(await page.locator(".lab4-stale").isVisible(), true);
      await page
        .getByRole("button", { name: "运行分类任务", exact: true })
        .click();
      await page.waitForFunction(
        () => !document.querySelector(".lab4-run").disabled,
        null,
        { timeout: 90000 },
      );
      assert.equal(await page.locator(".lab4-support-images img").count(), 25);
      assert.equal(await page.locator(".lab4-run-tabs button").count(), 1);
    },
  );
  await check("responsive and accessible from 320 to 1366 pixels", async () => {
    for (const width of [320, 390, 768, 960, 1366]) {
      await page.setViewportSize({ width, height: 844 });
      await page.locator(".lab4-query-toolbar").scrollIntoViewIfNeeded();
      assert.ok(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth + 1,
        ),
        `overflow ${width}`,
      );
      if (width === 390) {
        await page.emulateMedia({ reducedMotion: "reduce" });
        await page.locator(".lab4-query-tile").nth(2).click();
        await page.waitForFunction(
          () => document.activeElement?.classList.contains("lab4-inspector"),
          null,
          { timeout: 3000 },
        );
        const top = await page
          .locator(".lab4-inspector")
          .evaluate((e) => e.getBoundingClientRect().top);
        assert.ok(
          top >= 0 && top < 220,
          `mobile inspector should be directly visible: ${top}`,
        );
        await page
          .getByRole("button", { name: "返回图片列表", exact: true })
          .click();
        assert.equal(
          await page
            .locator(".lab4-query-tile[aria-pressed=true]")
            .evaluate((e) => e === document.activeElement),
          true,
        );
        await page.emulateMedia({ reducedMotion: "no-preference" });
      }
      await audit(`responsive ${width}`);
      await screenshot(`query-${width}`);
    }
  });
  await check(
    "delayed inference exposes a loading state and cancels on leaving",
    async () => {
      await page.route("**/api/labs/lab04/run", async (route) => {
        await new Promise((resolve) => setTimeout(resolve, 800));
        try {
          await route.fulfill({
            status: 503,
            contentType: "application/json",
            body: JSON.stringify({
              error: { message: "delayed fixture", code: "fixture" },
            }),
          });
        } catch {}
      });
      await page
        .getByRole("button", { name: "运行分类任务", exact: true })
        .click();
      await page.locator(".lab4-progress").waitFor();
      assert.equal(await page.locator(".lab4-run").isDisabled(), true);
      await page.getByRole("link", { name: "方法与协议", exact: true }).click();
      await page.getByRole("heading", { name: "数据与任务划分" }).waitFor();
      await page.getByRole("link", { name: "现场推理", exact: true }).click();
      assert.equal(await page.locator(".lab4-progress").count(), 0);
      assert.equal(
        await page
          .getByRole("button", { name: "运行分类任务", exact: true })
          .isEnabled(),
        true,
      );
      await new Promise((resolve) => setTimeout(resolve, 900));
      await page.unroute("**/api/labs/lab04/run");
    },
  );
  await check("keyboard focus and reduced motion are supported", async () => {
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.getByRole("textbox", { name: "任务种子", exact: true }).focus();
    await page.keyboard.press("Tab");
    assert.equal(
      await page
        .getByRole("button", { name: "随机生成任务种子" })
        .evaluate((e) => e === document.activeElement),
      true,
    );
    await page.getByRole("button", { name: "随机生成任务种子" }).press("Enter");
    assert.ok(
      Number(
        await page
          .getByRole("textbox", { name: "任务种子", exact: true })
          .inputValue(),
      ) >= 0,
    );
    await page.emulateMedia({ reducedMotion: "no-preference" });
  });
  await check("unavailable model never displays invented scores", async () => {
    const snapshot = await (
      await page.request.get(base + "/api/labs/lab04/snapshot")
    ).json();
    await page.route("**/api/labs/lab04/snapshot", (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ ...snapshot, available: false }),
      }),
    );
    await page.reload();
    await page.getByRole("heading", { name: "模型尚未就绪" }).waitFor();
    assert.equal(
      await page
        .getByRole("button", { name: "运行分类任务", exact: true })
        .isDisabled(),
      true,
    );
    assert.equal(await page.locator(".lab4-metrics").count(), 0);
    await page.unroute("**/api/labs/lab04/snapshot");
    await page.reload();
    await page.locator(".lab4-setup").waitFor();
  });
  await check(
    "verified graph extension preserves source metrics and paired evidence",
    async () => {
      const snapshot = await (
        await page.request.get(base + "/api/labs/lab04/snapshot")
      ).json();
      if (!snapshot.graph_extension?.available) return false;
      await page.getByRole("radio", { name: /特征校准/ }).check();
      await page
        .getByRole("combobox", { name: "校准策略", exact: true })
        .selectOption("graph");
      const response = page.waitForResponse(
        (r) => r.url().endsWith("/api/labs/lab04/run") && r.ok(),
      );
      await page
        .getByRole("button", { name: "运行分类任务", exact: true })
        .click();
      const value = await (await response).json();
      assert.equal(value.calibration, "graph");
      assert.equal(value.metric, "graph_and_prototype_soft_assignment");
      assert.equal(value.transductive, true);
      assert.ok(value.config.whitening_exponent > 0);
      await page.waitForFunction(
        () => !document.querySelector(".lab4-run").disabled,
      );
      await audit("graph prediction");
      await page.getByRole("link", { name: "对比与消融", exact: true }).click();
      await page.getByText(/同骨干增量改进/).waitFor();
      assert.equal(snapshot.graph_extension.available, true);
      assert.ok(
        snapshot.comparisons.some(
          (row) => row.method === "graph" && row.episodes === 600,
        ),
      );
      await page.getByRole("link", { name: "现场推理", exact: true }).click();
    },
  );
  await check("multiview scores and decision rule remain honest", async () => {
    const snapshot = await (
      await page.request.get(base + "/api/labs/lab04/snapshot")
    ).json();
    if (!snapshot.multiview_extension?.available) return false;
    await page.getByRole("radio", { name: /特征校准/ }).check();
    await page
      .getByRole("combobox", { name: "校准策略", exact: true })
      .selectOption("multiview");
    const response = page.waitForResponse(
      (r) => r.url().endsWith("/api/labs/lab04/run") && r.ok(),
    );
    await page
      .getByRole("button", { name: "运行分类任务", exact: true })
      .click();
    const value = await (await response).json();
    assert.equal(value.calibration, "multiview");
    assert.equal(value.metric, "multiview_graph_assignment");
    assert.ok(["mean", "concat", "original"].includes(value.config.views));
    for (const q of value.queries) {
      const index = value.classes.findIndex((c) => c.id === q.predicted_id);
      assert.ok(Math.abs(q.confidence - q.probabilities[index]) < 1e-9);
    }
    if (value.decision_rule === "balanced_capacity")
      for (const c of value.classes)
        assert.equal(
          value.queries.filter((q) => q.predicted_id === c.id).length,
          15,
        );
    await page.waitForFunction(
      () => !document.querySelector(".lab4-run").disabled,
    );
    await audit("multiview prediction");
    await page.getByRole("link", { name: "对比与消融", exact: true }).click();
    await page
      .getByRole("combobox", { name: "重点查看", exact: true })
      .selectOption("multiview");
    await page.getByText(/多视图增量实验/).waitFor();
    await page.getByRole("link", { name: "现场推理", exact: true }).click();
  });
  await check(
    "evaluation focus and score reliability are real, not decorative",
    async () => {
      await page.getByRole("link", { name: "对比与消融", exact: true }).click();
      const snapshot = await (
        await page.request.get(base + "/api/labs/lab04/snapshot")
      ).json();
      assert.equal(snapshot.verification.verified, true);
      assert.equal(snapshot.verification.core_tasks, 1200);
      if (snapshot.multiview_extension?.available) {
        await page
          .getByRole("combobox", { name: "重点查看", exact: true })
          .selectOption("multiview");
        await page.getByRole("button", { name: "5-shot", exact: true }).click();
        const summary = await page
          .locator(".lab4-evaluation-facts")
          .innerText();
        const expected = snapshot.comparisons.find(
          (r) => r.method === "multiview" && r.shot === 5,
        );
        assert.ok(summary.includes((100 * expected.accuracy).toFixed(2) + "%"));
        await page.getByRole("link", { name: "可靠性", exact: true }).click();
        await page.locator(".lab4-score-reliability summary").click();
        const count = await page
          .locator(".lab4-score-reliability tbody td:nth-child(2)")
          .evaluateAll((els) =>
            els.reduce(
              (sum, e) => sum + Number(e.textContent.replaceAll(",", "")),
              0,
            ),
          );
        assert.equal(count, 45000);
        assert.ok(
          (await page.locator(".lab4-reliability").innerText()).includes(
            "并非同等数量的独立图片",
          ) ||
            (await page.locator(".lab4-reliability").innerText()).includes(
              "并非",
            ),
        );
        await audit("reliability table");
      }
      for (const width of [320, 390, 768, 1366]) {
        await page.setViewportSize({ width, height: 844 });
        assert.ok(
          await page.evaluate(
            () => document.documentElement.scrollWidth <= innerWidth + 1,
          ),
          `evaluation overflow at ${width}`,
        );
      }
      await page.getByRole("link", { name: "现场推理", exact: true }).click();
    },
  );
  await check(
    "independent evidence pages preserve state and support history navigation",
    async () => {
      const before = runCount;
      const taskTabs = await page
        .locator(".lab4-run-tabs button")
        .allTextContents();
      await page.getByRole("link", { name: "对比与消融", exact: true }).click();
      const focus = page.getByRole("combobox", {
        name: "重点查看",
        exact: true,
      });
      await focus.selectOption("multiview");
      await page.getByRole("button", { name: "5-shot", exact: true }).click();
      await page
        .locator(".lab4-accuracy-plot button")
        .filter({ hasText: "Baseline" })
        .click();
      assert.equal(await focus.inputValue(), "baseline");
      assert.equal(await page.locator(".lab4-conclusion:visible").count(), 0);
      await focus.selectOption("multiview");
      assert.equal(
        await page.locator(".lab4-score-reliability:visible").count(),
        0,
      );
      assert.equal(await page.locator(".lab4-paired-plot:visible").count(), 0);
      const pages = [
        ["模块消融", "ablations", ".lab4-paired-plot"],
        ["可靠性", "reliability", ".lab4-calibration-plot"],
        ["端侧性能", "deployment", ".lab4-latency-plot"],
        ["历史与来源", "history", ".lab4-provenance"],
      ];
      for (const [label, id, selector] of pages) {
        await page.getByRole("link", { name: label, exact: true }).click();
        assert.ok(page.url().endsWith("#/labs/lab04/" + id));
        assert.equal(await page.locator(selector).isVisible(), true);
        if (id === "deployment") {
          assert.equal(
            await page
              .getByText(
                "这里是Windows本机CPU的端侧可行性示范，不代表手机、ARM或其他硬件的实测速度；加载、校验文件和HTTP传输不在上表计时内。",
                { exact: true },
              )
              .count(),
            0,
          );
          const data = page.locator(
            ".lab4-evaluation-section .lab4-data-details",
          );
          await data.locator("summary").click();
          assert.ok((await data.locator("tbody tr").count()) >= 3);
          await data.locator("table").scrollIntoViewIfNeeded();
          await screenshot("deployment-without-table-footer");
          await data.locator("summary").click();
        }
        assert.equal(
          await page.locator(".lab4-accuracy-plot:visible").count(),
          0,
        );
        assert.equal(
          await page
            .getByRole("button", { name: "5-shot", exact: true })
            .getAttribute("aria-pressed"),
          "true",
        );
        for (const width of [320, 390, 768, 1366]) {
          await page.setViewportSize({ width, height: 900 });
          assert.ok(
            await page.evaluate(
              () => document.documentElement.scrollWidth <= innerWidth + 1,
            ),
            `${id} overflow at ${width}`,
          );
        }
        await audit("independent " + id);
        await page.locator(".lab4-evaluation-header").scrollIntoViewIfNeeded();
        await screenshot("page-" + id);
      }
      await page.goBack();
      assert.ok(page.url().endsWith("/deployment"));
      await page.goForward();
      assert.ok(page.url().endsWith("/history"));
      await page.getByRole("link", { name: "性能对比", exact: true }).click();
      assert.equal(await focus.inputValue(), "multiview");
      assert.equal(
        await page
          .getByRole("button", { name: "5-shot", exact: true })
          .getAttribute("aria-pressed"),
        "true",
      );
      assert.deepEqual(
        await page.locator(".lab4-section-nav a").allTextContents(),
        ["现场推理", "方法与协议", "对比与消融"],
      );
      assert.equal(
        await page
          .locator(
            'a[href="#/labs/lab04/requirements"], a[href="#/labs/lab04/materials"]',
          )
          .count(),
        0,
      );
      await page.getByRole("link", { name: "现场推理", exact: true }).click();
      assert.deepEqual(
        await page.locator(".lab4-run-tabs button").allTextContents(),
        taskTabs,
      );
      assert.equal(
        runCount,
        before,
        "Evidence navigation must not run inference",
      );
    },
  );
  await check(
    "module contrasts use their own paired controls and export exact evidence",
    async () => {
      const before = runCount;
      const snapshot = await (
        await page.request.get(base + "/api/labs/lab04/snapshot")
      ).json();
      await page.getByRole("link", { name: "对比与消融", exact: true }).click();
      await page.getByRole("link", { name: "模块消融", exact: true }).click();
      await page
        .getByRole("button", { name: "模块直接对照", exact: true })
        .click();
      for (const shots of [1, 5]) {
        await page
          .getByRole("button", { name: `${shots}-shot`, exact: true })
          .click();
        assert.equal(
          await page.locator(".lab4-module-contrasts .lab4-paired-row").count(),
          6,
        );
        const text = await page.locator(".lab4-module-contrasts").innerText();
        assert.ok(text.includes("原型更新＋均衡转导"));
        assert.ok(text.includes("多重比较"));
        const entries = snapshot.module_contrasts.rows.filter(
          (r) => r.shot === shots,
        );
        const row = entries.find((r) => r.id === "prototype_balanced");
        assert.ok(text.includes((row.paired_delta * 100).toFixed(2)));
        assert.ok(text.includes((row.paired_ci95 * 100).toFixed(2)));
      }
      await page.locator(".lab4-contrast-details summary").click();
      assert.equal(
        await page.locator(".lab4-contrast-details tbody tr").count(),
        6,
      );
      assert.ok(
        (await page.locator(".lab4-contrast-details").innerText()).includes(
          "不能全部归因于原型单一模块",
        ),
      );
      await page.locator(".lab4-contrast-details summary").click();
      await page
        .getByRole("button", { name: "相对 Baseline", exact: true })
        .click();
      assert.equal(await page.locator(".lab4-module-contrasts").count(), 0);
      assert.ok(
        (
          await page.locator(".lab4-paired-plot figcaption").innerText()
        ).includes("相对同版 Baseline"),
      );
      await page
        .getByRole("button", { name: "模块直接对照", exact: true })
        .click();
      for (const width of [320, 390, 768, 1366]) {
        await page.setViewportSize({ width, height: 900 });
        assert.ok(
          await page.evaluate(
            () => document.documentElement.scrollWidth <= innerWidth + 1,
          ),
          `module comparisons overflow ${width}`,
        );
        await audit("module comparisons " + width);
        if (width === 390 || width === 1366) {
          await page.locator(".lab4-module-contrasts").scrollIntoViewIfNeeded();
          await screenshot("module-pairs-" + width);
        }
      }
      const pending = page.waitForEvent("download");
      await page.getByRole("button", { name: "导出评测", exact: true }).click();
      const downloaded = await pending;
      const content = JSON.parse(
        await readFile(await downloaded.path(), "utf8"),
      );
      assert.deepEqual(content.module_contrasts, snapshot.module_contrasts);
      assert.equal(runCount, before);
      await page.getByRole("link", { name: "现场推理", exact: true }).click();
    },
  );
  await check("cannot silently mix changed artifact revisions", async () => {
    await page.route("**/api/labs/lab04/run", async (route) => {
      const response = await route.fetch();
      const value = await response.json();
      value.artifact_revision = "f".repeat(64);
      await route.fulfill({ response, json: value });
    });
    const before = await page.locator(".lab4-run-tabs button").count();
    await page
      .getByRole("button", { name: "运行分类任务", exact: true })
      .click();
    await page
      .getByRole("alert")
      .filter({ hasText: "模型版本在本次请求前已更新" })
      .waitFor();
    assert.equal(await page.locator(".lab4-run-tabs button").count(), before);
    assert.equal(
      await page
        .getByRole("button", { name: "运行分类任务", exact: true })
        .isEnabled(),
      true,
    );
    await page.unroute("**/api/labs/lab04/run");
  });
  await check(
    "expanded samples remain navigable after collapsing and sorting",
    async () => {
      await page.getByRole("button", { name: /^全部/ }).click();
      await page.locator("#lab4-class-filter").selectOption("all");
      await page.locator("#lab4-query-sort").selectOption("original");
      const expand = page.locator(".lab4-expand");
      if ((await expand.innerText()).includes("展开")) await expand.click();
      assert.equal(await page.locator(".lab4-query-tile").count(), 75);
      await page.locator(".lab4-query-tile").nth(50).click();
      await expand.click();
      assert.equal(await page.locator(".lab4-query-tile").count(), 12);
      assert.equal(
        await page.locator(".lab4-query-tile[aria-pressed=true]").count(),
        1,
      );
      await page.locator("#lab4-query-sort").selectOption("confidence");
      assert.equal(
        await page.locator(".lab4-query-tile[aria-pressed=true]").count(),
        1,
      );
    },
  );
  await check(
    "optional wider backbone discloses capacity and preserves paired samples",
    async () => {
      const snapshot = await (
        await page.request.get(base + "/api/labs/lab04/snapshot")
      ).json();
      if (!snapshot.wide_extension?.available) return false;
      await page.getByRole("radio", { name: /特征校准/ }).check();
      await page
        .getByRole("combobox", { name: "校准策略", exact: true })
        .selectOption("wide");
      const response = page.waitForResponse(
        (r) => r.url().endsWith("/api/labs/lab04/run") && r.ok(),
      );
      await page
        .getByRole("button", { name: "运行分类任务", exact: true })
        .click();
      const value = await (await response).json();
      assert.equal(value.metric, "wider_backbone_graph");
      assert.equal(value.calibration, "wide");
      assert.equal(value.config.larger_backbone, true);
      assert.equal(value.config.additional_training, true);
      assert.deepEqual(
        value.config.backbone_widths,
        snapshot.wide_extension.model.widths,
      );
      assert.ok(
        snapshot.wide_extension.model.parameters > snapshot.model.parameters,
      );
      await page.waitForFunction(
        () => !document.querySelector(".lab4-run").disabled,
      );
      await audit("wider study live results");
      await page.getByRole("link", { name: "对比与消融", exact: true }).click();
      await page
        .getByRole("combobox", { name: "重点查看", exact: true })
        .selectOption("wide");
      await page.getByText(/独立较宽模型研究/).waitFor();
      assert.ok(
        (await page.locator(".lab4-incremental-note").innerText()).includes(
          "不能归于校准算法",
        ),
      );
      await page.getByRole("link", { name: "现场推理", exact: true }).click();
    },
  );
  await check(
    "wider two-view improvement uses the same weights and paired live samples",
    async () => {
      const snapshot = await (
        await page.request.get(base + "/api/labs/lab04/snapshot")
      ).json();
      if (!snapshot.wideview_extension?.available) return false;
      const calibration = page.getByRole("combobox", {
        name: "校准策略",
        exact: true,
      });
      await calibration.selectOption("wideview");
      const pending = page.waitForResponse(
        (r) => r.url().endsWith("/api/labs/lab04/run") && r.ok(),
      );
      await page
        .getByRole("button", { name: "运行分类任务", exact: true })
        .click();
      const value = await (await pending).json();
      await page.waitForFunction(
        () => !document.querySelector(".lab4-run").disabled,
      );
      assert.equal(value.calibration, "wideview");
      assert.equal(value.metric, "wider_backbone_multiview");
      assert.equal(value.config.encodings_per_image, 2);
      assert.equal(value.config.additional_training_vs_wide, false);
      assert.equal(value.projection.feature_dimensions, 128);
      assert.ok(
        (await page.locator(".lab4-setup").innerText()).includes(
          "没有新增训练",
        ),
      );
      const reference = await (
        await page.request.post(base + "/api/labs/lab04/run", {
          data: {
            method: "calibrated",
            calibration: "wide",
            shots: value.shots,
            seed: value.seed,
          },
        })
      ).json();
      assert.equal(value.checkpoint_sha256, reference.checkpoint_sha256);
      assert.equal(value.artifact_revision, reference.artifact_revision);
      assert.deepEqual(
        value.queries.map((q) => q.id),
        reference.queries.map((q) => q.id),
      );
      await audit("wideview real prediction");
      await page.getByRole("link", { name: "对比与消融", exact: true }).click();
      await page
        .getByRole("combobox", { name: "重点查看", exact: true })
        .selectOption("wideview");
      await page.getByRole("button", { name: "5-shot", exact: true }).click();
      const expected = snapshot.comparisons.find(
        (r) => r.method === "wideview" && r.shot === 5,
      );
      assert.ok(
        (await page.locator(".lab4-evaluation-facts").innerText()).includes(
          (expected.accuracy * 100).toFixed(2) + "%",
        ),
      );
      assert.ok(
        (await page.locator(".lab4-incremental-note").innerText()).includes(
          "不新增训练",
        ),
      );
      await page.locator(".lab4-accuracy-plot").scrollIntoViewIfNeeded();
      await screenshot("wideview-evidence-plot");
      await page.getByRole("link", { name: "端侧性能", exact: true }).click();
      assert.ok(
        (await page.locator(".lab4-latency-plot").innerText()).includes(
          "较宽双视图",
        ),
      );
      await page.getByRole("link", { name: "现场推理", exact: true }).click();
      await calibration.selectOption("wide");
    },
  );
  await check(
    "continued model keeps capacity, discloses extra training and preserves tasks",
    async () => {
      const snapshot = await (
        await page.request.get(base + "/api/labs/lab04/snapshot")
      ).json();
      if (!snapshot.continued_extension?.available) return false;
      const mode = page.getByRole("combobox", {
        name: "校准策略",
        exact: true,
      });
      await mode.selectOption("continued");
      const pending = page.waitForResponse(
        (r) => r.url().endsWith("/api/labs/lab04/run") && r.ok(),
      );
      await page
        .getByRole("button", { name: "运行分类任务", exact: true })
        .click();
      const value = await (await pending).json();
      await page.waitForFunction(
        () => !document.querySelector(".lab4-run").disabled,
      );
      assert.equal(value.metric, "continued_backbone_multiview");
      assert.equal(value.config.additional_epochs, 16);
      assert.equal(value.config.selected_additional_epoch, 12);
      assert.equal(value.config.same_capacity_as_wide, true);
      assert.equal(value.config.additional_training_vs_wide, true);
      assert.equal(
        snapshot.continued_extension.model.parameters,
        snapshot.wide_extension.model.parameters,
      );
      assert.ok(
        (await page.locator(".lab4-setup").innerText()).includes(
          "额外训练16轮",
        ),
      );
      const old = await (
        await page.request.post(base + "/api/labs/lab04/run", {
          data: {
            method: "calibrated",
            calibration: "wideview",
            shots: value.shots,
            seed: value.seed,
          },
        })
      ).json();
      assert.notEqual(value.checkpoint_sha256, old.checkpoint_sha256);
      assert.equal(value.artifact_revision, old.artifact_revision);
      assert.deepEqual(
        value.queries.map((q) => q.id),
        old.queries.map((q) => q.id),
      );
      await audit("continued true inference");
      await page.getByRole("link", { name: "对比与消融", exact: true }).click();
      await page
        .getByRole("combobox", { name: "重点查看", exact: true })
        .selectOption("continued");
      await page.getByRole("button", { name: "5-shot", exact: true }).click();
      const expected = snapshot.comparisons.find(
        (r) => r.method === "continued" && r.shot === 5,
      );
      assert.ok(
        (await page.locator(".lab4-evaluation-facts").innerText()).includes(
          (expected.accuracy * 100).toFixed(2) + "%",
        ),
      );
      assert.ok(
        (await page.locator(".lab4-incremental-note").innerText()).includes(
          "不是",
        ) ||
          (await page.locator(".lab4-incremental-note").innerText()).includes(
            "不归为",
          ),
      );
      await page.locator(".lab4-accuracy-plot").scrollIntoViewIfNeeded();
      await screenshot("continued-evidence");
      await page.getByRole("link", { name: "模块消融", exact: true }).click();
      const modelRows = await page
        .locator(".lab4-module-contrasts")
        .innerText();
      assert.equal(modelRows.includes("同容量续训"), false);
      await page.getByRole("link", { name: "现场推理", exact: true }).click();
      await mode.selectOption("wide");
    },
  );
  await check(
    "malformed API results never replace verified results or crash charts",
    async () => {
      const data = {
        method: "calibrated",
        calibration: await page
          .getByRole("combobox", { name: "校准策略", exact: true })
          .inputValue(),
        shots: 1,
        seed: Number(
          await page
            .getByRole("textbox", { name: "任务种子", exact: true })
            .inputValue(),
        ),
      };
      const response = await page.request.post(base + "/api/labs/lab04/run", {
        data,
      });
      assert.equal(response.ok(), true);
      const valid = await response.json();
      const metrics = await page.locator(".lab4-metrics").innerText();
      const tabs = await page
        .locator(".lab4-run-tabs button")
        .allTextContents();
      for (const mutate of [
        (v) => (v.classes[0] = null),
        (v) => (v.queries[0] = null),
        (v) => (v.points = null),
        (v) => (v.prototypes[0].before = [null, 0]),
        (v) => (v.steps[0].detail = {}),
        (v) => (v.projection.explained_variance = null),
        (v) => delete v.artifact_revision,
      ]) {
        const malformed = structuredClone(valid);
        mutate(malformed);
        await page.route("**/api/labs/lab04/run", (route) =>
          route.fulfill({ json: malformed }),
        );
        await page
          .getByRole("button", { name: "运行分类任务", exact: true })
          .click();
        await page
          .getByRole("alert")
          .filter({ hasText: "推理结果与请求或评分不一致" })
          .waitFor();
        assert.equal(await page.locator(".lab4-metrics").innerText(), metrics);
        assert.deepEqual(
          await page.locator(".lab4-run-tabs button").allTextContents(),
          tabs,
        );
        assert.equal(await page.locator(".lab4-run").isEnabled(), true);
        assert.deepEqual(errors, []);
        await page.unroute("**/api/labs/lab04/run");
      }
    },
  );
  await check(
    "cancel preserves completed comparisons and never starts the remaining method",
    async () => {
      const calls = [];
      let release, started, finished;
      const gate = new Promise((resolve) => (release = resolve));
      const held = new Promise((resolve) => (started = resolve));
      const settled = new Promise((resolve) => (finished = resolve));
      await page.route("**/api/labs/lab04/run", async (route) => {
        calls.push(route.request().postDataJSON());
        if (calls.length === 1) {
          const response = await route.fetch();
          await route.fulfill({ response });
          return;
        }
        started();
        await gate;
        try {
          await route.fulfill({
            status: 503,
            json: {
              error: { message: "canceled stale response", code: "fixture" },
            },
          });
        } catch {
          // A canceled fetch may already have closed this intercepted request.
        } finally {
          finished();
        }
      });
      await page.setViewportSize({ width: 390, height: 844 });
      await page
        .getByRole("button", { name: "同任务对比三种方法", exact: true })
        .click();
      await held;
      await page
        .locator(".lab4-progress-message")
        .filter({ hasText: "2 / 3" })
        .waitFor();
      const metrics = await page.locator(".lab4-metrics").innerText();
      const tabs = await page
        .locator(".lab4-run-tabs button")
        .allTextContents();
      const cancel = page.getByRole("button", {
        name: "取消等待",
        exact: true,
      });
      await cancel.scrollIntoViewIfNeeded();
      assert.ok(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth + 1,
        ),
      );
      await audit("cancel during partial comparison, narrow screen");
      await screenshot("cancel-pending-390");
      await cancel.press("Enter");
      await page
        .locator(".lab4-operation-note")
        .filter({ hasText: "服务器当前计算可能仍会完成" })
        .waitFor();
      await page.waitForFunction(
        () => document.activeElement === document.querySelector(".lab4-run"),
      );
      assert.equal(await page.locator(".lab4-progress").count(), 0);
      assert.equal(await page.locator(".lab4-run").isEnabled(), true);
      release();
      await settled;
      assert.deepEqual(
        calls.map((v) => v.method),
        ["baseline", "calibrated"],
      );
      assert.equal(await page.locator(".lab4-metrics").innerText(), metrics);
      assert.deepEqual(
        await page.locator(".lab4-run-tabs button").allTextContents(),
        tabs,
      );
      assert.equal(await page.getByRole("alert").count(), 0);
      await page.unroute("**/api/labs/lab04/run");
      await page.setViewportSize({ width: 1366, height: 900 });
      await page.locator(".lab4-operation-note").scrollIntoViewIfNeeded();
      await screenshot("canceled-comparison");
    },
  );
  await check(
    "immediate retry is isolated from a canceled late response",
    async () => {
      const seed = page.getByRole("textbox", { name: "任务种子", exact: true });
      const data = {
        method: "calibrated",
        shots: 1,
        seed: 314159,
        calibration: await page
          .getByRole("combobox", { name: "校准策略", exact: true })
          .inputValue(),
      };
      const response = await page.request.post(base + "/api/labs/lab04/run", {
        data,
      });
      assert.equal(response.ok(), true);
      const valid = await response.json();
      const gates = [0, 1].map(() => {
        let release, started, finished;
        return {
          gate: new Promise((resolve) => (release = resolve)),
          held: new Promise((resolve) => (started = resolve)),
          settled: new Promise((resolve) => (finished = resolve)),
          release: () => release(),
          started: () => started(),
          finished: () => finished(),
        };
      });
      let requests = 0;
      await page.route("**/api/labs/lab04/run", async (route) => {
        const index = requests++;
        assert.ok(index < 2);
        gates[index].started();
        await gates[index].gate;
        try {
          await route.fulfill({ json: index === 0 ? { broken: true } : valid });
        } catch {
          // First request was canceled; its completion must not affect the retry.
        } finally {
          gates[index].finished();
        }
      });
      const run = page.getByRole("button", {
        name: "运行分类任务",
        exact: true,
      });
      await run.click();
      await gates[0].held;
      await page.getByRole("button", { name: "取消等待", exact: true }).click();
      await seed.fill("314159");
      await run.click();
      await gates[1].held;
      gates[0].release();
      await gates[0].settled;
      assert.equal(await page.locator(".lab4-run").isDisabled(), true);
      assert.equal(await page.locator(".lab4-progress").count(), 1);
      assert.equal(await page.getByRole("alert").count(), 0);
      gates[1].release();
      await gates[1].settled;
      await page.waitForFunction(
        () => !document.querySelector(".lab4-run").disabled,
      );
      assert.ok(
        (await page.locator(".lab4-result-heading").innerText()).includes(
          "314159",
        ),
      );
      assert.equal(await page.locator(".lab4-run-tabs button").count(), 1);
      assert.equal(await page.locator(".lab4-operation-note").count(), 0);
      assert.equal(await page.getByRole("alert").count(), 0);
      assert.equal(requests, 2);
      await page.unroute("**/api/labs/lab04/run");
      await audit("recovered after cancel");
    },
  );
  await check("offline failure keeps usable error state", async () => {
    await page.route("**/api/labs/lab04/run", (route) =>
      route.fulfill({
        status: 503,
        contentType: "application/json",
        body: JSON.stringify({
          error: { message: "测试：模型暂不可用", code: "test_unavailable" },
        }),
      }),
    );
    await page
      .getByRole("button", { name: "运行分类任务", exact: true })
      .click();
    await page
      .getByRole("alert")
      .filter({ hasText: "测试：模型暂不可用" })
      .waitFor();
    assert.equal(
      await page
        .getByRole("button", { name: "运行分类任务", exact: true })
        .isEnabled(),
      true,
    );
    await page.unroute("**/api/labs/lab04/run");
  });
  await check(
    "malformed snapshot refresh is rejected while verified state remains usable",
    async () => {
      await page.getByRole("link", { name: "对比与消融", exact: true }).click();
      const snapshot = await (
        await page.request.get(base + "/api/labs/lab04/snapshot")
      ).json();
      const before = await page.locator(".lab4-evaluation-facts").innerText();
      for (const mutate of [
        (v) => (v.model.channels = null),
        (v) => (v.comparisons[0].accuracy = 0.99),
        (v) => (v.diagnostics.rows[0].score_bins = null),
        (v) =>
          (v.edge_benchmark.rows[0].encoding_and_classification_p95_ms = -1),
      ]) {
        const broken = structuredClone(snapshot);
        mutate(broken);
        await page.route("**/api/labs/lab04/snapshot", (route) =>
          route.fulfill({ json: broken }),
        );
        await page
          .getByRole("button", { name: "重新读取实验状态", exact: true })
          .click();
        await page
          .getByRole("alert")
          .filter({ hasText: "实验状态数据不完整或统计不一致" })
          .waitFor();
        assert.ok(
          (await page.getByRole("alert").innerText()).includes("仍保留上一次"),
        );
        assert.equal(
          await page.locator(".lab4-evaluation-facts").innerText(),
          before,
        );
        assert.equal(
          await page
            .getByRole("button", { name: "重新读取实验状态", exact: true })
            .isEnabled(),
          true,
        );
        assert.deepEqual(errors, []);
        await page.unroute("**/api/labs/lab04/snapshot");
      }
      await page
        .getByRole("button", { name: "重新读取实验状态", exact: true })
        .click();
      await page.waitForFunction(
        () => !document.querySelector(".lab4-reload").disabled,
      );
      assert.equal(await page.getByRole("alert").count(), 0);
      await audit("snapshot refresh recovered");
    },
  );
  await check(
    "first-load snapshot failures have accessible retries on report and overview",
    async () => {
      const before = runCount;
      await page.route("**/api/labs/lab04/snapshot", (route) =>
        route.fulfill({ json: { available: true, model: { channels: null } } }),
      );
      await page.reload();
      await page
        .getByRole("alert")
        .filter({ hasText: "实验状态数据不完整或统计不一致" })
        .waitFor();
      assert.equal(await page.locator(".lab4-evaluation-facts").count(), 0);
      assert.equal(await page.locator(".lab4-page").count(), 1);
      await audit("invalid snapshot first load");
      await page.unroute("**/api/labs/lab04/snapshot");
      await page.getByRole("button", { name: "重新加载", exact: true }).click();
      await page.locator(".lab4-evaluation-facts").waitFor();
      await page.route("**/api/labs/lab04/snapshot", (route) =>
        route.fulfill({ json: { model: null } }),
      );
      await page.goto(base + "/#/labs/lab04");
      await page
        .getByRole("alert")
        .filter({ hasText: "实验状态数据不完整或统计不一致" })
        .waitFor();
      assert.equal(await page.locator(".lab4-overview-evidence").count(), 0);
      await page.unroute("**/api/labs/lab04/snapshot");
      await page
        .getByRole("button", { name: "重试实验状态", exact: true })
        .click();
      await page.locator(".lab4-overview-evidence").waitFor();
      assert.equal(await page.getByRole("alert").count(), 0);
      assert.deepEqual(errors, []);
      assert.equal(runCount, before);
      await page.goto(base + "/#/labs/lab04/evaluation");
      await page.locator(".lab4-evaluation-facts").waitFor();
    },
  );
  await check(
    "each new page opens directly and refreshes without inference",
    async () => {
      const before = runCount;
      for (const [id, selector] of [
        ["evaluation", ".lab4-accuracy-plot"],
        ["ablations", ".lab4-paired-plot"],
        ["reliability", ".lab4-calibration-plot"],
        ["deployment", ".lab4-latency-plot"],
        ["history", ".lab4-provenance"],
      ]) {
        await page.goto(base + "/#/labs/lab04/" + id);
        await page.locator(selector).waitFor();
        await page.reload();
        await page.locator(selector).waitFor();
        assert.ok(page.url().endsWith("/" + id));
      }
      assert.equal(runCount, before);
    },
  );
  await check(
    "retired delivery entries are absent and old links open the workbench",
    async () => {
      const before = runCount;
      const retiredLinks =
        'a[href="#/labs/lab04/requirements"], a[href="#/labs/lab04/materials"]';
      for (const route of ["", "/inference", "/protocol", "/evaluation"]) {
        await page.goto(base + "/#/labs/lab04" + route);
        await page
          .locator(route ? ".lab4-section-nav" : ".lab4-overview-shortcuts")
          .waitFor();
        assert.equal(await page.locator(retiredLinks).count(), 0);
        if (route) {
          assert.deepEqual(
            await page.locator(".lab4-section-nav a").allTextContents(),
            ["现场推理", "方法与协议", "对比与消融"],
          );
          assert.equal(
            await page
              .locator('.lab4-section-nav a[aria-current="page"]')
              .count(),
            1,
          );
        }
      }
      for (const retired of ["requirements", "materials"]) {
        await page.goto(base + "/#/labs/lab04/" + retired);
        await page.locator(".lab4-setup").waitFor();
        await page.reload();
        await page.locator(".lab4-setup").waitFor();
        assert.equal(await page.locator(retiredLinks).count(), 0);
        assert.equal(
          await page
            .locator(".lab4-requirements, .lab4-materials-page, video")
            .count(),
          0,
        );
        assert.equal(
          await page
            .locator('.lab4-section-nav a[aria-current="page"]')
            .innerText(),
          "现场推理",
        );
      }
      for (const width of [320, 390, 768, 1366]) {
        await page.setViewportSize({ width, height: 900 });
        assert.ok(
          await page.evaluate(
            () => document.documentElement.scrollWidth <= innerWidth + 1,
          ),
          `navigation overflow at ${width}`,
        );
        await audit("reduced navigation " + width);
      }
      await screenshot("three-section-navigation");
      const catalog = await (
        await page.request.get(base + "/api/course")
      ).json();
      const lab = catalog.labs.find((l) => l.id === "lab04");
      assert.ok(
        lab.modules.every((m) => !["requirements", "materials"].includes(m.id)),
      );
      const current = lab.files.filter(
        (f) =>
          f.id.startsWith("current-") &&
          ["report", "slides", "video"].includes(f.kind),
      );
      assert.equal(
        current.length,
        5,
        "Removing UI entries must not delete deliverable files",
      );
      for (const file of current.filter((f) => f.available)) {
        const response = await page.request.get(base + file.href, {
          headers: { Range: "bytes=0-127" },
        });
        assert.ok([200, 206].includes(response.status()));
        if (file.kind === "video") assert.equal(response.status(), 206);
      }
      assert.equal(
        runCount,
        before,
        "Navigation changes must never run inference",
      );
    },
  );
  await check("previous lab entries still load", async () => {
    for (const lab of ["lab01", "lab02", "lab03"]) {
      await page.goto(base + "/#/labs/" + lab);
      await page.locator(".module-list").waitFor();
      assert.ok((await page.locator(".module-row").count()) > 0);
    }
  });
  assert.deepEqual(errors, []);
  console.log(
    JSON.stringify(
      { passed: checks.length, skipped, screenshots: output },
      null,
      2,
    ),
  );
} finally {
  await context.close();
  await browser.close();
}
