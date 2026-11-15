/** Real-browser regression suite. Install: npm ci, then npx playwright install chromium.
 * Run with a local service: LAB03_TEST_URL=http://127.0.0.1:8765 npm run test:browser
 * Screenshots stay in the OS temp directory, not among submission materials.
 */
import assert from "node:assert/strict";
import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { chromium } from "playwright";
import AxeBuilder from "@axe-core/playwright";

const base = process.env.LAB03_TEST_URL || "http://127.0.0.1:8765";
const browser = await chromium.launch({
  headless: true,
  channel: process.env.PLAYWRIGHT_CHANNEL || undefined,
});
const context = await browser.newContext({
  viewport: { width: 1366, height: 900 },
});
const errors = [];
context.on("page", (page) =>
  page.on("pageerror", (error) => errors.push(error.message)),
);
const snapshots = await mkdtemp(join(tmpdir(), "lab03-ui-check-"));
const cases = [];
const runButton = (page) =>
  page.getByRole("button", { name: "运行路径搜索", exact: true });
async function ready(page) {
  await runButton(page).waitFor();
}
async function run(page) {
  await runButton(page).click();
  await page.getByRole("button", { name: "导出本次结果" }).waitFor();
}
async function check(name, fn) {
  await fn();
  cases.push(name);
  console.log("PASS", name);
}
const snapshot = await (
  await context.request.get(base + "/api/labs/lab03/snapshot")
).json();
assert.equal(snapshot.warnings.length, 0);
const page = await context.newPage();
try {
  await check(
    "desktop / mobile / tablet: four modules and accessibility",
    async () => {
      for (const width of [320, 390, 768, 920, 1366]) {
        await page.setViewportSize({ width, height: 900 });
        for (const module of [
          "search",
          "optimization",
          "assistant",
          "evaluation",
        ]) {
          await page.goto(base + "/#/labs/lab03/" + module);
          await page.locator(".lab3-page .task-section").first().waitFor();
          const overflow = await page.evaluate(
            () => document.documentElement.scrollWidth > innerWidth,
          );
          assert.equal(overflow, false, width + " " + module + " overflows");
          const audit = await new AxeBuilder({ page })
            .withTags(["wcag2a", "wcag2aa", "wcag21aa", "best-practice"])
            .analyze();
          assert.deepEqual(
            audit.violations.map((v) => ({
              id: v.id,
              nodes: v.nodes.map((n) => n.target),
            })),
            [],
            width + " " + module,
          );
          if (width === 390 || width === 1366)
            await page.screenshot({
              path: join(snapshots, module + "-" + width + ".png"),
            });
        }
      }
    },
  );
  await check(
    "real search: playback, focused branch, list and export",
    async () => {
      await page.goto(base + "/#/labs/lab03/search");
      await ready(page);
      await run(page);
      assert.match(
        await page.locator("#lab3-result-heading").innerText(),
        /可核验/,
      );
      assert.equal(
        await page.evaluate(() => document.activeElement?.id),
        "lab3-result-heading",
      );
      await page.getByRole("button", { name: "第一步", exact: true }).click();
      assert.equal(
        await page.getByLabel("搜索轨迹步骤", { exact: true }).inputValue(),
        "0",
      );
      await page.getByRole("button", { name: "下一步", exact: true }).click();
      assert.equal(
        await page.getByLabel("搜索轨迹步骤", { exact: true }).inputValue(),
        "1",
      );
      await page.getByRole("button", { name: "播放轨迹", exact: true }).click();
      await page.waitForFunction(
        () =>
          Number(
            document.querySelector('input[aria-label="搜索轨迹步骤"]').value,
          ) > 1,
      );
      await page.getByRole("button", { name: "暂停", exact: true }).click();
      await page.getByRole("button", { name: "最后一步", exact: true }).click();
      await page.setViewportSize({ width: 390, height: 844 });
      await page.getByRole("button", { name: "定位当前", exact: true }).click();
      assert.ok(
        await page
          .locator(".lab3-tree-scroll")
          .evaluate((el) => el.scrollLeft > 0),
      );
      await page.getByRole("button", { name: "步骤列表", exact: true }).click();
      assert.ok((await page.locator(".lab3-step-list li").count()) > 3);
      await page.screenshot({
        path: join(snapshots, "search-mobile-list.png"),
      });
      const downloadPromise = page.waitForEvent("download");
      await page
        .getByRole("button", { name: "导出本次结果", exact: true })
        .click();
      const download = await downloadPromise;
      const exported = JSON.parse(
        await readFile(await download.path(), "utf8"),
      );
      assert.equal(exported.result.valid, true);
      assert.equal(exported.result.method, "hybrid");
      assert.ok(exported.path_evidence.every((edge) => edge.eligible));
      assert.match(download.suggestedFilename(), /shared-3-hybrid-16/);
    },
  );
  await check("budget exhaustion gives a working budget increase", async () => {
    await page
      .getByLabel("实验三搜索方法", { exact: true })
      .selectOption("bfs");
    await page
      .getByLabel("实验三问题", { exact: true })
      .selectOption("test-7-0-Q0");
    await page.getByLabel("实验三扩展预算", { exact: true }).selectOption("4");
    await run(page);
    assert.match(
      await page.locator("#lab3-result-heading").innerText(),
      /预算/,
    );
    await page
      .getByRole("button", { name: "增加到 8 个节点并重试", exact: true })
      .click();
    await page.getByRole("button", { name: "导出本次结果" }).waitFor();
    assert.equal(
      await page.getByLabel("实验三扩展预算", { exact: true }).inputValue(),
      "8",
    );
    assert.match(
      await page.locator(".lab3-current-step").innerText(),
      /不使用h/,
    );
  });
  await check(
    "LLM replay blocks missing records and can switch to a real sample",
    async () => {
      await page
        .getByLabel("实验三问题", { exact: true })
        .selectOption("shared-3");
      await page
        .getByLabel("实验三搜索方法", { exact: true })
        .selectOption("hybrid_llm");
      assert.equal(await runButton(page).isDisabled(), true);
      await page
        .getByRole("button", { name: "切换到有实测记录的问题" })
        .click();
      await run(page);
      assert.match(
        await page.locator(".task-result").innerText(),
        /没有再次调用模型/,
      );
      await page
        .getByLabel("实验三LLM来源", { exact: true })
        .selectOption("remote");
      if (!snapshot.remote_available)
        assert.equal(await runButton(page).isDisabled(), true);
    },
  );
  await check(
    "snapshot failure has bounded loading and explicit retry",
    async () => {
      const p = await context.newPage();
      let first = true;
      await p.route("**/api/labs/lab03/snapshot", async (route) => {
        if (first) {
          first = false;
          await route.fulfill({
            status: 503,
            contentType: "application/json",
            body: JSON.stringify({
              error: { code: "unavailable", message: "测试：暂时不可用" },
            }),
          });
        } else await route.continue();
      });
      await p.goto(base + "/#/labs/lab03/search");
      await p.getByRole("button", { name: "重新加载", exact: true }).click();
      await ready(p);
      assert.equal(await p.locator(".course-skeleton").count(), 0);
      await p.close();
    },
  );
  await check(
    "leaving during snapshot load aborts it; returning reloads correctly",
    async () => {
      const p = await context.newPage();
      let began;
      const started = new Promise((r) => {
        began = r;
      });
      let first = true;
      let release;
      const gate = new Promise((r) => {
        release = r;
      });
      await p.route("**/api/labs/lab03/snapshot", async (route) => {
        if (first) {
          first = false;
          began();
          await gate;
        }
        await route
          .fulfill({
            status: 200,
            contentType: "application/json",
            body: JSON.stringify(snapshot),
          })
          .catch(() => {});
      });
      await p.goto(base + "/#/labs/lab03/search");
      await started;
      await p.locator('.course-navigation > a[href="#/labs/lab01"]').click();
      release();
      await p.goto(base + "/#/labs/lab03/search");
      await ready(p);
      assert.equal(await p.locator(".error-banner:visible").count(), 0);
      await p.close();
    },
  );
  await check(
    "cancelled response cannot overwrite a newer real run",
    async () => {
      const p = await context.newPage();
      await p.goto(base + "/#/labs/lab03/search");
      await ready(p);
      const old = await (
        await context.request.post(base + "/api/labs/lab03/run", {
          data: { query_id: "shared-3", method: "hybrid", budget: 16 },
        })
      ).json();
      let first = true,
        release;
      const gate = new Promise((r) => {
        release = r;
      });
      await p.route("**/api/labs/lab03/run", async (route) => {
        if (first) {
          first = false;
          await gate;
          await route
            .fulfill({
              status: 200,
              contentType: "application/json",
              body: JSON.stringify(old),
            })
            .catch(() => {});
        } else await route.continue();
      });
      await runButton(p).click();
      await p.getByRole("button", { name: "停止等待", exact: true }).click();
      await p.getByLabel("实验三搜索方法", { exact: true }).selectOption("bfs");
      await run(p);
      release();
      await p.waitForTimeout(150);
      assert.match(await p.locator(".lab3-result-meta").innerText(), /^BFS/);
      await p.close();
    },
  );
  await check("network error can retry without stale result", async () => {
    const p = await context.newPage();
    await p.goto(base + "/#/labs/lab03/search");
    await ready(p);
    let first = true;
    await p.route("**/api/labs/lab03/run", async (route) => {
      if (first) {
        first = false;
        await route.abort("failed");
      } else await route.continue();
    });
    await runButton(p).click();
    await p.getByRole("button", { name: "重新运行", exact: true }).click();
    await p.getByRole("button", { name: "导出本次结果" }).waitFor();
    assert.equal(await p.locator(".error-banner:visible").count(), 0);
    await p.close();
  });
  await check(
    "missing optional artifacts show empty states, not fabricated zeros",
    async () => {
      const p = await context.newPage();
      const missing = {
        ...snapshot,
        ga: {},
        learning: [],
        summary: [],
        manifest: {},
        paired: [],
        failures: [],
        llm_summary: {},
        llm_explanations: [],
        stress: {},
        warnings: ["测试：可选实验结果缺失"],
      };
      await p.route("**/api/labs/lab03/snapshot", (route) =>
        route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify(missing),
        }),
      );
      for (const module of ["optimization", "assistant", "evaluation"]) {
        await p.goto(base + "/#/labs/lab03/" + module);
        await p.locator(".lab3-page .task-section").first().waitFor();
        assert.ok((await p.locator(".task-empty").count()) > 0);
      }
      await p.close();
    },
  );
  await check(
    "reduced motion and keyboard scrolling remain usable",
    async () => {
      await page.emulateMedia({ reducedMotion: "reduce" });
      await page.goto(base + "/#/labs/lab03/search");
      await ready(page);
      await page
        .getByLabel("实验三搜索方法", { exact: true })
        .selectOption("bfs");
      await run(page);
      await page.getByRole("button", { name: "搜索树", exact: true }).click();
      const canvas = page.locator(".lab3-tree-scroll");
      await canvas.focus();
      await canvas.press("ArrowRight");
      assert.equal(
        await canvas.evaluate((el) => el === document.activeElement),
        true,
      );
      const a = await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze();
      assert.equal(a.violations.length, 0);
    },
  );
  await check(
    "previous labs still expose working paths and controls",
    async () => {
      for (const route of [
        "",
        "labs/lab01",
        "labs/lab02",
        "labs/lab01/demo",
        "labs/lab02/deterministic",
      ]) {
        await page.goto(base + "/#/" + route);
        await page.locator("main").filter({ visible: true }).first().waitFor();
        assert.equal(await page.locator(".error-banner:visible").count(), 0);
      }
      const one = await context.request.post(base + "/api/labs/lab01/run", {
        data: { task: "rules", source: "P1", target: "C3" },
      });
      assert.equal(one.ok(), true);
      assert.equal((await one.json()).stats.total_facts, 135);
    },
  );
  assert.deepEqual(errors, []);
  console.log(
    JSON.stringify(
      {
        passed: cases.length,
        cases,
        screenshots: snapshots,
        browserErrors: errors,
      },
      null,
      2,
    ),
  );
} finally {
  await browser.close();
}
