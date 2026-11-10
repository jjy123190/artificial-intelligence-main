/** Workbench interaction regressions. Uses the same local server and installed
 * browser as test:browser; only error/race cases intercept network responses. */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { chromium } from "playwright";
import AxeBuilder from "@axe-core/playwright";

const base = process.env.LAB03_TEST_URL || "http://127.0.0.1:8765";
const browser = await chromium.launch({
  headless: true,
  channel: process.env.PLAYWRIGHT_CHANNEL || undefined,
});
const context = await browser.newContext({
  viewport: { width: 1366, height: 950 },
});
const errors = [];
context.on("page", (page) =>
  page.on("pageerror", (error) => errors.push(error.message)),
);
const page = await context.newPage();
const snapshot = await (
  await context.request.get(base + "/api/labs/lab03/snapshot")
).json();
const cases = [];
async function check(name, fn) {
  await fn();
  cases.push(name);
  console.log("PASS", name);
}
async function exportJson(page, name) {
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name, exact: true }).click();
  const download = await downloadPromise;
  return JSON.parse(await readFile(await download.path(), "utf8"));
}
async function open(page) {
  await page.goto(base + "/#/labs/lab03/search");
  await page
    .getByRole("button", { name: "运行路径搜索", exact: true })
    .waitFor();
}
try {
  await check(
    "2/3/4-hop examples expose the requested state model",
    async () => {
      await open(page);
      for (const hops of [2, 4, 3]) {
        await page
          .getByRole("button", { name: `${hops} 跳示例`, exact: true })
          .click();
        const id = await page
          .getByLabel("实验三问题", { exact: true })
          .inputValue();
        const query = snapshot.queries.find((q) => q.id === id);
        assert.equal(query.hops, hops);
        assert.equal(
          await page.locator(".lab3-relation-chain li").count(),
          hops + 1,
        );
        assert.ok(
          (await page.locator(".lab3-query-model").innerText()).includes(
            query.start_name,
          ),
        );
      }
      await page.getByText("状态、动作与终止条件", { exact: true }).click();
      assert.match(
        await page.locator(".lab3-model-details").innerText(),
        /置信度 ≥ 0.55/,
      );
    },
  );
  await check(
    "tree nodes and timeline support keyboard replay without changing evidence",
    async () => {
      await page
        .getByRole("button", { name: "运行路径搜索", exact: true })
        .click();
      await page.getByRole("button", { name: "导出本次结果" }).waitFor();
      const answer = await page.locator(".lab3-path-answer").innerText();
      const node = page.locator(".lab3-tree-node").first();
      const label = await node.getAttribute("aria-label");
      const expected = Number(label.match(/第 (\d+) 步/)[1]) - 1;
      await node.focus();
      await node.press("Enter");
      assert.equal(
        Number(
          await page.getByLabel("搜索轨迹步骤", { exact: true }).inputValue(),
        ),
        expected,
      );
      const event = page.getByRole("button", { name: /^跳转到第 4 步/ });
      await event.click();
      assert.equal(
        await page.getByLabel("搜索轨迹步骤", { exact: true }).inputValue(),
        "3",
      );
      assert.equal(await page.locator(".lab3-path-answer").innerText(), answer);
      await page.getByRole("button", { name: "最后一步", exact: true }).click();
      const audit = await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze();
      assert.deepEqual(
        audit.violations.map((v) => ({
          id: v.id,
          targets: v.nodes.map((n) => n.target),
        })),
        [],
      );
    },
  );
  await check(
    "four real runs share a question and budget, export independently verifiable costs",
    async () => {
      await page
        .getByRole("button", { name: "对比四种算法", exact: true })
        .click();
      await page.getByRole("button", { name: "导出算法对比" }).waitFor();
      const comparison = await exportJson(page, "导出算法对比");
      assert.equal(comparison.complete, true);
      assert.deepEqual(
        comparison.rows.map((r) => r.method),
        ["bfs", "astar_fixed", "astar_ga", "mcts"],
      );
      for (const row of comparison.rows) {
        assert.equal(row.result.query.id, comparison.query.id);
        assert.equal(row.result.result.budget, comparison.budget);
        assert.equal(row.result.result.expert_enabled, true);
        const direct = await (
          await context.request.post(base + "/api/labs/lab03/run", {
            data: {
              query_id: comparison.query.id,
              method: row.method,
              budget: comparison.budget,
            },
          })
        ).json();
        for (const field of [
          "status",
          "valid",
          "expanded",
          "total_work",
          "rollout_steps",
          "hops",
        ])
          assert.equal(
            row.result.result[field],
            direct.result[field],
            row.method + " " + field,
          );
      }
      const audit = await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa"])
        .analyze();
      assert.deepEqual(
        audit.violations.map((v) => v.id),
        [],
      );
    },
  );
  await check(
    "one unavailable algorithm keeps other results and offers a complete retry",
    async () => {
      const p = await context.newPage();
      await p.route("**/api/labs/lab03/run", (route) => {
        if (route.request().postDataJSON().method === "astar_fixed")
          return route.fulfill({
            status: 503,
            contentType: "application/json",
            body: JSON.stringify({
              error: { message: "测试：工作进程暂不可用" },
            }),
          });
        return route.continue();
      });
      await open(p);
      await p
        .getByRole("button", { name: "对比四种算法", exact: true })
        .click();
      await p.getByRole("button", { name: "导出算法对比" }).waitFor();
      const result = await exportJson(p, "导出算法对比");
      assert.equal(result.rows.filter((row) => row.result).length, 3);
      assert.match(result.rows[1].error, /工作进程暂不可用/);
      assert.equal(
        await p
          .locator(".lab3-comparison-grid article")
          .nth(1)
          .locator(".lab3-compare-value")
          .innerText(),
        "—\n扩展节点",
      );
      await p.unroute("**/api/labs/lab03/run");
      await p.getByRole("button", { name: "重新对比算法" }).click();
      await p.getByRole("button", { name: "导出算法对比" }).waitFor();
      assert.ok(
        (await exportJson(p, "导出算法对比")).rows.every((row) => row.result),
      );
      await p.close();
    },
  );
  await check(
    "cancelled and changed-question comparisons discard late responses",
    async () => {
      const p = await context.newPage();
      await open(p);
      const stale = await (
        await context.request.post(base + "/api/labs/lab03/run", {
          data: { query_id: "shared-3", method: "bfs", budget: 16 },
        })
      ).json();
      let release;
      let reached;
      const gate = new Promise((resolve) => {
        release = resolve;
      });
      const intercepted = new Promise((resolve) => {
        reached = resolve;
      });
      let first = true;
      await p.route("**/api/labs/lab03/run", async (route) => {
        if (!first) return route.continue();
        first = false;
        reached();
        await gate;
        await route.fulfill({ json: stale }).catch(() => {});
      });
      await p
        .getByRole("button", { name: "对比四种算法", exact: true })
        .click();
      await intercepted;
      assert.equal(
        await p
          .getByRole("button", { name: "运行路径搜索", exact: true })
          .isDisabled(),
        true,
      );
      await p.getByRole("button", { name: "取消对比", exact: true }).click();
      await p.getByRole("button", { name: "2 跳示例", exact: true }).click();
      const id = await p.getByLabel("实验三问题", { exact: true }).inputValue();
      await p
        .getByRole("button", { name: "对比四种算法", exact: true })
        .click();
      release();
      await p.getByRole("button", { name: "导出算法对比" }).waitFor();
      const fresh = await exportJson(p, "导出算法对比");
      assert.equal(fresh.rows.length, 4);
      assert.ok(fresh.rows.every((row) => row.result.query.id === id));
      await p.close();
    },
  );
  await check(
    "evaluation export follows the selected split and budget",
    async () => {
      await page.goto(base + "/#/labs/lab03/evaluation");
      await page
        .getByLabel("实验三评估场景", { exact: true })
        .selectOption("ood");
      await page
        .getByLabel("实验三对比预算", { exact: true })
        .selectOption("4");
      const exported = await exportJson(page, "导出当前评估");
      assert.equal(exported.split, "ood");
      assert.equal(exported.budget, 4);
      assert.deepEqual(
        exported.metrics,
        snapshot.summary.filter(
          (row) => row.split === "ood" && row.budget === 4,
        ),
      );
      assert.deepEqual(
        exported.paired,
        snapshot.paired.filter(
          (row) => row.split === "ood" && row.budget === 4,
        ),
      );
    },
  );
  assert.deepEqual(errors, []);
  console.log(
    JSON.stringify(
      { passed: cases.length, cases, browserErrors: errors },
      null,
      2,
    ),
  );
} finally {
  await context.close();
  await browser.close();
}
