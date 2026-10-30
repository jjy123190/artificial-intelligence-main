import { useEffect, useRef, useState } from "react";
import cytoscape from "cytoscape";
import type { Core } from "cytoscape";
import { CornersOut, Plus, Minus } from "@phosphor-icons/react";
import type { LabOneResult } from "../types";
import { canvasColor } from "../graph-utils";
export default function KnowledgeDiagram({ data }: { data: LabOneResult }) {
  const host = useRef<HTMLDivElement>(null);
  const graph = useRef<Core | null>(null);
  const [selected, setSelected] = useState("C3");
  const [scope, setScope] = useState("holdings");
  useEffect(() => {
    if (!host.current) return;
    const relations = data.triples.filter(
      (r) => scope === "all" || r.relation === "HOLDS",
    );
    const cy = cytoscape({
      container: host.current,
      elements: [
        ...data.entities.map((e) => ({
          data: {
            id: "entity:" + e.eid,
            label: e.name,
            kind: e.kind,
            eid: e.eid,
          },
        })),
        ...relations.map((r, i) => ({
          data: {
            id: "relation:" + i,
            source: "entity:" + r.source,
            target: "entity:" + r.target,
            label:
              r.relation === "HOLDS"
                ? (Number(r.value) * 100).toFixed(0) + "%"
                : r.name,
          },
        })),
      ],
      style: [
        {
          selector: "node",
          style: {
            label: "data(label)",
            "background-color": canvasColor("--paper"),
            "border-color": canvasColor("--graph-line"),
            "border-width": 1.5,
            color: canvasColor("--ink"),
            shape: "roundrectangle",
            width: 132,
            height: 52,
            "font-size": 12,
            "text-valign": "center",
            "text-wrap": "wrap",
            "text-max-width": "118px",
          },
        },
        {
          selector: 'node[kind="Person"]',
          style: {
            shape: "ellipse",
            width: 76,
            height: 76,
            "background-color": canvasColor("--surface"),
          },
        },
        {
          selector: "edge",
          style: {
            label: "data(label)",
            width: 1.5,
            "line-color": canvasColor("--graph-line"),
            "target-arrow-color": canvasColor("--graph-line"),
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            "font-size": 11,
            color: canvasColor("--muted"),
            "text-background-color": canvasColor("--bg"),
            "text-background-opacity": 0.94,
            "text-background-padding": "3px",
            "text-rotation": "autorotate",
          },
        },
        {
          selector: "node:selected",
          style: {
            "background-color": canvasColor("--primary-soft"),
            "border-color": canvasColor("--primary"),
            "border-width": 3,
          },
        },
      ],
      layout: { name: "grid", cols: 4, padding: 34, spacingFactor: 1.4 },
      minZoom: 0.2,
      maxZoom: 2.5,
      wheelSensitivity: 0.2,
    });
    graph.current = cy;
    cy.getElementById("entity:" + selected).select();
    cy.on("tap", "node", (event) => setSelected(event.target.id().slice(7)));
    const observer = new ResizeObserver(() => {
      cy.resize();
      cy.fit(undefined, 32);
    });
    observer.observe(host.current);
    return () => {
      observer.disconnect();
      cy.destroy();
      graph.current = null;
    };
  }, [data, scope]);
  useEffect(() => {
    const cy = graph.current;
    if (!cy) return;
    cy.nodes().unselect();
    cy.getElementById("entity:" + selected).select();
  }, [selected]);
  function zoom(factor: number) {
    const cy = graph.current;
    if (cy)
      cy.zoom({
        level: cy.zoom() * factor,
        renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 },
      });
  }
  return (
    <section className="knowledge-diagram" aria-label="实验一原始知识图谱">
      <div className="knowledge-controls">
        <label>
          定位实体
          <select
            aria-label="实验一图谱实体"
            value={selected}
            onChange={(e) => setSelected(e.target.value)}
          >
            {data.entities.map((e) => (
              <option key={e.eid} value={e.eid}>
                {e.eid} · {e.name}
              </option>
            ))}
          </select>
        </label>
        <label>
          显示关系
          <select
            aria-label="实验一图谱关系"
            value={scope}
            onChange={(e) => setScope(e.target.value)}
          >
            <option value="holdings">持股关系</option>
            <option value="all">全部原始关系</option>
          </select>
        </label>
        <div>
          <button
            className="icon-button"
            aria-label="放大实验一图谱"
            onClick={() => zoom(1.2)}
          >
            <Plus size={18} />
          </button>
          <button
            className="icon-button"
            aria-label="缩小实验一图谱"
            onClick={() => zoom(1 / 1.2)}
          >
            <Minus size={18} />
          </button>
          <button
            className="icon-button"
            aria-label="适配实验一图谱"
            onClick={() => graph.current?.fit(undefined, 32)}
          >
            <CornersOut size={18} />
          </button>
        </div>
      </div>
      <div
        ref={host}
        className="knowledge-canvas"
        role="img"
        tabIndex={0}
        aria-label="实验一原始关系图。上方选择器可定位实体，加号放大、减号缩小、数字0适配画布。"
        onKeyDown={(e) => {
          if (["+", "=", "-", "0"].includes(e.key)) e.preventDefault();
          if (e.key === "+" || e.key === "=") zoom(1.2);
          if (e.key === "-") zoom(1 / 1.2);
          if (e.key === "0") graph.current?.fit(undefined, 32);
        }}
      />
      <p className="helper-text">
        矩形是企业，圆形是自然人。此处只有原始知识；规则任务新增的关系在知识增强结果中单独展示。
      </p>
    </section>
  );
}
