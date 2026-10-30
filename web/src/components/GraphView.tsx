import { useEffect, useMemo, useRef, useState } from "react";
import cytoscape from "cytoscape";
import type { Core, ElementDefinition } from "cytoscape";
import {
  Minus,
  Plus,
  CornersOut,
  Path as PathIcon,
  Pause,
  Play,
  SkipForward,
  DownloadSimple,
} from "@phosphor-icons/react";
import type { Analysis, CaseData } from "../types";
import { shortName } from "../api";
import {
  canvasColor,
  draftEdges,
  proofHighlights,
  proofOrder,
} from "../graph-utils";

const nodeId = (id: string) => "entity:" + id;
const edgeId = (id: string) => "relation:" + id;

type Props = {
  data: CaseData;
  analysis: Analysis | null;
  selected: string;
  onSelect: (id: string) => void;
  claim: string | null;
  activeFact: string | null;
  onActiveFact: (id: string | null) => void;
  dirty: boolean;
  busy: boolean;
  mode?: "source" | "reasoning";
  initialFocus?: boolean;
  initialActiveFact?: string | null;
};
export default function GraphView({
  data,
  analysis,
  selected,
  onSelect,
  claim,
  activeFact,
  onActiveFact,
  dirty,
  busy,
  mode = "reasoning",
  initialFocus = true,
  initialActiveFact = null,
}: Props) {
  const isSource = mode === "source";
  const container = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);
  const selectRef = useRef(onSelect);
  selectRef.current = onSelect;
  const [compact, setCompact] = useState(window.innerWidth < 1060);
  const [focus, setFocus] = useState(initialFocus);
  const [derived, setDerived] = useState(false);
  const [filters, setFilters] = useState({
    holding: true,
    guarantee: true,
    trade: true,
    kinship: true,
  });
  const [zoom, setZoom] = useState(100);
  const [playing, setPlaying] = useState(false);
  const [step, setStep] = useState(-1);
  const [hint, setHint] = useState(
    isSource
      ? "拖动节点整理布局；选中后可查看原始实体资料"
      : "拖动节点整理布局；选中后可跳转实验二分析",
  );
  const sourceEdges = useMemo(
    () =>
      isSource || dirty
        ? draftEdges(data)
        : analysis?.edges || draftEdges(data),
    [data, analysis, dirty, isSource],
  );
  const highlight = useMemo(
    () =>
      !isSource && !dirty
        ? proofHighlights(analysis, activeFact)
        : { nodes: new Set<string>(), edges: new Set<string>() },
    [analysis, activeFact, dirty, isSource],
  );
  const sequence = useMemo(
    () => (isSource || dirty ? [] : proofOrder(analysis, claim)),
    [analysis, claim, dirty, isSource],
  );
  const visible = useMemo(() => {
    const allowed = new Set<string>();
    if (!focus) {
      data.entities.forEach((e) => allowed.add(e.eid));
    } else {
      allowed.add(selected);
      for (const edge of sourceEdges) {
        if (edge.source === selected) allowed.add(edge.target);
        if (edge.target === selected) allowed.add(edge.source);
      }
      let frontier = [...allowed];
      for (let depth = 0; depth < 3; depth++) {
        const next: string[] = [];
        for (const h of data.holdings)
          if (frontier.includes(h.target) && !allowed.has(h.holder)) {
            allowed.add(h.holder);
            next.push(h.holder);
          }
        frontier = next;
      }
      for (const node of highlight.nodes) allowed.add(node);
    }
    const nodes = data.entities.filter((e) => allowed.has(e.eid));
    const edges = [
      ...sourceEdges.filter((e) => filters[e.kind as keyof typeof filters]),
      ...(!isSource && !dirty && derived ? analysis?.derived_edges || [] : []),
    ].filter((e) => allowed.has(e.source) && allowed.has(e.target));
    return { nodes, edges };
  }, [
    data,
    sourceEdges,
    selected,
    focus,
    filters,
    derived,
    analysis,
    dirty,
    highlight,
    isSource,
  ]);
  const topology =
    visible.nodes
      .map((n) => n.eid)
      .sort()
      .join("|") +
    ";" +
    visible.edges
      .map((e) => e.id + e.source + e.target)
      .sort()
      .join("|");
  useEffect(() => {
    setPlaying(false);
    setStep(-1);
    if (!isSource) onActiveFact(initialActiveFact);
  }, [analysis?.id, claim, dirty, isSource]);
  useEffect(() => {
    if (!playing || !sequence.length) return;
    const timer = setTimeout(() => {
      const next = step + 1;
      if (next >= sequence.length) {
        setPlaying(false);
        return;
      }
      setStep(next);
      onActiveFact(sequence[next]);
    }, 1250);
    const pause = () => {
      if (document.hidden) setPlaying(false);
    };
    document.addEventListener("visibilitychange", pause);
    return () => {
      clearTimeout(timer);
      document.removeEventListener("visibilitychange", pause);
    };
  }, [playing, step, sequence, onActiveFact]);
  useEffect(() => {
    if (!container.current) return;
    const primary = canvasColor("--primary");
    const ink = canvasColor("--ink");
    const muted = canvasColor("--muted");
    const border = canvasColor("--graph-line");
    const white = canvasColor("--bg");
    const cy = cytoscape({
      container: container.current,
      elements: [],
      style: [
        {
          selector: "node",
          style: {
            label: "data(label)",
            "font-family": "Microsoft YaHei, Segoe UI, sans-serif",
            "font-size": 13,
            "font-weight": 600,
            "text-wrap": "wrap",
            "text-max-width": "106px",
            "text-valign": "center",
            "text-halign": "center",
            color: ink,
            "background-color": white,
            "border-color": border,
            "border-width": 1.2,
            width: 120,
            height: 56,
            shape: "round-rectangle",
            "overlay-opacity": 0,
          },
        },
        {
          selector: 'node[kind="Person"]',
          style: {
            shape: "ellipse",
            width: 42,
            height: 42,
            "background-color": canvasColor("--surface"),
            "text-valign": "bottom",
            "text-margin-y": 9,
            "text-max-width": "110px",
            "font-size": 12,
          },
        },
        {
          selector: "edge",
          style: {
            width: 1.3,
            "line-color": border,
            "target-arrow-color": border,
            "target-arrow-shape": "triangle",
            "arrow-scale": 0.75,
            "curve-style": "bezier",
            label: "data(label)",
            "font-family": "Segoe UI, Microsoft YaHei, sans-serif",
            "font-size": 12,
            "text-background-color": white,
            "text-background-opacity": 0.94,
            "text-background-padding": "3px",
            "text-rotation": "autorotate",
            "text-margin-y": -5,
            color: muted,
            "overlay-opacity": 0,
          },
        },
        {
          selector: 'edge[kind="guarantee"]',
          style: {
            "line-color": canvasColor("--graph-guarantee"),
            "target-arrow-color": canvasColor("--graph-guarantee"),
            "line-style": "dashed",
            "line-dash-pattern": [6, 4],
          },
        },
        {
          selector: 'edge[kind="trade"]',
          style: {
            "line-color": canvasColor("--graph-trade"),
            "target-arrow-color": canvasColor("--graph-trade"),
            "line-style": "dotted",
            "control-point-step-size": 42,
          },
        },
        {
          selector: 'edge[kind="kinship"]',
          style: {
            "line-style": "dotted",
            "target-arrow-shape": "none",
            width: 1.5,
          },
        },
        {
          selector: 'edge[kind="control"]',
          style: {
            "line-color": primary,
            "target-arrow-color": primary,
            "line-style": "dashed",
            "control-point-step-size": 70,
            label: "控制",
          },
        },
        {
          selector: "node.selected",
          style: {
            "background-color": primary,
            "border-color": primary,
            "border-width": 2,
            color: canvasColor("--on-primary"),
            "font-weight": 700,
          },
        },
        { selector: 'node.selected[kind="Person"]', style: { color: ink } },
        {
          selector: "node.evidence",
          style: { "border-color": primary, "border-width": 2.5 },
        },
        {
          selector: "edge.evidence",
          style: {
            "line-color": primary,
            "target-arrow-color": primary,
            width: 3,
            color: primary,
            "z-index": 10,
          },
        },
      ],
      minZoom: 0.2,
      maxZoom: 2.5,
      layout: { name: "preset" },
    });
    cy.on("tap", "node", (event) =>
      selectRef.current(event.target.data("eid")),
    );
    cy.on("zoom", () => setZoom(Math.round(cy.zoom() * 100)));
    cy.on("mouseover", "node,edge", () => {
      if (container.current) container.current.style.cursor = "pointer";
    });
    cy.on("mouseout", "node,edge", () => {
      if (container.current) container.current.style.cursor = "grab";
    });
    cyRef.current = cy;
    const observer = new ResizeObserver(() => {
      cy.resize();
      setCompact(cy.width() < 600);
      if (cy.nodes().length) {
        cy.fit(undefined, cy.width() < 600 || cy.height() < 380 ? 22 : 44);
        if (cy.zoom() > 1.35) {
          cy.zoom(1.35);
          cy.center();
        }
      }
    });
    observer.observe(container.current);
    return () => {
      observer.disconnect();
      cy.destroy();
      cyRef.current = null;
    };
  }, []);
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    const previous = new Map(
      cy.nodes().map((n) => [n.data("eid"), n.position()]),
    );
    const elements: ElementDefinition[] = [
      ...visible.nodes.map((node, i) => ({
        data: {
          id: nodeId(node.eid),
          eid: node.eid,
          label:
            node.kind === "Person"
              ? (node.eid.length > 10 ? "…" + node.eid.slice(-7) : node.eid) +
                " · " +
                node.name.slice(0, 8)
              : (node.eid.length > 10 ? "…" + node.eid.slice(-7) : node.eid) +
                "\n" +
                (shortName(node.name).length > 8
                  ? shortName(node.name).slice(0, 8) + "…"
                  : shortName(node.name)),
          kind: node.kind,
        },
        position: previous.get(node.eid) || {
          x:
            Math.cos((i / Math.max(1, visible.nodes.length)) * Math.PI * 2) *
            220,
          y:
            Math.sin((i / Math.max(1, visible.nodes.length)) * Math.PI * 2) *
            160,
        },
      })),
      ...visible.edges.map((edge) => ({
        data: {
          ...edge,
          id: edgeId(edge.id),
          source: nodeId(edge.source),
          target: nodeId(edge.target),
        },
      })),
    ];
    cy.batch(() => {
      cy.elements().remove();
      cy.add(elements);
    });
    if (visible.nodes.length <= 10) {
      const others = cy.nodes().filter((n) => n.data("eid") !== selected);
      const rx = Math.max(
        compact ? 105 : 130,
        Math.min(compact ? 180 : 260, cy.width() * (compact ? 0.31 : 0.34)),
      );
      const ry = Math.max(95, Math.min(170, cy.height() * 0.28));
      cy.getElementById(nodeId(selected)).position({ x: 0, y: 0 });
      others.forEach((node, i) => {
        const angle =
          -Math.PI / 2 + (i * Math.PI * 2) / Math.max(1, others.length);
        node.position({ x: Math.cos(angle) * rx, y: Math.sin(angle) * ry });
      });
      cy.layout({
        name: "preset",
        fit: true,
        padding: compact || cy.height() < 380 ? 22 : 44,
        animate: false,
      }).run();
    } else {
      const people = cy.nodes().filter((n) => n.data("kind") === "Person");
      const companies = cy.nodes().filter((n) => n.data("kind") === "Company");
      const columns = Math.max(
        3,
        Math.min(5, Math.ceil(Math.sqrt(companies.length))),
      );
      const rows = Math.ceil(companies.length / columns);
      const spacingX = 175,
        spacingY = 120;
      companies.forEach((node, i) => {
        node.position({
          x: ((i % columns) - (columns - 1) / 2) * spacingX,
          y: (Math.floor(i / columns) - (rows - 1) / 2) * spacingY + 50,
        });
      });
      people.forEach((node, i) => {
        node.position({
          x: (i - (people.length - 1) / 2) * spacingX,
          y: (-(rows + 1) * spacingY) / 2,
        });
      });
      cy.layout({
        name: "preset",
        fit: true,
        padding: compact ? 24 : 48,
        animate: false,
      }).run();
    }
    if (cy.zoom() > 1.35) {
      cy.zoom(1.35);
      cy.center();
    }
    cy.getElementById(nodeId(selected)).addClass("selected");
  }, [topology, compact]);
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.batch(() => {
      for (const edge of visible.edges) {
        const el = cy.getElementById(edgeId(edge.id));
        if (el.length) el.data("label", edge.label);
      }
      cy.nodes().removeClass("selected evidence");
      cy.edges().removeClass("evidence");
      cy.getElementById(nodeId(selected)).addClass("selected");
      for (const id of highlight.nodes)
        cy.getElementById(nodeId(id)).addClass("evidence");
      for (const id of highlight.edges)
        cy.getElementById(edgeId(id)).addClass("evidence");
    });
  }, [selected, highlight, visible.edges]);
  function fit() {
    const cy = cyRef.current;
    if (cy) {
      cy.fit(undefined, cy.width() < 600 || cy.height() < 380 ? 22 : 40);
      if (cy.zoom() > 1.35) {
        cy.zoom(1.35);
        cy.center();
      }
    }
  }
  function zoomBy(scale: number) {
    const cy = cyRef.current;
    if (cy)
      cy.zoom({
        level: Math.max(0.2, Math.min(2.5, cy.zoom() * scale)),
        renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 },
      });
  }
  function advance() {
    setPlaying(false);
    const next = (step + 1) % Math.max(sequence.length, 1);
    setStep(next);
    onActiveFact(sequence[next] || null);
  }
  function play() {
    if (!sequence.length) return;
    if (playing) {
      setPlaying(false);
      return;
    }
    if (step >= sequence.length - 1) {
      setStep(-1);
      onActiveFact(null);
    }
    setPlaying(true);
  }
  async function exportGraph() {
    const cy = cyRef.current;
    if (!cy) return;
    const blob = await cy.png({
      output: "blob-promise",
      full: true,
      scale: 2,
      bg: canvasColor("--bg"),
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "equity-graph.png";
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    setHint("当前可见图谱已导出");
  }
  const current = step >= 0 ? analysis?.provenance[sequence[step]] : null;
  return (
    <section className="graph-panel" aria-label="交互式企业关系图">
      <div className="graph-toolbar">
        <div className="graph-title">
          <PathIcon size={19} />
          <h2>{isSource ? "原始关系图" : "企业关系图"}</h2>
          <span>
            {visible.nodes.length}/{data.entities.length} 实体
          </span>
        </div>
        <div className="segmented" aria-label="图谱范围">
          <button
            className={focus ? "active" : ""}
            onClick={() => setFocus(true)}
            aria-pressed={focus}
          >
            {isSource ? "当前实体关系" : "当前企业关系"}
          </button>
          <button
            className={!focus ? "active" : ""}
            onClick={() => setFocus(false)}
            aria-pressed={!focus}
          >
            整个案例
          </button>
        </div>
      </div>
      <div className="graph-stage">
        <div
          className="graph-canvas"
          ref={container}
          tabIndex={0}
          role="img"
          aria-label="可缩放、拖拽的股权图。使用上方定位实体选择节点；加号放大、减号缩小、数字0适配画布。"
          onKeyDown={(e) => {
            if (e.key === "+" || e.key === "=") {
              e.preventDefault();
              zoomBy(1.2);
            }
            if (e.key === "-") {
              e.preventDefault();
              zoomBy(1 / 1.2);
            }
            if (e.key === "0") {
              e.preventDefault();
              fit();
            }
          }}
        />
        <div className="graph-legend" aria-label="关系筛选">
          {(["holding", "guarantee", "trade", "kinship"] as const).map(
            (kind) => (
              <button
                key={kind}
                aria-pressed={filters[kind]}
                className={!filters[kind] ? "off" : ""}
                onClick={() =>
                  setFilters({ ...filters, [kind]: !filters[kind] })
                }
              >
                <i className={"legend-line " + kind} />
                {
                  {
                    holding: "持股",
                    guarantee: "担保",
                    trade: "交易",
                    kinship: "亲属",
                  }[kind]
                }
              </button>
            ),
          )}
        </div>
        <div className="graph-controls">
          <button
            className="icon-button"
            onClick={() => zoomBy(1.2)}
            title="放大"
            aria-label="放大关系图"
          >
            <Plus size={17} />
          </button>
          <output>{zoom}%</output>
          <button
            className="icon-button"
            onClick={() => zoomBy(1 / 1.2)}
            title="缩小"
            aria-label="缩小关系图"
          >
            <Minus size={17} />
          </button>
          <span className="control-divider" />
          <button
            className="icon-button"
            onClick={fit}
            title="适配画布"
            aria-label="适配画布"
          >
            <CornersOut size={18} />
          </button>
          <button
            className="icon-button"
            onClick={() => void exportGraph()}
            title="导出当前图谱"
            aria-label="导出当前图谱PNG"
          >
            <DownloadSimple size={18} />
          </button>
        </div>
        {dirty && (
          <div className="draft-label">
            {isSource ? "当前案例草稿" : "草案关系 · 结果待重新计算"}
          </div>
        )}
      </div>
      <div className="graph-meta">
        <span>{hint}</span>
        {!isSource && (
          <label className="inline-check">
            <input
              type="checkbox"
              checked={derived}
              disabled={dirty || !analysis}
              onChange={(e) => setDerived(e.target.checked)}
            />
            <span>推导控制关系</span>
          </label>
        )}
      </div>
      {!isSource && (
        <div className="proof-player">
          <div className="player-buttons">
            <button
              className={"icon-button " + (playing ? "selected" : "")}
              disabled={!sequence.length || dirty || busy}
              onClick={play}
              aria-label={playing ? "暂停推理演示" : "播放当前结论的证明"}
              title={playing ? "暂停" : "播放证明"}
            >
              {playing ? (
                <Pause size={19} weight="fill" />
              ) : (
                <Play size={19} weight="fill" />
              )}
            </button>
            <button
              className="icon-button"
              disabled={!sequence.length || dirty || busy}
              onClick={advance}
              aria-label="下一步证明"
              title="下一步"
            >
              <SkipForward size={19} />
            </button>
          </div>
          <div className="player-copy">
            <strong>
              {current
                ? current.rule
                  ? current.rule + " · " + current.rule_description
                  : "读取输入事实"
                : "让推理过程看得见"}
            </strong>
            <span>
              {current
                ? current.label
                : sequence.length
                  ? "播放当前结论的 " + sequence.length + " 个证明节点"
                  : "选中企业后，在证据中选择要演示的结论"}
            </span>
          </div>
          <span className="step-counter">
            {step >= 0 ? step + 1 : 0} / {sequence.length}
          </span>
        </div>
      )}
    </section>
  );
}
