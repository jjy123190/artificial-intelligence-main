import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowsOut,
  Crosshair,
  MagnifyingGlassMinus,
  MagnifyingGlassPlus,
} from "@phosphor-icons/react";
import type { Result } from "./types";
import { stateKey, treeStates } from "./tree-model";
import { relations, statuses } from "./presentation";

export default function SearchTree({
  result,
  index,
  onSelectStep,
}: {
  result: Result;
  index: number;
  onSelectStep: (index: number) => void;
}) {
  const viewport = useRef<HTMLDivElement>(null);
  const drag = useRef<{
    x: number;
    y: number;
    left: number;
    top: number;
  } | null>(null);
  const [zoom, setZoom] = useState(1);
  const [view, setView] = useState<"tree" | "list">("tree");
  const [width, setWidth] = useState(800);
  const visible = useMemo(
    () => treeStates(result.result.trace, index),
    [result, index],
  );
  const names = useMemo(
    () => Object.fromEntries(result.graph.nodes.map((n) => [n.id, n.name])),
    [result],
  );
  const columns = new Map<number, typeof visible.states>();
  visible.states.forEach((s) =>
    columns.set(s.depth, [...(columns.get(s.depth) || []), s]),
  );
  const positions = new Map<string, { x: number; y: number }>();
  const rows = Math.max(1, ...[...columns.values()].map((c) => c.length));
  columns.forEach((items, depth) =>
    items.forEach((s, i) =>
      positions.set(stateKey(s.edge_ids), {
        x: 36 + depth * 256,
        y: 58 + i * 90 + (rows - items.length) * 45,
      }),
    ),
  );
  const canvasWidth = Math.max(
    780,
    72 + Math.max(0, ...columns.keys()) * 256 + 180,
  );
  const canvasHeight = Math.max(440, 90 + rows * 90);
  const scale = Math.max(0.75, Math.min(1, width / canvasWidth)) * zoom;
  const currentKey = visible.current ? stateKey(visible.current.edge_ids) : "";
  const recorded = useMemo(() => {
    const indices = new Map<string, number>();
    result.result.trace
      .slice(0, index + 1)
      .forEach((event, position) =>
        indices.set(stateKey(event.edge_ids), position),
      );
    return indices;
  }, [result, index]);

  useEffect(() => {
    const container = viewport.current;
    if (!container) return;
    const resize = new ResizeObserver(([entry]) =>
      setWidth(entry.contentRect.width),
    );
    resize.observe(container);
    return () => resize.disconnect();
  }, [view]);

  function focusCurrent() {
    const container = viewport.current;
    const position = positions.get(currentKey);
    if (!container || !position) return;
    container.scrollTo({
      left: (position.x + 90) * scale - container.clientWidth / 2,
      top: (position.y + 36) * scale - container.clientHeight / 2,
      behavior: "instant",
    });
  }

  return (
    <div className="lab3-tree-view">
      <div className="lab3-tree-toolbar">
        <div className="segmented" aria-label="搜索过程视图">
          <button
            type="button"
            aria-pressed={view === "tree"}
            onClick={() => setView("tree")}
          >
            搜索树
          </button>
          <button
            type="button"
            aria-pressed={view === "list"}
            onClick={() => setView("list")}
          >
            步骤列表
          </button>
        </div>
        {view === "tree" && (
          <div className="lab3-zoom" aria-label="画布工具">
            <button
              type="button"
              className="icon-button"
              aria-label="缩小搜索树"
              disabled={zoom <= 0.75}
              onClick={() => setZoom((z) => Math.max(0.75, z - 0.25))}
            >
              <MagnifyingGlassMinus size={18} />
            </button>
            <span>{Math.round(scale * 100)}%</span>
            <button
              type="button"
              className="icon-button"
              aria-label="放大搜索树"
              disabled={zoom >= 1.75}
              onClick={() => setZoom((z) => Math.min(1.75, z + 0.25))}
            >
              <MagnifyingGlassPlus size={18} />
            </button>
            <button
              type="button"
              className="icon-button"
              aria-label="重置搜索树缩放"
              onClick={() => setZoom(1)}
            >
              <ArrowsOut size={18} />
            </button>
            <button
              type="button"
              className="button secondary compact"
              onClick={focusCurrent}
            >
              <Crosshair size={16} />
              定位当前
            </button>
          </div>
        )}
      </div>
      {view === "tree" ? (
        <>
          <div
            ref={viewport}
            className="lab3-tree-scroll"
            tabIndex={0}
            role="region"
            aria-label="搜索树画布，可用方向键滚动"
            onPointerDown={(event) => {
              if (
                event.pointerType !== "mouse" ||
                event.button !== 0 ||
                (event.target as Element).closest('[role="button"]')
              )
                return;
              drag.current = {
                x: event.clientX,
                y: event.clientY,
                left: event.currentTarget.scrollLeft,
                top: event.currentTarget.scrollTop,
              };
              event.currentTarget.setPointerCapture(event.pointerId);
            }}
            onPointerMove={(event) => {
              if (!drag.current) return;
              event.currentTarget.scrollLeft =
                drag.current.left + drag.current.x - event.clientX;
              event.currentTarget.scrollTop =
                drag.current.top + drag.current.y - event.clientY;
            }}
            onPointerUp={() => {
              drag.current = null;
            }}
            onLostPointerCapture={() => {
              drag.current = null;
            }}
          >
            <svg
              width={canvasWidth * scale}
              height={canvasHeight * scale}
              viewBox={"0 0 " + canvasWidth + " " + canvasHeight}
              role="group"
              aria-label="本次搜索轨迹：描边为当前节点，虚线为拦截，路径色表示最终返回的候选分支"
            >
              {[...columns.keys()].map((depth) => (
                <text
                  key={depth}
                  x={36 + depth * 256}
                  y={32}
                  fontSize={11}
                  fill="var(--muted)"
                >
                  {depth === 0
                    ? "起点 / 深度 0"
                    : "第 " +
                      depth +
                      " 跳 / " +
                      (relations[result.query.relations[depth - 1]] ||
                        "关系扩展")}
                </text>
              ))}
              {visible.states.map((s) => {
                const pos = positions.get(stateKey(s.edge_ids))!;
                const parent = positions.get(stateKey(s.edge_ids.slice(0, -1)));
                const onPath =
                  result.result.edge_ids.length > 0 &&
                  s.edge_ids.every((e, i) => e === result.result.edge_ids[i]) &&
                  visible.current?.status === "solution";
                return s.depth && parent ? (
                  <path
                    key={"edge" + stateKey(s.edge_ids)}
                    d={
                      "M " +
                      (parent.x + 180) +
                      " " +
                      (parent.y + 36) +
                      " C " +
                      (parent.x + 222) +
                      " " +
                      (parent.y + 36) +
                      ", " +
                      (pos.x - 42) +
                      " " +
                      (pos.y + 36) +
                      ", " +
                      pos.x +
                      " " +
                      (pos.y + 36)
                    }
                    fill="none"
                    stroke={onPath ? "var(--primary)" : "var(--graph-line)"}
                    strokeOpacity={onPath ? 1 : 0.7}
                    strokeWidth={onPath ? 2.5 : 1.2}
                  />
                ) : null;
              })}
              {visible.states.map((s) => {
                const pos = positions.get(stateKey(s.edge_ids))!;
                const current = stateKey(s.edge_ids) === currentKey;
                const rejected = s.status === "rejected";
                const label = names[s.node] || s.node;
                const eventIndex = recorded.get(stateKey(s.edge_ids));
                return (
                  <g
                    key={stateKey(s.edge_ids)}
                    transform={"translate(" + pos.x + "," + pos.y + ")"}
                    className={
                      eventIndex == null ? undefined : "lab3-tree-node"
                    }
                    role={eventIndex == null ? undefined : "button"}
                    tabIndex={eventIndex == null ? undefined : 0}
                    aria-label={
                      eventIndex == null
                        ? undefined
                        : `回看${label}，第 ${eventIndex + 1} 步`
                    }
                    aria-pressed={eventIndex == null ? undefined : current}
                    aria-description={`${statuses[s.status] || "路径节点"}，深度 ${s.g}${s.h == null ? "" : `，h ${s.h.toFixed(2)}，f ${s.f?.toFixed(2)}`}`}
                    onClick={() => {
                      if (eventIndex != null) onSelectStep(eventIndex);
                    }}
                    onKeyDown={(event) => {
                      if (
                        eventIndex != null &&
                        (event.key === "Enter" || event.key === " ")
                      ) {
                        event.preventDefault();
                        onSelectStep(eventIndex);
                      }
                    }}
                  >
                    <title>
                      {label +
                        " · " +
                        (statuses[s.status] || "路径节点（过程未保留）") +
                        (s.reason ? " · " + s.reason : "")}
                    </title>
                    <rect
                      width={180}
                      height={72}
                      rx={4}
                      fill={
                        rejected
                          ? "var(--danger-soft)"
                          : current
                            ? "var(--primary-soft)"
                            : "var(--node-bg)"
                      }
                      stroke={
                        rejected
                          ? "var(--danger)"
                          : current
                            ? "var(--primary)"
                            : "var(--node-border)"
                      }
                      strokeDasharray={rejected ? "4 3" : undefined}
                      strokeWidth={current ? 2 : 1}
                    />
                    <path
                      d="M1 29H179"
                      stroke={current ? "var(--primary)" : "var(--node-border)"}
                      strokeOpacity="0.5"
                    />
                    <circle
                      cx={0}
                      cy={36}
                      r={4}
                      fill="var(--node-bg)"
                      stroke={current ? "var(--primary)" : "var(--graph-line)"}
                    />
                    <circle
                      cx={180}
                      cy={36}
                      r={4}
                      fill="var(--node-bg)"
                      stroke={current ? "var(--primary)" : "var(--graph-line)"}
                    />
                    <text
                      aria-hidden="true"
                      x={12}
                      y={19}
                      fontSize={10}
                      fill={
                        rejected
                          ? "var(--danger)"
                          : current
                            ? "var(--primary)"
                            : "var(--muted)"
                      }
                    >
                      {statuses[s.status] || "路径节点"}
                    </text>
                    <text
                      aria-hidden="true"
                      x={168}
                      y={19}
                      textAnchor="end"
                      fontSize={10}
                      fill="var(--muted)"
                    >
                      g {s.g}
                    </text>
                    <text x={12} y={48} fontSize={13} fill="var(--ink)">
                      {label.length > 12 ? label.slice(0, 11) + "…" : label}
                    </text>
                    <text
                      aria-hidden="true"
                      x={12}
                      y={63}
                      fontSize={10}
                      fill="var(--muted)"
                    >
                      {s.h == null
                        ? s.depth === 0
                          ? "搜索起始状态"
                          : "候选关系路径"
                        : `h ${s.h.toFixed(2)}   /   f ${s.f?.toFixed(2)}`}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>
          <div className="lab3-tree-legend" aria-label="搜索树图例">
            <span>
              <i className="current" />
              当前节点
            </span>
            <span>
              <i className="rejected" />
              专家规则拦截
            </span>
            <span>
              <i />
              已访问的候选状态
            </span>
            <span className="lab3-pan-hint">拖动画布 · 点击节点回看</span>
          </div>
          {visible.hidden > 0 && (
            <p className="helper-text">
              当前显示 {visible.states.length} / {visible.total}{" "}
              个状态，保留当前分支。步骤列表和导出可查看全部已记录事件。
            </p>
          )}
        </>
      ) : (
        <div
          className="lab3-step-list"
          role="region"
          aria-label="搜索步骤列表"
          tabIndex={0}
        >
          <ol>
            {result.result.trace.slice(0, index + 1).map((s, i) => (
              <li key={s.id} aria-current={i === index ? "step" : undefined}>
                <button
                  type="button"
                  className="lab3-step-number"
                  aria-label={`回看第 ${i + 1} 步`}
                  onClick={() => onSelectStep(i)}
                >
                  {i + 1}
                </button>
                <div>
                  <strong>
                    {statuses[s.status] || s.status} · {names[s.node] || s.node}
                  </strong>
                  <p>{s.path.map((n) => names[n] || n).join(" → ")}</p>
                  <small>
                    深度 {s.g}
                    {s.h != null
                      ? " · h=" + s.h.toFixed(2) + " · f=" + s.f?.toFixed(2)
                      : ""}
                    {s.p != null
                      ? " · 学习概率 " + (s.p * 100).toFixed(1) + "%"
                      : ""}
                  </small>
                  {s.reason && <p className="lab3-rejection">{s.reason}</p>}
                </div>
              </li>
            ))}
          </ol>
        </div>
      )}
    </div>
  );
}
