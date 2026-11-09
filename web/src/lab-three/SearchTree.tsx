import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowsOut,
  Crosshair,
  MagnifyingGlassMinus,
  MagnifyingGlassPlus,
} from "@phosphor-icons/react";
import type { Result } from "./types";
import { stateKey, treeStates } from "./tree-model";
import { statuses } from "./presentation";

export default function SearchTree({
  result,
  index,
}: {
  result: Result;
  index: number;
}) {
  const viewport = useRef<HTMLDivElement>(null);
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
  columns.forEach((items, depth) =>
    items.forEach((s, i) =>
      positions.set(stateKey(s.edge_ids), {
        x: 24 + depth * 202,
        y: 46 + i * 62,
      }),
    ),
  );
  const canvasWidth = Math.max(
    600,
    48 + (Math.max(0, ...columns.keys()) + 1) * 202 - 40,
  );
  const canvasHeight = Math.max(
    220,
    62 + Math.max(0, ...[...columns.values()].map((c) => c.length)) * 62,
  );
  const scale = Math.max(0.75, Math.min(1, width / canvasWidth)) * zoom;
  const currentKey = visible.current ? stateKey(visible.current.edge_ids) : "";

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
      left: (position.x + 80) * scale - container.clientWidth / 2,
      top: (position.y + 20) * scale - container.clientHeight / 2,
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
          <p className="helper-text lab3-pan-hint">
            可横向、纵向滚动；小屏可切换步骤列表，阅读完整实体名称。
          </p>
          <div
            ref={viewport}
            className="lab3-tree-scroll"
            tabIndex={0}
            role="region"
            aria-label="搜索树画布，可用方向键滚动"
          >
            <svg
              width={canvasWidth * scale}
              height={canvasHeight * scale}
              viewBox={"0 0 " + canvasWidth + " " + canvasHeight}
              role="img"
              aria-label="本次搜索轨迹：描边为当前节点，虚线为拦截，路径色表示最终返回的候选分支"
            >
              {[...columns.keys()].map((depth) => (
                <text
                  key={depth}
                  x={24 + depth * 202}
                  y={25}
                  fontSize={12}
                  fill="var(--muted)"
                >
                  {depth === 0 ? "起点" : "第 " + depth + " 跳"}
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
                      (parent.x + 162) +
                      " " +
                      (parent.y + 21) +
                      " C " +
                      (parent.x + 186) +
                      " " +
                      (parent.y + 21) +
                      ", " +
                      (pos.x - 24) +
                      " " +
                      (pos.y + 21) +
                      ", " +
                      pos.x +
                      " " +
                      (pos.y + 21)
                    }
                    fill="none"
                    stroke={onPath ? "var(--primary)" : "var(--graph-line)"}
                    strokeOpacity={onPath ? 1 : 0.5}
                    strokeWidth={onPath ? 2.5 : 1.2}
                  />
                ) : null;
              })}
              {visible.states.map((s) => {
                const pos = positions.get(stateKey(s.edge_ids))!;
                const current = stateKey(s.edge_ids) === currentKey;
                const rejected = s.status === "rejected";
                const label = names[s.node] || s.node;
                return (
                  <g
                    key={stateKey(s.edge_ids)}
                    transform={"translate(" + pos.x + "," + pos.y + ")"}
                  >
                    <title>
                      {label +
                        " · " +
                        (statuses[s.status] || "路径节点（过程未保留）") +
                        (s.reason ? " · " + s.reason : "")}
                    </title>
                    <rect
                      width={162}
                      height={42}
                      rx={6}
                      fill={
                        rejected
                          ? "var(--danger-soft)"
                          : current
                            ? "var(--primary-soft)"
                            : "var(--paper)"
                      }
                      stroke={
                        rejected
                          ? "var(--danger)"
                          : current
                            ? "var(--primary)"
                            : "var(--border)"
                      }
                      strokeDasharray={rejected ? "4 3" : undefined}
                      strokeWidth={current ? 2 : 1}
                    />
                    <text
                      x={81}
                      y={26}
                      textAnchor="middle"
                      fontSize={12}
                      fill="var(--ink)"
                    >
                      {label.length > 12 ? label.slice(0, 11) + "…" : label}
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
                <span className="lab3-step-number">{i + 1}</span>
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
