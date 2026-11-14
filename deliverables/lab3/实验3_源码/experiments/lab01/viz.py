"""可视化：知识图谱（推理前 / 推理增强后）与效率实验折线图，均输出静态 PNG；
若安装了 pyvis，另输出可拖拽的交互式 HTML。

配色取自经色觉障碍（CVD）校验的分类调色板：节点类型只用 3 种色相（担保公司并入"企业"），
关系类型靠线型 + 边标签区分；风险预警用状态红圈 + 警示符 + 文字标注，不单靠颜色传达含义。
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from matplotlib.transforms import Bbox

from experiments.lab01.kr.knowledge_graph import INFERRED, SCHEMA

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Segoe UI Symbol", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
CRITICAL = "#d03b3b"                                     # 状态色，只用于风险预警
NODE_COLOR = {"企业": "#2a78d6", "自然人": "#eb6834", "上市公司": "#1baf7a"}
NODE_LEGEND = {"企业": "企业 / 担保公司", "自然人": "自然人", "上市公司": "上市公司"}
NODE_SIZE = 1000
EDGE_STYLE = {  # 关系: (线型, 标签格式)
    "HOLDS": ("solid", lambda d: f"{d['ratio']:.0%}"),
    "GUARANTEES": ("dashed", lambda d: f"担保{d['amount']}"),
    "TRADES": ("dotted", lambda d: f"交易{d['amount']}"),
    "KIN": ("dashdot", lambda d: d["relation"]),
    "CONTROLS": ("solid", lambda d: f"实控·{d['rule']}"),
}
LABEL_BOX = dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.9)
# 边标签候选位置 (垂直于边的偏移/磅, 沿边的比例)：先沿边滑动，都不行再往边的两侧挪
LABEL_SPOTS = [(shift, t) for shift in (0, 9, -9, 16, -16, 24, -24, 32, -32)
               for t in (0.5, 0.4, 0.6, 0.3, 0.7, 0.25, 0.75, 0.2, 0.8, 0.15, 0.85)]
WARN = "▲"      # 微软雅黑没有 ⚠ 字形，PNG 中改用 ▲；HTML 由浏览器渲染，仍用 ⚠


def short(name):
    return name.replace("有限公司", "").replace("股份", "")


def layout(kg):
    """在推理前的图上计算一次布局，推理前后两张图共用，便于对照。"""
    return nx.spring_layout(nx.Graph(kg.G), k=1.8, iterations=300, seed=7)


def avoid_overlap(fig, ax, labels, names, centers):
    """边标签默认在边的中点，可能压到节点名称或别的标签上。逐个沿各自的边滑动
    （label_pos 取 0.5 附近的候选值），选第一个不与节点、名称、已放标签重叠的位置。"""
    renderer = fig.canvas.get_renderer()
    r = NODE_SIZE ** 0.5 / 2 * fig.dpi / 72                 # 节点半径（像素）
    taken = [t.get_window_extent(renderer).padded(3) for t in names]
    taken += [Bbox.from_extents(x - r, y - r, x + r, y + r)
              for x, y in ax.transData.transform(centers)]

    to_disp, to_data = ax.transData.transform, ax.transData.inverted().transform

    def along(t, p):                                         # 边上比例 p 处的显示坐标
        t.label_pos = p
        return to_disp(type(t)._update_text_pos_angle(t, t.arrow)[:2])

    def place(t, shift, p):
        a, b = along(t, p - 0.01), along(t, p + 0.01)
        d = b - a
        normal = np.array([-d[1], d[0]]) / (np.hypot(*d) or 1)
        xy = to_data(along(t, p) + normal * shift * fig.dpi / 72)
        t._update_text_pos_angle = lambda arrow, xy=xy: (*xy, 0)   # 固定位置，draw 时不再按中点重算
        t.set_position(xy)
        return t.get_window_extent(renderer).padded(3)

    bad = 0
    circles = taken[len(names):]
    for t in labels:
        text = [b for b in taken if not any(b is c for c in circles)]
        cost = lambda box: 10 * sum(box.overlaps(b) for b in text) + sum(box.overlaps(c) for c in circles)
        best = min(LABEL_SPOTS, key=lambda s: (cost(place(t, *s)), LABEL_SPOTS.index(s)))
        box = place(t, *best)                                  # 找不到完全空的位置时，宁可压节点圆也不压文字
        bad += any(box.overlaps(b) for b in text)
        taken.append(box)
    return bad


def draw_kg(kg, path, pos, title, enriched=False):
    """enriched=False：原始图谱（知识表示）；
    enriched=True：推理增强后 —— 已知关系退为浅灰背景，突出回写的"实际控制"边与风险标签。"""
    G = kg.G
    fig, ax = plt.subplots(figsize=(14, 10.5), dpi=150, facecolor=SURFACE)   # 与保存时同 dpi，避让计算才一致
    ax.set_facecolor(SURFACE)

    label_groups = []                    # 边标签等节点和名称画完后再摆放，以便避让
    for rel, (style, fmt) in EDGE_STYLE.items():
        edges = [(u, v, d) for u, v, d in G.edges(data=True) if d["rel"] == rel]
        if not edges:
            continue
        inferred = rel in INFERRED
        color = INK if inferred else (BASELINE if enriched else INK_2)
        conn = f"arc3,rad={0.3 if inferred else 0.08}"
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, _ in edges], edge_color=color,
                               style=style, width=2.2 if inferred else 1.3, arrows=True, arrowsize=14,
                               node_size=NODE_SIZE, connectionstyle=conn, ax=ax)
        if inferred or not enriched:
            label_groups.append(({(u, v): fmt(d) for u, v, d in edges}, INK if inferred else INK_2, conn))

    nodes = list(G.nodes(data=True))
    risky = {n for n, d in nodes if enriched and d.get("risks")}
    nx.draw_networkx_nodes(G, pos, nodelist=[n for n, _ in nodes], node_size=NODE_SIZE,
                           node_color=[NODE_COLOR.get(d["category"], NODE_COLOR["企业"]) for _, d in nodes],
                           edgecolors=[CRITICAL if n in risky else SURFACE for n, _ in nodes],
                           linewidths=[3.5 if n in risky else 2 for n, _ in nodes], ax=ax)
    names = []
    for n, d in nodes:                   # 名称写在节点下方：长名字放不进圆内，也不压在色块上
        text = short(d["name"]) + (f"\n{WARN} {'、'.join(d['risks'])}" if n in risky else "")
        names.append(ax.annotate(text, pos[n], xytext=(0, -19), textcoords="offset points", ha="center",
                                 va="top", fontsize=9.5, color=INK, linespacing=1.3,
                                 bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.85)))

    labels = []
    for edge_labels, color, conn in label_groups[::-1]:   # 推理边排在最后，倒序后优先占位
        labels += nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, font_color=color,
                                               rotate=False, connectionstyle=conn, bbox=LABEL_BOX,
                                               node_size=NODE_SIZE, ax=ax).values()
    ax.margins(0.06)
    ax.set(xlim=ax.get_xlim(), ylim=ax.get_ylim())          # 先定死坐标范围，避让计算才准
    avoid_overlap(fig, ax, labels, names, [pos[n] for n, _ in nodes])

    handles = [Line2D([], [], marker="o", ls="", markersize=11, markerfacecolor=c, markeredgecolor=SURFACE,
                      label=NODE_LEGEND[k]) for k, c in NODE_COLOR.items()]
    if enriched:
        handles += [Line2D([], [], color=BASELINE, lw=1.3, label="已知关系"),
                    Line2D([], [], color=INK, lw=2.2, label="实际控制（推理回写，标注推出它的规则）"),
                    Line2D([], [], marker="o", ls="", markersize=11, markerfacecolor=SURFACE,
                           markeredgecolor=CRITICAL, markeredgewidth=2.5, label=f"{WARN} 风险预警")]
    else:
        handles += [Line2D([], [], color=INK_2, ls=style, lw=1.3, label=SCHEMA["relations"][rel][2])
                    for rel, (style, _) in EDGE_STYLE.items() if rel not in INFERRED]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(handles),
              frameon=False, fontsize=10, handlelength=2.2, columnspacing=1.6, borderaxespad=0.3)
    ax.set_title(title, loc="left", fontsize=16, color=INK, pad=34)
    ax.axis("off")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def draw_kg_html(kg, path):
    """可交互版本（需要 pip install pyvis），适合录 Demo 视频时拖拽展示。"""
    try:
        from pyvis.network import Network
    except ImportError:
        return False
    net = Network(height="800px", width="100%", directed=True, cdn_resources="in_line")
    for n, d in kg.G.nodes(data=True):
        risks = d.get("risks")
        net.add_node(n, label=d["name"] + (f"\n⚠ {'、'.join(risks)}" if risks else ""),
                     color={"background": NODE_COLOR.get(d["category"], NODE_COLOR["企业"]),
                            "border": CRITICAL if risks else SURFACE},
                     borderWidth=4 if risks else 1, title=f"{d['category']} {n}")
    for u, v, d in kg.G.edges(data=True):
        style, fmt = EDGE_STYLE[d["rel"]]
        net.add_edge(u, v, label=fmt(d), color=INK if d["rel"] in INFERRED else INK_2,
                     dashes=style != "solid")
    with open(path, "w", encoding="utf-8") as f:   # pyvis 的 write_html 在中文 Windows 上按 GBK 写入会报错
        f.write(net.generate_html())
    return True


# ---------------------------------------------------------------- 效率实验
SERIES_COLOR = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # 分类色按固定顺序分配，两张子图同一方法同一颜色


def fmt_ms(v):
    return f"{v:.2f} ms" if v < 1 else (f"{v:.1f} ms" if v < 100 else f"{v:,.0f} ms")


def draw_benchmark(path, panels, methods, title, note):
    """panels: [dict(title, subtitle, xlabel, x, log_x, series={方法: [毫秒...]})]
    methods: [(完整名称, 线尾简称)]，决定颜色顺序与图例顺序。"""
    color = {m: SERIES_COLOR[i] for i, (m, _) in enumerate(methods)}
    tag = dict(methods)
    fig, axes = plt.subplots(1, len(panels), figsize=(14, 6.4), facecolor=SURFACE)
    for ax, p in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for m, ys in p["series"].items():
            ax.plot(p["x"], ys, color=color[m], lw=2, marker="o", ms=8, mec=SURFACE, mew=2,
                    solid_joinstyle="round", solid_capstyle="round", zorder=3)
            ax.annotate(f"{tag[m]}  {fmt_ms(ys[-1])}", (p["x"][-1], ys[-1]), xytext=(9, 0),
                        textcoords="offset points", va="center", fontsize=9.5, color=INK_2)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}" if v < 1 else f"{v:,.0f}"))
        ax.yaxis.set_minor_locator(NullLocator())
        if p.get("log_x"):
            ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(p["x"]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.tick_params(colors=MUTED, labelsize=9.5, length=0)
        ax.set_xlabel(p["xlabel"], color=INK_2, fontsize=10.5)
        ax.set_ylabel("每次查询耗时（毫秒，对数坐标）", color=INK_2, fontsize=10.5)
        ax.set_title(f"{p['title']}\n", loc="left", fontsize=13, color=INK)
        ax.text(0, 1.02, p["subtitle"], transform=ax.transAxes, fontsize=9.5, color=INK_2)
        ax.margins(x=0.04)
    handles = [Line2D([], [], color=color[m], lw=2, marker="o", ms=8, mec=SURFACE, mew=2, label=m)
               for m, _ in methods]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.925), ncol=len(handles),
               frameon=False, fontsize=10.5, columnspacing=2)
    fig.suptitle(title, x=0.01, y=0.995, ha="left", fontsize=16, color=INK)
    fig.text(0.01, 0.01, note, fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 0.93, 0.87), w_pad=6)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
