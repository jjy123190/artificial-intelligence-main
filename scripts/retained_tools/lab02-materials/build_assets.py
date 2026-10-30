"""Render publication figures from the frozen, verified CSV snapshot."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import PercentFormatter

WORK = Path(__file__).resolve().parent
DATA = json.loads((WORK / "data-snapshot.json").read_text(encoding="utf-8"))["tables"]
ASSETS = WORK / "assets"
ASSETS.mkdir(exist_ok=True)
for path in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
    if Path(path).exists():
        font_manager.fontManager.addfont(path)
        FAMILY = font_manager.FontProperties(fname=path).get_name()
        break
else:
    FAMILY = "DejaVu Sans"
plt.rcParams.update({"font.family": FAMILY, "font.size": 10, "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
                     "figure.facecolor": "white", "savefig.facecolor": "white", "grid.alpha": .17})
COLORS = ["#8D979C", "#456F83", "#A55430", "#58836A"]


def save(fig, name):
    path = ASSETS / (name + ".png")
    fig.savefig(path, dpi=320, bbox_inches="tight", pad_inches=.10)
    fig.savefig(ASSETS / (name + ".svg"), bbox_inches="tight", pad_inches=.10)
    plt.close(fig)
    print(path.name, flush=True)


def architecture():
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    def box(x, y, w, h, title, detail, fill="#F6F1EA"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.006,rounding_size=.015",
                                    linewidth=.8, edgecolor="#CBBFB3", facecolor=fill))
        ax.text(x+w/2, y+h*.68, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color="#342D29")
        ax.text(x+w/2, y+h*.29, detail, ha="center", va="center", fontsize=8.5, color="#665E57")
    def arrow(a, b, color="#A55430"):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, linewidth=1.1, color=color))
    for x, title, detail in [(.015,"观测资料","持股／交易／担保"),(.27,"结构推理","27条规则与证明"),
                             (.525,"指标抽取","四维观测与缺测"),(.78,"数值评分","Bayes／Mamdani")]:
        box(x,.62,.205,.28,title,detail)
    for x in (.22,.475,.73): arrow((x,.76),(x+.044,.76))
    box(.27,.12,.205,.28,"已证结构 A","原始证明始终保留", "#EBF1F2")
    box(.59,.12,.395,.28,"最终复核行动","不锁定／硬锁定／校准复核", "#F5E7DD")
    arrow((.372,.62),(.372,.405))
    arrow((.475,.26),(.585,.26))
    arrow((.883,.62),(.883,.405))
    ax.text(.911,.51,"分数 s",fontsize=8.5,ha="center",color="#665E57")
    fig.tight_layout(pad=.2)
    save(fig,"architecture")


def deterministic():
    fig, axes = plt.subplots(1,2,figsize=(6.6,3.2))
    groups=[1,4,8,16]
    for i,(model,label) in enumerate([("naive_scan","全扫描"),("indexed_naive","索引朴素"),("semi_naive","半朴素")]):
        rows=[next(r for r in DATA["forward"] if int(r["groups"])==g and r["model"]==model) for g in groups]
        for ax,field in zip(axes,["elapsed_ms","candidate_checks"]):
            ax.errorbar(groups,[float(r[field+"_mean"]) for r in rows],yerr=[float(r[field+"_sd"]) for r in rows],
                        color=COLORS[i],marker="o",markersize=4,linewidth=1.7,capsize=2,label=label)
    for ax,title,ylabel in zip(axes,["(a) 推理耗时","(b) 候选检查"],["毫秒（对数轴）","次数（对数轴）"]):
        ax.set(yscale="log",xlabel="股权结构组数",ylabel=ylabel,title=title)
        ax.set_xticks(groups);ax.grid();ax.tick_params(labelsize=8.5)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=3,frameon=False,fontsize=9)
    fig.tight_layout(rect=(0,.10,1,1),pad=.5)
    save(fig,"deterministic")


def indicator():
    fig,ax=plt.subplots(figsize=(6.6,3.25))
    scenarios=["clean","noise_010","noise_020","noise_030","missing_010","missing_030","mixed","severe"]
    labels=["清洁","噪声.10","噪声.20","噪声.30","缺测.10","缺测.30","混合","重度"]
    for i,(model,name) in enumerate([("bayes_learned","Bayes学习CPT"),("fuzzy","Mamdani手工"),
                                    ("hybrid_fixed","固定混合"),("fuzzy_learned","Mamdani拟合")]):
        rows=[next(r for r in DATA["indicator"] if r["scenario"]==s and r["model"]==model) for s in scenarios]
        ax.errorbar(range(8),[float(r["f1_mean"]) for r in rows],yerr=[float(r["f1_sd"]) for r in rows],
                    marker="o",markersize=3.8,linewidth=1.6,capsize=2,color=[COLORS[1],COLORS[0],COLORS[2],COLORS[3]][i],label=name)
    ax.set_xticks(range(8),labels,rotation=20,ha="right",fontsize=9)
    ax.set(ylabel="F1",ylim=(.5,.97));ax.grid(axis="y")
    ax.legend(ncol=2,frameon=False,fontsize=8.5,loc="upper right")
    fig.tight_layout(pad=.5)
    save(fig,"indicator_stress")


def policy():
    fig,axes=plt.subplots(1,2,figsize=(6.6,3.35))
    scenarios=["clean","false_loop_020","severe"]
    labels=["清洁","错误闭环","重度"]
    configs=[("hybrid_adaptive_unlocked","不锁定",COLORS[1]),("hybrid_adaptive_locked","硬锁定",COLORS[0]),
             ("hybrid_adaptive_calibrated","校准复核",COLORS[2])]
    for i,(model,name,color) in enumerate(configs):
        rows=[next(r for r in DATA["graph"] if r["scenario"]==s and r["model"]==model) for s in scenarios]
        for ax,field in zip(axes,["recall","false_positive_rate"]):
            ax.bar([x+(i-1)*.24 for x in range(3)],[float(r[field+"_mean"]) for r in rows],width=.22,
                   yerr=[float(r[field+"_sd"]) for r in rows],capsize=2,color=color,label=name,error_kw={"elinewidth":.7})
    for ax,title in zip(axes,["(a) 召回率","(b) 误报率"]):
        ax.set_xticks(range(3),labels,fontsize=9);ax.set(title=title,ylim=(0,1.05))
        ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis="y");ax.tick_params(axis="y",labelsize=8.5)
    handles,names=axes[0].get_legend_handles_labels();fig.legend(handles,names,loc="lower center",ncol=3,frameon=False,fontsize=9)
    fig.tight_layout(rect=(0,.10,1,1),pad=.5)
    save(fig,"policy_tradeoff")


if __name__ == "__main__":
    architecture();deterministic();indicator();policy()
