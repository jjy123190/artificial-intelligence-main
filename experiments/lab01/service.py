"""Experiment 1 adapter: execute the original methods on read-only shared CSV data."""
from __future__ import annotations
import csv
import json
import time
from pathlib import Path
from core.kb.guard_display import describe_guard
from experiments.lab01.kr.knowledge_graph import EquityKG, SCHEMA, INFERRED
from experiments.lab01.kr.frames import build_company_frames
from experiments.lab01.kr.production import InferenceEngine
from experiments.lab01.reasoning.rules import RULES, RULE_DESC
from experiments.lab01.reasoning.graph_reasoning import penetrate, relation_path, find_cycles
ROOT=Path(__file__).resolve().parents[2]

def read_dataset():
    raw={}
    for name in ("persons","companies","holdings","guarantees","trades","kinships"):
        with (ROOT/"data"/"raw"/(name+".csv")).open(encoding="utf-8-sig",newline="") as f:
            raw[name]=list(csv.DictReader(f))
    return {"persons":[(r["id"],r["姓名"]) for r in raw["persons"]],
        "companies":[(r["id"],r["名称"],r["类型"],float(r["注册资本万元"])) for r in raw["companies"]],
        "holdings":[(r["股东"],r["被投企业"],float(r["持股比例"])) for r in raw["holdings"]],
        "guarantees":[(r["担保方"],r["被担保方"],float(r["金额万元"])) for r in raw["guarantees"]],
        "trades":[(r["卖方"],r["买方"],float(r["金额万元"])) for r in raw["trades"]],
        "kinships":[(r["自然人1"],r["自然人2"],r["关系"]) for r in raw["kinships"]]}

def frame_rows(frames):
    result=[]
    for cid,f in frames.items():
        result.append({"id":cid,"name":f.name,"parent":f.is_a.name if f.is_a else None,
            "slots":[{"name":slot,"value":f.get(slot),"unit":f.get(slot,"单位"),
                "inherited":"值" not in f.slots.get(slot,{})} for slot in f.all_slots()]})
    return result

def run_task(payload):
    started=time.perf_counter(); data=read_dataset(); kg=EquityKG(data); facts=kg.to_predicates()
    _,frames=build_company_frames(data)
    schema=[{"id":key,"name":spec[2],"head_types":list(spec[0]),"tail_types":list(spec[1]),"attribute":spec[3],"inferred":key in INFERRED} for key,spec in SCHEMA["relations"].items()]
    result={"experiment":"lab01","version":"original-lab01-methods","entities":[{"eid":id,"name":d["name"],"kind":d["type"]} for id,d in kg.G.nodes(data=True)],
        "triples":[{"source":u,"target":v,"relation":d["rel"],"name":SCHEMA["relations"][d["rel"]][2],"value":d[SCHEMA["relations"][d["rel"]][3]]} for u,v,d in kg.G.edges(data=True)],
        "schema":schema,"facts":sorted(facts,key=str),"rules":[{"name":r.name,"description":r.desc,"conditions":r.conditions,"conclusions":r.conclusions,"priority":r.priority,"has_guard":r.test is not None,**describe_guard(r.test)} for r in RULES],
        "frames_before":frame_rows(frames),"stats":{"entities":kg.G.number_of_nodes(),"raw_relations":kg.G.number_of_edges(),"input_facts":len(facts),"rules":len(RULES)},
        "note":"实验一的原始规则与简单路径方法，读取公共CSV但不改写；不使用实验二规则代替。"}
    task=payload.get("task","snapshot"); target=payload.get("target","C3"); source=payload.get("source","P1")
    if target not in kg.G or source not in kg.G: raise ValueError("请选择实验一案例中的实体")
    if task=="rules":
        if kg.G.nodes[target]["type"] != "Company": raise ValueError("控制查询目标必须是企业")
        engine=InferenceEngine(RULES,facts);cycles=engine.run()
        if cycles>=10000: raise ValueError("规则推理达到预算，未确认完成")
        goal=("Controls",source,target)
        proved=goal in engine.wm
        result.update({"goal":goal,"proved":proved,"proof":engine.explain(goal,rule_desc=RULE_DESC) if proved else [],
            "steps":[{"cycle":c,"rule":r,"fact":f,"premises":engine.why[f][1]} for c,r,f in engine.trace],
            "derived_facts":sorted(engine.wm.facts-facts,key=str),"controls":engine.query("ActualController","?c","?p"),"alerts":engine.query("Alert","?c","?t")})
        for _,c,p in engine.query("ActualController","?c","?p"):
            frames[c].set("实际控制人",kg.name(p));kg.add_relation(p,"CONTROLS",c,engine.why[("Controls",p,c)][0])
        for _,c,level in engine.query("RiskLevel","?c","?l"):frames[c].set("风险等级",level)
        for cid in frames:
            alerts=[f[2] for f in engine.query("Alert",cid,"?t")]
            if alerts:frames[cid].set("风险提示","、".join(alerts))
        result["frames_after"]=frame_rows(frames)
        result["enhanced_relations"]=[{"source":u,"target":v,"rule":d["rule"]} for u,v,d in kg.G.edges(data=True) if d["rel"]=="CONTROLS"]
        result["stats"].update({"cycles":cycles,"total_facts":len(engine.wm),"added_facts":len(engine.wm)-len(facts),"added_relations":len(result["enhanced_relations"])})
    elif task=="penetration":
        if kg.G.nodes[target]["type"]!="Company":raise ValueError("股权穿透目标必须是企业")
        result["owners"]=penetrate(kg,target,max_depth=payload.get("max_depth",10));result["target"]=target
        result["holding_cycles"]=find_cycles(kg,"HOLDS");result["guarantee_cycles"]=find_cycles(kg,"GUARANTEES")
        result["note"]+=" 循环仅按简单路径截断，不能当成无限路径经济权益；25%只是教学筛查阈值。"
    elif task=="path":
        result["path"]=relation_path(kg,source,target);result["source"]=source;result["target"]=target
        result["note"]+=" 最短关联路径忽略搜索方向、跨关系类型，展示仍保留原边箭头；不等于控制关系。"
    elif task!="snapshot":raise ValueError("未知实验一任务")
    benchmark=ROOT/"outputs"/"lab01"/"benchmark.json"
    result["benchmark"]=json.loads(benchmark.read_text(encoding="utf-8")) if benchmark.is_file() else None
    result["elapsed_ms"]=(time.perf_counter()-started)*1000
    return result
