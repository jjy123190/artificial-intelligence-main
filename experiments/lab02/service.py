"""Experiment 2 requirement-oriented tasks, using the existing verified engines."""
from __future__ import annotations
import time
from core.kb.graph_io import case_from_dict
from experiments.lab02.hybrid.pipeline import prepare_case, assess_company
from experiments.lab02.deterministic.backward import BackwardEngine
from experiments.lab02.deterministic.resolution import ResolutionEngine, equity_resolution_example
from backend.worker import load_model

def run_task(payload):
    case=case_from_dict(payload["case"]);task=payload["task"];target=payload["target"];source=payload["source"]
    if target not in {c.eid for c in case.companies}:raise ValueError("目标必须是当前案例中的企业")
    if source not in case.by_id():raise ValueError("查询主体不在当前案例中")
    started=time.perf_counter()
    if task=="resolution":
        clauses,goal=equity_resolution_example(case)
        result=ResolutionEngine(optimized=True,max_seconds=8,max_pairs=250000).prove(clauses,goal)
        return {"task":task,"goal":str(goal),"proved":result.proved,"status":result.status,"stats":result.stats,"proof":result.lines(),
            "elapsed_ms":(time.perf_counter()-started)*1000,"note":"独立P1/C1/C2/C3子案例。数值守卫先转为显式算术事实，不冒充全知识库归结。"}
    context=prepare_case(case,completeness=payload["completeness"])
    result={"task":task,"target":target,"source":source,"preparation_ms":context.preparation_ms,"input_facts":len(context.facts),"forward_stats":context.engine.stats()}
    if task=="modeling":
        result.update({"facts":[{"fact":f.as_tuple(),"formula":str(f),"source":f.source} for f in context.facts],
            "fact_counts":context.facts.counts(),"derived_facts":[{"fact":f,"rule":why[0],"premises":why[1]} for f,why in sorted(context.engine.why.items(),key=lambda row:str(row[0]))],
            "note":"输入事实保留来源，预处理派生也明确标注；一阶谓词表示、结构结论和评分指标分开。"})
    elif task=="deterministic":
        goal=("Controls",source,target)
        backward=BackwardEngine(context.engine.rules,context.facts.to_tuples(),max_rounds=200,max_tables=12000)
        proof=backward.solve(goal);forward=context.engine.holds(goal)
        if forward!=(proof is not None):raise RuntimeError("正反向结果不一致")
        result.update({"goal":goal,"proved":forward,"equivalent":True,"forward_proof":context.engine.explain(goal) if forward else [],
            "backward_proof":proof.lines({r.name:r.desc for r in context.engine.rules}) if proof else [],"backward_stats":backward.stats(),
            "steps":[{"cycle":s.cycle,"rule":s.rule,"facts":s.facts} for s in context.engine.steps],"note":"命题可证只表示在当前教学事实和规则下可推出；无法证明不等于现实中为假。"})
    else:
        model=load_model(payload["model_source"]);observation=context.observations[target]
        result["model_source"]=payload["model_source"]
        if task=="uncertain":
            t=time.perf_counter();bayes=model.bayes.predict(observation);bayes_ms=(time.perf_counter()-t)*1000
            t=time.perf_counter();fuzzy_score=model.fuzzy.predict(observation);fuzzy_ms=(time.perf_counter()-t)*1000
            # Time the public predict interfaces; the extra UI explanation call is outside them.
            fuzzy=model.fuzzy.explain(observation)
            result.update({"observation":observation.as_dict(),"bayes":{"score":bayes,"parameters":model.bayes.as_dict(),"elapsed_ms":bayes_ms},
                "fuzzy":{"score":fuzzy_score,"explanation":fuzzy,"parameters":model.fuzzy.as_dict(),"elapsed_ms":fuzzy_ms},
                "timing_scope":"单次预测接口耗时，不含共同事实构建和额外的页面说明调用。Mamdani的predict复用explain完成隶属度与聚合计算；稳定的性能比较见多轮正式实验。",
                "note":"两种方法读取同一观测指标；Mamdani评分不是概率，合成数据不能认证真实企业风险能力。"})
        elif task=="hybrid":
            result["decision_threshold"]=payload["threshold"]
            result.update({"decisions":[assess_company(context,target,model,method=method,threshold=payload["threshold"],include_explanation=True) for method in ("hybrid_unlocked","hybrid_locked","hybrid_calibrated")],
                "note":"结构锁定影响复核行动，不改写原始分数；校准保留证明，也可能暂缓复核；效果通过实测而非单个案例认定。"})
        else:raise ValueError("未知实验二任务")
    result["elapsed_ms"]=(time.perf_counter()-started)*1000
    return result
