"""第二次课堂汇报的统一入口。核心算法和测试只依赖 Python 标准库。"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import unittest
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.kb.domain import demo_case, load_from_equity_kg
from core.kb.facts import build_facts
from core.kb.features import RiskObservation, from_case
from core.kb.rules import build_rules, describe_rules
from experiments.lab02.deterministic.forward import ForwardEngine, InferenceLimitError
from experiments.lab02.deterministic.backward import BackwardEngine
from experiments.lab02.deterministic.resolution import ResolutionEngine, equity_resolution_example
from experiments.lab02.hybrid.model import HybridModel
from experiments.lab02.hybrid.pipeline import prepare_case, assess_company, METHODS
from core.kb.graph_io import case_from_dict
from experiments.lab02.eval.artifacts import execution_manifest, write_csv, write_json, seal_manifest


DISCLAIMER = "教学数据全部虚构；控制/受益人/风险均为模型内候选或复核线索，不是法律、监管或投资结论。"


def _case(name):
    return load_from_equity_kg() if name == "base" else demo_case()


def _hybrid(model_path: Path | None, seed: str | None = None, variant: str = "selected"):
    """加载已冻结的模型；v2只增加消融记录，预测参数与v1兼容。"""
    if model_path is None:
        return HybridModel(), "手工先验/CPT与固定融合权重（尚未从实验文件加载）"
    data = json.loads(model_path.read_text(encoding="utf-8-sig"))
    if (not isinstance(data, dict) or type(data.get("version")) is not int
            or data["version"] not in (1, 2)
            or not isinstance(data.get("models_by_seed"), dict) or not data["models_by_seed"]
            or any(not isinstance(bundle, dict) for bundle in data["models_by_seed"].values())):
        raise ValueError("模型文件应包含 version=1或2 与非空 models_by_seed（指标实验或端到端实验模型）")
    models = data["models_by_seed"]
    selected = str(seed) if seed is not None else next(iter(models))
    if selected not in models:
        raise ValueError(f"模型文件不含 seed={selected}")
    key = {"fixed": "fixed_model", "adaptive": "adaptive_model", "selected": "model"}[variant]
    if key not in models[selected]:
        raise ValueError("该模型文件没有保存固定权重版本，请使用selected或端到端实验模型文件")
    return HybridModel.from_dict(models[selected][key]), f"已加载 {model_path.resolve()}，seed={selected}，variant={variant}"


def run_demo(output: Path, *, case_name="extended", model_path=None, model_seed=None, model_variant="selected"):
    case = _case(case_name)
    context = prepare_case(case)
    rules = context.engine.rules
    facts = context.facts
    forward = context.engine
    backward = BackwardEngine(rules, facts.to_tuples())
    goals = [("Controls", "P1", "C3"), ("Alert", "C3", "关联交易待核查"),
             ("Alert", "C6", "循环持股"), ("Alert", "C7", "担保圈"), ("Controls", "P3", "C3")]
    queries = []
    for goal in goals:
        proof = backward.solve(goal)
        expected = forward.holds(goal)
        if expected != (proof is not None):
            raise AssertionError(f"正反向结论不一致: {goal}")
        queries.append({"goal": goal, "proved": proof is not None, "equivalent": True,
                        "stats": backward.stats(),
                        "proof": proof.lines({r.name: r.desc for r in rules}) if proof else []})
    clauses, goal = equity_resolution_example(case)
    resolution = ResolutionEngine().prove(clauses, goal)
    if not resolution.proved:
        raise InferenceLimitError(f"归结演示未完成证明: {resolution.status}, {resolution.reason}")
    model, model_source = _hybrid(model_path, model_seed, model_variant)
    observations = list(context.observations.values())
    scores, table = [], []
    for observation in observations:
        cid = observation.company_id
        established = context.established_for(cid)
        detail = assess_company(context, cid, model, method="hybrid_locked", include_explanation=True)
        scores.append({"observation": observation.as_dict(), "name": case.name(cid), **detail})
        table.append({"company_id": cid, "company_name": case.name(cid),
                      "bayes": detail["components"]["bayes"], "fuzzy": detail["components"]["fuzzy"],
                      "rule_score": detail["components"]["rule"], "hybrid": detail["score"],
                      "review_priority": detail["review_priority"],
                      "review_required": detail["review_required"], "risk_class": detail["risk_class"],
                      "lock_changed_action": detail["lock_changed_action"],
                      "alerts": ";".join(f[2] for f in established if f[0] == "Alert")})
    dest = output / "demo"
    dest.mkdir(parents=True, exist_ok=True)
    write_csv(dest / "company_risks.csv", table)
    write_json(dest / "summary.json", {"case": case.label, "source": case.source, "size": case.summary(),
               "disclaimer": DISCLAIMER, "model_source": model_source, "forward": forward.stats(),
               "backward_checks": queries, "resolution": {"status": resolution.status,
               "stats": resolution.stats, "proof": resolution.lines()}, "companies": scores})
    write_json(dest / "traces.json", {"input_facts": [asdict(f) for f in facts],
               "derived_facts": [{"fact": f, "rule": rule, "premises": premises}
                                 for f, (rule, premises) in forward.why.items()],
               "steps": [asdict(s) for s in forward.steps], "rounds": forward.rounds})
    (dest / "rules.txt").write_text(describe_rules(rules) + "\n", encoding="utf-8")
    lines = ["企业股权风险推理 · 第二次课堂汇报", "=" * 64, DISCLAIMER, "",
             case.label, case.source, str(case.summary()), f"规则数: {len(rules)}", "",
             "[1] 正向闭包", json.dumps(forward.stats(), ensure_ascii=False), forward.trace_text(), "",
             "[2] 正反向关键查询一致性"]
    for q in queries:
        lines.append(f"{q['goal']} → {'可证' if q['proved'] else '无法证明（不等于现实为假）'}")
    lines += ["", "P1 控制 C3 的反向证明：", *queries[0]["proof"], "",
              "[3] 独立归结反演（数值守卫已转为显式算术事实）", *resolution.lines(), "",
              "[4] 同一企业的 Bayes / Fuzzy / Hybrid", model_source,
              "ID    Bayes  Fuzzy  Hybrid  复核优先级  结构预警"]
    for row in table:
        lines.append(f"{row['company_id']:<5s} {row['bayes']:.3f}  {row['fuzzy']:.3f}  {row['hybrid']:.3f}  "
                     f"{row['review_priority']}  {row['alerts'] or '无已知预警'}")
    lines += ["", "高复核优先级可以由已证结构预警触发，不与数值评分混为一谈。",
              "无真实风险标签，不在这29个实体上伪报分类准确率。",
              f"完整事实、全部推理步骤和来源: {dest / 'traces.json'}", ""]
    text = "\n".join(lines)
    (dest / "demo.txt").write_text(text, encoding="utf-8")
    seal_manifest(dest, execution_manifest(ROOT))
    print(text)
    return dest


def run_check(output: Path) -> bool:
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests" / "lab02"), top_level_dir=str(ROOT))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    text = stream.getvalue()
    print(text)
    dest = output / "qa"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "test_results.txt").write_text(text, encoding="utf-8")
    write_json(dest / "test_summary.json", {"tests_run": result.testsRun,
               "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
               "successful": result.wasSuccessful(), **execution_manifest(ROOT)})
    return result.wasSuccessful()


def run_query(args):
    case = _case(args.case)
    facts = build_facts(case).to_tuples()
    rules = build_rules()
    def value(arg):
        if re.fullmatch(r"[-+]?\d+", arg):
            return int(arg)
        if re.fullmatch(r"[-+]?\d+\.\d+", arg):
            return float(arg)
        return arg
    goal = (args.predicate, *(value(a) for a in args.arguments))
    if args.engine == "forward":
        engine = ForwardEngine(rules, facts)
        engine.run()
        answers = engine.query(*goal)
    else:
        engine = BackwardEngine(rules, facts)
        answers = engine.ask(*goal)
    proofs = [{"fact": f, "proof": engine.explain(f)} for f in answers]
    write_json(args.output_dir / "query.json", {"query": goal, "engine": args.engine,
               "answers": proofs, "stats": engine.stats(), "disclaimer": DISCLAIMER})
    print(DISCLAIMER)
    print(f"找到 {len(answers)} 个答案；完整证明写入 {args.output_dir / 'query.json'}")
    for proof in proofs[:args.limit]:
        print("\n".join(proof["proof"]))
    if not answers:
        print("知识库无法证明该目标；这不是现实中的否定结论。")


def run_predict(args):
    raw = json.loads(args.input.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("输入文件应为非空的观测对象数组，参见 data/example_observations.json")
    model, source = _hybrid(args.model, args.model_seed, args.model_variant)
    rows = []
    for entry in raw:
        observation = RiskObservation(entry["company_id"], entry["values"],
                                      entry.get("source", "用户提供的教学输入"), entry.get("quality", 1.0))
        rows.append({"company_id": observation.company_id, **model.explain(observation)})
    write_json(args.output_dir / "predictions.json", {"model_source": source, "disclaimer": DISCLAIMER, "results": rows})
    print(DISCLAIMER)
    print(source)
    for row in rows[:20]:
        print(f"{row['company_id']}: score={row['score']:.4f} priority={row['review_priority']} coverage={row['coverage']:.0%}")
    print(f"全部 {len(rows)} 条结果已保存: {args.output_dir / 'predictions.json'}")


def run_analyze(args):
    data = json.loads(args.input.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("图输入必须是JSON对象")
    if "case" in data:
        if set(data) - {"case", "target_id", "completeness"}:
            raise ValueError("图请求只允许 case/target_id/completeness，不接收风险标签或干净图真值")
        case = case_from_dict(data["case"])
        completeness = data.get("completeness")
        target = args.target or data.get("target_id")
    else:
        case = case_from_dict(data)
        completeness, target = None, args.target
    context = prepare_case(case, completeness=completeness)
    model, source = _hybrid(args.model, args.model_seed, args.model_variant)
    targets = [target] if target else list(context.observations)
    if not targets:
        raise ValueError("输入图中没有可分析的企业")
    decisions = [assess_company(context, cid, model, method=args.method,
                               threshold=args.threshold, include_explanation=True) for cid in targets]
    write_json(args.output_dir / "graph_analysis.json", {"model_source": source,
        "disclaimer": DISCLAIMER, "graph": context.as_dict(), "decisions": decisions})
    print(DISCLAIMER)
    for result in decisions:
        print(f"{result['company_id']}: score={result['score']:.4f}, "
              f"review={result['review_required']}, priority={result['review_priority']}, "
              f"locked_change={result['lock_changed_action']}")
    print(f"完整输入事实、推理步骤和行动解释: {args.output_dir / 'graph_analysis.json'}")


def _plots(output, *, required=False):
    try:
        from experiments.lab02.eval.plots import make_figures
        files = make_figures(output / "experiments", output / "benchmarks", output / "figures")
        from experiments.lab02.eval.e2e_plots import make_e2e_figures
        files += make_e2e_figures(output / "end_to_end", output / "end_to_end" / "figures")
        if (output / "governance" / "equity_summary.csv").is_file():
            from experiments.lab02.eval.governance_experiments import make_governance_figure
            files.append(make_governance_figure(output / "governance", output / "governance" / "figures"))
    except ImportError:
        print("未安装可选绘图依赖。运行 python -m pip install -r requirements-plots.txt 后重试 plots。")
        if required:
            raise
        return []
    for p in files:
        print(f"[图] {p}")
    return files


def parser():
    p = argparse.ArgumentParser(description="企业股权风险推理：第二次课堂汇报代码与复现实验")
    sub = p.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "lab02", help="生成结果目录，默认本项目的 output")
    for name, help_ in [("web", "启动交互式网页工作台"), ("demo", "同源案例全流程演示"), ("query", "带完整证明的目标查询"),
                        ("predict", "读取自定义风险指标 JSON"), ("analyze", "从股权图执行完整推理及复核决策"),
                        ("e2e", "完整股权图链路与锁定/不锁定配对实验"), ("check", "运行全部自动化测试"),
                        ("experiments", "训练/验证/测试、噪声、敏感性、消融、D-S冲突"),
                        ("governance", "循环权益扩容、证据失效撤回和权利冲突实测"),
                        ("benchmark", "确定性推理效率对照"), ("plots", "只从已有CSV重新绘图"),
                        ("verify", "验收代码/结果指纹、数据划分、指标重算与模型回放"),
                        ("all", "测试 + 演示 + 指标实验 + 端到端图实验 + 基准 + 图表 + 验收")]:
        sp = sub.add_parser(name, parents=[common], help=help_)
        if name == "verify":
            sp.add_argument("--full-replay", action="store_true", help="重算全部保存预测；默认每种条件首条")
        if name == "web":
            sp.add_argument("--port", type=int, default=8765)
            sp.add_argument("--open", action="store_true", dest="open_browser", help="服务就绪后打开浏览器")
        if name in ("demo", "query"):
            sp.add_argument("--case", choices=["base", "extended"], default="extended")
        if name in ("demo", "predict", "analyze"):
            sp.add_argument("--model", type=Path, help="显式加载 experiments/models.json；默认使用手工参数")
            sp.add_argument("--model-seed", help="指定模型文件中的种子")
            sp.add_argument("--model-variant", choices=["selected", "fixed", "adaptive"], default="selected",
                            help="selected为验证选择模型；fixed为端到端实验保存的固定权重模型")
        if name == "query":
            sp.add_argument("predicate")
            sp.add_argument("arguments", nargs="*")
            sp.add_argument("--engine", choices=["forward", "backward"], default="backward")
            sp.add_argument("--limit", type=int, default=20, help="仅限制终端展示，不裁剪保存的完整答案")
        if name in ("predict", "analyze"):
            sp.add_argument("--input", type=Path, required=True)
        if name == "analyze":
            sp.add_argument("--target", help="目标企业ID；省略则使用请求中的target_id或分析全部企业")
            sp.add_argument("--method", choices=METHODS, default="hybrid_locked")
            sp.add_argument("--threshold", type=float, default=.5)
        if name in ("experiments", "benchmark", "e2e", "governance", "all"):
            sp.add_argument("--quick", action="store_true", help="小规模冒烟，不作为正式实验")
            sp.add_argument("--no-plots", action="store_true")
    return p


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    args = parser().parse_args(argv or ["demo"])
    args.output_dir = args.output_dir.resolve()
    try:
        if args.command == "web":
            from backend.serve import serve
            serve(args.port, args.open_browser)
        elif args.command == "demo":
            run_demo(args.output_dir, case_name=args.case, model_path=args.model, model_seed=args.model_seed, model_variant=args.model_variant)
        elif args.command == "query":
            if args.limit < 1:
                raise ValueError("limit 必须为正数")
            run_query(args)
        elif args.command == "predict":
            run_predict(args)
        elif args.command == "analyze":
            run_analyze(args)
        elif args.command == "check":
            return 0 if run_check(args.output_dir) else 1
        elif args.command == "verify":
            from experiments.lab02.eval.verification import verify_results
            result = verify_results(args.output_dir, ROOT, full_replay=args.full_replay)
            write_json(args.output_dir / "qa" / "programming_acceptance.json", result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "plots":
            _plots(args.output_dir, required=True)
        else:
            from experiments.lab02.eval.experiments import ExperimentConfig, run_experiments
            from experiments.lab02.eval.benchmark import run_benchmarks
            from experiments.lab02.eval.end_to_end import EndToEndConfig, run_end_to_end
            from experiments.lab02.eval.governance_experiments import run_governance_experiments
            if args.command == "all":
                if not run_check(args.output_dir):
                    return 1
                run_demo(args.output_dir)
            if args.command in ("experiments", "all"):
                run_experiments(args.output_dir / "experiments", ExperimentConfig.smoke() if args.quick else ExperimentConfig())
            if args.command in ("e2e", "all"):
                run_end_to_end(args.output_dir / "end_to_end", EndToEndConfig.smoke() if args.quick else EndToEndConfig())
            if args.command in ("governance", "all"):
                run_governance_experiments(args.output_dir / "governance",
                    sizes=(4, 8) if args.quick else (8, 32, 64, 128), repeats=1 if args.quick else 3)
            if args.command in ("benchmark", "all"):
                run_benchmarks(args.output_dir / "benchmarks", groups=(1, 2) if args.quick else (1, 4, 8, 16),
                               repeats=1 if args.quick else 3)
            if not args.no_plots:
                _plots(args.output_dir)
            if args.command == "all":
                from experiments.lab02.eval.verification import verify_results
                result = verify_results(args.output_dir, ROOT)
                write_json(args.output_dir / "qa" / "programming_acceptance.json", result)
                print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, KeyError, OSError, InferenceLimitError, ImportError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
