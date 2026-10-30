"""Independent audit probes. Run from any cwd; never rewrites production code/results."""
from __future__ import annotations

import csv
import itertools
import json
import math
import random
import statistics
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from core.kb.domain import Entity, EquityCase, Holding
from core.kb.equity import economic_equity
from core.kb.features import FEATURES, RiskObservation
from core.kb.fol import Clause, Literal, atom
from core.kb.rules_dsl import Pattern, Rule, RuleSet, neq
from experiments.lab02.deterministic.backward import BackwardEngine
from experiments.lab02.deterministic.forward import ForwardEngine
from experiments.lab02.deterministic.resolution import ResolutionEngine
from experiments.lab02.eval.metrics import evaluate
from experiments.lab02.hybrid.model import HybridModel
from experiments.lab02.uncertain.bayes import BinaryBayesNet, BinaryNode, ImpossibleEvidence, RiskBayes
from experiments.lab02.uncertain.dempster_shafer import Mass, dempster, yager
from experiments.lab02.uncertain.fuzzy import MamdaniFuzzy


def close(a, b, tolerance=1e-10):
    assert math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance), (a, b)


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def datalog_ground_oracle():
    """Enumerate ground substitutions; use no production matching/join/unification."""
    domain = (0, 1, 2)
    arities = {"A": 1, "B": 1, "C": 1, "E": 2, "R": 2}
    comparisons = 0
    queries = 0
    for seed in range(120):
        rng = random.Random(8000 + seed)
        initial = {(pred, *args) for pred, arity in arities.items()
                   for args in itertools.product(domain, repeat=arity) if rng.random() < .20}
        specs, rules = [], []
        for index in range(8):
            body = []
            for _ in range(rng.randint(0, 3)):
                pred = rng.choice(tuple(arities))
                args = tuple(rng.choice(("?x", "?y", "?z", 0, 1, 2)) for _ in range(arities[pred]))
                body.append((pred, *args))
            variables = sorted({a for p in body for a in p[1:] if isinstance(a, str)})
            pred = rng.choice(tuple(arities))
            head = (pred, *(rng.choice([*variables, *domain]) for _ in range(arities[pred])))
            guard = tuple(rng.sample(variables, 2)) if len(variables) >= 2 and rng.random() < .3 else ()
            specs.append((body, head, variables, guard))
            rules.append(Rule(f"r{index}", "audit", tuple(Pattern.of(*p) for p in body),
                              Pattern.of(*head), neq(*guard) if guard else None))
        expected = set(initial)
        instantiate = lambda p, b: (p[0], *(b.get(a, a) for a in p[1:]))
        while True:
            added = set()
            for body, head, variables, guard in specs:
                for values in itertools.product(domain, repeat=len(variables)):
                    binding = dict(zip(variables, values))
                    if guard and binding[guard[0]] == binding[guard[1]]:
                        continue
                    if all(instantiate(p, binding) in expected for p in body):
                        added.add(instantiate(head, binding))
            if added <= expected:
                break
            expected |= added
        rs = RuleSet(rules)
        for indexed, semi in ((False, False), (True, False), (True, True)):
            engine = ForwardEngine(rs, sorted(initial), use_index=indexed)
            engine.run(semi_naive=semi)
            assert engine.wm.facts == expected, (seed, indexed, semi, engine.wm.facts ^ expected)
            comparisons += 1
        backward = BackwardEngine(rs, sorted(initial))
        for pred, arity in arities.items():
            patterns = [tuple(f"?q{i}" for i in range(arity))]
            patterns += list(itertools.product(domain, repeat=arity))
            if arity == 2:
                patterns.append(("?x", "?x"))
            for args in patterns:
                def matches(fact):
                    if fact[0] != pred:
                        return False
                    binding = {}
                    for pattern, value in zip(args, fact[1:]):
                        if isinstance(pattern, str):
                            if pattern in binding and binding[pattern] != value:
                                return False
                            binding[pattern] = value
                        elif pattern != value:
                            return False
                    return True
                target = {f for f in expected if matches(f)}
                actual = set(backward.ask(pred, *args))
                assert actual == target, (seed, pred, args, actual ^ target)
                queries += 1
    return {"programs": 120, "forward_closure_comparisons": comparisons, "backward_queries": queries}


def resolution_truth_table():
    comparisons = 0
    names = ("P", "Q", "R")
    for seed in range(180):
        rng = random.Random(9100 + seed)
        specs = [tuple((rng.choice(names), rng.choice((False, True)))
                       for _ in range(rng.randrange(1, 4))) for _ in range(rng.randrange(1, 9))]
        clauses = [Clause.make(Literal(atom(pred), neg) for pred, neg in items) for items in specs]
        satisfying = [dict(zip(names, bits)) for bits in itertools.product((False, True), repeat=3)]
        satisfying = [a for a in satisfying if all(any(a[p] != neg for p, neg in c) for c in specs)]
        for goal in names:
            expected = all(a[goal] for a in satisfying)
            for optimized in (False, True):
                result = ResolutionEngine(optimized=optimized, max_seconds=5).prove(clauses, atom(goal))
                assert result.status != "limit", result.stats
                assert result.proved == expected, (seed, goal, optimized, expected, result.status)
                comparisons += 1
    return {"knowledge_bases": 180, "entailment_comparisons": comparisons}


def bayes_joint_oracle():
    checks = 0
    for seed in range(12):
        rng = random.Random(seed + 2000)
        nodes = []
        for i in range(4):
            parents = tuple(f"V{j}" for j in range(i) if rng.random() < .6)
            table = {bits: rng.choice((0, .1, .3, .65, .9, 1))
                     for bits in itertools.product((0, 1), repeat=len(parents))}
            nodes.append(BinaryNode(f"V{i}", parents, table))
        net = BinaryBayesNet(list(reversed(nodes)))
        joint = []
        for bits in itertools.product((0, 1), repeat=4):
            state = dict(zip([n.name for n in nodes], bits))
            probability = 1.0
            for node in nodes:
                p = node.p_true[tuple(state[k] for k in node.parents)]
                probability *= p if state[node.name] else 1 - p
            joint.append((state, probability))
        for states in itertools.product((None, 0, 1), repeat=4):
            evidence = {f"V{i}": v for i, v in enumerate(states) if v is not None}
            compatible = [(a, p) for a, p in joint if all(a[k] == v for k, v in evidence.items())]
            total = sum(p for a, p in compatible)
            for query in net.names:
                if total == 0:
                    try:
                        net.query(query, evidence)
                    except ImpossibleEvidence:
                        pass
                    else:
                        raise AssertionError("Impossible evidence was accepted")
                else:
                    close(net.query(query, evidence), sum(p for a, p in compatible if a[query]) / total)
                checks += 1
    return {"random_dags": 12, "posterior_or_impossible_evidence_checks": checks}


def exact_fuzzy(values, high_start, medium_weight):
    """Exact piecewise-linear integration, independent from sampled-grid production code."""
    observed = {f: value for f, value in zip(FEATURES, values) if value is not None}
    if not observed:
        return .5
    importance = dict(zip(FEATURES, (.65, .90, .85, 1.0)))
    high_values = {f: max(0., (v - high_start) / (1 - high_start)) for f, v in observed.items()}
    hi = max(importance[f] * v for f, v in high_values.items())
    if len(observed) > 1:
        hi = max(hi, sorted(high_values.values(), reverse=True)[1])
    mid = medium_weight * max(1 - abs(2 * v - 1) for v in observed.values())
    low = min(max(0., 1 - v / (1 - high_start)) for v in observed.values())
    def membership(x, a, b, c):
        if x == b:
            return 1.
        if x < b:
            return max(0., (x - a) / (b - a)) if b > a else 0.
        return max(0., (c - x) / (c - b)) if c > b else 0.
    curves = [(0., 0., .45, low), (.2, .5, .8, mid), (.55, 1., 1., hi)]
    def heights(x):
        return [min(h, membership(x, a, b, c)) for a, b, c, h in curves]
    knots = {0., 1.}
    for a, b, c, h in curves:
        knots.update((a, b, c, a + h * (b - a), c - h * (c - b)))
    base = sorted(knots)
    for left, right in zip(base, base[1:]):
        lhs, rhs = heights(left), heights(right)
        for i, j in itertools.combinations(range(3), 2):
            dl, dr = lhs[i] - lhs[j], rhs[i] - rhs[j]
            if dl * dr < 0:
                knots.add(left + (right - left) * dl / (dl - dr))
    knots = sorted(knots)
    area = moment = 0.
    for left, right in zip(knots, knots[1:]):
        yl, yr = max(heights(left)), max(heights(right))
        width = right - left
        area += width * (yl + yr) / 2
        moment += width * ((2 * left + right) * yl + (left + 2 * right) * yr) / 6
    centroid = moment / area if area else .5
    coverage = len(observed) / 4
    return coverage * centroid + (1 - coverage) * .5


def fuzzy_integration_oracle():
    rng = random.Random(991)
    values = list(itertools.product((None, 0., .5, 1.), repeat=4))
    values += [tuple(rng.random() for _ in FEATURES) for _ in range(100)]
    max_error = 0.
    for start, medium in ((.45, 1.), (.15, .35), (.75, .65)):
        model = MamdaniFuzzy(high_start=start, medium_weight=medium)
        for row in values:
            expected = exact_fuzzy(row, start, medium)
            actual = model.predict(RiskObservation("audit", dict(zip(FEATURES, row))))
            error = abs(actual - expected)
            max_error = max(max_error, error)
            assert error < .002, (start, medium, row, expected, actual)
    return {"observations": len(values) * 3, "max_abs_error_vs_exact_integral": max_error,
            "production_grid_points": 101, "allowed_absolute_error": .002}


def ds_set_intersection_oracle():
    rng = random.Random(883)
    max_error = 0.
    frame = frozenset((0, 1))
    sets = (frozenset((1,)), frozenset((0,)), frame)
    for _ in range(600):
        weights = [[rng.random() for _ in range(3)] for _ in range(2)]
        normalized = [[x / sum(w) for x in w] for w in weights]
        masses = [Mass(*w) for w in normalized]
        combined = Counter()
        for (a, pa), (b, pb) in itertools.product(zip(sets, normalized[0]), zip(sets, normalized[1])):
            combined[a & b] += pa * pb
        conflict = combined[frozenset()]
        for robust in (False, True):
            actual = yager(*masses) if robust else dempster(*masses)
            expected = [combined[s] + (conflict if s == frame and robust else 0.) for s in sets]
            if not robust:
                expected = [x / (1 - conflict) for x in expected]
            for a, b in zip((actual.mass.review, actual.mass.normal, actual.mass.ignorance), expected):
                max_error = max(max_error, abs(a - b))
                close(a, b)
    return {"source_pairs": 600, "fusion_comparisons": 1200, "maximum_abs_error": max_error}


def metrics_pairwise_oracle():
    rng = random.Random(712)
    for _ in range(200):
        labels = [rng.randrange(2) for _ in range(25)]
        scores = [rng.choice((0., .2, .5, .7, 1.)) for _ in labels]
        threshold = rng.choice((.2, .5, .7))
        result = evaluate(labels, scores, threshold)
        confusion = Counter((y, int(s >= threshold)) for y, s in zip(labels, scores))
        tp, tn, fp, fn = [confusion[k] for k in ((1, 1), (0, 0), (0, 1), (1, 0))]
        assert (result["tp"], result["tn"], result["fp"], result["fn"]) == (tp, tn, fp, fn)
        close(result["accuracy"], (tp + tn) / len(labels))
        close(result["f1"], 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
        close(result["brier"], statistics.mean((y - s) ** 2 for y, s in zip(labels, scores)))
        positive = [s for y, s in zip(labels, scores) if y]
        negative = [s for y, s in zip(labels, scores) if not y]
        if positive and negative:
            auc = statistics.mean(float(a > b) + .5 * (a == b) for a, b in itertools.product(positive, negative))
            close(result["auc"], auc)
        else:
            assert result["auc"] is None
    return {"confusion_and_tie_aware_auc_cases": 200}


def equity_iteration_oracle():
    rng = random.Random(519)
    comparisons = 0
    maximum_error = 0.
    for index in range(70):
        n = rng.randrange(2, 7)
        people = ["P0", "P1"]
        companies = [f"C{i}" for i in range(n)]
        holdings = []
        for target in companies:
            for holder in [*people, *companies]:
                if rng.random() < .55:
                    holdings.append(Holding(holder, target, rng.uniform(.005, .08)))
        case = EquityCase(entities=[Entity(p, p, "Person") for p in people] +
                          [Entity(c, c, "Company", capital=1000.) for c in companies], holdings=holdings)
        direct = {c: [sum(h.ratio for h in holdings if h.target == c and h.holder == p) for p in people] +
                  [1 - sum(h.ratio for h in holdings if h.target == c)] for c in companies}
        values = {c: [0.] * 3 for c in companies}
        for iteration in range(500):
            updated = {c: [direct[c][i] + sum(h.ratio * values[h.holder][i] for h in holdings
                                            if h.target == c and h.holder in values)
                           for i in range(3)] for c in companies}
            change = max(abs(updated[c][i] - values[c][i]) for c in companies for i in range(3))
            values = updated
            if change < 1e-14:
                break
        else:
            raise AssertionError("Independent contraction iteration did not converge")
        for target in companies:
            actual = economic_equity(case, target)
            assert actual["status"] == "solved", actual
            by_person = {o["entity_id"]: o["share"] for o in actual["owners"]}
            observed = [by_person.get(p, 0.) for p in people] + [actual["unknown_share"]]
            for a, b in zip(observed, values[target]):
                maximum_error = max(maximum_error, abs(a - b))
                close(a, b)
            close(actual["total_share"], 1.)
            comparisons += 1
    return {"graphs": 70, "company_equity_checks": comparisons, "max_abs_error": maximum_error}


def training_records_audit():
    """Reconstruct Bayes CPT counts and independently re-evaluate the saved fuzzy grid."""
    from core.kb.graph_io import case_from_dict
    from experiments.lab02.hybrid.pipeline import prepare_case
    families = []
    indicator = ROOT / "outputs/lab02/experiments"
    graph = ROOT / "outputs/lab02/end_to_end"
    samples = csv_rows(indicator / "samples.csv")
    graph_samples = [json.loads(line) for line in (graph / "reference_graphs.jsonl").read_text(encoding="utf-8").splitlines()]
    for folder, source in ((indicator, samples), (graph, graph_samples)):
        bundles = json.loads((folder / "models.json").read_text(encoding="utf-8"))["models_by_seed"]
        for seed, bundle in bundles.items():
            rows = [r for r in source if str(r["seed"]) == seed and r["split"] == "train"]
            observations, labels = [], []
            for row in rows:
                if folder == indicator:
                    observation = RiskObservation(row["sample_id"], {f: float(row[f]) if row[f] else None for f in FEATURES})
                else:
                    observation = prepare_case(case_from_dict(row["case"])).observations[row["target_id"]]
                observations.append(observation)
                labels.append(int(row["label"]))
            bayes = bundle["model"]["bayes"]
            close(bayes["prior"], (sum(labels) + 1) / (len(labels) + 2))
            for feature in FEATURES:
                for label in (0, 1):
                    selected = [o.values[feature] for o, y in zip(observations, labels)
                                if y == label and o.values.get(feature) is not None]
                    expected = (sum(v >= bayes["threshold"] for v in selected) + 1) / (len(selected) + 2)
                    close(bayes["cpt"][feature][label], expected)
            fuzzy = bundle["fuzzy_model"]
            losses = []
            for start in (.15, .30, .45, .60, .75):
                for medium in (.35, .65, 1.0):
                    model = MamdaniFuzzy(high_start=start, medium_weight=medium)
                    loss = math.fsum((model.predict(o) - y) ** 2 for o, y in zip(observations, labels)) / len(labels)
                    losses.append((loss, start, medium))
            best = min(losses, key=lambda r: (r[0], abs(r[1] - .45) + abs(r[2] - 1)))
            assert (fuzzy["high_start"], fuzzy["medium_weight"]) == best[1:]
            close(fuzzy["training_brier"], best[0])
            families.append({"family": folder.name, "seed": seed, "training_rows": len(rows),
                             "bayes_cpt_recounted": True, "fuzzy_candidates_recomputed": len(losses)})
    return {"training_bundles": families}


def boundary_findings():
    """Record counterexamples separately from supported-domain oracle results."""
    from core.kb import fol
    from core.kb.fol import Exists, Not, Var
    results = []
    original = 2 ** 53 + 1
    converted = atom("Amount", original).args[0].value
    clauses = [Clause.make([Literal(atom("Amount", original))])]
    result = ResolutionEngine().prove(clauses, atom("Amount", original - 1))
    results.append({"id": "fol_large_integer_precision", "reproduced": converted != original and result.proved,
                    "input_integer": original, "converted_value": converted,
                    "incorrect_neighbor_goal_proved": result.proved,
                    "scope": "Generic FOL numeric inputs beyond binary64 exact integers; bundled coursework values are smaller."})
    old_counter = fol._skolem_counter
    try:
        fol._skolem_counter = itertools.count(1)
        clauses = fol.to_clauses(Exists((Var("x"),), atom("P", "?x")))
        clauses += fol.to_clauses(Not(atom("P", "sk1")))
        result = ResolutionEngine().prove(clauses, atom("Unrelated"))
        results.append({"id": "skolem_collision_across_formulas", "reproduced": result.proved,
                        "separate_inputs": ["exists x: P(x)", "not P(sk1)"],
                        "converted_clauses": [str(c) for c in clauses],
                        "incorrect_unrelated_goal_proved": result.proved,
                        "scope": "Multiple independently clausified formulas with a user constant matching a generated Skolem name; course rules contain no existential quantifiers."})
    finally:
        fol._skolem_counter = old_counter
    bundles = json.loads((ROOT / "outputs/lab02/experiments/models.json").read_text(encoding="utf-8"))["models_by_seed"]
    for seed, bundle in bundles.items():
        model = HybridModel.from_dict(bundle["model"])
        if model.weights == (1., 0., 0.):
            ablated = model.without("bayes")
            results.append({"id": "ablation_fallback_activates_disabled_component", "reproduced": ablated.weights[1] == 1.,
                            "seed": seed, "before": dict(zip(model.NAMES, model.weights)),
                            "after_removing_bayes": dict(zip(ablated.NAMES, ablated.weights)),
                            "scope": "Saved official ablation experiment; pure-fuzzy fallback differs from documented delete-and-renormalize semantics."})
    return results


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "seed_policy": "fixed audit seeds",
              "production_code_modified": False, "checks": [], "findings": []}
    for check in (datalog_ground_oracle, resolution_truth_table, bayes_joint_oracle,
                  fuzzy_integration_oracle, ds_set_intersection_oracle, metrics_pairwise_oracle,
                  equity_iteration_oracle, training_records_audit):
        started = time.perf_counter()
        try:
            details = check()
            item = {"name": check.__name__, "passed": True, "details": details}
        except Exception:
            item = {"name": check.__name__, "passed": False, "error": traceback.format_exc()}
        item["elapsed_seconds"] = time.perf_counter() - started
        report["checks"].append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    report["findings"] = boundary_findings()
    report["supported_domain_oracles_passed"] = all(r["passed"] for r in report["checks"])
    output = Path(__file__).with_name("independent-review.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"supported_domain_oracles_passed": report["supported_domain_oracles_passed"],
                      "findings": report["findings"], "report": str(output)}, ensure_ascii=False), flush=True)
    return 0 if report["supported_domain_oracles_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
