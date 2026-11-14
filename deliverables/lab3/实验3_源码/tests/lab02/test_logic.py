import itertools
import unittest
from unittest.mock import patch
from core.kb.fol import (And, Atom, Clause, Const, Exists, Forall, Func, Iff, Implies, Literal,
                    Not, Num, Or, Var, atom, standardize_apart, substitute, substitute_term, to_clauses)
from core.kb.unify import (factor, factors, match_atom, resolve, resolve_term, subsumes, unify_atoms,
                      unify_terms)
from core.kb.domain import load_from_equity_kg
from experiments.lab02.deterministic.resolution import ResolutionEngine, alpha_key, equity_resolution_example
from experiments.lab02.optimize.search import relevant_rules
from core.kb.rules import build_rules
from core.kb.facts import build_facts
from experiments.lab02.deterministic.forward import ForwardEngine


def C(*literals):
    return Clause.make(literals)


class LogicTests(unittest.TestCase):
    def test_occurs_and_chain(self):
        x, y = Var("x"), Var("y")
        self.assertIsNone(unify_terms(x, Func("f", (x,))))
        s = unify_atoms(atom("P", x, y), atom("P", y, "a"))
        self.assertEqual(resolve_term(x, s), Const("a"))
        self.assertEqual(resolve_term(y, s), Const("a"))
        self.assertIsNone(unify_terms(Num(1), Const("1")))

    def test_cnf_truth_table(self):
        p, q, r = atom("P"), atom("Q"), atom("R")
        formulas = [Iff(p, q), Not(Or((p, q))), Or((And((p, q)), r)),
                    Implies(And((p, Implies(p, q))), q), Not(Not(p)),
                    Or(()), And(()), Implies(p, And(()))]
        def value(f, assignment):
            if isinstance(f, Atom):
                return assignment[f.pred]
            if isinstance(f, Not):
                return not value(f.sub, assignment)
            if isinstance(f, And):
                return all(value(x, assignment) for x in f.subs)
            if isinstance(f, Or):
                return any(value(x, assignment) for x in f.subs)
            if isinstance(f, Implies):
                return not value(f.lhs, assignment) or value(f.rhs, assignment)
            return value(f.lhs, assignment) == value(f.rhs, assignment)
        for f in formulas:
            clauses = to_clauses(f)
            for bits in itertools.product([False, True], repeat=3):
                assignment = dict(zip("PQR", bits))
                cnf = all(any(assignment[l.atom.pred] != l.negated for l in c) for c in clauses)
                self.assertEqual(value(f, assignment), cnf, (f, bits))

    def test_quantifier_shadowing(self):
        x = Var("x")
        f = Forall((x,), And((Exists((x,), atom("P", x)), atom("Q", x))))
        g = standardize_apart(f)
        inner, q = g.sub.subs
        self.assertEqual(q.args[0], g.vars[0])
        self.assertNotEqual(inner.vars[0], g.vars[0])

    def test_capture_avoiding_substitution(self):
        x, y = Var("x"), Var("y")
        g = substitute(Forall((x,), atom("P", y, x)), {"y": x})
        self.assertEqual(g.sub.args[0], x)
        self.assertNotEqual(g.vars[0], x)
        self.assertEqual(g.sub.args[1], g.vars[0])

    def test_free_variable_skolem_dependency(self):
        x, y = Var("x"), Var("y")
        c = to_clauses(Exists((y,), atom("R", x, y)))[0]
        a = c.literals[0].atom.args
        self.assertIsInstance(a[1], Func)
        self.assertEqual(a[1].args, (a[0],))

    def test_skolem_constant_not_existing_symbol(self):
        from core.kb import fol
        old = fol._skolem_counter
        try:
            fol._skolem_counter = itertools.count(1)
            c = to_clauses(Exists((Var("x"),), atom("R", "sk1", "?x")))[0]
            self.assertNotEqual(*c.literals[0].atom.args)
        finally:
            fol._skolem_counter = old

    def test_separately_clausified_formulas_do_not_capture_user_constant(self):
        for user_first in (False, True):
            with self.subTest(user_first=user_first), patch("core.kb.fol._skolem_counter", itertools.count(1)):
                negative = to_clauses(Not(atom("P", "sk1"))) if user_first else []
                positive = to_clauses(Exists((Var("x"),), atom("P", "?x")))
                if not user_first:
                    negative = to_clauses(Not(atom("P", "sk1")))
                witness = positive[0].literals[0].atom.args[0]
                self.assertNotEqual(witness, Const("sk1"))
                self.assertIsNone(unify_terms(witness, Const("sk1")))
                self.assertNotEqual(alpha_key(positive[0]), alpha_key(C(Literal(atom("P", "sk1")))))
                for optimized in (False, True):
                    engine = ResolutionEngine(optimized=optimized)
                    self.assertEqual(engine.prove(positive + negative, atom("Unrelated")).status, "saturated")
                    self.assertTrue(engine.prove(positive + to_clauses(Not(atom("P", witness))), atom("Unrelated")).proved)

    def test_skolem_identity_does_not_depend_on_display_number(self):
        with patch("core.kb.fol._skolem_counter", itertools.count(1)):
            positive = to_clauses(Exists((Var("x"),), atom("P", "?x")))
        with patch("core.kb.fol._skolem_counter", itertools.count(1)):
            negative = to_clauses(Exists((Var("x"),), Not(atom("P", "?x"))))
        self.assertEqual(positive[0].literals[0].atom.args[0].name, negative[0].literals[0].atom.args[0].name)
        self.assertEqual(ResolutionEngine().prove(positive + negative, atom("Unrelated")).status, "saturated")

    def test_skolem_function_identity_survives_substitution_and_resolution(self):
        with patch("core.kb.fol._skolem_counter", itertools.count(1)):
            clauses = to_clauses(Forall((Var("x"),), Exists((Var("y"),), atom("P", "?x", "?y"))))
        x, witness = clauses[0].literals[0].atom.args
        bound = substitute_term(witness, {x.name: Const("a")})
        self.assertEqual(bound, resolve_term(witness, {x.name: Const("a")}))
        ordinary = Func(witness.name, (Const("a"),))
        self.assertIsNone(unify_terms(bound, ordinary))
        self.assertIsNone(match_atom(atom("T", witness), atom("T", ordinary)))
        self.assertFalse(subsumes(C(Literal(atom("T", witness))), C(Literal(atom("T", ordinary)))))
        self.assertNotEqual(alpha_key(C(Literal(atom("T", bound)))), alpha_key(C(Literal(atom("T", ordinary)))))
        self.assertEqual(len(C(Literal(atom("T", bound)), Literal(atom("T", ordinary)))), 2)
        for optimized in (False, True):
            engine = ResolutionEngine(optimized=optimized)
            self.assertTrue(engine.prove(clauses, atom("P", "a", bound)).proved)
            negative = to_clauses(Not(atom("P", "a", ordinary)))
            self.assertEqual(engine.prove(clauses + negative, atom("Unrelated")).status, "saturated")

    def test_big_integer_terms_remain_distinct_through_resolution(self):
        for value in (2 ** 53 + 1, -(2 ** 53 + 1), 10 ** 400 + 1):
            with self.subTest(value=value):
                given = atom("Amount", value)
                neighbor = atom("Amount", value - 1)
                self.assertEqual(given.args[0].value, value)
                self.assertIsInstance(given.args[0].value, int)
                self.assertEqual(str(given.args[0]), str(value))
                self.assertIsNone(unify_atoms(given, neighbor))
                clauses = [C(Literal(given))]
                for optimized in (False, True):
                    engine = ResolutionEngine(optimized=optimized)
                    self.assertTrue(engine.prove(clauses, given).proved)
                    self.assertEqual(engine.prove(clauses, neighbor).status, "saturated")
        self.assertEqual(Num(1), Num(1.0))
        self.assertEqual(len({Num(1), Num(1.0)}), 1)
        self.assertIsNotNone(unify_terms(Num(1), Num(1.0)))

    def test_subsumption_preserves_all_bindings(self):
        general = C(*(Literal(atom(p, "?x")) for p in "PQR"))
        specific = C(Literal(atom("P", "a")), Literal(atom("Q", "a")), Literal(atom("R", "b")))
        self.assertFalse(subsumes(general, specific))
        self.assertTrue(subsumes(C(), specific))
        self.assertFalse(subsumes(specific, C()))
        self.assertTrue(subsumes(C(Literal(atom("P", "?x")), Literal(atom("P", "?y"))),
                                 C(Literal(atom("P", "a")))))

    def test_matching_target_variables_are_rigid(self):
        self.assertIsNone(match_atom(atom("P", "?x", "?x"), atom("P", "?u", "?v")))
        p = atom("P", Func("f", (Var("x"),)), "?x")
        self.assertIsNotNone(match_atom(p, atom("P", Func("f", (Const("a"),)), "a")))
        self.assertIsNone(match_atom(p, atom("P", Func("f", (Var("u"),)), "?v")))

    def test_tautology_not_just_unifiability(self):
        c = C(Literal(atom("P", "?x")), Literal(atom("P", "a"), True))
        self.assertFalse(c.is_tautology)
        c = C(Literal(atom("P", "?x")), Literal(atom("P", "?x"), True))
        self.assertTrue(c.is_tautology)

    def test_resolution_factors_needed(self):
        positive = C(Literal(atom("P", "?x")), Literal(atom("P", "?y")))
        negative = C(Literal(atom("P", "?u"), True), Literal(atom("P", "?v"), True))
        for optimized in (False, True):
            result = ResolutionEngine(optimized=optimized).prove([positive, negative], atom("Any"))
            self.assertTrue(result.proved, result.stats)
            self.assertTrue(result.proof_steps()[-1].clause.is_empty)

    def test_equity_resolution(self):
        clauses, goal = equity_resolution_example(load_from_equity_kg())
        for optimized in (False, True):
            result = ResolutionEngine(optimized=optimized).prove(clauses, goal)
            self.assertTrue(result.proved, (result.status, result.stats))
            for step in result.proof_steps():
                if step.kind == "resolution":
                    a, b = (result.steps[p].clause for p in step.parents)
                    candidates = resolve(a, b, do_factor=False)
                elif step.kind == "factor":
                    candidates = factors(result.steps[step.parents[0]].clause)
                else:
                    continue
                self.assertTrue(any(subsumes(c, step.clause) and subsumes(step.clause, c) for c in candidates),
                                f"Invalid proof step {step.cid}")
        result = ResolutionEngine(max_clauses=1).prove(clauses, goal)
        self.assertEqual(result.status, "limit")

    def test_resolution_satisfiable_and_slicing(self):
        self.assertEqual(ResolutionEngine().prove([C(Literal(atom("P", "a")))], atom("Q")).status,
                         "saturated")
        rules = build_rules()
        facts = build_facts(load_from_equity_kg()).to_tuples()
        full, sliced = ForwardEngine(rules, facts), ForwardEngine(relevant_rules(rules, {"Controls"}), facts)
        full.run()
        sliced.run()
        self.assertLess(len(sliced.rules), len(rules))
        self.assertEqual(full.query("Controls", "?x", "?y"), sliced.query("Controls", "?x", "?y"))

    def test_nonfinite_term_rejected(self):
        for x in [float("nan"), float("inf"), True]:
            with self.assertRaises(ValueError):
                Num(x)


if __name__ == "__main__":
    unittest.main()
