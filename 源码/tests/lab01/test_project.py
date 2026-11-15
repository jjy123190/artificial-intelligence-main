import unittest
from experiments.lab01.dataset import demo_dataset, synthetic_dataset
from experiments.lab01.kr.knowledge_graph import EquityKG
from experiments.lab01.kr.production import InferenceEngine, unify
from experiments.lab01.kr.frames import build_company_frames
from experiments.lab01.kr.relational import RelationalStore
from experiments.lab01.reasoning.rules import RULES
from experiments.lab01.reasoning.graph_reasoning import penetrate, upstream_paths, find_cycles

class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.data = demo_dataset()
        self.kg = EquityKG(self.data)
        self.engine = InferenceEngine(RULES, self.kg.to_predicates())
        self.engine.run()

    def test_control(self):
        self.assertIn(('ActualController','C3','P1'),self.engine.wm)
        self.assertIn(('ActualController','C4','P1'),self.engine.wm)

    def test_equity_is_not_voting_control(self):
        self.assertAlmostEqual(penetrate(self.kg,'C3')[0]['ratio'],.342)

    def test_alert_sets(self):
        self.assertEqual({f[1] for f in self.engine.query('Alert','?c','循环持股')},{'C6','C7','C8'})
        self.assertEqual({f[1] for f in self.engine.query('Alert','?c','担保圈')},{'C7','C10','C11'})
        self.assertEqual(self.engine.query('Alert','?c','关联交易待核查'),[('Alert','C3','关联交易待核查')])

    def test_cycle_algorithms_agree(self):
        self.assertEqual(set(find_cycles(self.kg,'HOLDS')[0]),{'C6','C7','C8'})
        self.assertEqual(set(find_cycles(self.kg,'GUARANTEES')[0]),{'C7','C10','C11'})

    def test_no_false_control(self):
        self.assertFalse(self.engine.query('ActualController','C12','?p'))

    def test_frame_override(self):
        _,fs=build_company_frames(self.data)
        self.assertEqual(fs['C3'].get('关联交易披露'),'必须公告披露')
        fs['C3'].set('风险等级','高')
        self.assertEqual(fs['C3'].get('风险等级'),'高')
        with self.assertRaises(ValueError): fs['C3'].set('风险等级','未知等级')

    def test_depth_cutoff_not_person(self):
        self.assertFalse(any(o['beneficial'] for o in penetrate(self.kg,'C3',max_depth=1)))

    def test_half_not_control(self):
        e=InferenceEngine(RULES,[('Holds','P','C',.5),('Person','P')]); e.run()
        self.assertFalse(e.query('Controls','?a','?b'))

    def test_unification_repeated_variable(self):
        self.assertIsNone(unify(('R','?x','?x'),('R','A','B'),{}))

    def test_fixed_point(self):
        self.assertEqual(self.engine.run(),0)

    def test_sql_graph_same_on_dag(self):
        d=synthetic_dataset(n_layers=4,width=10,n_persons=5)
        h=EquityKG(d).relation_graph('HOLDS'); sql=RelationalStore(d['holdings'])
        canon=lambda rows: sorted((p,round(r,12),n) for p,r,n in rows)
        a=canon(upstream_paths(h,'L4C1',4))
        self.assertEqual(a,canon(sql.upstream_recursive('L4C1',4)))
        self.assertEqual(a,canon(sql.upstream_by_joins('L4C1',4)))
        sql.conn.close()

if __name__=='__main__': unittest.main(verbosity=2)
