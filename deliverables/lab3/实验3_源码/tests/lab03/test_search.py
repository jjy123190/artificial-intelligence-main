"""Meaningful invariants: independent search, valid witnesses, budgets, and split leakage."""
from tests.lab03.support import requires_artifacts
import json,unittest,random,math
from pathlib import Path
from unittest.mock import patch
from experiments.lab03.domain import Graph,Node,Edge,Query,State,enumerate_paths,expert_legal
from experiments.lab03.search import search as engine_search, FIXED, GA_METHODS, ML_METHODS
from experiments.lab03.domain import path_key
from experiments.lab03.dataset import load_dataset,unpack,shared_graph
from experiments.lab03.learning import probability,fit_classifier
from experiments.lab03.genetic import optimize
from experiments.lab03.service import run_task,tasks
from backend.schemas import LabThreeTask


def search(graph, query, method='bfs', budget=24, **kwargs):
    """Explicit synthetic parameters for engine fixtures, never service fallbacks."""
    if method in GA_METHODS:
        kwargs.setdefault('weights', list(FIXED))
    if method in ML_METHODS:
        kwargs.setdefault('model', {'coef': [0.] * 6, 'intercept': 0., 'seed': kwargs.get('seed', 7)})
    return engine_search(graph, query, method, budget, **kwargs)


def fixture():
    nodes=[Node('p','人',kind='Person'),Node('a','甲'),Node('b','乙'),Node('c','目标',sector='制造',listed=True),Node('x','失效目标',sector='制造',listed=True)]
    edges=[Edge('e1','p','a','HOLDS'),Edge('e2','a','b','HOLDS'),Edge('e3','b','c','HOLDS'),Edge('bad','a','x','HOLDS',verified=False),Edge('cycle','b','a','HOLDS')]
    return Graph(nodes,edges),Query('q','p',('HOLDS','HOLDS','HOLDS'),'制造')


class SearchTests(unittest.TestCase):
    def test_bfs_and_admissible_astar_match_exhaustive_goals(self):
        for record in load_dataset()[:4]:
            graph,qs=unpack(record)
            for q in qs:
                expected={s.node for s in enumerate_paths(graph,q)}
                for method in ['bfs','astar_admissible']:
                    r=search(graph,q,method,256)
                    self.assertEqual(bool(expected),r['found'])
                    if r['found']:self.assertIn(r['answer'],expected);self.assertTrue(r['valid'])

    def test_all_engines_return_existing_valid_witness(self):
        g,q=fixture()
        for method in ['bfs','astar_admissible','astar_fixed','astar_ga','astar_ml','mcts','hybrid']:
            r=search(g,q,method,64)
            self.assertTrue(r['valid'],method);self.assertEqual(r['edge_ids'],['e1','e2','e3'])
            self.assertEqual(r['hops'],3)

    def test_budget_exhaustion_is_not_logical_false(self):
        g,q=fixture();r=search(g,q,'bfs',1)
        self.assertEqual(r['status'],'budget_exhausted');self.assertIsNone(r['answer']);self.assertEqual(r['expanded'],1)

    def test_exhausted_empty_graph(self):
        g,q=fixture();g=Graph(list(g.nodes.values()),[])
        for m in ['bfs','astar_ga','mcts']:
            r=search(g,q,m,8);self.assertEqual(r['status'],'exhausted');self.assertFalse(r['found'])

    def test_cycle_cannot_repeat_node(self):
        g,q=fixture();s=State(('p','a','b'),('e1','e2'))
        self.assertNotIn('cycle',[e.id for e in g.successors(s,q)])

    def test_expert_rejects_invalid_path_even_if_model_rewards_it(self):
        g,_=fixture();q=Query('q','p',('HOLDS','HOLDS'),'制造')
        r=search(g,q,'hybrid_llm',32,llm_scores={path_key(('e1','bad')):1.,path_key(('e1','e2')):0.})
        self.assertFalse(r['found'])
        # The informed search now proves this parent has no legal goal before
        # it is ever expanded. Proof pruning is not an expansion-time rejection.
        self.assertEqual(r['status'],'exhausted');self.assertEqual(r['proof_pruned'],1)
        self.assertEqual(r['expert_rejections'],0);self.assertEqual(r['total_edge_checks'],3)
        self.assertTrue(any(s['status']=='proof_pruned' for s in r['trace']))
        # Independently exercise the original expert rejection even with a
        # maximal model reward: zero weights disable the informed proof shortcut.
        r=search(g,q,'hybrid_llm',32,weights=[0.]*6,
                 llm_scores={path_key(('e1','bad')):1.,path_key(('e1','e2')):0.})
        self.assertFalse(r['found']);self.assertGreater(r['expert_rejections'],0)
        self.assertEqual(r['proof_pruned'],0)
        self.assertTrue(any(s['status']=='rejected' and s['edge_ids']==['e1','bad'] for s in r['trace']))
        r=search(g,q,'hybrid_no_expert',32)
        self.assertTrue(r['found']);self.assertFalse(r['valid'])

    def test_rules_cover_stale_unverified_and_low_confidence(self):
        for kwargs in [{'active':False},{'verified':False},{'confidence':.54}]:self.assertFalse(expert_legal(Edge('e','a','b','HOLDS',**kwargs)))
        self.assertTrue(expert_legal(Edge('e','a','b','HOLDS',confidence=.55)))

    def test_search_never_calls_gold_oracle(self):
        g,q=fixture()
        with patch('experiments.lab03.domain.enumerate_paths',side_effect=AssertionError('oracle leaked')):
            self.assertTrue(search(g,q,'hybrid',32)['valid'])

    def test_mcts_reproducible_and_rollouts_accounted(self):
        g,q=fixture();a=search(g,q,'mcts',16,seed=19);b=search(g,q,'mcts',16,seed=19)
        self.assertEqual(a['path'],b['path']);self.assertEqual(a['trace'],b['trace'])
        self.assertGreater(a['rollout_steps'],0);self.assertEqual(a['total_work'],a['expanded']+a['rollout_steps'])

    def test_invalid_query_and_budget_rejected(self):
        g,q=fixture()
        for budget in [0,257]:
            with self.assertRaises(ValueError):search(g,q,'bfs',budget)
        with self.assertRaises(ValueError):search(g,Query('x','missing',q.relations,'制造'))

    def test_path_validation_rejects_forged_edges(self):
        g,q=fixture();self.assertFalse(g.valid_path(State(('p','a','b','c'),('e1','e2','made-up')),q))

    def test_schema_bounds_and_extra_fields(self):
        from pydantic import ValidationError
        for kwargs in [{'budget':0},{'budget':999},{'budget':True},{'method':'unknown'},{'answer':'c'},{'api_key':'x'}]:
            with self.assertRaises(ValidationError):LabThreeTask(**kwargs)

    def test_shared_case_preserves_original_edges(self):
        g,qs=shared_graph();r=search(g,qs[1],'bfs',128)
        self.assertEqual(r['path'],['P1','C1','C2','C3'])
        product=math.prod(g.edges[e].value for e in r['edge_ids'])
        self.assertAlmostEqual(product,.182)


class DataTests(unittest.TestCase):
    def test_graphs_and_entities_do_not_leak_across_splits(self):
        data=load_dataset();seen=set();count=0
        for record in data:
            g,qs=unpack(record);self.assertFalse(seen&set(g.nodes));seen.update(g.nodes)
            for q in qs:self.assertTrue(enumerate_paths(g,q));count+=1
        self.assertEqual(count,405)

    def test_model_features_do_not_contain_ids_or_gold(self):
        from experiments.lab03.domain import FEATURE_NAMES
        self.assertEqual(len(FEATURE_NAMES),6)
        m={str(seed):fit_classifier(load_dataset(),seed) for seed in (7,19,43)}
        for model in m.values():
            self.assertTrue(all(x.startswith('train-') for x in model['train_graph_ids']))
            self.assertEqual(len(model['coef']),6)
            self.assertTrue(0<=probability(model,[.5,1,.5,1,1,0])<=1)

    def test_ga_elitism_and_validation_only(self):
        ga={str(seed):optimize(load_dataset(),seed,population=4,generations=2) for seed in (7,19,43)}
        for result in ga.values():
            self.assertTrue(all(x.startswith('validation-') for x in result['validation_graph_ids']))
            values=[r['fitness'] for r in result['history']]
            self.assertTrue(all(b>=a-1e-12 for a,b in zip(values,values[1:])))

    def test_web_query_cannot_override_graph_or_answers(self):
        with self.assertRaises(ValueError):run_task({'query_id':'../../secrets'})
        r=run_task({'query_id':'shared-3','method':'astar_fixed','budget':16})
        self.assertTrue(r['result']['valid']);self.assertEqual(r['llm']['mode'],'off')

    def test_missing_llm_replay_fails_explicitly(self):
        with self.assertRaises(ValueError):run_task({'query_id':'shared-3','method':'hybrid_llm','budget':8})

class LLMTests(unittest.TestCase):
    @requires_artifacts('models.json','ga_history.json','manifest.json','llm_cache.json','llm_audit.json','llm_evidence_readout.json')
    def test_real_cached_scores_are_bounded_and_graph_bound(self):
        from experiments.lab03.service import read
        from experiments.lab03.domain import digest
        cache=read('llm_cache.json',{})
        self.assertEqual(len(cache),9)
        for qid,record in cache.items():
            g,q,_=tasks()[qid]
            self.assertEqual(record['graph_sha256'],digest(g.to_dict()))
            self.assertEqual(record['query_sha256'],digest(q.__dict__))
            self.assertTrue(all(g.valid_prefix(g.state_for_edges(q,json.loads(e)),q,expert=False) and 0<=v<=1 for e,v in record['scores'].items()))
            result=run_task({'query_id':qid,'method':'hybrid_llm','budget':8})
            self.assertEqual(result['llm']['mode'],'replay')

    @requires_artifacts('models.json','ga_history.json','manifest.json','llm_cache.json','llm_audit.json','llm_evidence_readout.json')
    def test_replay_rejects_tampered_fingerprint(self):
        import experiments.lab03.service as service
        original=service.read
        def modified(name,default=None):
            result=original(name,default)
            if name=='llm_cache.json':
                result['test-7-0-Q0']['graph_sha256']='not-the-graph'
            return result
        with patch.object(service,'read',side_effect=modified):
            with self.assertRaises(ValueError):service.run_task({'query_id':'test-7-0-Q0','method':'hybrid_llm'})

    def test_llm_prompt_contains_no_gold_or_answer_id(self):
        from experiments.lab03.llm import prompt_for
        g,q=fixture();prompt=prompt_for(g,q,g.edges['e1'])
        self.assertNotIn('目标',prompt)
        self.assertNotIn('oracle',prompt)
        self.assertIn('Do NOT answer',prompt)
        self.assertIn('verified=',prompt)

    def test_remote_never_uses_ambient_developer_credentials(self):
        import os
        from experiments.lab03.llm import remote_score
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaises(ValueError):remote_score('candidate')

    def test_synthetic_person_actions_have_valid_schema(self):
        for record in load_dataset():
            g,qs=unpack(record)
            for edge in g.edges.values():
                if g.nodes[edge.source].kind=='Person':self.assertEqual(edge.relation,'HOLDS')
            for node in g.nodes.values():
                if node.kind=='Company':self.assertTrue(node.name.startswith(node.sector))

if __name__=='__main__':unittest.main()