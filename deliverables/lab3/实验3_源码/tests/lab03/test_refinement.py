"""Refinement regressions: exact one-hop evidence, cached work and model boundaries."""
from tests.lab03.support import requires_artifacts
import json
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from experiments.lab03.domain import Graph, Node, Edge, Query, State, features, path_key
from experiments.lab03.search import search, FIXED
from experiments.lab03 import llm
from experiments.lab03.genetic import optimize, warm_start
from experiments.lab03.dataset import load_dataset


def fixture():
    graph=Graph([Node('p','起点'),Node('a','中间'),Node('b','旁路'),Node('z','终点',sector='制造',listed=True)],
                [Edge('pa','p','a','HOLDS'),Edge('pb','p','b','HOLDS'),
                 Edge('az','a','z','HOLDS'),Edge('bz','b','z','HOLDS')])
    return graph,Query('q','p',('HOLDS','HOLDS'),'制造')


class ExactOneHopTests(TestCase):
    def test_observed_completion_does_not_bypass_expansion_budget(self):
        g,q=fixture()
        self.assertFalse(search(g,q,'astar_fixed',2)['found'])
        r=search(g,q,'astar_fixed',3)
        self.assertTrue(r['valid'])
        self.assertEqual(r['expanded'],3)
        self.assertEqual(r['edge_ids'],['pa','az'])

    def test_llm_cannot_override_observed_one_hop_completion(self):
        g,q=fixture();model={'seed':7,'coef':[0.]*6,'intercept':0.}
        scores={path_key(['pa']):0.,path_key(['pb']):1.}
        r=search(g,q,'hybrid_llm',3,FIXED,model,llm_scores=scores)
        self.assertTrue(r['valid'])
        self.assertEqual(r['edge_ids'],['pa','az'])
        event=next(e for e in r['trace'] if e['edge_ids']==['pa'])
        self.assertEqual(event['h'],1.)

    def test_zero_heuristic_still_preserves_fifo_lower_bound(self):
        g,q=fixture()
        bfs=search(g,q,'bfs',3)
        lower=search(g,q,'astar_admissible',3)
        zeros=search(g,q,'astar_ga',3,weights=[0.]*6)
        self.assertFalse(bfs['found'])
        for r in (lower,zeros):
            self.assertEqual(r['expanded'],bfs['expanded'])
            self.assertEqual(r['found'],bfs['found'])
            self.assertEqual(r['heuristic_evaluations'],0)

    def test_successors_are_evaluated_once_and_reused(self):
        g,q=fixture();calls=[]
        original=g.successors
        def observed(state,*args):
            calls.append(state.edges)
            return original(state,*args)
        with patch.object(g,'successors',side_effect=observed):r=search(g,q,'astar_fixed',16)
        self.assertEqual(len(calls),len(set(calls)))
        self.assertGreater(r['successor_cache_hits'],0)
        self.assertEqual(r['total_edge_checks'],4)
        self.assertEqual(r['successor_checks'],0)
        self.assertEqual(r['heuristic_edge_checks'],4)

    def test_cache_is_not_shared_between_queries_or_expert_modes(self):
        g,q=fixture();g=Graph(g.nodes.values(),[Edge('pa','p','a','HOLDS',verified=False),g.edges['az']])
        safe=search(g,q,'astar_fixed',16,expert=True)
        unsafe=search(g,q,'astar_fixed',16,expert=False)
        safe_again=search(g,q,'astar_fixed',16,expert=True)
        self.assertFalse(safe['found']);self.assertTrue(unsafe['found']);self.assertFalse(unsafe['valid'])
        self.assertEqual(safe['total_edge_checks'],safe_again['total_edge_checks'])

    def test_goal_rejects_wrong_start_or_relation_not_just_correct_endpoint(self):
        g,q=fixture();state=g.state_for_edges(q,['pa','az'])
        self.assertTrue(g.goal(state,q))
        self.assertFalse(g.goal(state,Query('bad','b',q.relations,q.sector)))
        self.assertFalse(g.goal(state,Query('bad','p',('HOLDS','TRADES'),q.sector)))
        self.assertFalse(g.goal(state,None))

    def test_core_request_types_fail_as_value_errors(self):
        g,q=fixture()
        for query in [None,{},Query('q','p',['HOLDS','HOLDS'],'制造'),Query('q','p',([],),'制造')]:
            with self.subTest(query=query),self.assertRaises(ValueError):search(g,query)
        for method in [None,[],{}]:
            with self.subTest(method=method),self.assertRaises(ValueError):search(g,q,method)

    def test_web_adapter_rejects_malformed_payload_without_type_error(self):
        from experiments.lab03.service import run_task
        for payload in [None,[],{'query_id':[]},{'method':[]}]:
            with self.subTest(payload=payload),self.assertRaises(ValueError):run_task(payload)


class WarmStartTests(TestCase):
    def test_starting_weights_are_validated(self):
        for initial in [None,[[True]*6],[[0.]*5],[[0.]*5+[1.]],[[float('nan')]*6],[[0.]*6]*3]:
            with self.subTest(initial=initial),self.assertRaises(ValueError):
                optimize([],7,population=3,initial_weights=initial)

    def test_warm_start_is_validation_bound_and_training_independent(self):
        records=load_dataset()
        only_validation=[r for r in records if r['split']=='validation']
        self.assertEqual(warm_start(records,7),warm_start(only_validation,7))
        with self.assertRaises(ValueError):warm_start([r for r in records if r['split']=='train'],7)

    def test_provided_incumbent_appears_in_initial_generation(self):
        records=[r for r in load_dataset() if r['split']=='validation' and r['seed']==7]
        weights=warm_start(records,7)
        result=optimize(records,7,population=3,generations=0,initial_weights=weights)
        self.assertEqual(result['initial_weights'],weights)
        self.assertEqual(result['validation_graph_ids'],sorted(r['graph']['id'] for r in records))


class ControlledExplanationTests(TestCase):
    def test_controlled_texts_are_complete_factual_paraphrases(self):
        g,q=fixture();r=search(g,q,'astar_fixed',16)
        options=llm.controlled_explanation_options(g,q,r)
        self.assertEqual(len(options),3)
        for text in options:
            for node in r['path']:self.assertIn(g.nodes[node].name,text)
            self.assertIn('2跳关系顺序',text)
            self.assertIn('持股',text)
        self.assertEqual(options[0],llm.path_explanation(g,q,r))

    def test_model_only_chooses_wording_and_cannot_write_facts(self):
        g,q=fixture();r=search(g,q,'astar_fixed',16)
        with patch.object(llm,'local_predict',return_value=([.1,.8,.1],12,False)) as predict:
            explanation=llm.controlled_explanation(g,q,r)
        self.assertEqual(explanation['controlled_choice'],'B')
        self.assertEqual(explanation['controlled_text'],llm.controlled_explanation_options(g,q,r)[1])
        self.assertTrue(explanation['controlled_grounded'])
        self.assertEqual(predict.call_args.args[1],('A','B','C'))
        # Same text would not be certified as unrestricted free-form inference.
        self.assertFalse(llm.audit_explanation(g,q,r,explanation['controlled_text'])['grounded'])

    def test_invalid_witness_never_requests_model_explanation(self):
        g,q=fixture();r={'valid':True,'path':['p','a','z'],'edge_ids':['fake','az']}
        with patch.object(llm,'local_predict',side_effect=AssertionError('invalid witness sent to LLM')):
            self.assertFalse(llm.controlled_explanation(g,q,r)['controlled_generated'])

    def test_fabricated_claim_stays_uncertified(self):
        g,q=fixture();r=search(g,q,'astar_fixed',16)
        text=llm.path_explanation(g,q,r)+'因此该公司经营稳定。'
        self.assertFalse(llm.audit_explanation(g,q,r,text)['grounded'])


class MemoizedModelTests(TestCase):
    def test_identical_predictions_reuse_model_but_different_letters_do_not(self):
        import torch
        class Tokenizer:
            def apply_chat_template(self,*args,**kwargs):return args[0][0]['content']
            def __call__(self,text,**kwargs):return {'input_ids':torch.tensor([[1,2]])}
            def encode(self,letter,**kwargs):return [{'A':0,'B':1,'C':2,'True':3,'False':4}[letter]]
        class Model:
            def __init__(self):self.calls=[]
            def __call__(self,**kwargs):
                self.calls.append(kwargs)
                return type('Output',(),{'logits':torch.tensor([[[2.,1.,0.,1.,2.]]])})()
        tok,model=Tokenizer(),Model()
        with llm._prediction_lock:llm._prediction_cache.clear()
        with patch.object(llm,'local_runtime',return_value=(tok,model)):
            first=llm.local_predict('facts',('A','B','C'))
            second=llm.local_predict('facts',('A','B','C'))
            third=llm.local_predict('facts',('True','False'))
        self.assertFalse(first[2]);self.assertTrue(second[2]);self.assertFalse(third[2])
        self.assertEqual(first[:2],second[:2]);self.assertEqual(len(model.calls),2)
        self.assertTrue(all(c['logits_to_keep']==1 for c in model.calls))
        with llm._prediction_lock:llm._prediction_cache.clear()


@requires_artifacts('models.json','ga_history.json','manifest.json','refinement/protocol.json','refinement/confirmation.json','optimization/baseline_sources.zip')
class ConfirmationAuditTests(TestCase):
    def test_precommitted_confirmation_has_disjoint_ids_and_full_method_budget_matrix(self):
        from experiments.lab03.refinement import validate_frozen_strategy, FOLDER, read
        protocol=validate_frozen_strategy()
        confirmation=read(FOLDER/'confirmation.json')
        self.assertEqual(confirmation['question_count'],216)
        self.assertEqual(confirmation['paired_runs'],5184)
        old_ids={r['graph']['id'] for r in load_dataset()}
        self.assertTrue(all(g['id'] not in old_ids for g in confirmation['graphs']))
        keys=[(r['query_id'],r['method'],r['budget']) for r in confirmation['rows']]
        self.assertEqual(len(keys),len(set(keys)))
        self.assertEqual(set(protocol['confirmation_plan']['seeds']),{101,307,911})

    def test_refinement_saved_results_recompute_without_changing_old_stage(self):
        from experiments.lab03.refinement import audit_refinement, FOLDER, read
        from experiments.lab03.optimization import audit_optimization, FOLDER as FIRST
        self.assertEqual(audit_refinement(),read(FOLDER/'comparison.json'))
        self.assertEqual(audit_optimization(),read(FIRST/'comparison.json'))


class ReproductionCommandTests(TestCase):
    def test_all_rebuilds_current_stage_before_reporting(self):
        from experiments.lab03.main import main
        calls=[]
        with patch('pathlib.Path.is_file',return_value=True), \
             patch('experiments.lab03.evaluate.run_benchmark',side_effect=lambda:calls.append('benchmark')), \
             patch('experiments.lab03.evaluate.run_llm',side_effect=lambda:calls.append('llm')), \
             patch('experiments.lab03.reliability.rebuild',side_effect=lambda:calls.append('rebuild')), \
             patch('experiments.lab03.posthoc.analyze_structural_failures',side_effect=lambda:calls.append('failures')), \
             patch('experiments.lab03.report.build_report',side_effect=lambda:calls.append('report')):
            self.assertEqual(main(['all']),0)
        self.assertEqual(calls,['benchmark','llm','rebuild','report'])
