"""Requirements audit: no silent fallbacks, path-scoped LLM, strict evidence and provenance."""
import copy
import io
import json
import os
import unittest
from unittest.mock import patch

from experiments.lab03.domain import Graph, Node, Edge, State, Query, path_key
from experiments.lab03.search import search, heuristic, FIXED
from experiments.lab03 import llm, service
from experiments.lab03.genetic import optimize
from experiments.lab03.learning import fit_classifier
from experiments.lab03.evaluate import boundary_evaluation
from tests.lab03.test_search import fixture


def model(seed=7):
    return {'coef': [0.] * 6, 'intercept': 0., 'seed': seed}


class RequirementLogicTests(unittest.TestCase):
    def test_learned_method_requires_its_actual_prerequisites(self):
        graph, query = fixture()
        for method in ['astar_ga', 'hybrid', 'hybrid_no_expert', 'hybrid_llm']:
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, 'GA参数'):
                search(graph, query, method)
        for method in ['astar_ml', 'hybrid', 'hybrid_no_expert', 'hybrid_llm']:
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, '分类器'):
                search(graph, query, method, weights=FIXED)
        with self.assertRaisesRegex(ValueError, '评分'):
            search(graph, query, 'hybrid_llm', weights=FIXED, model=model())

    def test_classifier_seed_cannot_silently_cross_experiment_split(self):
        graph, query = fixture()
        with self.assertRaisesRegex(ValueError, '种子'):
            search(graph, query, 'astar_ml', model=model(19), seed=7)

    def test_graph_rejects_truthy_strings_and_nonfinite_evidence(self):
        for kwargs in [{'verified': 'False'}, {'active': 1}, {'confidence': True},
                       {'confidence': float('nan')}, {'value': float('inf')}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Graph([Node('a','甲'), Node('b','乙')], [Edge('e','a','b','HOLDS', **kwargs)])
        with self.assertRaises(ValueError):
            Graph([Node('a','甲', listed='false')], [])

    def test_transition_rejects_forged_or_disconnected_edges(self):
        graph, _ = fixture()
        for state, edge in [(State(('p',)), graph.edges['e2']),
                            (State(('p',)), Edge('not-real','p','a','HOLDS')),
                            (State(('p','a','b'), ('e1','e2')), graph.edges['cycle'])]:
            with self.subTest(edge=edge), self.assertRaises(ValueError):
                graph.apply(state, edge)

    def test_no_answer_and_unsafe_boundary_matrix(self):
        result = boundary_evaluation()
        self.assertEqual(result['total'], 56)
        self.assertEqual(result['passed'], result['total'])
        self.assertTrue(any(not r['answerable'] for r in result['cases']))

    def test_training_and_ga_fail_on_missing_or_invalid_data(self):
        with self.assertRaisesRegex(ValueError, '训练'): fit_classifier([], 7)
        with self.assertRaisesRegex(ValueError, '验证'): optimize([], 7)
        for kwargs in [{'population':2},{'generations':-1},{'budget':True}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):optimize([],7,**kwargs)

    def test_path_rewards_do_not_leak_between_prefixes_sharing_an_edge(self):
        graph = Graph([Node('p','起点'),Node('a','甲'),Node('b','乙'),Node('c','汇合'),Node('z','末端')],
                      [Edge('a','p','a','HOLDS'),Edge('b','p','b','HOLDS'),Edge('ac','a','c','HOLDS'),
                       Edge('bc','b','c','HOLDS'),Edge('cz','c','z','HOLDS')])
        query = Query('q','p',('HOLDS',)*4,'制造')
        first = graph.state_for_edges(query,['a','ac','cz'])
        second = graph.state_for_edges(query,['b','bc','cz'])
        scores={path_key(first.edges): 1.}
        low=heuristic(graph,first,query,[0.]*6,llm=scores)
        other=heuristic(graph,second,query,[0.]*6,llm=scores)
        self.assertEqual(other-low,1.)

    def test_path_score_key_must_be_canonical_and_structurally_valid(self):
        graph, query = fixture()
        for key in ['e1', '[]', '["e2"]', '["e1", "e2"]', '["e1","missing"]']:
            with self.subTest(key=key), self.assertRaises(ValueError):
                search(graph,query,'hybrid_llm',weights=FIXED,model=model(),llm_scores={key:.5})
        for score in [True, float('nan'), -1, 1.01]:
            with self.subTest(score=score), self.assertRaises(ValueError):
                search(graph,query,'hybrid_llm',weights=FIXED,model=model(),llm_scores={path_key(['e1']):score})

    def test_llm_prompt_contains_path_context_and_actual_query_predicates(self):
        graph, query = fixture()
        query = Query('q','p',query.relations,'制造',listed=False)
        state = graph.state_for_edges(query,['e1'])
        prompt = llm.prompt_for(graph,query,graph.edges['e2'],state)
        self.assertIn('PATH PREFIX',prompt)
        self.assertIn('2/3',prompt)
        self.assertIn('listed=False',prompt)
        self.assertIn('remaining relations',prompt)
        self.assertNotIn('目标',prompt)
        self.assertNotIn('all evidence eligible=',prompt)
        self.assertNotIn('expert_legal=',prompt)
        with self.assertRaises(ValueError):llm.prompt_for(graph,query,graph.edges['e3'],state)

    def test_candidate_lookahead_has_bounded_path_context_not_goal_labels(self):
        graph, query = fixture()
        with patch('experiments.lab03.domain.enumerate_paths',side_effect=AssertionError('oracle leaked')):
            candidates=llm.candidate_actions(graph,query)
        self.assertLessEqual(len(candidates),8)
        self.assertTrue(all(s.depth<2 for s,e in candidates))
        self.assertTrue(all(e in graph.successors(s,query,False) for s,e in candidates))

    def test_scoring_records_bind_prefix_and_reject_unknown_backend(self):
        graph, query = fixture()
        candidates=llm.candidate_actions(graph,query)
        from experiments.lab03.domain import digest
        readout={'fixture':'explicit synthetic evaluator'}
        fake={'score':.6,'legal':True,'calibrated_legal':True,'calibrated_confidence':.8,
              'readout_sha256':digest(readout),'inference_ms':1,'protocol':llm.SCORE_PROTOCOL}
        with patch.object(llm,'local_score',return_value=fake), patch('experiments.lab03.calibration.load_readout',return_value=readout):
            records=llm.score_actions(graph,query,candidates)
        self.assertEqual(set(llm.bound_scores(records)),{path_key(s.edges+(e.id,)) for s,e in candidates})
        with self.assertRaises(ValueError):llm.score_actions(graph,query,candidates,'fake')

    def test_model_explanation_cannot_certify_wrong_relations_with_correct_names(self):
        graph, query = fixture()
        result=search(graph,query,'bfs',64)
        names='、'.join(graph.nodes[n].name for n in result['path'])
        wrong=names+'存在亲属关系，保证企业绝对安全。'
        checked=llm.audit_explanation(graph,query,result,wrong)
        self.assertFalse(checked['grounded'])
        self.assertEqual(checked['missing_path_entities'],[])
        self.assertIn('持股',checked['verified_text'])
        self.assertTrue(llm.audit_explanation(graph,query,result,checked['verified_text'])['grounded'])

    def test_explanation_revalidates_result_not_just_valid_boolean(self):
        graph, query = fixture()
        fake={'valid':True,'path':['p','a','b','c'],'edge_ids':['e1','e2','fabricated']}
        with patch.object(llm,'local_runtime',side_effect=AssertionError('should not call model')):
            self.assertFalse(llm.explain_verified_path(graph,query,fake)['generated'])
        self.assertEqual(llm.path_explanation(graph,query,fake),'')

    def test_remote_rejects_entity_answers_and_incomplete_output(self):
        env={'LAB03_LLM_API_KEY':'unit-test-only','LAB03_LLM_MODEL':'unit-test-only'}
        values=[{'score':.7,'legal':True,'reason':'answer is entity N99'},
                {'score':True,'legal':True,'reason':'promising'},
                {'score':.7,'legal':True,'reason':'promising','answer':'N99'},
                {'score':.7,'legal':'true','reason':'promising'}, []]
        for value in values:
            response={'status':'completed','output':[{'content':[{'type':'output_text','text':json.dumps(value)}]}]}
            with self.subTest(value=value), patch.dict(os.environ,env,clear=True), patch.object(llm.urllib.request,'urlopen',return_value=io.BytesIO(json.dumps(response).encode())):
                with self.assertRaises(ValueError):llm.remote_score('candidate')
        for malformed in [[], {}, {'output':[None]}, {'output':[{'content':[None]}]}]:
            with patch.dict(os.environ,env,clear=True), patch.object(llm.urllib.request,'urlopen',return_value=io.BytesIO(json.dumps(malformed).encode())):
                with self.assertRaises(ValueError):llm.remote_score('candidate')
        response={'status':'incomplete','output':[]}
        with patch.dict(os.environ,env,clear=True), patch.object(llm.urllib.request,'urlopen',return_value=io.BytesIO(json.dumps(response).encode())):
            with self.assertRaises(ValueError):llm.remote_score('candidate')

    def test_remote_valid_schema_and_same_bounded_candidates(self):
        env={'LAB03_LLM_API_KEY':'unit-test-only','LAB03_LLM_MODEL':'unit-test-only'}
        response={'status':'completed','output':[{'content':[{'type':'output_text','text':json.dumps({'score':.7,'legal':True,'reason':'promising'})}]}]}
        with patch.dict(os.environ,env,clear=True), patch.object(llm.urllib.request,'urlopen',return_value=io.BytesIO(json.dumps(response).encode())) as call:
            result=llm.remote_score('candidate')
            self.assertEqual(result['score'],.7)
            payload=json.loads(call.call_args.args[0].data)
            self.assertFalse(payload['store'])
            self.assertEqual(payload['text']['format']['schema']['properties']['reason']['enum'],['promising','uncertain','unsafe'])

    def test_stale_artifacts_are_unavailable_without_disabling_baselines(self):
        from tests.lab03.support import artifact_fixture
        original=service.read
        def corrupt(name,default=None):
            value=artifact_fixture(name,original(name,default))
            if name in ['models.json','ga_history.json']:
                for entry in value.values():entry['provenance']['dataset_sha256']='stale'
            if name=='manifest.json':value['dataset_sha256']='stale'
            return value
        with patch.object(service,'read',side_effect=corrupt):
            self.assertTrue(service.run_task({'method':'bfs','budget':128})['result']['valid'])
            with self.assertRaises(ValueError):service.run_task({'method':'hybrid'})
            snapshot=service.snapshot()
            self.assertEqual(snapshot['summary'],[])
            self.assertTrue(snapshot['warnings'])

    def test_old_edge_only_llm_cache_is_rejected(self):
        graph, query = fixture()
        record={'graph_sha256':llm.digest(graph.to_dict()),'query_sha256':llm.digest(query.__dict__),
                'scores':{'e1':.5},'model':'old','scored_edges':1,'llm_ms':1}
        with self.assertRaisesRegex(ValueError,'协议'):service.validate_replay(record,graph,query)


if __name__=='__main__':unittest.main()
