"""Verify raw predictions, supervised readout evidence and exact safe-pruning semantics."""
from tests.lab03.support import requires_artifacts
import json,tempfile,hashlib,os
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
from experiments.lab03.domain import Graph,Node,Edge,Query,State,path_key
from experiments.lab03.search import search,FIXED
from experiments.lab03 import llm,calibration


def fixture():
    nodes=[Node('p','Start'),Node('a','Dead'),Node('b','Live'),Node('z','Goal',sector='制造',listed=True)]
    return Graph(nodes,[Edge('01','p','a','HOLDS'),Edge('02','p','b','HOLDS'),Edge('bz','b','z','HOLDS')]),Query('q','p',('HOLDS','HOLDS'),'制造')


class ProofPruningTests(TestCase):
    def test_proven_dead_child_is_recorded_not_counted_as_queue_pop(self):
        g,q=fixture();r=search(g,q,'astar_fixed',3)
        self.assertTrue(r['valid']);self.assertEqual(r['expanded'],3)
        self.assertEqual(r['proof_pruned'],1);self.assertGreater(r['proof_checks'],0)
        step=next(e for e in r['trace'] if e['status']=='proof_pruned')
        self.assertEqual(step['node'],'a');self.assertTrue(step['reason']);self.assertIsNone(step['h'])
        self.assertFalse(any(e['status']=='expanded' and e['node']=='a' for e in r['trace']))
        self.assertEqual(r['expert_rejections'],0)

    def test_goal_is_not_returned_free_even_when_entire_suffix_is_seen(self):
        g,q=fixture()
        for budget in (1,2):self.assertFalse(search(g,q,'astar_fixed',budget)['valid'])

    def test_no_answer_terminates_from_checked_children(self):
        g,q=fixture();q=Query('q','p',q.relations,'absent')
        r=search(g,q,'astar_fixed',1)
        self.assertEqual(r['status'],'exhausted');self.assertFalse(r['found']);self.assertEqual(r['expanded'],1)
        self.assertEqual(r['proof_pruned'],2);self.assertEqual(r['total_edge_checks'],3)

    def test_lower_bound_and_blind_algorithms_preserve_no_pruning_semantics(self):
        g,q=fixture()
        for method in ('bfs','astar_admissible','mcts'):
            r=search(g,q,method,64)
            self.assertEqual(r['proof_pruned'],0);self.assertEqual(r['proof_checks'],0)
        r=search(g,q,'astar_ga',64,weights=[0.]*6)
        self.assertEqual(r['proof_pruned'],0);self.assertEqual(r['proof_checks'],0)

    def test_truncated_suffix_is_unknown_not_a_delete_decision(self):
        nodes=[Node(str(i),str(i),sector='制造' if i==6 else '其他',listed=i==6) for i in range(7)]
        edges=[Edge('01','0','1','HOLDS'),Edge('02','1','2','HOLDS'),Edge('03','1','3','HOLDS'),Edge('04','1','4','HOLDS'),Edge('05','4','6','HOLDS')]
        g=Graph(nodes,edges);q=Query('q','0',('HOLDS',)*3,'制造')
        r=search(g,q,'astar_fixed',16)
        self.assertTrue(r['valid']);self.assertEqual(r['edge_ids'],['01','04','05'])
        self.assertFalse(any(e['status']=='proof_pruned' and e['edge_ids']==['01'] for e in r['trace']))

    def test_neither_llm_nor_ml_can_prove_a_branch_dead(self):
        g,q=fixture();model={'seed':7,'coef':[0.]*6,'intercept':-1000.}
        scores={path_key(['01']):1.,path_key(['02']):0.}
        r=search(g,q,'hybrid_llm',3,FIXED,model,llm_scores=scores)
        self.assertTrue(r['valid']);self.assertEqual(r['edge_ids'],['02','bz'])

    def test_trace_has_no_effect_on_cost_or_pruning(self):
        g,q=fixture();a=search(g,q,'astar_fixed',16,trace=False);b=search(g,q,'astar_fixed',16,trace=True)
        for k in ('proof_checks','proof_pruned','peak_frontier','total_edge_checks','adjacency_edge_scans','expanded'):
            self.assertEqual(a[k],b[k])


class CalibratedEvidenceTests(TestCase):
    def test_raw_and_supervised_predictions_are_both_retained(self):
        model={'hidden_dim':2,'mean':[0.,0.],'scale':[1.,1.],'coef':[-1.,0.],'intercept':0.}
        g,q=fixture();prompt=llm.prompt_for(g,q,g.edges['02'])
        with patch.object(llm,'local_runtime',return_value=(None,None)), \
             patch.object(llm,'local_predict',return_value=([.8,.1,.1],10,False)), \
             patch.object(llm,'evidence_prediction',return_value=([2.,1.],[.9,.1],7,False)):
            r=llm.local_score(prompt,readout=model)
        self.assertTrue(r['legal']);self.assertFalse(r['calibrated_legal'])
        self.assertEqual(r['evidence_hidden'],[2.,1.]);self.assertEqual(r['processed_prompt_tokens'],17)
        self.assertEqual(r['readout_sha256'],llm.digest(model))
        self.assertIn('不是',r['calibration_note']) if '不是' in r['calibration_note'] else self.assertIn('另列',r['calibration_note'])

    def test_score_binding_uses_readout_not_expert_label(self):
        r={'path_key':path_key(['x']),'score':.8,'legal':True,'calibrated_legal':False,'expert_legal':True}
        self.assertAlmostEqual(llm.bound_scores([r])[r['path_key']],.2)
        r['calibrated_legal']=True;r['expert_legal']=False
        self.assertAlmostEqual(llm.bound_scores([r])[r['path_key']],.8)

    def test_missing_readout_never_silently_uses_raw_model(self):
        g,q=fixture()
        with patch('experiments.lab03.calibration.load_readout',side_effect=ValueError('missing readout')), \
             patch.object(llm,'local_score') as model:
            with self.assertRaises(ValueError):llm.score_actions(g,q,llm.candidate_actions(g,q))
        model.assert_not_called()

    def test_fit_does_not_accept_bad_features_or_one_class_labels(self):
        for x,y in [([[0.,1.]],[1.]),([[float('nan'),0.],[1.,1.]],[0.,1.])]:
            with self.assertRaises(ValueError):calibration.fit_from_hidden(x,y)

    def test_probability_depends_only_on_learned_parameters_and_hidden_vector(self):
        model={'hidden_dim':2,'mean':[0.,0.],'scale':[1.,1.],'coef':[2.,-1.],'intercept':0.}
        self.assertGreater(calibration.probability(model,[1.,0.]),.5)
        self.assertLess(calibration.probability(model,[0.,1.]),.5)
        for x in ([1.],[float('nan'),0.]):
            with self.assertRaises(ValueError):calibration.probability(model,x)

    def test_evidence_cache_is_prompt_specific_and_reports_reuse(self):
        tok,model=object(),object();llm.clear_prediction_caches()
        with patch.object(llm,'local_runtime',return_value=(tok,model)), \
             patch.object(llm,'evidence_representation',return_value=([1.,2.],[.9,.1],7)) as forward:
            first=llm.evidence_prediction('facts');second=llm.evidence_prediction('facts');third=llm.evidence_prediction('other')
        self.assertFalse(first[3]);self.assertTrue(second[3]);self.assertFalse(third[3]);self.assertEqual(forward.call_count,2)
        llm.clear_prediction_caches()


@requires_artifacts('models.json','ga_history.json','manifest.json','llm_cache.json','llm_audit.json','llm_evidence_readout.json','llm_summary.json','runs.csv','reliability/confirmation.json','reliability/llm_confirmation.json','reliability/protocol.json','reliability/stage_three_baseline.zip')
class ReliabilityArtifactTests(TestCase):
    def test_readout_training_and_predictions_reconstruct_from_real_saved_states(self):
        from experiments.lab03.calibration import verify_readout,load_readout
        v=verify_readout();self.assertEqual(v['train_prefixes'],1296);self.assertEqual(v['train_unique_prompts'],301)
        model=load_readout();self.assertEqual(model['l2'],.1)
        wrong={**model,'train_graph_ids':['test-7-0']}
        with self.assertRaises(ValueError):calibration.validate_readout(wrong)
        wrong={**model,'scale':[0.]*896}
        with self.assertRaises(ValueError):calibration.validate_readout(wrong)

    def test_fourth_stage_confirmation_freeze_has_all_cases_and_no_label_tuning(self):
        from experiments.lab03.reliability import validate_frozen_strategy,read,FOLDER
        p=validate_frozen_strategy();v=read(FOLDER/'confirmation.json')
        self.assertEqual(p['confirmation_plan']['seeds'],[5039,7001,9001])
        self.assertEqual(v['question_count'],216);self.assertEqual(v['paired_runs'],5184)
        self.assertEqual(v['negative_questions'],216);self.assertEqual(v['negative_pairs'],864)
        for kind in ('rows','negative_rows'):
            keys=[(r['query_id'],r['method'],r['budget']) for r in v[kind]]
            self.assertEqual(len(keys),len(set(keys)))
        self.assertTrue(all(not r['after']['found'] for r in v['negative_rows']))

    def test_new_llm_and_precision_shift_results_recompute_without_model_calls(self):
        from experiments.lab03.verification_reliability import verify_llm_confirmation
        with patch.object(llm,'local_runtime',side_effect=AssertionError('replay must not call model')):
            result=verify_llm_confirmation()
        self.assertEqual(result,{'fresh_questions':18,'fresh_prefixes':144,'numeric_cases':100,'paired_searches':54})

    def test_readout_corruption_blocks_replay_instead_of_silently_falling_back(self):
        from experiments.lab03.service import tasks,validate_replay,read
        graph,query,_=tasks()['test-7-0-Q0'];record=read('llm_cache.json')['test-7-0-Q0']
        wrong={**record,'readout_sha256':'wrong'}
        with self.assertRaisesRegex(ValueError,'分类头'):validate_replay(wrong,graph,query)
        with patch('experiments.lab03.calibration.load_readout',side_effect=ValueError('missing classifier')):
            with self.assertRaisesRegex(ValueError,'missing'):validate_replay(record,graph,query)


@requires_artifacts('models.json','ga_history.json','manifest.json','llm_cache.json','llm_audit.json','llm_evidence_readout.json','llm_summary.json','runs.csv','reliability/confirmation.json','reliability/llm_confirmation.json','reliability/protocol.json','reliability/stage_three_baseline.zip')
class ConfirmationDisplayTests(TestCase):
    def test_snapshot_exposes_disjoint_confirmations_without_hidden_vectors(self):
        from experiments.lab03.service import snapshot
        value=snapshot();self.assertEqual(value['warnings'],[])
        self.assertEqual(value['confirmation']['negative_questions'],216)
        fresh=value['llm_confirmation'];self.assertEqual(fresh['prefixes'],144)
        self.assertEqual(fresh['numeric_raw']['false_accept'],12)
        self.assertEqual(fresh['numeric_calibrated']['false_accept'],1)
        self.assertEqual(fresh['numeric_calibrated']['false_reject'],2)
        self.assertLess(fresh['numeric_calibrated']['balanced_accuracy'],fresh['numeric_raw']['balanced_accuracy'])
        for raw in ('audit','numeric_challenge','hidden','rows','negative_rows'):
            self.assertNotIn(raw,fresh);self.assertNotIn(raw,value['confirmation'])

    def test_stale_or_corrupt_confirmation_is_hidden_without_disabling_search(self):
        from experiments.lab03 import service
        import copy
        original=service.read
        for name in ('reliability/confirmation.json','reliability/llm_confirmation.json'):
            for corruption in ('source','matrix'):
                def corrupt(filename,default=None):
                    value=original(filename,default)
                    if filename==name:
                        value=copy.deepcopy(value)
                        if corruption=='source':value['source_sha256']={}
                        elif name.endswith('/confirmation.json'):value['negative_summary']=[{}]*8
                        else:value['numeric_calibrated']['false_reject']=0
                    return value
                with self.subTest(name=name,corruption=corruption),patch.object(service,'read',side_effect=corrupt):
                    value=service.snapshot();self.assertTrue(value['warnings'])
                    key='confirmation' if name.endswith('/confirmation.json') else 'llm_confirmation'
                    self.assertEqual(value[key],{})
                    self.assertTrue(service.run_task({'method':'bfs','budget':128})['result']['valid'])
