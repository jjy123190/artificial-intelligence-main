"""Second requirement audit: independent kernels, evidence identity and honest diagnostics."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.lab03 import llm
from experiments.lab03.domain import Graph,Node,Edge,Query,State,path_key
from experiments.lab03.search import search,FIXED
from experiments.lab03.diagnostics import legality_metrics,llm_diagnostics,controlled_counterexamples
from tests.lab03.test_search import fixture


def model(): return {'seed':7,'coef':[0.]*6,'intercept':0.}


def completed():
    return {'status':'completed','output':[{'status':'completed','content':[
        {'type':'output_text','text':json.dumps({'score':.7,'legal':True,'reason':'promising'})}]}]}


class KernelBoundaryReview(unittest.TestCase):
    def test_bfs_never_calls_heuristic_or_classifier(self):
        g,q=fixture()
        with patch('experiments.lab03.search.heuristic',side_effect=AssertionError('BFS used h')),\
             patch('experiments.lab03.search.probability',side_effect=AssertionError('BFS used classifier')):
            self.assertTrue(search(g,q,'bfs',64)['valid'])

    def test_mcts_never_calls_astar_heuristic(self):
        g,q=fixture()
        with patch('experiments.lab03.search.heuristic',side_effect=AssertionError('MCTS used h')):
            self.assertTrue(search(g,q,'mcts',256)['valid'])

    def test_malformed_witness_types_fail_closed(self):
        g,q=fixture()
        invalid=[None,{},State(None,None),State(('p',None),('e1',)),
                 State(('p',[]),('e1',)),State(('p','a'),({},)),
                 State(['p','a'],['e1']),State(('p','a','b'),('e2','e1'))]
        for state in invalid:
            with self.subTest(state=state):
                self.assertFalse(g.valid_state(state))
                self.assertFalse(g.valid_prefix(state,q))
                self.assertFalse(g.valid_path(state,q))

    def test_apply_does_not_extend_forged_history(self):
        g,q=fixture()
        for state in [State(('p','a'),('fabricated',)),State(('p','a'),()),State(('missing','a'),('e1',))]:
            with self.subTest(state=state), self.assertRaises(ValueError):
                g.apply(state,g.edges['e2'])

    def test_llm_empty_scores_are_allowed_only_if_no_actions_exist(self):
        g,q=fixture()
        with self.assertRaisesRegex(ValueError,'评分'):
            search(g,q,'hybrid_llm',weights=FIXED,model=model(),llm_scores={})
        empty=Graph(list(g.nodes.values()),[])
        result=search(empty,q,'hybrid_llm',weights=FIXED,model=model(),llm_scores={})
        self.assertFalse(result['found'])
        self.assertEqual(result['status'],'exhausted')

    def test_classifier_and_ga_change_search_order_without_oracle(self):
        nodes=[Node('p','人'),Node('a','干扰',listed=False),Node('b','路径',listed=True),
               Node('z','终点',sector='制造',listed=True)]
        g=Graph(nodes,[Edge('first','p','a','HOLDS'),Edge('second','p','b','HOLDS'),Edge('end','b','z','HOLDS')])
        q=Query('q','p',('HOLDS','HOLDS'),'制造')
        self.assertFalse(search(g,q,'astar_ga',3,weights=[0.]*6)['valid'])
        self.assertTrue(search(g,q,'astar_ga',3,weights=[0.,0.,0.,4.,0.,0.])['valid'])
        trained={'seed':7,'coef':[0.,0.,0.,0.,5.,0.],'intercept':-2.5}
        self.assertTrue(search(g,q,'astar_ml',3,model=trained)['valid'])


    def test_random_graphs_match_raw_edge_oracle_for_all_kernels(self):
        # Deliberately do not reuse Graph.successors/apply/goal/valid_path or enumerate_paths.
        import random
        methods=['bfs','astar_admissible','astar_fixed','astar_ga','astar_ml','hybrid','hybrid_llm','mcts']
        for seed in range(120):
            rng=random.Random(18000+seed)
            nodes=[Node(str(i),str(i),sector=rng.choice(['制造','综合']),listed=rng.choice([True,False])) for i in range(6)]
            edges=[]
            for i in range(6):
                for j in range(6):
                    if rng.random()<.25:
                        edges.append(Edge(str(len(edges)),str(i),str(j),rng.choice(['HOLDS','TRADES']),
                                          confidence=rng.choice([.54,.55,1.]),verified=rng.random()>.15,active=rng.random()>.1))
            hops=rng.choice([2,3,4]);relations=tuple(rng.choice(['HOLDS','TRADES']) for _ in range(hops))
            q=Query(str(seed),'0',relations,'制造',listed=rng.choice([True,False]))
            if seed%2==0:
                chain=['0']+rng.sample(['1','2','3','4','5'],hops)
                nodes[int(chain[-1])]=Node(chain[-1],chain[-1],sector='制造',listed=q.listed)
                for a,b,r in zip(chain,chain[1:],relations):edges.append(Edge(str(len(edges)),a,b,r))
            g=Graph(nodes,edges)
            todo=[(('0',),())];witnesses=set()
            while todo:
                visited,prefix=todo.pop();depth=len(prefix)
                if depth==hops:
                    node=next(n for n in nodes if n.id==visited[-1])
                    if node.kind=='Company' and node.sector==q.sector and node.listed==q.listed:witnesses.add(prefix)
                else:
                    for e in edges:
                        if (e.source==visited[-1] and e.target not in visited and e.relation==relations[depth]
                                and e.verified and e.active and e.confidence>=.55):
                            todo.append((visited+(e.target,),prefix+(e.id,)))
            candidates=llm.candidate_actions(g,q)
            scores={path_key(st.edges+(e.id,)):rng.random() for st,e in candidates}
            for method in methods:
                result=search(g,q,method,256,weights=FIXED,model=model(),llm_scores=scores,trace=False)
                with self.subTest(seed=seed,method=method):
                    self.assertEqual(result['valid'],bool(witnesses))
                    if result['found']:self.assertIn(tuple(result['edge_ids']),witnesses)
                    else:self.assertEqual(result['status'],'exhausted')


class LLMProtocolReview(unittest.TestCase):
    def invoke(self,response,prompt='candidate'):
        env={'LAB03_LLM_API_KEY':'unit-test-only','LAB03_LLM_MODEL':'unit-test-only'}
        with patch.dict(os.environ,env,clear=True), patch.object(llm.urllib.request,'urlopen',return_value=io.BytesIO(json.dumps(response).encode())) as call:
            result=llm.remote_score(prompt)
            return result,json.loads(call.call_args.args[0].data)

    def test_remote_and_local_share_explicit_legality_rule(self):
        g,q=fixture();prompt=llm.prompt_for(g,q,g.edges['e1'])
        result,payload=self.invoke(completed(),prompt)
        self.assertIn(llm.EVIDENCE_POLICY,payload['instructions'])
        self.assertIn(llm.EVIDENCE_POLICY,llm.legality_prompt_for(prompt))
        self.assertNotIn('Choose only A, B or C',payload['input'])
        self.assertEqual(payload['text']['format']['schema']['properties']['score'],
                         {'type':'number','minimum':0,'maximum':1})
        self.assertEqual(result['request_prompt'],payload['input'])
        self.assertNotIn('unit-test-only',result['instructions'])

    def test_refusal_cannot_hide_beside_valid_json(self):
        response=completed();response['output'][0]['content'].append({'type':'refusal','refusal':'No'})
        with self.assertRaisesRegex(ValueError,'拒绝'):self.invoke(response)

    def test_incomplete_message_cannot_be_used_in_completed_envelope(self):
        response=completed();response['output'][0]['status']='incomplete'
        with self.assertRaisesRegex(ValueError,'未完成'):self.invoke(response)

    def test_ambiguous_multiple_outputs_are_rejected(self):
        response=completed();response['output']*=2
        with self.assertRaisesRegex(ValueError,'唯一'):self.invoke(response)

    def test_api_error_and_incomplete_details_are_rejected(self):
        for field in ['error','incomplete_details']:
            response=completed();response[field]={'message':'not complete'}
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'未完成|错误'):self.invoke(response)

    def test_legality_prompt_requires_actual_facts(self):
        with self.assertRaises(ValueError):llm.legality_prompt_for('fake prefix')

    def test_mismatched_local_checkpoint_is_not_called_qwen(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory,'model.safetensors').write_bytes(b'not the published model')
            with patch.dict(os.environ,{'LAB03_LOCAL_MODEL':directory}):
                with self.assertRaisesRegex(ValueError,'固定Qwen版本'):llm.verified_local_source()

    def test_explicit_missing_model_does_not_fallback_to_other_cache(self):
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,{'LAB03_LOCAL_MODEL':directory}):
            with self.assertRaisesRegex(ValueError,'权重目录'):llm.verified_local_source()

    def test_checkpoint_identity_requires_matching_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            content=b'unit test checkpoint';Path(directory,'model.safetensors').write_bytes(content)
            Path(directory,'config.json').write_bytes(b'config')
            manifest={'config.json':hashlib.sha256(b'config').hexdigest()}
            with patch.dict(os.environ,{'LAB03_LOCAL_MODEL':directory}), \
                 patch.object(llm,'WEIGHTS_SHA256',hashlib.sha256(content).hexdigest()), \
                 patch.object(llm,'MODEL_FILE_HASHES',manifest):
                self.assertEqual(llm.verified_local_source(),str(Path(directory).resolve()))

    def test_nontext_explanation_is_not_certified(self):
        g,q=fixture();result=search(g,q,'bfs',64)
        for value in [None,False,{}]:
            checked=llm.audit_explanation(g,q,result,value)
            self.assertFalse(checked['grounded'])
            self.assertTrue(checked['verified_text'])


class ExperimentalInterpretationReview(unittest.TestCase):
    def test_majority_only_legality_prediction_is_exposed(self):
        records=[{'legal':True,'expert_legal':True}]*61+[{'legal':True,'expert_legal':False}]*11
        metrics=legality_metrics(records)
        self.assertEqual(metrics['accuracy'],61/72)
        self.assertEqual(metrics['always_legal_accuracy'],61/72)
        self.assertEqual(metrics['ineligible_recall'],0.)
        self.assertEqual(metrics['false_accept_rate'],1.)
        self.assertEqual(metrics['balanced_accuracy'],.5)
        self.assertEqual(metrics['false_accept'],11)

    def test_absent_class_is_missing_not_zero_accuracy(self):
        metrics=legality_metrics([{'legal':True,'expert_legal':True}])
        self.assertIsNone(metrics['ineligible_recall'])
        self.assertIsNone(metrics['balanced_accuracy'])
        self.assertIsNone(legality_metrics([])['accuracy'])
        with self.assertRaises(ValueError):legality_metrics([{'legal':1,'expert_legal':True}])

    def test_paired_diagnostics_separate_search_cost_from_answer_loss(self):
        evaluation={'legal':True,'expert_legal':False,'path_key':path_key(['e']),
                    'path_edges':['e'],'score':.9}
        audit=[{'query_id':'q','evaluations':[evaluation]}]
        base={'query_id':'q','valid':True,'expanded':3,'status':'found'}
        rows=[{**base,'method':'hybrid'},{**base,'method':'hybrid_llm','expanded':5}]
        result=llm_diagnostics(audit,rows)
        self.assertEqual(result['extra_expansions'],2)
        self.assertEqual(result['lost_answers'],0)
        self.assertEqual(len(result['legality_errors']),1)
        with self.assertRaises(ValueError):llm_diagnostics(audit,rows+[rows[0]])

    def test_expert_ablation_exposes_invalid_evidence(self):
        result=controlled_counterexamples()['unsafe_branch']
        self.assertTrue(result['expert_on']['valid'])
        self.assertTrue(result['expert_off']['found'])
        self.assertFalse(result['expert_off']['valid'])

    def test_wrong_scores_can_lose_answer_even_with_valid_evidence(self):
        result=controlled_counterexamples()['budget_trap']
        self.assertTrue(result['neutral_scores']['valid'])
        self.assertFalse(result['wrong_scores']['found'])
        self.assertEqual(result['wrong_scores']['status'],'budget_exhausted')
        self.assertEqual(result['wrong_scores']['expert_rejections'],0)


if __name__=='__main__':unittest.main()
