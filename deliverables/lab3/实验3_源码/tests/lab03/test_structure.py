"""Bounded local suffix evidence and honest total-work/model boundaries."""
from tests.lab03.support import requires_artifacts
import hashlib,json,os,tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
from experiments.lab03.domain import Graph,Node,Edge,Query,State,features,SUFFIX_PROBE_CAP,path_key
from experiments.lab03.search import search,heuristic,FIXED
from experiments.lab03 import llm


def two_hop():
    nodes=[Node('p','Start',kind='Person'),Node('a','Other sector'),Node('b','Dead one'),Node('c','Dead two'),
           Node('z','Goal',sector='制造',listed=True)]
    edges=[Edge('01','p','a','HOLDS'),Edge('az','a','z','HOLDS')]
    return Graph(nodes,edges),Query('q','p',('HOLDS','HOLDS'),'制造')


class BoundedSuffixTests(TestCase):
    def test_two_hop_evidence_is_structural_not_sector_homophily(self):
        g,q=two_hop();reads=[]
        with patch('experiments.lab03.domain.enumerate_paths',side_effect=AssertionError('oracle used')):
            x=features(g,State(('p',)),q,account=reads.append)
        self.assertEqual(x[1],1.);self.assertEqual(x[4],1.);self.assertEqual(x[5],0.)
        self.assertEqual(reads,[1,1])
        self.assertEqual(heuristic(g,State(('p',)),q,FIXED),2.)

    def test_unknown_capped_branch_is_not_misreported_dead(self):
        g,q=two_hop();g=Graph(g.nodes.values(),[Edge('01','p','b','HOLDS'),Edge('02','p','c','HOLDS'),
                                             Edge('03','p','a','HOLDS'),g.edges['az']])
        probes=[]
        x=features(g,State(('p',)),q,probe=lambda:probes.append(1))
        self.assertEqual(len(probes),SUFFIX_PROBE_CAP)
        self.assertLess(x[1],1.);self.assertEqual(x[5],0.)
        self.assertTrue(search(g,q,'astar_fixed',64)['valid'])

    def test_exhaustive_small_suffix_can_prove_failure_without_returning_answer(self):
        g,q=two_hop();g=Graph(g.nodes.values(),[Edge('01','p','b','HOLDS')])
        x=features(g,State(('p',)),q)
        self.assertEqual(x[1],0.);self.assertEqual(x[5],1.)
        self.assertFalse(search(g,q,'astar_fixed',64)['found'])

    def test_probe_respects_entire_path_cycle_and_evidence_constraints(self):
        g,q=two_hop();state=State(('p','a'),('01',));q=Query('q','p',('HOLDS',)*3,'制造')
        g=Graph(g.nodes.values(),[g.edges['01'],Edge('back','a','p','HOLDS'),
                                 Edge('unsafe','a','b','HOLDS',verified=False),Edge('bz','b','z','HOLDS')])
        self.assertEqual(features(g,state,q,True)[1],0.)
        self.assertEqual(features(g,state,q,False)[1],1.)
        self.assertFalse(search(g,q,'astar_fixed',64)['found'])

    def test_probe_only_runs_for_exactly_two_remaining_relations(self):
        g,q=two_hop()
        for hops in (3,4):
            scans=[]
            with patch.object(g,'successors',wraps=g.successors) as succ:
                features(g,State(('p',)),Query('q','p',('HOLDS',)*hops,'制造'),probe=lambda:scans.append(1))
            self.assertEqual(succ.call_count,1);self.assertFalse(scans)

    def test_probe_never_returns_solution_without_paying_queue_budget(self):
        g,q=two_hop()
        for budget in (1,2):self.assertFalse(search(g,q,'astar_fixed',budget)['found'])
        r=search(g,q,'astar_fixed',3)
        self.assertTrue(r['valid']);self.assertEqual(r['expanded'],3)
        self.assertEqual(r['heuristic_probe_states'],1)

    def test_all_scanned_outgoing_edges_are_counted_even_if_filtered(self):
        g,q=two_hop();g=Graph(g.nodes.values(),list(g.edges.values())+[Edge('other','p','b','TRADES')])
        r=search(g,q,'astar_fixed',64)
        self.assertEqual(r['total_edge_checks'],2)
        self.assertEqual(r['adjacency_edge_scans'],3)
        self.assertEqual(r['heuristic_probe_states'],1)

    def test_blind_kernels_have_no_probe_work(self):
        g,q=two_hop()
        for method in ('bfs','mcts','astar_admissible'):
            r=search(g,q,method,64)
            self.assertEqual(r['heuristic_probe_states'],0)
            self.assertEqual(r['heuristic_edge_checks'],0)

    def test_trace_cannot_invent_additional_probe_or_scan_work(self):
        g,q=two_hop()
        for method in ('astar_fixed','bfs','mcts'):
            a=search(g,q,method,64,trace=True);b=search(g,q,method,64,trace=False)
            for key in ('heuristic_probe_states','adjacency_edge_scans','total_edge_checks','expanded','edge_ids'):
                self.assertEqual(a[key],b[key])


class LLMNeutralityTests(TestCase):
    def test_missing_and_half_score_have_exactly_zero_preference(self):
        g,q=two_hop();q=Query('q','p',('HOLDS',)*4,'制造');s=g.apply(State(('p',)),g.edges['01'])
        base=heuristic(g,s,q,FIXED)
        self.assertEqual(base,heuristic(g,s,q,FIXED,llm={path_key(['01']):.5}))
        self.assertEqual(base,heuristic(g,s,q,FIXED,llm={path_key(['unseen']):1.}))
        self.assertLess(heuristic(g,s,q,FIXED,llm={path_key(['01']):1.}),base)
        self.assertGreater(heuristic(g,s,q,FIXED,llm={path_key(['01']):0.}),base)

    def test_neutral_actual_scores_preserve_whole_search_order(self):
        g,q=two_hop();model={'coef':[0.]*6,'intercept':0.,'seed':7}
        values={path_key(st.edges+(e.id,)):.5 for st,e in llm.candidate_actions(g,q)}
        a=search(g,q,'hybrid',16,FIXED,model)
        b=search(g,q,'hybrid_llm',16,FIXED,model,llm_scores=values)
        for k in ('valid','expanded','edge_ids','total_edge_checks','trace'):self.assertEqual(a[k],b[k])

    def test_llm_does_not_override_two_hop_observed_evidence(self):
        g,q=two_hop();self.assertEqual(heuristic(g,State(('p',)),q,FIXED,llm={}),2.)
        g=Graph(g.nodes.values(),list(g.edges.values())+[Edge('pa','b','p','HOLDS')])
        q=Query('q','b',('HOLDS',)*3,'制造');state=g.apply(State(('b',)),g.edges['pa'])
        self.assertEqual(heuristic(g,state,q,FIXED,llm={path_key(['pa']):0.}),2.)

    def test_candidate_preparation_reports_actual_neighborhood_scans(self):
        g,q=two_hop();scans=[]
        candidates=llm.candidate_actions(g,q,account=scans.append)
        self.assertEqual(scans,[1,1]);self.assertEqual(len(candidates),2)
        with patch.object(g,'successors',side_effect=AssertionError('unaccounted scan')):
            for s,e in candidates:llm.prompt_for(g,q,e,s)

    def test_model_fact_evaluation_stays_independent_of_expert_decision(self):
        g,q=two_hop()
        with patch.object(llm,'local_predict',side_effect=[([.8,.1,.1],10,False),([.1,.9],7,False)]), \
             patch.object(llm,'local_runtime',return_value=(None,None)):
            result=llm.local_score(llm.prompt_for(g,q,g.edges['01']))
        self.assertFalse(result['legal']);self.assertAlmostEqual(result['score'],.85)
        self.assertEqual(result['processed_prompt_tokens'],17)


class ModelIdentityTests(TestCase):
    def test_tokenizer_and_config_mismatch_cannot_keep_pinned_model_label(self):
        for body in (None,b'altered'):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);root.joinpath('model.safetensors').write_bytes(b'weights')
                if body is not None:root.joinpath('tokenizer.json').write_bytes(body)
                with patch.dict(os.environ,{'LAB03_LOCAL_MODEL':directory}), \
                     patch.object(llm,'WEIGHTS_SHA256',hashlib.sha256(b'weights').hexdigest()), \
                     patch.object(llm,'MODEL_FILE_HASHES',{'tokenizer.json':hashlib.sha256(b'correct').hexdigest()}):
                    with self.assertRaisesRegex(ValueError,'固定Qwen版本'):llm.verified_local_source()

    def test_manifest_covers_all_runtime_components(self):
        self.assertEqual(set(llm.MODEL_FILE_HASHES),{'config.json','generation_config.json','tokenizer_config.json','tokenizer.json','vocab.json','merges.txt'})
        self.assertTrue(all(len(v)==64 for v in llm.MODEL_FILE_HASHES.values()))


@requires_artifacts('models.json','ga_history.json','manifest.json','llm_cache.json','llm_audit.json','llm_evidence_readout.json','llm_summary.json','runs.csv','reliability/confirmation.json','reliability/llm_confirmation.json','structure/protocol.json','structure/confirmation.json','structure/stage_two_baseline.zip')
class FrozenStructureArtifactTests(TestCase):
    def test_frozen_new_seed_matrix_is_complete_disjoint_and_honest(self):
        from experiments.lab03.structure import validate_frozen_strategy,FOLDER,read,METHODS
        from experiments.lab03.dataset import load_dataset
        protocol=validate_frozen_strategy();v=read(FOLDER/'confirmation.json')
        self.assertEqual(protocol['confirmation_plan']['seeds'],[1201,2027,4051])
        self.assertEqual(v['question_count'],216);self.assertEqual(v['paired_runs'],5184)
        self.assertEqual(set(r['method'] for r in v['rows']),set(METHODS))
        keys=[(r['query_id'],r['method'],r['budget']) for r in v['rows']]
        self.assertEqual(len(keys),len(set(keys)))
        old={r['graph']['id'] for r in load_dataset()}
        self.assertFalse(old.intersection(g['id'] for g in v['graphs']))
        for r in v['rows']:
            self.assertGreaterEqual(r['after_adjacency_edge_scans'],r['after']['total_edge_checks'])
            if r['method'] in ('bfs','astar_admissible','mcts'):self.assertEqual(r['before'],r['after'])

    def test_new_comparison_recomputes_and_retains_all_regressions(self):
        from experiments.lab03.structure import audit_structure,FOLDER,read
        v=audit_structure();self.assertEqual(v,read(FOLDER/'comparison.json'))
        self.assertEqual(v['pairs'],3888)
        self.assertEqual(sum(r['lost'] for r in v['summary']),len(v['regressions']))

    def test_llm_preparation_costs_and_model_scores_recompute(self):
        from experiments.lab03.dataset import ROOT
        from experiments.lab03.service import tasks,validate_replay
        caches=json.loads((ROOT/'outputs/lab03/llm_cache.json').read_text(encoding='utf-8'))
        runs=json.loads((ROOT/'outputs/lab03/llm_runs.json').read_text(encoding='utf-8'))
        summary=json.loads((ROOT/'outputs/lab03/llm_summary.json').read_text(encoding='utf-8'))
        for qid,cache in caches.items():
            graph,query,_=tasks()[qid];scans=[]
            llm.candidate_actions(graph,query,account=scans.append)
            self.assertEqual(cache['preparation_edge_scans'],sum(scans))
            self.assertEqual(cache['preparation_states'],len(scans))
            self.assertEqual(validate_replay(cache,graph,query),cache['scores'])
            for field,value in [('preparation_edge_scans',-1),('preparation_edge_scans',float(sum(scans))),('preparation_states',True)]:
                bad={**cache,field:value}
                with self.assertRaises(ValueError):validate_replay(bad,graph,query)
        self.assertEqual(summary['llm_preparation_edge_scans'],sum(r['llm_preparation_edge_scans'] for r in runs))
        self.assertEqual(summary['llm_preparation_states'],sum(r['llm_preparation_states'] for r in runs))


@requires_artifacts('models.json','ga_history.json','manifest.json','llm_cache.json','llm_audit.json','llm_evidence_readout.json','llm_summary.json','runs.csv','reliability/confirmation.json','reliability/llm_confirmation.json','structure/protocol.json','structure/confirmation.json','structure/stage_two_baseline.zip')
class PosthocFailureTests(TestCase):
    def test_all_selected_failures_and_prefix_evidence_replay(self):
        from experiments.lab03.posthoc import analyze_structural_failures
        from experiments.lab03.structure import FOLDER,read
        v=analyze_structural_failures(write_result=False)
        self.assertEqual(v,read(FOLDER/'failure_analysis.json'))
        matrix=read(FOLDER/'confirmation.json')
        expected={(r['query_id'],r['method']) for r in matrix['rows'] if r['budget']==8 and r['method'] in ('astar_fixed','hybrid') and not r['after']['valid']}
        self.assertEqual({(r['query_id'],r['method']) for r in v['cases']},expected)
        self.assertEqual(len(v['cases']),len(expected))
        for r in v['cases']:
            self.assertFalse(r['budget8']['valid'])
            self.assertEqual(len(r['expanded_prefixes']),r['budget8']['expanded'])
