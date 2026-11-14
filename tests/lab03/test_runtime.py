"""Runtime corruption injection tests; never rewrite frozen experimental results."""
from tests.lab03.support import requires_artifacts
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import tempfile
import threading
from unittest import TestCase
from unittest.mock import patch

from experiments.lab03 import service, calibration, llm, runtime_integrity as integrity


@requires_artifacts('models.json','ga_history.json','manifest.json','llm_cache.json','llm_audit.json','llm_evidence_readout.json')
class RuntimeReplayTests(TestCase):
    def setUp(self):
        self.graph,self.query,_=service.tasks()['test-7-0-Q0']
        self.cache=copy.deepcopy(service.read('llm_cache.json')[self.query.id])

    def test_finite_changed_cache_score_and_cost_are_not_real_model_evidence(self):
        for field in ('scores','llm_ms'):
            value=copy.deepcopy(self.cache)
            if field=='scores':value[field][next(iter(value[field]))]=.123456
            else:value[field]+=1
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'原始推断'):
                service.validate_replay(value,self.graph,self.query)

    def test_modified_audit_score_is_checked_against_original_probabilities(self):
        original=service.read
        def corrupt(name,default=None):
            value=original(name,default)
            if name=='llm_audit.json':
                value=copy.deepcopy(value);value[0]['evaluations'][0]['score']=.123456
            return value
        with patch.object(service,'read',side_effect=corrupt),self.assertRaisesRegex(ValueError,'概率'):
            service.validate_replay(self.cache,self.graph,self.query)

    def test_missing_duplicate_and_nonobject_audits_block_replay(self):
        original=service.read
        for variant in (None,[],[None],service.read('llm_audit.json')[:1]*2):
            def corrupt(name,default=None):return variant if name=='llm_audit.json' else original(name,default)
            with self.subTest(variant=str(variant)[:40]),patch.object(service,'read',side_effect=corrupt),self.assertRaises(ValueError):
                service.validate_replay(self.cache,self.graph,self.query)

    def test_changed_hidden_state_or_boolean_cannot_masquerade_as_probability(self):
        original=service.read
        for change in ('hidden','boolean','identity'):
            def corrupt(name,default=None):
                value=original(name,default)
                if name=='llm_audit.json':
                    value=copy.deepcopy(value);row=value[0]['evaluations'][0]
                    if change=='hidden':row['evidence_hidden']=[0.]*896
                    elif change=='boolean':row['calibrated_confidence']=True
                    else:row['model']='another-model'
                return value
            with self.subTest(change=change),patch.object(service,'read',side_effect=corrupt),self.assertRaises(ValueError):
                service.validate_replay(self.cache,self.graph,self.query)


@requires_artifacts('models.json','ga_history.json','manifest.json','llm_cache.json','llm_audit.json','llm_evidence_readout.json','llm_summary.json','runs.csv','reliability/confirmation.json','reliability/llm_confirmation.json')
class RuntimeDisplayTests(TestCase):
    @classmethod
    def setUpClass(cls):
        integrity.clear_verified_inputs()
        # Corruption tests deliberately start after a valid warm cache hit.
        cls.reference = service.snapshot()
        assert not cls.reference['warnings']

    def corrupt(self,name,mutate):
        original=service.read
        def load(filename,default=None):
            value=original(filename,default)
            return mutate(copy.deepcopy(value)) if filename==name else value
        return patch.object(service,'read',side_effect=load)

    def test_valid_snapshot_replays_without_loading_or_inferencing_qwen(self):
        with patch.object(llm,'local_runtime',side_effect=AssertionError('runtime verification is not model inference')):
            value=service.snapshot()
        self.assertEqual(value['warnings'],[])
        self.assertTrue(value['llm_summary']);self.assertTrue(value['llm_confirmation'])
        self.assertEqual(value['llm_confirmation']['numeric_calibrated']['false_reject'],2)

    def test_summary_accuracy_and_paired_success_cannot_be_edited_independently(self):
        def mutate(v):v['legal_check_accuracy']=1.;v['results'][1]['success_rate']=.5;return v
        with self.corrupt('llm_summary.json',mutate):value=service.snapshot()
        self.assertEqual(value['llm_summary'],{});self.assertTrue(value['warnings'])
        self.assertTrue(service.run_task({'method':'bfs','budget':128})['result']['valid'])

    def test_changed_search_run_is_not_hidden_behind_a_recomputed_summary(self):
        def mutate(v):v[0]['expanded']+=1;return v
        with self.corrupt('llm_runs.json',mutate):value=service.snapshot()
        self.assertEqual(value['llm_summary'],{});self.assertTrue(value['warnings'])

    def test_explanation_certification_and_facts_are_rechecked(self):
        for field,text in [('grounded',True),('controlled_text','图中不存在的事实')]:
            def mutate(v):v[0][field]=text;return v
            with self.subTest(field=field),self.corrupt('llm_explanations.json',mutate):value=service.snapshot()
            self.assertEqual(value['llm_summary'],{});self.assertEqual(value['llm_explanations'],[])
            self.assertTrue(value['warnings'])

    def test_fresh_search_summary_and_novelty_metrics_match_raw_records(self):
        for field in ('paired_summary','novel_raw_metrics','inference_ms'):
            def mutate(v):
                if field=='paired_summary':v[field][0]['expanded']+=1
                elif field=='novel_raw_metrics':v[field]['false_accept']=0
                else:v[field]+=1
                return v
            with self.subTest(field=field),self.corrupt('reliability/llm_confirmation.json',mutate):value=service.snapshot()
            self.assertEqual(value['llm_confirmation'],{});self.assertTrue(value['warnings'])

    def test_numeric_logits_and_hidden_predictions_are_independently_recomputed(self):
        for field in ('raw_probabilities','calibrated_probability','hidden'):
            def mutate(v):
                r=v['numeric_challenge'][0]
                r[field]={'raw_probabilities':[0.,1.],'calibrated_probability':.1,'hidden':[0.]*896}[field]
                return v
            with self.subTest(field=field),self.corrupt('reliability/llm_confirmation.json',mutate):value=service.snapshot()
            self.assertEqual(value['llm_confirmation'],{});self.assertTrue(value['warnings'])

    def test_finite_changed_confirmation_aggregates_do_not_pass(self):
        for kind in ('summary','negative_summary','duplicate_row'):
            def mutate(v):
                if kind=='summary':v['summary'][0]['after']['success_rate']=.123
                elif kind=='negative_summary':v['negative_summary'][0]['after']['correct_exhausted']=0
                else:v['rows'][-1]=copy.deepcopy(v['rows'][0])
                return v
            with self.subTest(kind=kind),self.corrupt('reliability/confirmation.json',mutate):value=service.snapshot()
            self.assertEqual(value['confirmation'],{});self.assertTrue(value['warnings'])

    def test_original_csv_summary_and_graph_intervals_cannot_drift(self):
        for name in ('summary.json','paired_effects.json','failure_cases.json'):
            def mutate(v):
                if name=='summary.json':v[0]['success_rate']=.123456
                elif name=='paired_effects.json':v[0]['graph_bootstrap_low']=.123456
                else:v[0]['result']['expanded']+=1
                return v
            with self.subTest(name=name),self.corrupt(name,mutate):value=service.snapshot()
            self.assertEqual(value['summary'],[]);self.assertEqual(value['paired'],[])
            self.assertTrue(value['warnings'])

    def test_graph_cluster_bootstrap_matches_original_frozen_implementation(self):
        from experiments.lab03.verification import converted_rows
        from experiments.lab03.evaluate import paired_effects
        rows=converted_rows();computed=paired_effects(rows)
        self.assertEqual(computed,service.read('paired_effects.json'))
        integrity.benchmark_tables((service.OUT/'runs.csv').read_text(encoding='utf-8-sig'),
            service.read('summary.json'),computed,service.read('failure_cases.json'),service.read('manifest.json'))

    def test_malformed_fresh_structures_return_unavailable_not_http500(self):
        def mutate(v):v['audit']=[None]*18;return v
        with self.corrupt('reliability/llm_confirmation.json',mutate):value=service.snapshot()
        self.assertEqual(value['llm_confirmation'],{});self.assertTrue(value['warnings'])


class RequestScopeTests(TestCase):
    def test_readout_and_fingerprints_are_loaded_once_per_snapshot_not_once_forever(self):
        # Minimal metadata triggers the same request caches without requiring Qwen
        # weights, hidden states, old reports or saved experiment outputs.
        from tests.lab03.support import artifact_fixture
        fixtures={name:artifact_fixture(name) for name in ('models.json','ga_history.json','manifest.json')}
        def load(name,default=None):return copy.deepcopy(fixtures.get(name,default))
        with patch.object(service,'read',side_effect=load), \
             patch.object(calibration,'load_readout',side_effect=ValueError('declared missing readout')) as readout, \
             patch.object(service,'_artifact_source_hashes',wraps=service._artifact_source_hashes) as hashes, \
             patch.object(service,'_artifact_provenance',wraps=service._artifact_provenance) as source:
            service.snapshot()
            self.assertEqual(readout.call_count,1);self.assertEqual(hashes.call_count,1);self.assertEqual(source.call_count,1)
            service.snapshot()
            self.assertEqual(readout.call_count,2);self.assertEqual(hashes.call_count,2);self.assertEqual(source.call_count,2)

    def test_changed_file_is_observed_on_next_request_and_return_value_is_not_shared(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(service,'OUT',Path(temp)):
            p=Path(temp)/'x.json';p.write_text('{"value":1}',encoding='utf-8')
            @integrity.request_scoped
            def read():
                first=service.read('x.json');second=service.read('x.json')
                self.assertIs(first,second);return first
            old=read();old['value']=999;p.write_text('{"value":2}',encoding='utf-8')
            self.assertEqual(read(),{'value':2})

    def test_question_graphs_refresh_between_requests(self):
        original=service.load_dataset
        graph,query,_=service.tasks()['test-7-0-Q0'];start=query.start;old=graph.nodes[start].name
        def changed():
            value=copy.deepcopy(original())
            for record in value:
                if record['graph']['id']==graph.id:
                    for node in record['graph']['nodes']:
                        if node['id']==start:node['name']='Changed in current dataset'
            return value
        with patch.object(service,'load_dataset',side_effect=changed):
            updated,_,_=service.tasks()['test-7-0-Q0']
            self.assertEqual(updated.nodes[start].name,'Changed in current dataset')
        self.assertEqual(service.tasks()['test-7-0-Q0'][0].nodes[start].name,old)

    def test_nonfinite_or_duplicate_json_is_rejected_before_it_can_break_http(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(service,'OUT',Path(temp)):
            path=Path(temp)/'bad.json'
            for text in ('{"x":NaN}','{"x":Infinity}','{"x":-Infinity}','{"x":1,"x":2}'):
                path.write_text(text,encoding='utf-8')
                with self.subTest(text=text),self.assertRaisesRegex(ValueError,'损坏'):
                    service.read('bad.json')

    def test_error_cleanup_does_not_poison_later_request(self):
        calls=[]
        def bad():calls.append('bad');raise ValueError('missing')
        @integrity.request_scoped
        def fail():
            for _ in range(3):
                with self.assertRaises(ValueError):integrity.once('key',bad)
        fail();self.assertEqual(calls,['bad'])
        @integrity.request_scoped
        def success():return integrity.once('key',lambda:42)
        self.assertEqual(success(),42);self.assertIsNone(integrity._request_values.get())

    def test_simultaneous_requests_have_isolated_values(self):
        barrier=threading.Barrier(2)
        @integrity.request_scoped
        def run(value):
            first=integrity.once('same-key',lambda:value);barrier.wait(timeout=5)
            second=integrity.once('same-key',lambda:None);return first,second
        with ThreadPoolExecutor(max_workers=2) as pool:
            first=pool.submit(run,'a');second=pool.submit(run,'b')
            self.assertEqual(first.result(),('a','a'));self.assertEqual(second.result(),('b','b'))


class VerifiedTokenCacheTests(TestCase):
    def setUp(self):integrity.clear_verified_inputs()
    def tearDown(self):integrity.clear_verified_inputs()

    def test_caches_success_only_and_keys_every_current_value(self):
        calls=[]
        value={'scores':[.4,.5],'enabled':True}
        def check():calls.append('checked')
        integrity.verified_inputs('kind',[value],check)
        integrity.verified_inputs('kind',[value],check)
        self.assertEqual(calls,['checked'])
        value['scores'][0]=.6
        integrity.verified_inputs('kind',[value],check)
        self.assertEqual(calls,['checked','checked'])
        value['enabled']=1
        integrity.verified_inputs('kind',[value],check)
        self.assertEqual(len(calls),3)
        integrity.verified_inputs('another-kind',[value],check)
        self.assertEqual(len(calls),4)

    def test_failure_is_not_cached_or_converted_to_success(self):
        calls=[]
        def fail():calls.append(1);raise ValueError('corrupt')
        for _ in range(2):
            with self.assertRaises(ValueError):integrity.verified_inputs('bad',[{}],fail)
        self.assertEqual(len(calls),2);self.assertEqual(len(integrity._verified_inputs),0)

    def test_cache_is_bounded_and_contains_only_digest_verdicts(self):
        for i in range(50):integrity.verified_inputs('kind',[{'payload':[i]*500}],lambda:None)
        self.assertEqual(len(integrity._verified_inputs),integrity.VERDICT_CACHE_LIMIT)
        self.assertTrue(all(len(k[1])==64 and v is True for k,v in integrity._verified_inputs.items()))

    def test_checking_does_not_mutate_input_and_never_deserializes_binary(self):
        source={'nested':[{'x':True},.123456,None,'中文']};before=copy.deepcopy(source)
        with patch.object(integrity.marshal,'loads',side_effect=AssertionError('no binary input deserialization')):
            integrity.verified_inputs('builtins',[source],lambda:None)
        self.assertEqual(source,before)
