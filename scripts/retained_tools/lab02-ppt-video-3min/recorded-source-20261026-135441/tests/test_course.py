"""Unified course adapters: original-version semantics, input guards and safe files."""
from __future__ import annotations
import copy
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from fastapi.testclient import TestClient
from backend.api import create_app
from backend.course import FILES, catalog, file_path
from backend.schemas import LabTwoTask
from backend.worker import preset_case
from experiments.lab01.service import read_dataset, run_task as one
from experiments.lab02.service import run_task as two
ROOT=Path(__file__).resolve().parents[1]

class CourseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client=TestClient(create_app())
        cls.snapshot=one({'task':'snapshot'})
        cls.original=one({'task':'rules','source':'P1','target':'C3'})
        cls.payload=LabTwoTask(case=preset_case('base'),task='deterministic').model_dump()

    def test_catalog_has_requirement_entries_and_code_positions(self):
        value=self.client.get('/api/course').json()
        self.assertEqual([l['id'] for l in value['labs']],['lab01','lab02'])
        self.assertEqual([len(l['modules']) for l in value['labs']],[4,5])
        for lab in value['labs']:
            self.assertTrue((ROOT/lab['code']).is_dir())
            self.assertTrue(all(m['requirement'] and m['description'] for m in lab['modules']))

    def test_catalog_modules_are_experiment_tasks_not_submission_checklists(self):
        value=self.client.get('/api/course').json()
        expected={
            'lab01':['representation','industry','demo','limitations'],
            'lab02':['modeling','deterministic','uncertain','optimization','hybrid'],
        }
        for lab in value['labs']:
            self.assertEqual([module['id'] for module in lab['modules']],expected[lab['id']])
            for module in lab['modules']:
                self.assertNotIn('成果',module['title'])
                self.assertNotIn('提交',module['title'])

    def test_raw_case_validation_has_no_inference_or_result_fields(self):
        source={'case':preset_case('base'),'completeness':{'holdings':True,'trades':True,'guarantees':True,'registry':False}}
        with patch.object(self.client.app.state.inference,'analyze',side_effect=AssertionError('Raw validation must not infer')) as infer:
            response=self.client.post('/api/cases/validate',json=source)
            self.assertEqual(response.status_code,200,response.text)
            infer.assert_not_called()
        value=response.json()
        self.assertEqual(set(value),{'id','case','completeness'})
        self.assertEqual(value['case'],source['case'])
        self.assertFalse(value['completeness']['registry'])
        for key in ('decisions','provenance','score','derived_edges','parameters'):
            self.assertNotIn(key,value)

    def test_raw_case_validation_rejects_bad_graph_and_experiment_arguments(self):
        source={'case':preset_case('base')}
        invalid=copy.deepcopy(source);invalid['case']['holdings'][0]['target']='MISSING'
        duplicate=copy.deepcopy(source);duplicate['case']['entities'].append(duplicate['case']['entities'][0])
        for payload in (invalid,duplicate,{**source,'method':'hybrid_locked'},{**source,'label':1},{**source,'completeness':{'holdings':'false'}}):
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post('/api/cases/validate',json=payload).status_code,422)
        self.assertEqual(self.client.post('/api/cases/validate',content='{}',headers={'content-type':'text/plain'}).status_code,415)

    def test_raw_case_validation_does_not_write_source_data(self):
        before={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in (ROOT/'data/raw').glob('*.csv')}
        changed=preset_case('base');changed['holdings'][0]['ratio']=.01
        response=self.client.post('/api/cases/validate',json={'case':changed})
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(before,{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in (ROOT/'data/raw').glob('*.csv')})

    def test_lab01_snapshot_exposes_original_guard_conditions(self):
        by_name={rule['name']:rule for rule in self.snapshot['rules']}
        self.assertTrue(all(rule['guard_complete'] for rule in self.snapshot['rules']))
        self.assertEqual(by_name['R1']['guard_conditions'],['?r > 0.5'])
        self.assertEqual(by_name['R2']['guard_conditions'],['?rel = "配偶"'])
        self.assertEqual(by_name['R3']['guard_conditions'],['?r1 ≥ ?r2','(?r1 + ?r2) > 0.5'])
        self.assertEqual(by_name['R6']['guard_conditions'],[])

    def test_lab01_preserves_original_rules_not_lab02_rules(self):
        self.assertEqual(self.snapshot['stats']['rules'],15)
        self.assertEqual(self.original['stats'],dict(entities=16,raw_relations=25,input_facts=53,rules=15,cycles=17,total_facts=135,added_facts=82,added_relations=9))
        self.assertTrue(self.original['proved'])
        self.assertEqual(len(self.client.get('/api/rules').json()['items']),27)

    def test_knowledge_enhancement_never_changes_input_csv(self):
        before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data/raw').glob('*.csv')}
        value=one({'task':'rules','source':'P3','target':'C3'})
        self.assertFalse(value['proved'])
        self.assertEqual(value['proof'],[])
        self.assertEqual(self.snapshot['stats']['raw_relations'],25)
        self.assertEqual(before,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data/raw').glob('*.csv')})

    def test_frame_defaults_and_override_are_distinguished(self):
        frame=next(f for f in self.original['frames_before'] if f['id']=='C3')
        self.assertEqual(next(s for s in frame['slots'] if s['name']=='实际控制人')['value'],'未识别')
        changed=next(f for f in self.original['frames_after'] if f['id']=='C3')
        slot=next(s for s in changed['slots'] if s['name']=='实际控制人')
        self.assertEqual(slot['value'],'王建国');self.assertFalse(slot['inherited'])

    def test_lab01_simple_path_equity_and_cutoff(self):
        value=one({'task':'penetration','target':'C3','max_depth':10})
        owner=next(o for o in value['owners'] if o['owner']=='P1')
        self.assertAlmostEqual(owner['ratio'],.342)
        cut=one({'task':'penetration','target':'C3','max_depth':1})
        self.assertFalse(any(o['beneficial'] for o in cut['owners']))
        self.assertEqual(set(value['holding_cycles'][0]),{'C6','C7','C8'})

    def test_shortest_relation_path_preserves_arrows_and_same_endpoint(self):
        value=one({'task':'path','source':'C3','target':'C9'})
        self.assertTrue(value['path']);self.assertEqual(value['path'][0][0],'C3');self.assertEqual(value['path'][-1][2],'C9')
        self.assertTrue(all(s[3] in ('→','←') for s in value['path']))
        self.assertEqual(one({'task':'path','source':'P1','target':'P1'})['path'],[])

    def test_source_dataset_is_same_case(self):
        data=read_dataset();case=preset_case('base')
        self.assertEqual(len(data['persons'])+len(data['companies']),len(case['entities']))
        self.assertEqual(data['holdings'],[(h['holder'],h['target'],h['ratio']) for h in case['holdings']])

    def test_modeling_preserves_formula_and_fact_sources(self):
        result=two({**self.payload,'task':'modeling'})
        self.assertIn('Holds',result['fact_counts'])
        self.assertTrue(all(f['source'] and f['formula'] for f in result['facts']))
        self.assertTrue(result['derived_facts'])

    def test_deterministic_forward_backward_have_same_goal_and_proof(self):
        result=two(self.payload)
        self.assertTrue(result['proved']);self.assertTrue(result['equivalent'])
        self.assertTrue(result['forward_proof']);self.assertTrue(result['backward_proof'])
        negative=two({**self.payload,'source':'P3'})
        self.assertFalse(negative['proved']);self.assertEqual(negative['forward_proof'],[])

    def test_uncertain_scores_match_existing_models_and_keep_unknown(self):
        from backend.worker import load_model
        from experiments.lab02.hybrid.pipeline import prepare_case
        from core.kb.graph_io import case_from_dict
        result=two({**self.payload,'task':'uncertain'})
        context=prepare_case(case_from_dict(self.payload['case']))
        model=load_model('manual')
        self.assertAlmostEqual(result['bayes']['score'],model.bayes.predict(context.observations['C3']))
        self.assertAlmostEqual(result['fuzzy']['score'],model.fuzzy.predict(context.observations['C3']))
        partial=two({**self.payload,'task':'uncertain','completeness':dict(holdings=True,trades=True,guarantees=True,registry=False)})
        self.assertIsNone(partial['observation']['values']['opacity'])

    def test_hybrid_does_not_change_score_or_discard_proofs(self):
        result=two({**self.payload,'task':'hybrid'})
        self.assertEqual(len(result['decisions']),3)
        a,b,c=result['decisions'];self.assertEqual(a['score'],b['score']);self.assertEqual(b['score'],c['score'])
        self.assertFalse(a['review_required']);self.assertTrue(b['review_required'])
        self.assertTrue(all(d['proofs'] for d in result['decisions']))

    def test_resolution_is_independent_and_rejects_missing_case_input(self):
        result=two({**self.payload,'task':'resolution'})
        self.assertTrue(result['proved']);self.assertEqual(result['status'],'proved');self.assertTrue(result['proof'])
        changed=copy.deepcopy(self.payload);changed['task']='resolution';changed['case']['holdings']=[h for h in changed['case']['holdings'] if (h['holder'],h['target'])!=('C2','C3')]
        with self.assertRaises(ValueError):two(changed)

    def test_input_guards_reject_paths_labels_unknown_tasks_and_large_depth(self):
        for payload in ({'task':'python'},{'task':'rules','path':'C:/secret'},{'task':'penetration','max_depth':99},{'task':'rules','source':True}):
            self.assertEqual(self.client.post('/api/labs/lab01/run',json=payload).status_code,422)
        bad={**self.payload,'label':1}
        self.assertEqual(self.client.post('/api/labs/lab02/run',json=bad).status_code,422)
        self.assertEqual(self.client.post('/api/labs/lab02/run',json={**self.payload,'model_source':'../../secret'}).status_code,422)

    def test_file_allowlist_and_bytes_match_existing_deliverables(self):
        path=file_path('lab01','report');response=self.client.get('/api/course/files/lab01/report')
        self.assertEqual(response.status_code,200);self.assertEqual(response.content,path.read_bytes())
        for url in ('/api/course/files/lab99/report','/api/course/files/lab01/unknown','/api/course/files/lab02/report','/api/course/files/lab01/%2e%2e%2fmain.py'):
            self.assertEqual(self.client.get(url).status_code,404)
        self.assertIn('attachment',response.headers['content-disposition'])

    def test_lab01_catalog_lists_only_existing_files(self):
        lab=next(l for l in catalog()['labs'] if l['id']=='lab01')
        expected={identifier for identifier,(_,relative,_) in FILES['lab01'].items() if (ROOT/relative).is_file()}
        self.assertEqual({file['id'] for file in lab['files']},expected)
        self.assertTrue(all(file['available'] and (ROOT/file['path']).is_file() for file in lab['files']))

    def test_lab01_deleted_file_is_removed_without_placeholder(self):
        with tempfile.TemporaryDirectory(prefix='equity_materials_tests_') as directory:
            root=Path(directory).resolve()
            self.assertEqual(root.parent,Path(tempfile.gettempdir()).resolve())
            self.assertTrue(root.name.startswith('equity_materials_tests_'))
            kept=root/FILES['lab01']['report'][1]
            removed=root/FILES['lab01']['presentation-audio'][1]
            kept.parent.mkdir(parents=True,exist_ok=True)
            kept.write_bytes(b'retained report')
            removed.write_bytes(b'temporary audio')
            with patch('backend.course.ROOT',root):
                before=next(l for l in catalog()['labs'] if l['id']=='lab01')
                self.assertEqual({file['id'] for file in before['files']},{'report','presentation-audio'})
                self.assertTrue(removed.resolve().is_relative_to(root))
                removed.unlink()
                after=next(l for l in catalog()['labs'] if l['id']=='lab01')
                self.assertEqual([file['id'] for file in after['files']],['report'])
                with self.assertRaises(FileNotFoundError):file_path('lab01','presentation-audio')
                lab02=next(l for l in catalog()['labs'] if l['id']=='lab02')
                self.assertIn('report',{file['id'] for file in lab02['files']})
                self.assertFalse(next(file for file in lab02['files'] if file['id']=='report')['available'])

    def test_lab01_empty_retained_inventory_is_allowed(self):
        with tempfile.TemporaryDirectory(prefix='equity_materials_tests_') as directory:
            with patch('backend.course.ROOT',Path(directory).resolve()):
                lab=next(l for l in catalog()['labs'] if l['id']=='lab01')
                self.assertEqual(lab['files'],[])

    def test_lab02_materials_dont_claim_report_and_slides_are_done(self):
        lab=next(l for l in catalog()['labs'] if l['id']=='lab02')
        self.assertFalse(next(f for f in lab['files'] if f['id']=='report')['available'])
        self.assertFalse(next(f for f in lab['files'] if f['id']=='slides')['available'])
        self.assertTrue(next(f for f in lab['files'] if f['id']=='methods')['available'])

    def test_real_bounded_lab01_endpoint(self):
        response=self.client.post('/api/labs/lab01/run',json={'task':'rules','source':'P1','target':'C3'})
        self.assertEqual(response.status_code,200,response.text);self.assertTrue(response.json()['proved'])

    def test_real_bounded_lab02_endpoint(self):
        response=self.client.post('/api/labs/lab02/run',json={**self.payload,'task':'uncertain'})
        self.assertEqual(response.status_code,200,response.text)
        self.assertGreater(response.json()['fuzzy']['score'],0)

    def test_experiment_data_has_all_optimization_groups(self):
        result=self.client.get('/api/experiments').json()
        for key in ('summary','indicator_summary','sensitivity','goal_slicing','resolution','conflict_summary'):
            self.assertTrue(result[key],key)

if __name__=='__main__':unittest.main()
