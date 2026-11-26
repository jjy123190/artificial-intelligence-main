"""Raw-score and registration boundary tests for the new offline study runners."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.lab04 import study_literature as study


class LiteratureStudyTests(unittest.TestCase):
    def record(self):
        tasks=[{'query_labels':[0,1,2,3,4]},{'query_labels':[0,1,2,3,4]}]
        record={'plan_sha256':'frozen','group':'search/1','candidate':'candidate',
                'predictions':[[0,1,2,3,4],[4,3,2,1,0]],'accuracies':[1.,.2]}
        return record,tasks

    def test_prediction_based_score_recalculation(self):
        r,tasks=self.record()
        self.assertEqual(study.validate_scores(r,'search/1','candidate',tasks,'frozen'),[1.,.2])

    def test_tampering_rejected(self):
        record,tasks=self.record()
        mutations=[{'plan_sha256':'another'},{'candidate':'other'},{'group':'test/1'},
                   {'accuracies':[1.,1.]},{'predictions':[[0,1,2,3,4]]},
                   {'predictions':[[0.,1.,2.,3.,4.],[4.,3.,2.,1.,0.]]},
                   {'predictions':[[0,1,2,3,9],[4,3,2,1,0]]}]
        for change in mutations:
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    study.validate_scores({**record,**change},'search/1','candidate',tasks,'frozen')

    def test_pairing_by_image_identity_not_position(self):
        pool={'ids':np.array(['val/a/1','val/a/2','val/b/1']), 'labels':np.array([0,0,1])}
        sub={'ids':pool['ids'][[2,0]],'labels':np.array([1,0])}
        np.testing.assert_array_equal(study.aligned_indices(pool,sub),[2,0])
        with self.assertRaises(ValueError):study.aligned_indices(pool,{**sub,'labels':np.array([0,1])})
        with self.assertRaises(ValueError):study.aligned_indices(pool,{**sub,'ids':np.array(['val/a/9','val/a/1'])})

    def test_pool_validation_rejects_test_and_duplicates(self):
        pool={'ids':np.array(['val/a/1','val/a/2','val/b/1','val/b/2']),
              'labels':np.array([0,0,1,1]),'features':np.ones((4,3))}
        study.validate_pool(pool,'val',2,2)
        with self.assertRaises(ValueError):study.validate_pool(pool,'train',2,2)
        with self.assertRaises(ValueError):study.validate_pool({**pool,'ids':np.array(['val/a/1']*4)},'val',2,2)
        with self.assertRaises(ValueError):study.validate_pool({**pool,'labels':np.array([0,1,0,1])},'val',2,2)

    def test_registration_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            args=SimpleNamespace(output=directory)
            inputs={'model':'original'};prov={'source':'author'};tasks={'groups':{}}
            first=study.register(args,inputs,prov,tasks)
            self.assertEqual(first,study.register(args,inputs,prov,tasks))
            with self.assertRaises(ValueError):study.register(args,{'model':'changed'},prov,tasks)
            with self.assertRaises(ValueError):study.register(args,inputs,prov,{'groups':{'extra':[]}})

    def test_query_truth_does_not_enter_prediction(self):
        task={'support_indices':[0,1,2,3,4],'query_indices':[5,6,7,8,9],
              'support_labels':[0,1,2,3,4],'query_labels':[0,1,2,3,4]}
        features=np.concatenate([np.eye(5),np.eye(5)])
        seen=[]
        def predict(s,y,q,cfg):
            seen.append((s.copy(),y.copy(),q.copy()))
            return {'predictions':np.arange(5)}
        records=[]
        for truth in ([0,1,2,3,4],[4,3,2,1,0]):
            with tempfile.TemporaryDirectory() as directory:
                (Path(directory)/'scores').mkdir()
                changed={**task,'query_labels':truth}
                with patch.object(study,'graph_predict',side_effect=predict):
                    args=SimpleNamespace(output=directory)
                    study.score(args,'locked','search/1',{'id':'test','algorithm':'graph'},features,[changed,changed],None)
                records.append(study.read(Path(directory)/'scores/search_1_test.json'))
        self.assertEqual(records[0]['predictions'],records[1]['predictions'])
        self.assertNotEqual(records[0]['accuracies'],records[1]['accuracies'])
        for a,b in zip(seen[0],seen[2]):np.testing.assert_array_equal(a,b)

    def test_tasks_disjoint_and_deterministic(self):
        labels=np.repeat(np.arange(16),600)
        pool={'val':{'ids':np.array([f'val/{i//600}/{i}' for i in range(9600)]),'labels':labels}}
        ext={'val':{'ids':np.array([f'val/{i//200}/{i}' for i in range(3200)]),'labels':np.repeat(np.arange(16),200)}}
        a=study.make_tasks(pool,ext);b=study.make_tasks(pool,ext)
        self.assertEqual(study.digest_json(a),study.digest_json(b))
        self.assertFalse(set(a['search_indices'])&set(a['confirmation_indices']))
        for group,tasks in a['groups'].items():
            for task in tasks:
                self.assertFalse(set(task['support_indices'])&set(task['query_indices']))
                self.assertEqual(np.bincount(task['query_labels']).tolist(),[15]*5)

    def test_no_test_split_argument(self):
        source=Path(study.__file__).read_text(encoding='utf-8')
        self.assertNotIn('test_features.npz',source)
        self.assertNotIn('test_views.npz',source)
        self.assertNotIn('query_labels,',source)
