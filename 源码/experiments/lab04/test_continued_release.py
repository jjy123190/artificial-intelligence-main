"""Continued-model release, consistency, and fail-closed test boundary checks."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from argparse import Namespace
import numpy as np
import torch
from .data import normalized_tensor,sample_episode,digest_file
from .experiment import load_checkpoint
from .inference import run_episode,load_demo_pool,clear_cache
from .continued_extension import inference_inputs
from .graph_calibration import apply_transform,predict
from .multiview import combine_views
from .continued_audit import FILES,verify

ROOT=Path(__file__).resolve().parents[2]


class ContinuedReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory=next((p for p in (ROOT/'outputs/lab04',ROOT/'.local/lab04-continued-release-20261125') if (p/'continued_results.json').is_file()),None)
        if cls.directory is None:raise unittest.SkipTest('Optional continued release not present')
        torch.set_num_threads(2)

    def tearDown(self):clear_cache()

    def test_live_outputs_match_fixed_offline_features_and_same_capacity(self):
        directory=self.directory;pool=load_demo_pool(directory)
        for shot in (1,5):
            model,transform,config,views,metadata=inference_inputs(directory,shot)
            task=sample_episode(pool['labels'],np.random.default_rng(20261124),shots=shot)
            indices=np.concatenate((task['support_indices'],task['query_indices']));n=5*shot
            with torch.inference_mode():
                x=normalized_tensor(pool['images'][indices]);normal=model.features(x).numpy();flipped=model.features(x.flip(-1)).numpy()
            features=apply_transform(combine_views(normal,flipped,views),transform)
            expected=predict(features[:n],task['support_labels'],features[n:],config)
            actual=run_episode(directory,shots=shot,seed=20261124,method='calibrated',calibration='continued')
            self.assertEqual(actual['metric'],'continued_backbone_multiview')
            self.assertEqual(actual['config']['additional_epochs'],16);self.assertEqual(actual['config']['selected_additional_epoch'],12)
            self.assertTrue(actual['config']['same_capacity_as_wide']);self.assertTrue(actual['config']['additional_training_vs_wide'])
            self.assertEqual(actual['checkpoint_sha256'],digest_file(directory/'continued.pt'))
            self.assertNotEqual(actual['checkpoint_sha256'],digest_file(directory/'wide.pt'))
            self.assertEqual(metadata['model']['parameters'],509856)
            self.assertEqual([q['prediction'] for q in actual['query']],expected['predictions'].tolist())
            np.testing.assert_allclose([q['probabilities'] for q in actual['query']],expected['probabilities'],atol=3e-6,rtol=1e-5)
            previous=run_episode(directory,shots=shot,seed=20261124,method='calibrated',calibration='wideview')
            self.assertEqual([q['id'] for q in actual['query']],[q['id'] for q in previous['query']])
            self.assertEqual(actual['artifact_revision'],previous['artifact_revision'])

    def test_api_snapshot_and_accounting_preserve_old_results(self):
        from backend.lab_four import LabFourService
        snapshot=LabFourService(self.directory).snapshot()
        self.assertTrue(snapshot['continued_extension']['available'])
        self.assertEqual(snapshot['continued_extension']['training']['additional_epochs'],16)
        self.assertEqual(snapshot['verification']['extension_tasks'],6000)
        expected=json.loads((self.directory/'continued_results.json').read_text(encoding='utf-8'))
        for row in expected['rows']:
            rendered=next(r for r in snapshot['comparisons'] if r['method']=='continued' and r['shot']==row['shots'])
            for key in ('accuracy','ci95','paired_vs_wideview','paired_vs_wideview_ci95'):self.assertEqual(rendered[key],row[key])
        self.assertEqual(len([r for r in snapshot['comparisons'] if r['method']=='wideview']),2)

    def test_tampering_even_with_updated_hash_does_not_pass_scoring_audit(self):
        original=self.directory
        with tempfile.TemporaryDirectory(prefix='lab04_continued_audit_') as name:
            target=Path(name).resolve();self.assertEqual(target.parent,Path(tempfile.gettempdir()).resolve())
            for file in (*FILES,'continued_results.json'):shutil.copy2(original/file,target/file)
            self.assertTrue(verify(target,original)['verified'])
            manifest=target/'continued_results.json';raw=manifest.read_bytes();data=json.loads(raw)
            data['rows'][0]['accuracy']+=.01;manifest.write_text(json.dumps(data),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'evidence mismatch'):verify(target,original)
            manifest.write_bytes(raw)
            predictions=target/'continued_episodes.json';v=json.loads(predictions.read_text(encoding='utf-8'));v['episodes'][0]['predictions'][0]=(v['episodes'][0]['predictions'][0]+1)%5
            predictions.write_text(json.dumps(v),encoding='utf-8');data=json.loads(raw);data['files']['continued_episodes.json']['sha256']=digest_file(predictions);manifest.write_text(json.dumps(data),encoding='utf-8')
            with self.assertRaises(ValueError):verify(target,original)


class ContinuedEvaluationGateTests(unittest.TestCase):
    def test_failing_validation_cannot_open_test_images_or_create_outputs(self):
        from .evaluate_continuation import run
        with tempfile.TemporaryDirectory(prefix='lab04_continued_gate_') as name:
            root=Path(name);args=Namespace(candidate=str(root/'candidate'),validation=str(root/'validation'),artifacts=str(root/'core'),output=str(root/'out'))
            with patch('experiments.lab04.evaluate_continuation.audit_validation',return_value={'test_eligible':False}),patch('experiments.lab04.evaluate_continuation.load_split') as load:
                with self.assertRaisesRegex(ValueError,'gate did not pass'):run(args)
                load.assert_not_called();self.assertFalse((root/'out').exists())


if __name__=='__main__':unittest.main()
