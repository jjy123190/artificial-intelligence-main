"""Exact-file and live/offline invariants for the optional wide-view release."""
from __future__ import annotations
from argparse import Namespace
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .artifact_audit import audit_publication, clear_audit_cache
from .data import digest_file, normalized_tensor, sample_episode
from .experiment import load_checkpoint
from .graph_calibration import GraphConfig, apply_transform, predict
from .inference import run_episode, load_demo_pool, clear_cache
from .multiview import combine_views
from .wideview_audit import FILES, verify

ROOT=Path(__file__).resolve().parents[2]


def publication():
    for path in (ROOT/'outputs/lab04',ROOT/'.local/lab04-wideview-release-20261125'):
        if (path/'wideview_results.json').is_file():return path
    return None


class WideViewLiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory=publication()
        if cls.directory is None:raise unittest.SkipTest('Optional wide-view release not present')
        torch.set_num_threads(2)

    def tearDown(self):clear_cache()

    def test_two_view_live_matches_offline_for_both_shots_and_shuffled_queries(self):
        directory=self.directory;pool=load_demo_pool(directory);network,_=load_checkpoint(directory/'wide.pt');network.eval()
        configuration=json.loads((directory/'wideview_configuration.json').read_text(encoding='utf-8'))
        for shot in (1,5):
            task=sample_episode(pool['labels'],np.random.default_rng(20261124),shots=shot)
            ids=np.concatenate((task['support_indices'],task['query_indices']))
            with torch.inference_mode():
                tensor=normalized_tensor(pool['images'][ids]);original=network.features(tensor).numpy();flipped=network.features(tensor.flip(-1)).numpy()
            selected=configuration['selected'][str(shot)]
            with np.load(directory/'wideview_transform.npz',allow_pickle=False) as source:
                prefix=f'shot{shot}_';transform={k[len(prefix):]:source[k] for k in source.files if k.startswith(prefix)}
            embedded=apply_transform(combine_views(original,flipped,selected['views']),transform);n=5*shot
            config=GraphConfig(**selected['config']);offline=predict(embedded[:n],task['support_labels'],embedded[n:],config)
            live=run_episode(directory,shots=shot,seed=20261124,method='calibrated',calibration='wideview')
            self.assertEqual(live['metric'],'wider_backbone_multiview')
            self.assertEqual(live['config']['encodings_per_image'],2)
            self.assertFalse(live['config']['additional_training_vs_wide'])
            self.assertEqual(live['checkpoint_sha256'],digest_file(directory/'wide.pt'))
            self.assertEqual([q['prediction'] for q in live['query']],offline['predictions'].tolist())
            np.testing.assert_allclose([q['probabilities'] for q in live['query']],offline['probabilities'],atol=2e-6,rtol=1e-5)
            order=np.random.default_rng(9).permutation(75)
            reordered=predict(embedded[:n],task['support_labels'],embedded[n:][order],config)
            np.testing.assert_allclose(reordered['probabilities'],offline['probabilities'][order],atol=1e-10)
            self.assertTrue(all(q['feature_before']==q['feature_after'] for q in live['query']))

    def test_website_scores_pair_with_all_original_methods(self):
        from backend.lab_four import LabFourService
        service=LabFourService(self.directory);snapshot=service.snapshot()
        self.assertTrue(snapshot['wideview_extension']['available'])
        expected=json.loads((self.directory/'wideview_results.json').read_text(encoding='utf-8'))
        for row in expected['rows']:
            rendered=next(r for r in snapshot['comparisons'] if r['method']=='wideview' and r['shot']==row['shots'])
            for key in ('accuracy','ci95','paired_vs_wide','paired_vs_wide_ci95'):self.assertEqual(rendered[key],row[key])
        a=service.run({'method':'calibrated','calibration':'wideview','shots':1,'seed':1})
        b=service.run({'method':'calibrated','calibration':'wide','shots':1,'seed':1})
        for key in ('artifact_revision','pool_hash','checkpoint_sha256'):self.assertEqual(a[key],b[key])
        self.assertEqual([q['id'] for q in a['queries']],[q['id'] for q in b['queries']])

    def test_modified_summary_predictions_and_configuration_fail(self):
        # Only copy the six candidate files; core references remain read-only.
        directory=self.directory
        with tempfile.TemporaryDirectory(prefix='lab04_wideview_audit_') as name:
            target=Path(name).resolve();self.assertEqual(target.parent,Path(tempfile.gettempdir()).resolve())
            for filename in (*FILES,'wideview_results.json'):shutil.copy2(directory/filename,target/filename)
            self.assertTrue(verify(target,directory)['verified'])
            manifest=target/'wideview_results.json';original_manifest=manifest.read_bytes();value=json.loads(original_manifest)
            value['rows'][0]['accuracy']+=.01;manifest.write_text(json.dumps(value),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Measured evidence mismatch'):verify(target,directory)
            manifest.write_bytes(original_manifest)
            episodes=target/'wideview_episodes.json';original_episodes=episodes.read_bytes();value=json.loads(original_episodes)
            value['episodes'][0]['predictions'][0]=(value['episodes'][0]['predictions'][0]+1)%5
            episodes.write_text(json.dumps(value),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):verify(target,directory)
            manifest_value=json.loads(original_manifest);manifest_value['files']['wideview_episodes.json']['sha256']=digest_file(episodes)
            manifest.write_text(json.dumps(manifest_value),encoding='utf-8')
            with self.assertRaises(ValueError):verify(target,directory)
            episodes.write_bytes(original_episodes);manifest.write_bytes(original_manifest)
            config=target/'wideview_configuration.json';value=json.loads(config.read_text(encoding='utf-8'));value['selected']['1']['views']='concat'
            config.write_text(json.dumps(value),encoding='utf-8');manifest_value=json.loads(original_manifest);manifest_value['files']['wideview_configuration.json']['sha256']=digest_file(config)
            manifest.write_text(json.dumps(manifest_value),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'validated config'):verify(target,directory)


class WideViewSelectionBoundaryTests(unittest.TestCase):
    def test_incomplete_selection_never_loads_test(self):
        from .evaluate_wide_views import run
        with tempfile.TemporaryDirectory(prefix='lab04_wideview_boundary_') as name:
            root=Path(name);source=root/'selection';source.mkdir()
            for file,value in (('selection_lock.json',{'status':'selecting','test_accessed':False}),('selection.json',{'status':'selecting'}),('selection_plan.json',{})):
                (source/file).write_text(json.dumps(value),encoding='utf-8')
            args=Namespace(candidate=str(source),artifacts=str(root/'core'),output=str(root/'out'),data_root='not-accessed',threads=1)
            with patch('experiments.lab04.evaluate_wide_views.load_split') as load:
                with self.assertRaisesRegex(ValueError,'validation lock'):run(args)
                load.assert_not_called()


if __name__=='__main__':unittest.main()
