"""Distillation execution safety tests; no real model training."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.lab04 import train_feature_distillation as runner


class FeatureDistillationRunnerTests(unittest.TestCase):
    def test_training_output_cannot_overwrite_publication(self):
        source=Path(runner.__file__).resolve().parents[2]
        for path in [source/'outputs/lab04',source,source/'.local',source/'experiments']:
            with self.assertRaises(ValueError):runner.safe_output(path)
        self.assertEqual(runner.safe_output(source/'.local/isolated-study'),source/'.local/isolated-study')

    def test_teacher_and_student_views_share_pixels(self):
        rgb=torch.rand(3,3,84,84)
        mean=torch.tensor([.485,.456,.406])[None,:,None,None];std=torch.tensor([.229,.224,.225])[None,:,None,None]
        student=(rgb-mean)/std
        teacher=runner.normalize_same_view(student)
        tm=torch.tensor([120.39586422,115.59361427,104.54012653])[None,:,None,None]/255
        ts=torch.tensor([70.68188272,68.27635443,72.54505529])[None,:,None,None]/255
        torch.testing.assert_close(teacher*ts+tm,rgb,atol=2e-7,rtol=1e-6)

    def test_test_data_read_denied(self):
        with self.assertRaises(ValueError):runner.pixels_for([],[],'test')
        with self.assertRaises(ValueError):runner.pixels_for([('a','ignored','test/a/image.jpg')],[0],'train')

    def test_default_register_does_not_load_models(self):
        args=SimpleNamespace(run=False)
        with tempfile.TemporaryDirectory() as d,patch.object(runner,'register',return_value=(Path(d),{},None,None,None,None)),patch.object(runner,'load_checkpoint') as model:
            runner.train(args)
            model.assert_not_called()

    def test_atomic_checkpoint_readable_safely(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'resume.pt';runner.atomic_torch(p,{'tensor':torch.ones(3),'metadata':{'test_accessed':False}})
            loaded=torch.load(p,weights_only=True,map_location='cpu')
            torch.testing.assert_close(loaded['tensor'],torch.ones(3))
            self.assertFalse(p.with_suffix('.tmp').exists())

    def test_training_record_identity_no_test_listing(self):
        def fake(root,split):
            self.assertIn(split,('train','val'))
            count=64 if split=='train' else 16
            return [(f'{split}-class{c:02d}',Path('dummy'),f'{split}/{split}-class{c:02d}/{i}.jpg') for c in range(count) for i in range(600)]
        with patch.object(runner,'image_records',side_effect=fake):
            records,labels,classes=runner.data_identity('dummy')
        self.assertEqual(len(records['train']),38400);self.assertEqual(len(records['val']),9600)
        self.assertFalse(set(classes['train'])&set(classes['val']))
