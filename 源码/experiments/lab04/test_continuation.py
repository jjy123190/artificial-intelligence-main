"""Boundary and deterministic-resume tests for bounded continuation research."""
from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

from .train_continuation import guard_output,read_batch,read_train_records,train
from .validate_continuation import extract


class TinyNetwork(nn.Module):
    widths=(16,32,64,128)
    def __init__(self):
        super().__init__();self.encoder=nn.Linear(3,8);self.classifier=nn.Linear(8,64)
    def forward(self,x,return_features=False):
        features=self.encoder(x.mean((-2,-1)));logits=self.classifier(features)
        return (logits,features) if return_features else logits


class ContinuationBoundaryTests(unittest.TestCase):
    def test_test_images_rejected_before_loading_pixels(self):
        with patch('PIL.Image.open',side_effect=AssertionError('Must reject before reading')):
            with self.assertRaisesRegex(ValueError,'Only train'):
                read_batch([('n1',Path('do-not-read'),'test/n1/image.jpg')],[0])
        with self.assertRaisesRegex(ValueError,'Only train/validation'):
            extract(Namespace(),'test')

    def test_training_output_cannot_replace_published_weights(self):
        core=Path(__file__).resolve().parents[2]/'outputs/lab04'
        with self.assertRaisesRegex(ValueError,'Never train'):
            guard_output(core,core/'wide.pt')
        with self.assertRaisesRegex(ValueError,'Never train'):
            guard_output(core/'nested',core/'wide.pt')

    def test_streamed_image_preprocessing_is_finite_and_ordered(self):
        with tempfile.TemporaryDirectory(prefix='lab04_stream_fixture_') as name:
            root=Path(name).resolve();self.assertEqual(root.parent,Path(tempfile.gettempdir()).resolve())
            paths=[]
            for index,color in enumerate((10,230)):
                file=root/f'{index}.png';Image.fromarray(np.full((84,84,3),color,dtype=np.uint8)).save(file);paths.append(('n1',file,f'train/n1/{index}.png'))
            actual=read_batch(paths,[1,0])
            self.assertEqual(actual.shape,(2,84,84,3));self.assertEqual(actual.dtype,np.uint8)
            self.assertTrue((actual[0]==230).all());self.assertTrue((actual[1]==10).all())

    def test_watcher_rejects_reused_pid_or_wrong_command(self):
        from .after_continuation import is_training_live
        from unittest.mock import Mock
        process=Mock();process.is_running.return_value=True;process.create_time.return_value=100.;process.cmdline.return_value=['python','-m','experiments.lab04.train_continuation']
        with patch('experiments.lab04.after_continuation.psutil.Process',return_value=process):
            self.assertTrue(is_training_live(9,100.))
            self.assertFalse(is_training_live(9,99.))
            process.cmdline.return_value=['python','unrelated.py']
            self.assertFalse(is_training_live(9,100.))

    def test_invalid_budget_rejected_before_files(self):
        with self.assertRaisesRegex(ValueError,'bounded'):
            train(Namespace(epochs=0))
        with self.assertRaisesRegex(ValueError,'bounded'):
            train(Namespace(epochs=True))

    def test_epoch_boundary_resume_matches_uninterrupted_optimization(self):
        with tempfile.TemporaryDirectory(prefix='lab04_continue_resume_') as name:
            root=Path(name).resolve();initial=root/'initial.pt';initial.write_bytes(b'fixture')
            labels=np.repeat(np.arange(16),4)
            images=np.random.default_rng(3).integers(0,256,(64,12,12,3),dtype=np.uint8)
            validation=Namespace(images=images,labels=labels,ids=[f'val/n{i}/sample' for i in range(64)],class_names=[f'val{i}' for i in range(16)])
            args=lambda folder:Namespace(output=str(root/folder),initial=str(initial),data_root='never-read',epochs=2,learning_rate=.0002,metric_weight=.25,threads=1,seed=41)
            records=[('base',Path('never-read'),f'train/base/{i}') for i in range(64)]
            def score(model,*_):return {'1shot':.4,'5shot':.6,'mean':.5}
            # The fake network makes this an optimizer/RNG test, not a benchmark.
            with patch('experiments.lab04.train_continuation.read_train_records',return_value=(records,labels,['base'])),                 patch('experiments.lab04.train_continuation.load_split',return_value=validation),                 patch('experiments.lab04.train_continuation.load_checkpoint',side_effect=lambda *_:(TinyNetwork(),{})),                 patch('experiments.lab04.train_continuation.score_validation',side_effect=score),                 patch('experiments.lab04.train_continuation.balanced_batches',side_effect=lambda *_:[np.arange(64)]),                 patch('experiments.lab04.train_continuation.tasks_for',return_value=[]),                 patch('experiments.lab04.train_continuation.log'):
                with patch('experiments.lab04.train_continuation.read_batch',return_value=images):train(args('full'))
                with patch('experiments.lab04.train_continuation.read_batch',side_effect=[images,RuntimeError('simulated interruption')]):
                    with self.assertRaisesRegex(RuntimeError,'simulated interruption'):train(args('resume'))
                status=json.loads((root/'resume/training.json').read_text())
                self.assertEqual(len(status['rows']),1);self.assertEqual(status['status'],'running')
                with patch('experiments.lab04.train_continuation.read_batch',return_value=images):train(args('resume'))
            full=torch.load(root/'full/resume.pt',weights_only=True,map_location='cpu');resumed=torch.load(root/'resume/resume.pt',weights_only=True,map_location='cpu')
            for key in full['state']:torch.testing.assert_close(full['state'][key],resumed['state'][key],atol=0,rtol=0)
            self.assertEqual(full['scheduler'],resumed['scheduler']);self.assertEqual(full['best_epoch'],0)
            torch.testing.assert_close(full['generator'],resumed['generator'],atol=0,rtol=0)
            torch.testing.assert_close(full['torch_rng'],resumed['torch_rng'],atol=0,rtol=0)
            for key in ('ce_loss','metric_loss','training_accuracy','validation','next_learning_rate'):
                self.assertEqual(full['history'][-1][key],resumed['history'][-1][key])
            complete=json.loads((root/'resume/training.json').read_text());self.assertEqual(complete['status'],'complete')
            with patch('experiments.lab04.train_continuation.read_train_records',side_effect=AssertionError('Complete run must not restart')):
                train(args('resume'))
            changed=args('resume');changed.epochs=3
            with self.assertRaisesRegex(ValueError,'Immutable continuation plan'):train(changed)


if __name__=='__main__':unittest.main()
