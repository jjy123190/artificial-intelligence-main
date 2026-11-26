import unittest
import numpy as np
import torch
from experiments.lab04.learned_calibration import (
 LearnedCalibration,LearnedCalibrationConfig,episode_objective,objective,encode_calibration)

class LearnedCalibrationTests(unittest.TestCase):
 def setUp(self):torch.manual_seed(51);torch.set_num_threads(1)
 def test_identity_initialization(self):
  for kind in ('linear','residual'):
   m=LearnedCalibration(LearnedCalibrationConfig(architecture=kind));x=torch.randn(12,128)
   torch.testing.assert_close(m(x),torch.nn.functional.normalize(x,dim=1))
 def test_gradient_reaches_adapter(self):
  m=LearnedCalibration();x=torch.nn.functional.normalize(torch.randn(64,128),dim=1);y=torch.arange(8).repeat_interleave(8)
  head=torch.randn(64,128,requires_grad=True)
  losses=objective(m,head,x,x,x,y,shots=5,generator=torch.Generator().manual_seed(2));losses['total'].backward()
  self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in m.parameters()))
  self.assertIsNotNone(head.grad);self.assertAlmostEqual(float(losses['alignment']),0.)
 def test_five_way_episode_shots(self):
  x=torch.eye(8).repeat_interleave(8,dim=0);labels=torch.arange(8).repeat_interleave(8)
  for shot in (1,5):self.assertLess(float(episode_objective(x,labels,shots=shot)),.01)
 def test_no_shared_prediction_batch_statistics(self):
  m=LearnedCalibration().eval();x=np.random.default_rng(5).normal(size=(31,128))
  a=encode_calibration(m,x,batch_size=31);b=encode_calibration(m,x,batch_size=1)
  np.testing.assert_allclose(a,b,atol=1e-6)
 def test_invalid_arguments(self):
  for values in ({'architecture':'other'},{'learning_rate':0},{'hidden':True},{'anchor_weight':float('nan')}):
   with self.assertRaises(ValueError):LearnedCalibrationConfig(**values)
  with self.assertRaises(ValueError):episode_objective(torch.ones(4,5),torch.zeros(4),shots=5)
  with self.assertRaises(ValueError):encode_calibration(LearnedCalibration(),np.ones((4,128)))
  with self.assertRaises(ValueError):encode_calibration(LearnedCalibration().eval(),np.ones((4,128))*np.nan)
