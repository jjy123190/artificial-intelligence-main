import copy,unittest
import torch
from experiments.lab04.models import ResNet12
from experiments.lab04.metric_training import widen_resnet12
from experiments.lab04.channel_diversity import diversify_fanout,pair_correlation_penalty,cosine_episode_loss

class ChannelDiversityTests(unittest.TestCase):
 def setUp(self):torch.manual_seed(42);torch.set_num_threads(1)
 def test_exact_duplicate_forward_preserved(self):
  original=ResNet12(widths=(4,8,12,16));wide=widen_resnet12(original);before=copy.deepcopy(wide)
  changed=diversify_fanout(wide,relative_scale=.5);self.assertTrue(changed)
  x=torch.randn(3,3,32,32)
  torch.testing.assert_close(before(x),wide(x),atol=3e-5,rtol=3e-5)
 def test_gradients_break_identical_pairs(self):
  small=ResNet12(widths=(4,8,12,16));wide=widen_resnet12(small);diversify_fanout(wide,relative_scale=.5)
  torch.nn.functional.cross_entropy(wide(torch.randn(4,3,32,32)),torch.arange(4)).backward()
  grad=wide.encoder[3].convolutions[6].weight.grad
  self.assertGreater(float((grad[::2]-grad[1::2]).abs().max()),1e-5)
 def test_duplicate_penalty(self):
  x=torch.randn(64,16);duplicated=x.repeat_interleave(2,dim=1)
  self.assertGreater(float(pair_correlation_penalty(duplicated)),.99)
  self.assertLess(float(pair_correlation_penalty(torch.randn(64,32))),.15)
 def test_noop_rgb_and_deterministic(self):
  a=ResNet12(widths=(4,8,12,16));b=copy.deepcopy(a);rgb=a.encoder[0].convolutions[0].weight.detach().clone()
  diversify_fanout(a);diversify_fanout(b)
  for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y)
  torch.testing.assert_close(rgb,a.encoder[0].convolutions[0].weight)
 def test_metric_only_base_labels(self):
  x=torch.eye(8).repeat_interleave(8,dim=0);y=torch.arange(8).repeat_interleave(8)
  for shots in (1,5):self.assertLess(float(cosine_episode_loss(x,y,shots=shots,generator=torch.Generator().manual_seed(2))),.01)
 def test_invalid(self):
  with self.assertRaises(ValueError):diversify_fanout(ResNet12(),relative_scale=float('nan'))
  with self.assertRaises(ValueError):pair_correlation_penalty(torch.ones(2,4))
