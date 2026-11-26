import unittest
import numpy as np
from experiments.lab04.local_reconstruction import reconstruct_predict

class LocalReconstructionTests(unittest.TestCase):
 def fixture(self):
  rng=np.random.default_rng(2);centers=np.eye(5,16)
  s=np.stack([centers[c]+rng.normal(0,.01,(9,16)) for c in range(5)])
  q=np.stack([centers[c]+rng.normal(0,.01,(9,16)) for c in range(5) for _ in range(3)])
  return s,np.arange(5),q
 def test_reconstruct_separated(self):
  s,y,q=self.fixture();r=reconstruct_predict(s,y,q)
  np.testing.assert_array_equal(r['predictions'],np.repeat(np.arange(5),3));self.assertFalse(r['transductive'])
 def test_query_independent(self):
  s,y,q=self.fixture();full=reconstruct_predict(s,y,q)['scores'];one=np.concatenate([reconstruct_predict(s,y,x[None])['scores'] for x in q])
  np.testing.assert_allclose(full,one,atol=1e-12)
 def test_support_position_permutation(self):
  s,y,q=self.fixture();a=reconstruct_predict(s,y,q)['scores'];b=reconstruct_predict(s[::-1,::-1],y[::-1],q[:,::-1])['scores']
  np.testing.assert_allclose(a,b,atol=1e-12)
 def test_no_query_label_parameter(self):
  import inspect
  self.assertNotIn('query_labels',inspect.signature(reconstruct_predict).parameters)
 def test_invalid(self):
  s,y,q=self.fixture()
  with self.assertRaises(ValueError):reconstruct_predict(s,y+1,q)
  with self.assertRaises(ValueError):reconstruct_predict(s,y,q,ridge=0)
  with self.assertRaises(ValueError):reconstruct_predict(s,y,q,normalization='none')
  with self.assertRaises(ValueError):reconstruct_predict(s*np.nan,y,q)
