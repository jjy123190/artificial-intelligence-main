from __future__ import annotations
import unittest
import numpy as np
from backend.lab_four_projection import project


class ProjectionTests(unittest.TestCase):
    def test_identical_before_and_after_samples_do_not_move(self):
        rng=np.random.default_rng(71);x=rng.normal(size=(80,32));pa=x[:5];pb=pa+rng.normal(0,.3,pa.shape)
        result=project(x,x,pa,pb)
        np.testing.assert_allclose(result["before"],result["after"])
        self.assertTrue(result["isotropic"])
        self.assertTrue(0<=sum(result["explained_variance"])<=1+1e-12)

    def test_two_dimensional_geometry_uses_one_scale(self):
        x=np.array([[0.,0.],[4.,0.],[0.,1.],[4.,1.],[-2.,.5]])
        result=project(x,x,x[:2],x[:2])
        distances=np.linalg.norm(x[:,None]-x[None,:],axis=2)
        y=result["before"];new=np.linalg.norm(y[:,None]-y[None,:],axis=2)
        ratios=new[distances>1e-9]/distances[distances>1e-9]
        np.testing.assert_allclose(ratios,np.full_like(ratios,ratios[0]),atol=1e-12)
        self.assertAlmostEqual(sum(result["explained_variance"]),1.)

    def test_constant_features_are_finite(self):
        x=np.ones((10,8));result=project(x,x,x[:5],x[:5])
        self.assertTrue(np.isfinite(result["before"]).all())
        np.testing.assert_allclose(result["before"],.5)
        self.assertEqual(sum(result["explained_variance"]),0)

    def test_invalid_shapes_do_not_make_misleading_points(self):
        with self.assertRaises(ValueError):project(np.ones((3,4)),np.ones((2,4)),np.ones((2,4)),np.ones((2,4)))
        with self.assertRaises(ValueError):project(np.full((3,4),np.nan),np.ones((3,4)),np.ones((2,4)),np.ones((2,4)))

if __name__=="__main__":unittest.main()
