"""Synthetic checks, not mini-ImageNet accuracy selection."""
import unittest
import numpy as np
from experiments.lab04.teacher_geometry import fit_teacher_geometry, apply_teacher_geometry, unit


class TeacherGeometryTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(32)
        x=rng.normal(size=(80,8)); y=x@rng.normal(size=(8,13))+rng.normal(size=(80,13))*.02
        return x,y

    def test_gram_equivalence_of_compression(self):
        x,y=self.fixture();t=fit_teacher_geometry(x,y,mix=1.)
        a=apply_teacher_geometry(x,t)
        b=unit((unit(x)-t['center'])@t['regression'])
        np.testing.assert_allclose(a@a.T,b@b.T,atol=1e-11)
        self.assertEqual(a.shape,x.shape)

    def test_identity_endpoint(self):
        x,y=self.fixture();t=fit_teacher_geometry(x,y,mix=0.)
        np.testing.assert_allclose(apply_teacher_geometry(x,t),unit(unit(x)-unit(x).mean(0)),atol=1e-11)

    def test_train_order_invariance(self):
        x,y=self.fixture();a=fit_teacher_geometry(x,y);b=fit_teacher_geometry(x[::-1],y[::-1])
        np.testing.assert_allclose(a['matrix'],b['matrix'],atol=1e-11)

    def test_query_independence_and_no_mutation(self):
        x,y=self.fixture();saved=x.copy();t=fit_teacher_geometry(x,y)
        a=apply_teacher_geometry(x,t);b=apply_teacher_geometry(x[:1],t)
        np.testing.assert_allclose(a[:1],b,atol=1e-11);np.testing.assert_array_equal(x,saved)

    def test_bad_inputs(self):
        x,y=self.fixture()
        for kw in [{'mix':float('nan')},{'mix':1.1},{'ridge_ratio':0},{'ridge_ratio':True}]:
            with self.assertRaises(ValueError):fit_teacher_geometry(x,y,**kw)
        with self.assertRaises(ValueError):fit_teacher_geometry(x,y[:-1])
        with self.assertRaises(ValueError):fit_teacher_geometry(x*np.nan,y)
        with self.assertRaises(ValueError):fit_teacher_geometry(np.zeros_like(x),np.zeros_like(y))
        with self.assertRaises(ValueError):apply_teacher_geometry(x,{'center':np.zeros(7),'matrix':np.eye(8)})
