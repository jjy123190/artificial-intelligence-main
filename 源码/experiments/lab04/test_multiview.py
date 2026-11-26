from __future__ import annotations
import itertools
import inspect
import unittest
import numpy as np
import torch
from .multiview import combine_views, encode_views, minimum_assignment, capacity_predictions, predict
from .models import ResNet12
from .graph_calibration import GraphConfig


class MultiViewTests(unittest.TestCase):
    def test_solver_matches_brute_force(self):
        rng=np.random.default_rng(41)
        for n in range(1,7):
            costs=rng.normal(size=(n,n))
            cols=minimum_assignment(costs)
            expected=min(sum(costs[i,j] for i,j in enumerate(p)) for p in itertools.permutations(range(n)))
            self.assertAlmostEqual(float(costs[np.arange(n),cols].sum()),expected,places=10)
            self.assertEqual(len(set(cols)),n)

    def test_capacity_counts_and_query_permutation(self):
        rng=np.random.default_rng(18)
        probs=rng.uniform(.01,1,(75,5));probs/=probs.sum(1,keepdims=True)
        features=rng.normal(size=(75,8))
        labels=capacity_predictions(probs,features)
        np.testing.assert_array_equal(np.bincount(labels,minlength=5),[15]*5)
        order=rng.permutation(75)
        np.testing.assert_array_equal(labels[order],capacity_predictions(probs[order],features[order]))

    def test_capacity_does_not_forge_one_hot_confidence(self):
        rng=np.random.default_rng(15)
        support=rng.normal(size=(5,8));query=rng.normal(size=(75,8));labels=np.arange(5)
        cfg=GraphConfig(graph_weight=0,prototype_steps=2,balanced=True)
        soft=predict(support,labels,query,cfg,'soft')
        constrained=predict(support,labels,query,cfg,'balanced_capacity')
        np.testing.assert_allclose(soft['probabilities'],constrained['probabilities'])
        np.testing.assert_allclose(constrained['assigned_scores'],constrained['probabilities'][np.arange(75),constrained['predictions']])
        self.assertNotIn('query_labels',inspect.signature(predict).parameters)

    def test_multiview_uses_same_image_and_is_seed_free(self):
        torch.set_num_threads(1);torch.manual_seed(1)
        model=ResNet12().eval();images=np.random.default_rng(2).integers(0,256,(3,84,84,3),dtype=np.uint8)
        first=encode_views(model,images,mode='mean',batch_size=2)
        second=encode_views(model,images,mode='mean',batch_size=3)
        np.testing.assert_allclose(first,second,atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(first,axis=1),1,atol=1e-8)
        model.train()
        with self.assertRaises(ValueError):encode_views(model,images)

    def test_train_residual_covariance_does_not_use_query_labels(self):
        from .supervised_whitening import fit
        from .graph_calibration import apply_transform,centered_whitening
        rng=np.random.default_rng(21);train=rng.normal(size=(80,12));labels=np.repeat(np.arange(8),10)
        query=rng.normal(size=(9,12))
        expected=centered_whitening(train,1,.5);plain=fit(train,labels,strength=1,exponent=.5,residual_mix=0)
        np.testing.assert_allclose(plain['matrix'],expected['matrix'],atol=1e-10)
        transform=fit(train,labels,residual_mix=1)
        whole=apply_transform(query,transform)
        separate=np.concatenate([apply_transform(query[i:i+1],transform) for i in range(len(query))])
        np.testing.assert_allclose(whole,separate,atol=1e-10)

    def test_bad_capacity_shape_is_rejected(self):
        with self.assertRaises(ValueError):capacity_predictions(np.ones((7,5)),np.ones((7,8)))
        with self.assertRaises(ValueError):minimum_assignment(np.ones((2,3)))
        with self.assertRaises(ValueError):combine_views(np.ones((2,4)),np.ones((2,5)),'mean')

if __name__=='__main__':unittest.main()
