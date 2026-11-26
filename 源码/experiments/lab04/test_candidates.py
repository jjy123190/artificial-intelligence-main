from __future__ import annotations
import inspect
import unittest
import numpy as np
import torch
from .models import ResNet12
from .metric_training import widen_resnet12,balanced_batches,episode_loss,augment_images
from .graph_calibration import GraphConfig,predict,centered_whitening,apply_transform


class CandidateTrainingTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_channel_widening_preserves_network_function(self):
        torch.manual_seed(12)
        old=ResNet12(widths=(8,16,32,64)).eval();new=widen_resnet12(old).eval()
        x=torch.randn(3,3,84,84)
        with torch.no_grad():torch.testing.assert_close(old(x),new(x),atol=3e-6,rtol=3e-5)
        self.assertEqual(new.widths,(16,32,64,128))

    def test_balanced_sampling_uses_each_training_image_once(self):
        labels=np.repeat(np.arange(64),600)
        first=list(balanced_batches(labels,np.random.default_rng(9)))
        second=list(balanced_batches(labels,np.random.default_rng(9)))
        np.testing.assert_array_equal(np.concatenate(first),np.concatenate(second))
        self.assertEqual(len(set(np.concatenate(first))),len(labels))
        for ids in first:
            _,counts=np.unique(labels[ids],return_counts=True)
            np.testing.assert_array_equal(counts,np.full(16,4))

    def test_metric_training_loss_updates_features(self):
        x=torch.randn(64,16,requires_grad=True);labels=torch.arange(16).repeat_interleave(4)
        loss=episode_loss(x,labels);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(float(x.grad.abs().sum()),0)
        with self.assertRaises(ValueError):episode_loss(x,labels.roll(1))

    def test_augmentation_is_seeded_finite_and_independent(self):
        image=np.random.default_rng(3).integers(0,256,(1,84,84,3),dtype=np.uint8)
        images=np.repeat(image,8,axis=0)
        first=augment_images(images,torch.Generator().manual_seed(9))
        second=augment_images(images,torch.Generator().manual_seed(9))
        torch.testing.assert_close(first,second)
        self.assertFalse(torch.equal(first[0],first[1]))
        self.assertTrue(torch.isfinite(first).all())


class CandidateGraphTests(unittest.TestCase):
    def test_no_query_labels_and_query_permutation_equivariance(self):
        self.assertNotIn('query_labels',inspect.signature(predict).parameters)
        rng=np.random.default_rng(7);support=rng.normal(size=(25,16));query=rng.normal(size=(75,16));labels=np.repeat(np.arange(5),5)
        config=GraphConfig(graph_weight=.5)
        first=predict(support,labels,query,config);order=rng.permutation(75)
        second=predict(support,labels,query[order],config)
        np.testing.assert_allclose(first['probabilities'][order],second['probabilities'],atol=1e-9)
        np.testing.assert_allclose(first['probabilities'].sum(1),1)
        self.assertTrue(first['transductive'])

    def test_training_transform_is_inductive(self):
        rng=np.random.default_rng(8)
        train=rng.normal(size=(100,16));query=rng.normal(size=(10,16))
        transform=centered_whitening(train,1,.5)
        first=apply_transform(query,transform)
        single=np.concatenate([apply_transform(query[i:i+1],transform) for i in range(10)])
        np.testing.assert_allclose(first,single,atol=1e-10)

    def test_separated_clusters_predict_correctly(self):
        rng=np.random.default_rng(19);centers=np.eye(5)
        support=np.repeat(centers,2,axis=0)+rng.normal(0,.01,(10,5))
        query=np.repeat(centers,15,axis=0)+rng.normal(0,.01,(75,5))
        result=predict(support,np.repeat(np.arange(5),2),query)
        np.testing.assert_array_equal(result['predictions'],np.repeat(np.arange(5),15))

if __name__=='__main__':unittest.main()
