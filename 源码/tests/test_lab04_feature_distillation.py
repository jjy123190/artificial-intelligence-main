"""Training primitive tests use synthetic features only; no accuracy claims."""
import unittest
import torch
from torch import nn
from experiments.lab04.feature_distillation import (
    FeatureDistillationConfig,FeatureProjector,distillation_terms,training_loss)


class FeatureDistillationTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(29)

    def test_teacher_stop_gradient(self):
        student=torch.randn(8,4,requires_grad=True);teacher=torch.randn(8,7,requires_grad=True)
        projector=FeatureProjector(4,7)
        terms=distillation_terms(student,teacher,projector)
        (terms['feature']+terms['relation']).backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNotNone(student.grad);self.assertIsNotNone(projector.linear.weight.grad)

    def test_identical_features_zero(self):
        features=torch.randn(8,4)
        terms=distillation_terms(features,features,nn.Identity())
        self.assertLess(abs(float(terms['feature'])),1e-6)
        self.assertAlmostEqual(float(terms['relation']),0,places=7)

    def test_permutation_invariance(self):
        student=torch.randn(8,4);teacher=torch.randn(8,7);projector=FeatureProjector(4,7)
        order=torch.randperm(8)
        a=distillation_terms(student,teacher,projector);b=distillation_terms(student[order],teacher[order],projector)
        for key in a:torch.testing.assert_close(a[key],b[key])

    def test_weighted_loss_and_gradients(self):
        logits=torch.randn(8,5,requires_grad=True);labels=torch.arange(8)%5
        first=torch.randn(8,4,requires_grad=True);second=torch.randn(8,4,requires_grad=True)
        ta=torch.randn(8,7,requires_grad=True);tb=torch.randn(8,7,requires_grad=True)
        projector=FeatureProjector(4,7);metric=first.square().mean();cfg=FeatureDistillationConfig()
        values=training_loss(logits,labels,first,second,ta,tb,projector,metric,cfg)
        expected=values['ce']+values['feature']+.5*values['relation']+.5*values['alignment']+.25*values['metric']
        torch.testing.assert_close(values['total'],expected)
        values['total'].backward();self.assertIsNone(ta.grad);self.assertIsNone(tb.grad)
        self.assertIsNotNone(first.grad);self.assertIsNotNone(second.grad);self.assertIsNotNone(logits.grad)

    def test_zero_auxiliary_is_ce(self):
        logits=torch.randn(8,5);labels=torch.arange(8)%5;x=torch.randn(8,4);y=torch.randn(8,7)
        cfg=FeatureDistillationConfig(feature_weight=0,relation_weight=0,alignment_weight=0,metric_weight=0)
        values=training_loss(logits,labels,x,x,y,y,FeatureProjector(4,7),torch.tensor(0.),cfg)
        torch.testing.assert_close(values['total'],values['ce'])
        self.assertEqual(float(values['alignment']),0.)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):FeatureProjector(0,4)
        with self.assertRaises(ValueError):FeatureDistillationConfig(feature_weight=float('nan'))
        with self.assertRaises(ValueError):FeatureDistillationConfig(smoothing=1)
        with self.assertRaises(ValueError):distillation_terms(torch.ones(1,4),torch.ones(1,7),FeatureProjector(4,7))
        with self.assertRaises(ValueError):distillation_terms(torch.ones(4,4)*float('nan'),torch.ones(4,7),FeatureProjector(4,7))
