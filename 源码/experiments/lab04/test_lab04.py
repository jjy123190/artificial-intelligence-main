from __future__ import annotations

import inspect
import unittest
from copy import deepcopy
import numpy as np
import torch
from torch.nn import functional as F

from .calibration import CalibrationConfig, base_statistics, calibrated_predict
from .data import sample_episode
from .models import ResNet12, maml_adapt


class InferenceSafetyTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)
        self.base = self.rng.normal(size=(60, 8))
        self.prior = base_statistics(self.base, np.repeat(np.arange(6), 10))
        self.support = self.rng.normal(size=(5, 8))
        self.query = self.rng.normal(size=(15, 8))

    def test_one_shot_variance_and_predictions_are_finite(self):
        result = calibrated_predict(self.support, np.arange(5), self.query,
                                    self.prior, mean=True, variance=True, prototype=True)
        self.assertTrue(np.isfinite(result["variances"]).all())
        self.assertTrue((result["variances"] > 0).all())
        self.assertTrue(np.isfinite(result["probabilities"]).all())
        np.testing.assert_allclose(result["probabilities"].sum(1), 1)

    def test_prediction_api_has_no_query_labels(self):
        self.assertNotIn("query_labels", inspect.signature(calibrated_predict).parameters)
        # A query permutation only permutes outputs, including transduction.
        first = calibrated_predict(self.support, np.arange(5), self.query,
                                   self.prior, mean=True, variance=True, prototype=True)
        order = self.rng.permutation(len(self.query))
        second = calibrated_predict(self.support, np.arange(5), self.query[order],
                                    self.prior, mean=True, variance=True, prototype=True)
        np.testing.assert_allclose(first["probabilities"][order], second["probabilities"], atol=1e-10)

    def test_inductive_predictions_do_not_use_other_queries(self):
        whole = calibrated_predict(self.support, np.arange(5), self.query,
                                   self.prior, mean=True, variance=True)
        single = calibrated_predict(self.support, np.arange(5), self.query[:1],
                                    self.prior, mean=True, variance=True)
        np.testing.assert_allclose(whole["probabilities"][:1], single["probabilities"])

    def test_support_query_sampling_is_disjoint_and_seeded(self):
        labels = np.repeat(np.arange(20), 40)
        first = sample_episode(labels, np.random.default_rng(12), shots=5)
        second = sample_episode(labels, np.random.default_rng(12), shots=5)
        self.assertFalse(set(first["support_indices"]) & set(first["query_indices"]))
        np.testing.assert_array_equal(first["query_indices"], second["query_indices"])
        for label in range(5):
            self.assertEqual(int((first["support_labels"] == label).sum()), 5)
            self.assertEqual(int((first["query_labels"] == label).sum()), 15)


class MAMLTests(unittest.TestCase):
    def test_second_order_gradient_differs_from_first_order_approximation(self):
        torch.manual_seed(19)
        initial = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2))
        support = torch.randn(6, 2)
        query = torch.randn(6, 2)
        labels = torch.tensor([0, 1, 0, 1, 0, 1])
        gradients = []
        for second_order in (False, True):
            model = deepcopy(initial)
            adapted = maml_adapt(model, support, labels, learning_rate=0.3,
                                 second_order=second_order)
            logits = torch.func.functional_call(model, adapted, (query,))
            outer = torch.autograd.grad(F.cross_entropy(logits, labels), tuple(model.parameters()))
            gradients.append(torch.cat([value.flatten() for value in outer]))
        self.assertGreater(float((gradients[1] - gradients[0]).norm()), 1e-5)

    def test_second_order_outer_gradient_updates_backbone(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        model = ResNet12(classes=5)
        support = torch.randn(5, 3, 32, 32)
        labels = torch.arange(5)
        query = torch.randn(5, 3, 32, 32)
        before = next(model.encoder.parameters()).detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        adapted = maml_adapt(model, support, labels, second_order=True)
        output = torch.func.functional_call(model, adapted, (query,))
        F.cross_entropy(output, labels).backward()
        backbone_gradient = next(model.encoder.parameters()).grad
        self.assertIsNotNone(backbone_gradient)
        self.assertTrue(torch.isfinite(backbone_gradient).all())
        self.assertGreater(float(backbone_gradient.abs().sum()), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before, next(model.encoder.parameters()).detach()))


if __name__ == "__main__":
    unittest.main()
