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


class ImprovedCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(83)
        self.base = self.rng.normal(size=(180, 16))
        self.prior = base_statistics(self.base, np.repeat(np.arange(6), 30))
        self.support = self.rng.normal(size=(25, 16))
        self.labels = np.repeat(np.arange(5), 5)
        self.query = self.rng.normal(size=(75, 16))

    def test_config_rejects_nonfinite_and_invalid_values(self):
        for config in ({"temperature": 0}, {"variance_blend": -1}, {"mean_weight": float("nan")},
                       {"variance_prior_strength": 0}, {"prototype_steps": 31}, {"neighbors": 0}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                CalibrationConfig(**config)

    def test_zero_variance_blend_equals_euclidean_predictions(self):
        a = calibrated_predict(self.support, self.labels, self.query, self.prior)
        b = calibrated_predict(self.support, self.labels, self.query, self.prior,
                               variance=True, config=CalibrationConfig(variance_blend=0))
        np.testing.assert_allclose(a["probabilities"], b["probabilities"], atol=1e-12)

    def test_balanced_assignment_is_finite_and_permutation_equivariant(self):
        from .calibration import balanced_soft_assignments
        logits = self.rng.normal(size=(75, 5)) * 4
        weights = balanced_soft_assignments(logits, iterations=200)
        self.assertTrue(np.isfinite(weights).all())
        np.testing.assert_allclose(weights.sum(1), 1)
        np.testing.assert_allclose(weights.sum(0), 15, atol=1e-4)
        order = self.rng.permutation(75)
        permuted = balanced_soft_assignments(logits[order], iterations=200)
        np.testing.assert_allclose(weights[order], permuted, atol=1e-10)

    def test_balanced_transduction_does_not_consume_query_labels(self):
        cfg = CalibrationConfig(balanced_queries=True, prototype_steps=4, variance_blend=.1)
        first = calibrated_predict(self.support, self.labels, self.query, self.prior,
                                   prototype=True, variance=True, config=cfg)
        order = self.rng.permutation(75)
        second = calibrated_predict(self.support, self.labels, self.query[order], self.prior,
                                    prototype=True, variance=True, config=cfg)
        np.testing.assert_allclose(first["probabilities"][order], second["probabilities"], atol=1e-9)
        self.assertTrue(first["balanced_queries"])

    def test_training_center_transform_is_inductive_and_reproducible(self):
        from .feature_space import fit_feature_space, transform_features
        prior = fit_feature_space(self.base, np.repeat(np.arange(6), 30), center_strength=1, power=.5)
        whole = transform_features(self.query, prior)
        separate = np.concatenate([transform_features(self.query[i:i+1], prior) for i in range(75)])
        np.testing.assert_allclose(whole, separate, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(whole, axis=1), 1)
        untouched = transform_features(self.query, {"center": np.zeros(16), "power": np.array(1)})
        np.testing.assert_allclose(untouched, self.query / np.linalg.norm(self.query, axis=1, keepdims=True))


class VersionedInferenceTests(unittest.TestCase):
    def test_select_prior_does_not_apply_transform_to_baseline(self):
        from .feature_space import select_prior
        prior = {"means": np.zeros((3, 4)), "variances": np.ones((3, 4)),
                 "shot1_means": np.full((3, 4), 2), "shot1_variances": np.full((3, 4), 3),
                 "shot1_center": np.ones(4), "shot1_power": np.array(1)}
        plain = select_prior(prior, 1, transformed=False)
        self.assertNotIn("center", plain)
        np.testing.assert_array_equal(plain["means"], np.zeros((3, 4)))
        transformed = select_prior(prior, 1, transformed=True)
        np.testing.assert_array_equal(transformed["center"], np.ones(4))
        np.testing.assert_array_equal(transformed["means"], np.full((3, 4), 2))

    def test_malformed_features_fail_with_explanatory_value_error(self):
        prior = {"means": np.ones((5, 4)), "variances": np.ones((5, 4))}
        for support, labels in ((np.ones(4), np.arange(5)), (np.ones((5,4)), np.arange(5)/2)):
            with self.assertRaises(ValueError):
                calibrated_predict(support, labels, np.ones((5, 4)), prior)
        with self.assertRaises(ValueError):
            calibrated_predict(np.ones((5,4)), np.arange(5), np.ones((5,4)),
                               {"means": np.ones((5,4)), "variances": -np.ones((5,4))})


class EvidenceVerifierTests(unittest.TestCase):
    def test_verifier_detects_corrupted_summary_with_intact_weights(self):
        import json
        import tempfile
        from pathlib import Path
        from .data import digest_file
        from .verify import verify_artifacts
        with tempfile.TemporaryDirectory(prefix="lab04_verifier_fixture_") as directory:
            root = Path(directory)
            entries = {}
            for name in ("baseline", "aligned", "maml", "demo_pool"):
                path = root / (name + ".bin")
                path.write_bytes(name.encode())
                entries[name] = {"path":path.name,"sha256":digest_file(path)}
            records = []
            for shots in (1,5):
                for episode in (0,1):
                    records.append({"shots":shots,"episode":episode,"classes":list(range(5)),
                        "support_ids":[f"s{n}" for n in range(5*shots)],"query_ids":[f"q{n}" for n in range(75)],
                        "methods":{"baseline":{"accuracy":.4,"seconds":.01}}})
            summary = {"status":"complete","schema_version":1,
                "weights":{k:entries[k] for k in ("baseline","aligned","maml")},"demo_pool":entries["demo_pool"],
                "protocol":{"seed":10,"ways":5,"queries_per_class":15,"test_episodes_per_shot":2},
                "rows":[{"method":"baseline","shots":s,"accuracy":.4,"ci95":0,"paired_delta":0,"paired_ci95":0,"mean_episode_seconds":.01} for s in (1,5)]}
            (root/"episodes.json").write_text(json.dumps({"seed":210,"episodes":records}),encoding="utf-8")
            (root/"results.json").write_text(json.dumps(summary),encoding="utf-8")
            self.assertTrue(verify_artifacts(root)["verified"])
            summary["rows"][0]["accuracy"] = .9
            (root/"results.json").write_text(json.dumps(summary),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"Metric mismatch"):
                verify_artifacts(root)


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
