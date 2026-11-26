"""Safety and equation tests for the isolated literature-guided candidates."""
import dataclasses
import inspect
import unittest

import numpy as np
import torch

from experiments.lab04.literature_methods import (
    MAPConfig, TIMConfig, balanced_transport, map_predict_batch,
    prepare_episode_batch, tim_objective, tim_predict_batch,
)


class LiteratureMethodsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self):
        rng = np.random.default_rng(47)
        centers = np.eye(5, 8)
        support = np.repeat(centers, 2, 0) + rng.normal(0, .025, (10, 8))
        query = np.repeat(centers, 3, 0) + rng.normal(0, .025, (15, 8))
        return support[None], np.repeat(np.arange(5), 2)[None], query[None]

    def test_map_separable(self):
        s, y, q = self.fixture()
        result = map_predict_batch(s, y, q)
        np.testing.assert_array_equal(result['predictions'][0], np.repeat(np.arange(5), 3))
        self.assertLess(result['max_transport_residual'], 1e-5)
        np.testing.assert_allclose(result['probabilities'].sum(1), 3, atol=1e-5)

    def test_tim_separable(self):
        s, y, q = self.fixture()
        result = tim_predict_batch(s, y, q, TIMConfig(steps=20))
        np.testing.assert_array_equal(result['predictions'][0], np.repeat(np.arange(5), 3))
        self.assertLessEqual(result['objective_after'], result['objective_before'])
        np.testing.assert_allclose(result['probabilities'].sum(2), 1, atol=1e-10)

    def test_query_permutation(self):
        s, y, q = self.fixture()
        order = np.random.default_rng(3).permutation(q.shape[1])
        for fn, cfg in [(map_predict_batch, MAPConfig()), (tim_predict_batch, TIMConfig(steps=12))]:
            with self.subTest(method=fn.__name__):
                a = fn(s, y, q, cfg)['probabilities']
                b = fn(s, y, q[:, order], cfg)['probabilities']
                np.testing.assert_allclose(a[:, order], b, atol=1e-10)

    def test_support_permutation(self):
        s, y, q = self.fixture()
        order = np.arange(s.shape[1])[::-1]
        for fn, cfg in [(map_predict_batch, MAPConfig()), (tim_predict_batch, TIMConfig(steps=12))]:
            a = fn(s, y, q, cfg)['probabilities']
            b = fn(s[:, order], y[:, order], q, cfg)['probabilities']
            np.testing.assert_allclose(a, b, atol=1e-10)

    def test_task_isolation(self):
        s, y, q = self.fixture()
        for fn, cfg in [(map_predict_batch, MAPConfig()), (tim_predict_batch, TIMConfig(steps=12))]:
            one = fn(s, y, q, cfg)['probabilities']
            many = fn(np.concatenate([s, -s]), np.concatenate([y, y]),
                      np.concatenate([q, -q]), cfg)['probabilities']
            np.testing.assert_allclose(one[0], many[0], atol=1e-10)

    def test_no_query_label_argument(self):
        for fn in [map_predict_batch, tim_predict_batch, prepare_episode_batch, balanced_transport]:
            self.assertNotIn('query_labels', inspect.signature(fn).parameters)

    def test_inputs_not_mutated(self):
        s, y, q = self.fixture()
        originals = [a.copy() for a in (s, y, q)]
        map_predict_batch(s, y, q)
        tim_predict_batch(s, y, q, TIMConfig(steps=5))
        prepare_episode_batch(s, q, signed_power=.5, task_center=True)
        for a, b in zip((s, y, q), originals):
            np.testing.assert_array_equal(a, b)

    def test_bad_inputs(self):
        s, y, q = self.fixture()
        for fn in [map_predict_batch, tim_predict_batch]:
            for a, b, c in [(s[0], y, q), (s, y.astype(float), q), (s, y + 1, q),
                             (s, y, q[:, :, :-1]), (s, y, q[:, :0]), (s * np.nan, y, q)]:
                with self.subTest(method=fn.__name__, shapes=[x.shape for x in (a,b,c)]):
                    with self.assertRaises(ValueError):
                        fn(a, b, c)

    def test_configuration_bounds(self):
        for cls, kwargs in [(MAPConfig, {'learning_rate': float('nan')}),
                             (MAPConfig, {'steps': True}), (MAPConfig, {'inverse_temperature': 0}),
                             (TIMConfig, {'steps': -1}), (TIMConfig, {'temperature': float('inf')}),
                             (TIMConfig, {'support_weight': 0})]:
            with self.assertRaises(ValueError):
                cls(**kwargs)
        self.assertEqual(MAPConfig(**MAPConfig().to_dict()), MAPConfig())
        self.assertEqual(TIMConfig(**TIMConfig().to_dict()), TIMConfig())

    def test_sinkhorn_matches_positive_scaling(self):
        x = np.random.default_rng(8).normal(size=(2, 25, 5))
        p = np.exp(x)
        for _ in range(200):
            p /= p.sum(2, keepdims=True)
            p *= 5 / p.sum(1, keepdims=True)
        observed, residual = balanced_transport(x)
        self.assertLess(residual, 1e-6)
        np.testing.assert_allclose(p, observed, atol=1e-7)

    def test_sinkhorn_large_offsets(self):
        x = np.random.default_rng(8).normal(size=(2, 25, 5))
        first, _ = balanced_transport(x)
        second, _ = balanced_transport(x + 10000)
        np.testing.assert_allclose(first, second, atol=1e-10)

    def test_signed_power_is_explicit_and_finite(self):
        s, _, q = self.fixture()
        s -= .05; q -= .05
        before = s.copy()
        first, second = prepare_episode_batch(s, q, signed_power=.5, task_center=True)
        self.assertTrue(np.isfinite(first).all() and np.isfinite(second).all())
        np.testing.assert_array_equal(before, s)
        with self.assertRaises(ValueError):
            prepare_episode_batch(s, q, signed_power=0)

    def test_tim_objective_formula(self):
        s, y, q = self.fixture()
        ls = torch.tensor(s[:, :, :5], dtype=torch.float64)
        lq = torch.tensor(q[:, :, :5], dtype=torch.float64)
        labels = torch.tensor(y)
        cfg = TIMConfig()
        ps = ls.softmax(-1).numpy(); pq = lq.softmax(-1).numpy()
        ce = -np.log(ps[0, np.arange(y.shape[1]), y[0]]).mean()
        cond = -(pq * np.log(pq)).sum(-1).mean()
        marg = -(pq.mean(1) * np.log(pq.mean(1))).sum()
        expected = .1 * ce - marg + .1 * cond
        self.assertAlmostEqual(float(tim_objective(ls, labels, lq, cfg)), expected, places=12)

    def test_zero_iterations_and_inference_context(self):
        s, y, q = self.fixture()
        for fn, cfg in [(map_predict_batch, MAPConfig(steps=0)), (tim_predict_batch, TIMConfig(steps=0))]:
            with torch.inference_mode():
                result = fn(s, y, q, cfg)
            np.testing.assert_allclose(result['prototypes_before'], result['prototypes_after'])


if __name__ == '__main__':
    unittest.main()
