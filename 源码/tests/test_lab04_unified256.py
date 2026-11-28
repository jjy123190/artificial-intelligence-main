"""Regression tests for independent ablations, support adaptation, query privacy and the audit."""
from dataclasses import asdict
import unittest
import torch
from torch.nn import functional as F

from experiments.lab04.models import maml_adapt
from experiments.lab04.train_medium_gpu import medium_model
from experiments.lab04.unified256_methods import (METHODS, RULES, calibrated, local_bonus,
                                                   maml_predictions, probabilities, task_hash)
from experiments.lab04.unified256_train import init_maml
from experiments.lab04.unified256_audit import verify_prediction


class UnifiedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_single_interventions(self):
        full = asdict(METHODS["full"])
        for name, field in (("no_alignment", "encoder"), ("no_mean", "mean"), ("no_variance", "variance"),
                            ("no_update", "update"), ("no_balance", "balance"), ("full_local", "local"),
                            ("full_flip", "flip")):
            other = asdict(METHODS[name])
            self.assertEqual([k for k in full if full[k] != other[k]], [field])
        self.assertEqual(sum(asdict(METHODS["full_flip_local"])[k] != full[k] for k in full), 2)

    def test_sinkhorn_without_prototype_update(self):
        from unittest.mock import patch
        import experiments.lab04.unified256_methods as methods
        torch.manual_seed(7)
        support = F.normalize(torch.randn(1, 5, 256), dim=-1)
        y = torch.arange(5)[None]
        q = F.normalize(torch.randn(1, 75, 256), dim=-1)
        prior = {"means": torch.randn(64, 256)*.03, "variances": torch.full((64, 256), .003)}
        for name, calls, flag in (("no_update", 1, True), ("no_balance", methods.UPDATES+1, False)):
            with patch.object(methods, "probabilities", wraps=probabilities) as mock:
                ans = calibrated(support, y, q, prior, RULES[0], METHODS[name])
                self.assertEqual(tuple(ans.shape), (1, 75))
                self.assertEqual(mock.call_count, calls)
                self.assertTrue(all(c.args[1] == flag for c in mock.call_args_list))
        weights = probabilities(torch.randn(2, 75, 5), True)
        torch.testing.assert_close(weights.sum(1), torch.full((2, 5), 15.))
        torch.testing.assert_close(weights.sum(2), torch.ones(2, 75))

    def test_query_order_equivariance_and_inductive_baseline(self):
        torch.manual_seed(8)
        y = torch.arange(5)[None]
        s, q = (F.normalize(torch.randn(1, n, 256), dim=-1) for n in (5, 75))
        p = {"means": torch.rand(64, 256)*.1, "variances": torch.rand(64, 256)*.01}
        for name in ("full", "baseline"):
            a = calibrated(s, y, q, p, RULES[1], METHODS[name])
            perm = torch.randperm(75)
            b = calibrated(s, y, q[:, perm], p, RULES[1], METHODS[name])
            self.assertTrue(a[:, perm].equal(b))
            if name == "baseline":
                one = calibrated(s, y, q[:, 4:5], p, RULES[1], METHODS[name])
                self.assertTrue(a[:, 4:5].equal(one))
        sp, qp = torch.randn(1, 5, 9, 128), torch.randn(1, 7, 9, 128)
        forward = local_bonus(sp, y, qp).flip([1])
        reverse = local_bonus(sp, y, qp.flip([1]))
        torch.testing.assert_close(forward, reverse, atol=1e-6, rtol=1e-5)
        self.assertTrue(forward.argmax(-1).equal(reverse.argmax(-1)))

    def test_all_param_second_order_and_task_isolation(self):
        torch.manual_seed(91)
        model = init_maml(medium_model().state_dict())
        buffers = {k: v.clone() for k, v in model.named_buffers()}
        s, q = torch.randn(5, 3, 32, 32), torch.randn(5, 3, 32, 32)
        y = torch.arange(5)
        fast = maml_adapt(model, s, y, steps=1, learning_rate=.06, second_order=True)
        g = torch.autograd.grad(F.cross_entropy(torch.func.functional_call(model, fast, (q,)), y),
                                model.encoder[0].convolutions[0].weight)[0]
        first_order = maml_adapt(model, s, y, steps=1, learning_rate=.06, second_order=False)
        g_fo = torch.autograd.grad(F.cross_entropy(torch.func.functional_call(model, first_order, (q,)), y),
                                   model.encoder[0].convolutions[0].weight)[0]
        self.assertGreater((g-g_fo).abs().max().item(), 0.)
        self.assertEqual(set(fast), set(dict(model.named_parameters())))
        self.assertTrue(all(torch.equal(buffers[k], v) for k, v in model.named_buffers()))
        self.assertEqual(len(fast), len(list(model.parameters())))
        model.train()
        with self.assertRaises(ValueError):
            maml_predictions(model, None, None, "cpu")

    def test_scorer_rejects_wrong_order_or_misreported_accuracy(self):
        r = {"shots": 1, "episode": 0, "support_ids": ["x"]*5, "query_ids": [str(i) for i in range(75)]}
        y = [0, 1, 2, 3, 4]
        h = task_hash(r, y)
        ordered = {(1, 0): (h, [0]*75)}
        pred = {"shots": 1, "episode": 0, "predictions": [0]*75, "correct": 75, "accuracy": 1., "task_sha256": h}
        self.assertEqual(verify_prediction(pred, ordered, 1, 0), 75)
        with self.assertRaises(ValueError):
            verify_prediction({**pred, "correct": 74}, ordered, 1, 0)
        r["query_ids"].reverse()
        with self.assertRaises(ValueError):
            verify_prediction({**pred, "task_sha256": task_hash(r, y)}, ordered, 1, 0)


if __name__ == "__main__":
    unittest.main()
