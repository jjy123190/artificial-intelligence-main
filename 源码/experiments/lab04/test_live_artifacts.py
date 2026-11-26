"""Integration tests with ephemeral random-image artifacts, never benchmark data."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from .calibration import CalibrationConfig, base_statistics, calibrated_predict
from .data import digest_file, normalized_tensor, sample_episode
from .experiment import extract_features, save_checkpoint
from .feature_space import fit_feature_space, select_prior, transform_features
from .inference import clear_cache, run_episode
from .models import ResNet12


class PortableInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory(prefix="lab04_random_fixture_")
        cls.directory = Path(cls.temp.name).resolve()
        assert cls.directory.parent == Path(tempfile.gettempdir()).resolve()
        rng = np.random.default_rng(42)
        cls.images = rng.integers(0, 256, size=(125, 84, 84, 3), dtype=np.uint8)
        cls.labels = np.repeat(np.arange(5), 25)
        cls.priors = {}
        cls.features = {}
        for i, name in enumerate(("baseline", "aligned", "maml")):
            torch.manual_seed(41+i)
            model = ResNet12(classes=5 if name == "maml" else 64)
            model.eval()
            save_checkpoint(cls.directory / f"{name}.pt", model, purpose="random fixture", metadata={})
            if name != "maml":
                features = extract_features(model, cls.images, batch_size=25)
                cls.features[name] = features
                priors = base_statistics(features, cls.labels)
                for n in (1, 5):
                    transformed = fit_feature_space(features, cls.labels, center_strength=1, power=1)
                    priors.update({f"shot{n}_{key}": value for key,value in transformed.items()})
                cls.priors[name] = priors
                np.savez_compressed(cls.directory / f"{name}_prior.npz", **priors)
        ids = np.array([f"fixture/{i}.jpg" for i in range(125)])
        np.savez_compressed(cls.directory / "demo_pool.npz", images=cls.images, labels=cls.labels, ids=ids,
            class_wnids=np.array([f"n{i:08}" for i in range(5)]),class_names=np.array([f"class{i}" for i in range(5)]))
        config = CalibrationConfig(mean_weight=.1, variance_blend=.1, balanced_queries=True)
        cls.meta = {"schema_version":2,"status":"complete","study":{"version":"fixture-v2"},
            "weights":{name:{"path":f"{name}.pt","sha256":digest_file(cls.directory/f"{name}.pt")} for name in ("baseline","aligned","maml")},
            "priors":{name:{"path":f"{name}_prior.npz","sha256":digest_file(cls.directory/f"{name}_prior.npz")} for name in ("baseline","aligned")},
            "demo_pool":{"path":"demo_pool.npz","sha256":digest_file(cls.directory/"demo_pool.npz")},
            "calibration":{str(n):config.to_dict() for n in (1,5)},"runtime":{"threads":2},
            "maml_adaptation":{"1":{"steps":1,"learning_rate":.025},"5":{"steps":1,"learning_rate":.05}}}
        (cls.directory / "results.json").write_text(json.dumps(cls.meta),encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        clear_cache()
        assert cls.directory.parent == Path(tempfile.gettempdir()).resolve()
        assert cls.directory.name.startswith("lab04_random_fixture_")
        cls.temp.cleanup()

    def test_live_predictions_match_cached_features_and_shot_transform(self):
        for shots in (1,5):
            task = sample_episode(self.labels, np.random.default_rng(19), shots=shots, queries=15)
            cfg = CalibrationConfig(**self.meta["calibration"][str(shots)])
            for method, mode in (("baseline","none"),("calibrated","none"),("calibrated","full")):
                source = "baseline" if method == "baseline" else "aligned"
                prior = select_prior(self.priors[source], shots, transformed=mode=="full")
                features = transform_features(self.features[source],prior)
                expected = calibrated_predict(features[task["support_indices"]],task["support_labels"],features[task["query_indices"]],prior,
                    mean=mode=="full",variance=mode=="full",prototype=mode=="full",config=cfg)
                live = run_episode(self.directory, shots=shots,seed=19,method=method,calibration=mode)
                self.assertEqual([q["prediction"] for q in live["query"]],expected["predictions"].tolist())
                np.testing.assert_allclose([q["probabilities"] for q in live["query"]],expected["probabilities"],atol=1e-5)
                self.assertEqual(live["feature_motion"],"unchanged")
                np.testing.assert_allclose([q["feature_before"] for q in live["query"]],[q["feature_after"] for q in live["query"]])

    def test_method_comparison_keeps_samples_and_revision(self):
        runs = [run_episode(self.directory,shots=1,seed=7,method=method) for method in ("baseline","maml","calibrated")]
        for run in runs:
            self.assertEqual(run["pool_hash"],runs[0]["pool_hash"])
            self.assertEqual(run["artifact_revision"],runs[0]["artifact_revision"])
            self.assertEqual([q["id"] for q in run["query"]],[q["id"] for q in runs[0]["query"]])
        self.assertEqual(runs[1]["config"]["inner_steps"],1)
        self.assertEqual(runs[1]["config"]["inner_learning_rate"],.025)

    def test_multiview_live_scores_match_offline_pipeline(self):
        from .multiview import combine_views, predict as multiview_predict
        from .graph_calibration import centered_whitening, apply_transform, GraphConfig
        from .data import normalized_tensor
        from .experiment import load_checkpoint
        model,_=load_checkpoint(self.directory/'aligned.pt')
        with torch.inference_mode():
            normal=model.features(normalized_tensor(self.images)).numpy()
            flipped=model.features(normalized_tensor(self.images).flip(-1)).numpy()
        combined=combine_views(normal,flipped,'concat')
        transform=centered_whitening(combined,1,.5)
        cfg=GraphConfig(neighbors=5,graph_weight=.5,balanced=True,prototype_steps=2)
        plan={'status':'locked','selected':{str(n):{'views':'concat','assignment':'balanced_capacity','space':{'strength':1.,'exponent':.5},'config':cfg.to_dict()} for n in (1,5)}}
        config_path=self.directory/'multiview_configuration.json'
        config_path.write_text(json.dumps(plan),encoding='utf-8')
        transform_path=self.directory/'multiview_transform.npz'
        np.savez_compressed(transform_path,**{f'shot{n}_{k}':v for n in (1,5) for k,v in transform.items()})
        episodes=self.directory/'multiview_episodes.json';episodes.write_text('{}',encoding='utf-8')
        names=('multiview_configuration.json','multiview_transform.npz','multiview_episodes.json')
        manifest=self.directory/'multiview_results.json'
        manifest.write_text(json.dumps({'status':'complete','base_weights_sha256':digest_file(self.directory/'aligned.pt'),
            'files':{name:{'path':name,'sha256':digest_file(self.directory/name)} for name in names}}),encoding='utf-8')
        try:
            features=apply_transform(combined,transform)
            for shots in (1,5):
                task=sample_episode(self.labels,np.random.default_rng(19),shots=shots)
                expected=multiview_predict(features[task['support_indices']],task['support_labels'],features[task['query_indices']],cfg,'balanced_capacity')
                live=run_episode(self.directory,shots=shots,seed=19,method='calibrated',calibration='multiview')
                self.assertEqual(live['decision_rule'],'balanced_capacity')
                self.assertEqual([q['prediction'] for q in live['query']],expected['predictions'].tolist())
                np.testing.assert_allclose([q['probabilities'] for q in live['query']],expected['probabilities'],atol=1e-5)
                for q in live['query']:
                    self.assertAlmostEqual(q['confidence'],q['probabilities'][q['prediction']])
                    self.assertEqual(q['feature_before'],q['feature_after'])
        finally:
            for name in (*names,'multiview_results.json'):
                target=self.directory/name
                if not target.resolve().is_relative_to(self.directory):raise AssertionError('Invalid fixture cleanup')
                target.unlink(missing_ok=True)
            clear_cache()

    def test_boolean_and_malformed_direct_inputs_fail_before_loading(self):
        for kwargs in ({"shots":True},{"seed":True},{"seed":-1},{"method":[]},{"calibration":{}},{"shots":1.0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                run_episode(self.directory,**kwargs)

    def test_optional_wider_backbone_is_versioned_and_uses_same_episode(self):
        from .experiment import load_checkpoint
        from .graph_calibration import centered_whitening,apply_transform,GraphConfig,predict
        from .data import normalized_tensor
        from .wide_extension import read_extension
        model,_=load_checkpoint(self.directory/'aligned.pt')
        save_checkpoint(self.directory/'wide.pt',model,purpose="wider-path fixture, not a benchmark",metadata={})
        with torch.inference_mode(): features=model.features(normalized_tensor(self.images)).numpy()
        transform=centered_whitening(features,1,.5)
        cfg=GraphConfig(graph_weight=.5,neighbors=5)
        np.savez_compressed(self.directory/'wide_transform.npz',**{f'shot{n}_{k}':v for n in (1,5) for k,v in transform.items()})
        config={'selected':{str(n):{'space':{'strength':1.,'exponent':.5},'config':cfg.to_dict()} for n in (1,5)}}
        (self.directory/'wide_configuration.json').write_text(json.dumps(config),encoding='utf-8')
        for name in ('wide_episodes.json','wide_validation.json','wide_training_plan.json','wide_training.json'):
            (self.directory/name).write_text('{}',encoding='utf-8')
        names=('wide.pt','wide_transform.npz','wide_configuration.json','wide_episodes.json','wide_validation.json','wide_training_plan.json','wide_training.json')
        manifest={'status':'complete','candidate_checkpoint_sha256':digest_file(self.directory/'wide.pt'),
            'files':{n:{'path':n,'sha256':digest_file(self.directory/n)} for n in names}}
        (self.directory/'wide_results.json').write_text(json.dumps(manifest),encoding='utf-8')
        try:
            self.assertIsNotNone(read_extension(self.directory))
            task=sample_episode(self.labels,np.random.default_rng(123),shots=1)
            x=apply_transform(features,transform)
            expected=predict(x[task['support_indices']],task['support_labels'],x[task['query_indices']],cfg)
            live=run_episode(self.directory,shots=1,seed=123,method='calibrated',calibration='wide')
            self.assertEqual([q['prediction'] for q in live['query']],expected['predictions'].tolist())
            self.assertEqual(live['checkpoint_sha256'],manifest['candidate_checkpoint_sha256'])
            self.assertTrue(live['config']['larger_backbone'])
            self.assertEqual(live['metric'],'wider_backbone_graph')
            baseline=run_episode(self.directory,shots=1,seed=123,method='baseline')
            self.assertEqual([q['id'] for q in live['query']],[q['id'] for q in baseline['query']])
            self.assertEqual(live['artifact_revision'],baseline['artifact_revision'])
        finally:
            for name in (*names,'wide_results.json'):
                path=self.directory/name
                if not path.resolve().is_relative_to(self.directory):raise AssertionError('Unsafe fixture cleanup')
                path.unlink(missing_ok=True)
            clear_cache()

    def test_fingerprint_rejects_a_modified_checkpoint_before_prediction(self):
        path = self.directory / "aligned.pt"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"tamper")
            with self.assertRaisesRegex(ValueError,"Checksum mismatch"):
                run_episode(self.directory,method="calibrated")
        finally:
            path.write_bytes(original)
            clear_cache()


if __name__ == "__main__":
    unittest.main()
