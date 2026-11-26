"""Small exact-byte fixtures to prove audit checks and cache invalidation."""
from __future__ import annotations
import hashlib,json,math,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from experiments.lab04.artifact_audit import audit_publication,clear_audit_cache
from backend.lab_four import LabFourService
from backend.service import WebError


def write(path, value):
    path.write_text(json.dumps(value),encoding="utf-8")


class PublicationAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="lab04_audit_fixture_")
        self.root=Path(self.temp.name).resolve()
        if self.root.parent!=Path(tempfile.gettempdir()).resolve():raise AssertionError("Unexpected fixture root")
        hashes={}
        for name in ("baseline.pt","aligned.pt","maml.pt","baseline_prior.npz","aligned_prior.npz","demo_pool.npz"):
            raw=name.encode();(self.root/name).write_bytes(raw);hashes[name]={"path":name,"sha256":hashlib.sha256(raw).hexdigest()}
        records=[]
        for shots in (1,5):
            for episode in (0,1):
                targets=list(range(5))*15
                preds=[(v+1)%5 for v in targets]
                for i in range(30):preds[i]=targets[i]
                records.append({"shots":shots,"episode":episode,"classes":list(range(5)),
                    "support_ids":[f"s{i}" for i in range(shots*5)],"query_ids":[f"q{i}" for i in range(75)],
                    "query_labels":targets,"methods":{m:{"accuracy":.4,"seconds":.01,"correct":30,"predictions":preds.copy()} for m in ("baseline","full","maml")}})
        self.records=records
        self.result={"schema_version":1,"status":"complete","completed_utc":"2026-11-24T00:00:00+00:00",
            "protocol":{"seed":10,"ways":5,"shots":[1,5],"queries_per_class":15,"test_episodes_per_shot":2,
                "evaluation_pool_per_class":100,"dataset":"mini-ImageNet","image_size":[84,84],"training_classes":64,
                "validation_classes":16,"test_classes":20,"train_per_class":600},
            "model":{"name":"Fixture","widths":[8,16,24,32],"parameters_baseline":50352},
            "weights":{n:hashes[n+".pt"] for n in ("baseline","aligned","maml")},
            "priors":{n:hashes[n+"_prior.npz"] for n in ("baseline","aligned")},"demo_pool":hashes["demo_pool.npz"],
            "rows":[{"method":m,"shots":s,"accuracy":.4,"ci95":0.,"paired_delta":0.,"paired_ci95":0.,"episodes":2,"mean_episode_seconds":.01} for s in (1,5) for m in ("baseline","full","maml")],
            "training":{"warm_ce":{"epochs":[]},"aligned":{"alignment_weight":.5}}}
        write(self.root/"results.json",self.result)
        write(self.root/"episodes.json",{"seed":210,"episodes":records})

    def tearDown(self):
        clear_audit_cache()
        if not self.root.name.startswith("lab04_audit_fixture_"):raise AssertionError("Unexpected cleanup")
        self.temp.cleanup()

    def test_valid_artifacts_are_recomputed_and_cache_reused(self):
        first=audit_publication(self.root)
        self.assertTrue(first["verified"]);self.assertTrue(first["ready"])
        self.assertEqual(first["episodes_recomputed"],4)
        with patch("experiments.lab04.artifact_audit.audit_core",side_effect=AssertionError("Cache should be reused")):
            second=audit_publication(self.root)
            self.assertEqual(first,second)
        second["core"]["rows"][0]["accuracy"] = .9
        self.assertEqual(audit_publication(self.root)["core"]["rows"][0]["accuracy"],.4)

    def test_editing_summary_only_fails_even_when_weights_unchanged(self):
        audit_publication(self.root)
        self.result["rows"][1]["accuracy"] = .99
        write(self.root/"results.json",self.result)
        with self.assertRaisesRegex(ValueError,"Measured evidence mismatch"):
            audit_publication(self.root)
        with self.assertRaises(WebError):LabFourService(self.root).snapshot()

    def test_editing_predictions_invalidates_existing_cached_audit(self):
        audit_publication(self.root)
        self.records[0]["methods"]["full"]["predictions"][0]=4
        write(self.root/"episodes.json",{"seed":210,"episodes":self.records})
        with self.assertRaisesRegex(ValueError,"correct count"):
            audit_publication(self.root)

    def test_duplicate_rows_are_not_silently_displayed(self):
        self.result["rows"][-1]=dict(self.result["rows"][-2])
        write(self.root/"results.json",self.result)
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            audit_publication(self.root)

    def test_invalid_nan_and_negative_uncertainty_are_rejected(self):
        for bad in (float("nan"),-.1):
            self.result["rows"][0]["ci95"]=bad
            write(self.root/"results.json",self.result)
            with self.assertRaises(ValueError):audit_publication(self.root)

    def test_weight_tampering_invalidates_a_cached_audit(self):
        audit_publication(self.root)
        (self.root/"aligned.pt").write_bytes(b"changed weights")
        with self.assertRaisesRegex(ValueError,"checksum"):
            audit_publication(self.root)

    def test_path_traversal_is_rejected(self):
        self.result["weights"]["aligned"]["path"]="../outside.pt"
        write(self.root/"results.json",self.result)
        with self.assertRaisesRegex(ValueError,"path"):
            audit_publication(self.root)

    def test_missing_optional_models_allows_results_without_readiness(self):
        (self.root/"aligned.pt").unlink()
        audit=audit_publication(self.root)
        self.assertFalse(audit["ready"]);self.assertTrue(audit["verified"])
        snapshot=LabFourService(self.root).snapshot()
        self.assertFalse(snapshot["available"])
        self.assertEqual(len(snapshot["comparisons"]),6)
        self.assertTrue(snapshot["verification"]["verified"])

    def test_empty_graph_dependency_manifest_cannot_skip_hash_checks(self):
        graph={"status":"complete","base_weights_sha256":self.result["weights"]["aligned"]["sha256"],"validated_files":{}}
        write(self.root/"graph_results.json",graph)
        with self.assertRaisesRegex(ValueError,"dependencies"):
            audit_publication(self.root)


    def test_missing_core_checkpoint_is_not_required_to_read_verified_old_scores(self):
        (self.root/"maml.pt").unlink()
        result=LabFourService(self.root).snapshot()
        self.assertFalse(result["available"])
        self.assertTrue(result["verification"]["verified"])

    def test_graph_configuration_must_match_locked_selection(self):
        # Even a self-consistent hash manifest cannot authenticate a config
        # that differs from the original pre-test selection record.
        config={"configurations":{"1":{"config":{"neighbors":5}},"5":{"config":{"neighbors":5}}}}
        lock={"test_accessed":False,"weights_sha256":self.result["weights"]["aligned"]["sha256"],
              "selected":{"1":{"config":{"neighbors":10}},"5":{"config":{"neighbors":5}}}}
        deps={"graph_configuration.json":config,"graph_lock.json":lock,"graph_episodes.json":{"episodes":[]}}
        for name,value in deps.items():write(self.root/name,value)
        (self.root/"graph_transform.npz").write_bytes(b"not loaded by lightweight verifier")
        names=(*deps,"graph_transform.npz")
        extension={"status":"complete","base_weights_sha256":self.result["weights"]["aligned"]["sha256"],
                   "validated_files":{name:{"path":name,"sha256":hashlib.sha256((self.root/name).read_bytes()).hexdigest()} for name in names}}
        write(self.root/"graph_results.json",extension)
        with self.assertRaisesRegex(ValueError,"inference config"):
            audit_publication(self.root)


    def test_changed_files_during_recomputation_are_rejected(self):
        from experiments.lab04 import artifact_audit as module
        original=module.audit_core
        def changing(directory):
            value=original(directory)
            with (self.root/"episodes.json").open("a",encoding="utf-8") as stream:
                stream.write("\n")
            return value
        with patch.object(module,"audit_core",side_effect=changing):
            with self.assertRaisesRegex(ValueError,"changed during"):
                module.audit_publication(self.root)

    def test_snapshot_does_not_eagerly_import_ml(self):
        import subprocess,sys
        code="from pathlib import Path; from experiments.lab04.artifact_audit import audit_publication; import sys; audit_publication(Path(sys.argv[1])); assert 'torch' not in sys.modules and 'numpy' not in sys.modules"
        result=subprocess.run([sys.executable,"-B","-c",code,str(self.root)],capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)


    def test_missing_wide_reference_does_not_appear_as_ready(self):
        write(self.root/"wide_results.json",{"status":"complete","files":{}})
        with self.assertRaisesRegex(ValueError,"frozen extension references"):
            audit_publication(self.root)

    def test_core_scores_do_not_change_when_optional_wide_files_absent(self):
        result=audit_publication(self.root)
        self.assertIsNone(result["wide"])
        self.assertEqual(result["core"]["rows"],self.result["rows"])


    def test_wider_optional_study_is_recomputed_against_all_frozen_controls(self):
        from experiments.lab04.artifact_audit import audit_wide,audit_core
        core,reference,_=audit_core(self.root)
        # Four fixed tasks, all predicting 30/75 correctly, exactly like core.
        records=[];controls={}
        for shots in (1,5):
            for episode in (0,1):
                old=self.records[(shots==5)*2+episode]
                predicted=old["methods"]["full"]["predictions"]
                probabilities=[]
                for value in predicted:
                    scores=[.05]*5;scores[value]=.8;probabilities.append(scores)
                record={"shots":shots,"episode":episode,"support_ids":old["support_ids"],"query_ids":old["query_ids"],
                    "query_labels":old["query_labels"],"predictions":predicted,"raw_predictions":predicted,
                    "probabilities":probabilities,"accuracy":.4,"raw_accuracy":.4,"baseline_accuracy":.4,
                    "full_accuracy":.4,"graph_accuracy":.4,"multiview_accuracy":.4,"seconds":.01}
                records.append(record);controls[(shots,episode)]={"accuracy":.4}
        (self.root/"wide.pt").write_bytes(b"wider model fixture")
        (self.root/"wide_transform.npz").write_bytes(b"transform fixture")
        weight=hashlib.sha256((self.root/"wide.pt").read_bytes()).hexdigest()
        chosen={str(n):{"space":{"strength":1,"exponent":.5},"config":{"neighbors":5}} for n in (1,5)}
        validation={"status":"complete","test_accessed":False,"shots":{str(n):{"selected":chosen[str(n)]} for n in (1,5)}}
        write(self.root/"wide_validation.json",validation)
        write(self.root/"wide_training_plan.json",{"seed":10,"epochs":2})
        write(self.root/"wide_training.json",{"status":"complete","test_accessed":False,"best_epoch":2,"rows":[{},{}]})
        validation_hash=hashlib.sha256((self.root/"wide_validation.json").read_bytes()).hexdigest()
        lock={"status":"locked","test_accessed":False,"checkpoint_sha256":weight,"locked_utc":"2026-11-24T00:00:00+00:00",
            "selection_sha256":validation_hash,"training_plan_sha256":hashlib.sha256((self.root/"wide_training_plan.json").read_bytes()).hexdigest(),"selected":chosen}
        write(self.root/"wide_configuration.json",lock)
        lock_hash=hashlib.sha256((self.root/"wide_configuration.json").read_bytes()).hexdigest()
        write(self.root/"wide_episodes.json",{"episodes":records})
        names=("wide.pt","wide_transform.npz","wide_configuration.json","wide_episodes.json","wide_validation.json","wide_training_plan.json","wide_training.json")
        rows=[{"method":"wide","shots":n,"accuracy":.4,"ci95":0.,"episodes":2,"raw_accuracy":.4,"mean_episode_seconds":.01,
            "paired_delta":0.,"paired_ci95":0.,"paired_vs_full":0.,"paired_vs_full_ci95":0.,"paired_vs_graph":0.,"paired_vs_graph_ci95":0.,"paired_vs_multiview":0.,"paired_vs_multiview_ci95":0.} for n in (1,5)]
        result={"status":"complete","files":{name:{"path":name,"sha256":hashlib.sha256((self.root/name).read_bytes()).hexdigest()} for name in names},
            "candidate_checkpoint_sha256":weight,"selection_lock_sha256":lock_hash,"completed_utc":"2026-11-24T01:00:00+00:00",
            "training":{"selected_epoch":2,"epochs_completed":2},"rows":rows,
            "evaluation_plan":{"selection_lock_sha256":lock_hash,"baseline_checkpoint_sha256":core["weights"]["baseline"]["sha256"],"seed":10,"validation_reference_sha256":validation_hash}}
        write(self.root/"wide_results.json",result)
        self.assertEqual(audit_wide(self.root,core,reference,controls,controls)["rows"],rows)
        result["rows"][0]["accuracy"]=.9
        write(self.root/"wide_results.json",result)
        with self.assertRaisesRegex(ValueError,"evidence mismatch"):
            audit_wide(self.root,core,reference,controls,controls)


    def test_timing_medians_cannot_disagree_with_recorded_samples(self):
        revision=audit_publication(self.root)["artifact_revision"]
        row={"method":"baseline","shots":1,"repetitions":3,"weights_sha256":self.result["weights"]["baseline"]["sha256"],
             "artifact_revision":revision,"cold_wall_ms":50.,"wall_samples_ms":[10.,20.,30.],"compute_samples_ms":[5.,10.,15.],
             "warm_wall_median_ms":20.,"warm_wall_p95_ms":29.,"encoding_and_classification_median_ms":10.,"encoding_and_classification_p95_ms":14.5}
        write(self.root/"edge_benchmark.json",{"rows":[row]})
        self.assertIsNotNone(audit_publication(self.root)["edge_benchmark"])
        row["encoding_and_classification_median_ms"]=1.
        write(self.root/"edge_benchmark.json",{"rows":[row]})
        with self.assertRaisesRegex(ValueError,"median"):
            audit_publication(self.root)


if __name__=="__main__":unittest.main()
