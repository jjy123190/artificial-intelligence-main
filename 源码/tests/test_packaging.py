"""Full-project submission packaging with no private state or installed dependencies."""
from __future__ import annotations
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import zipfile
from scripts.package_project import build_package, project_files

class PackagingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='course_packaging_tests_')
        self.root=Path(self.temp.name).resolve()
        if self.root.parent!=Path(tempfile.gettempdir()).resolve() or not self.root.name.startswith('course_packaging_tests_'):
            raise AssertionError('Temporary directory outside intended test scope')
        for name in ('main.py','README.md','打开网站.cmd','experiments/lab01/service.py','experiments/lab02/service.py','experiments/lab03/service.py','experiments/lab03/打开网站.cmd','experiments/lab01/打开网站.cmd','experiments/lab02/打开网站.cmd','backend/api.py','core/kb/domain.py','web/src/App.tsx','web/dist/index.html','web/dist/assets/site.js','data/raw/holdings.csv','outputs/lab01/benchmark.json','outputs/lab02/experiments/models.json','deliverables/lab01/report.docx'):
            self.write(name,'test:'+name)
        for name in ('.state/private.json','.venv/env.json','web/node_modules/private.json','outputs/qa/capture.json','outputs/runtime/server.json','outputs/lab02/qa/test-summary.json','outputs/submissions/old.json','人工智能作业一/old.py','人工智能作业二/old.py','__pycache__/cache.pyc','temporary.tmp'):
            self.write(name,'excluded')

    def write(self,name,text):
        target=self.root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text,encoding='utf-8')

    def tearDown(self):
        if self.root.parent!=Path(tempfile.gettempdir()).resolve() or not self.root.name.startswith('course_packaging_tests_'):
            raise AssertionError('Cleanup outside test scope')
        self.temp.cleanup()

    def test_inventory_contains_both_experiments_and_static_site(self):
        paths={p.relative_to(self.root).as_posix() for p in project_files(self.root)}
        for name in ('experiments/lab01/service.py','experiments/lab02/service.py','web/dist/index.html','backend/api.py','core/kb/domain.py','data/raw/holdings.csv','deliverables/lab01/report.docx'):
            self.assertIn(name,paths)
        self.assertFalse(any('node_modules' in p or '.state' in p or '.venv' in p or '/qa/' in p or 'runtime' in p or '人工智能作业' in p for p in paths))

    def test_archive_has_exact_checksums_and_selected_entry(self):
        result=build_package('lab01',self.root)
        self.assertTrue(result['all_hashes_match']);self.assertTrue(result['private_state_excluded'])
        with zipfile.ZipFile(result['archive']) as archive:
            expected=json.loads(archive.read('SHA256.json'))
            self.assertEqual(set(archive.namelist()),set(expected)|{'SHA256.json'})
            self.assertIsNone(archive.testzip())
            for name,digest in expected.items():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(),digest)
            note=archive.read('本次实验提交说明.txt').decode('utf-8')
            self.assertIn('#/labs/lab01',note)
            self.assertIn('experiments/lab01/打开网站.cmd',note)

    def test_lab02_submission_still_contains_lab01(self):
        result=build_package('lab02',self.root)
        with zipfile.ZipFile(result['archive']) as archive:
            self.assertIn('experiments/lab01/service.py',archive.namelist())
            self.assertIn('experiments/lab02/service.py',archive.namelist())
            self.assertIn('#/labs/lab02',archive.read('本次实验提交说明.txt').decode('utf-8'))

    def test_lab03_submission_keeps_prior_labs_and_jsonl_replay_data(self):
        self.write('outputs/lab02/end_to_end/reference_graphs.jsonl','{"example":true}\n')
        result=build_package('lab03',self.root)
        with zipfile.ZipFile(result['archive']) as archive:
            for name in ['experiments/lab01/service.py','experiments/lab02/service.py','experiments/lab03/service.py','outputs/lab02/end_to_end/reference_graphs.jsonl']:
                self.assertIn(name,archive.namelist())
            self.assertIn('#/labs/lab03',archive.read('本次实验提交说明.txt').decode('utf-8'))

    def test_missing_build_is_not_silently_packaged(self):
        (self.root/'web/dist/index.html').unlink()
        with self.assertRaises(FileNotFoundError):build_package('lab01',self.root)

    def test_unknown_experiment_rejected_before_packaging(self):
        with self.assertRaises(ValueError):build_package('lab99',self.root)

if __name__=='__main__':unittest.main()


class LabFourPackagingTests(PackagingTests):
    def test_lab04_packages_model_and_prior_but_never_downloaded_data(self):
        self.write("experiments/lab04/打开网站.cmd", "launcher")
        self.write("experiments/lab04/inference.py", "inference")
        self.write("outputs/lab04/baseline.pt", "checkpoint")
        self.write("outputs/lab04/baseline_prior.npz", "statistics")
        self.write("outputs/lab04/demo_pool.npz", "demo pool")
        self.write(".local/mini-imagenet/test/private.jpg", "downloaded test data")
        self.write(".local/lab04-v2/resume.pt", "local training state")
        result=build_package("lab04",self.root)
        with zipfile.ZipFile(result["archive"]) as archive:
            self.assertIn("outputs/lab04/baseline.pt",archive.namelist())
            self.assertIn("outputs/lab04/baseline_prior.npz",archive.namelist())
            self.assertIn("outputs/lab04/demo_pool.npz",archive.namelist())
            self.assertFalse(any(name.startswith(".local/") for name in archive.namelist()))
            self.assertIn("#/labs/lab04",archive.read("本次实验提交说明.txt").decode("utf-8"))


    def test_packaged_inventory_describes_only_included_bytes(self):
        self.write("文件校验清单.json", json.dumps({"schema_version":1,"files":{".local/secret":{},"main.py":{}}}))
        self.write(".gitattributes", "* -text")
        result=build_package("lab04",self.root)
        with zipfile.ZipFile(result["archive"]) as archive:
            manifest=json.loads(archive.read("文件校验清单.json"))
            self.assertNotIn(".local/secret",manifest["files"])
            self.assertIn(".gitattributes",manifest["files"])
            for name,entry in manifest["files"].items():
                actual=archive.read(name)
                self.assertEqual(len(actual),entry["bytes"])
                self.assertEqual(hashlib.sha256(actual).hexdigest(),entry["sha256"])


class LiteratureReadingCachePackagingTests(PackagingTests):
    def test_literature_snapshots_not_redistributed_but_evidence_kept(self):
        prefix='outputs/lab04/next_study/literature_20261125/'
        for name in ('rfs_pdf.pdf','feat_paper.html','tim_code.txt','rfs_readme.md'):
            self.write(prefix+name,'third-party reading snapshot')
        for name in ('README.md','sources.json','verified_sources.json','reference_cache_spotcheck.json'):
            self.write(prefix+name,'source metadata / our evidence')
        self.write('outputs/lab04/next_study/literature_inference_20261125/selection.json','negative result')
        self.write(prefix+'cost_probe/result.json','our synthetic timing')
        self.write('experiments/lab04/LITERATURE_IMPROVEMENT_PLAN.md','our research plan')
        paths={p.relative_to(self.root).as_posix() for p in project_files(self.root)}
        for name in ('rfs_pdf.pdf','feat_paper.html','tim_code.txt','rfs_readme.md'):
            self.assertNotIn(prefix+name,paths)
        for name in ('README.md','sources.json','verified_sources.json','reference_cache_spotcheck.json'):
            self.assertIn(prefix+name,paths)
        self.assertIn(prefix+'cost_probe/result.json',paths)
        self.assertIn('outputs/lab04/next_study/literature_inference_20261125/selection.json',paths)
        self.assertIn('experiments/lab04/LITERATURE_IMPROVEMENT_PLAN.md',paths)


class RequirementResearchCachePackagingTests(PackagingTests):
    def test_raw_research_source_excluded_but_results_kept(self):
        prefix='outputs/lab04/next_study/requirement_accuracy_20261125/'
        self.write(prefix+'frn_model.txt','third-party source')
        self.write(prefix+'README.md','our explanation')
        self.write(prefix+'cuda_probe.json','our read-only diagnostic')
        self.write(prefix+'learned_calibration/result.json','our negative result')
        paths={p.relative_to(self.root).as_posix() for p in project_files(self.root)}
        self.assertNotIn(prefix+'frn_model.txt',paths)
        for name in ('README.md','cuda_probe.json','learned_calibration/result.json'):
            self.assertIn(prefix+name,paths)
