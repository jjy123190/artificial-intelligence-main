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
        for name in ('main.py','README.md','打开网站.cmd','experiments/lab01/service.py','experiments/lab02/service.py','experiments/lab01/打开网站.cmd','experiments/lab02/打开网站.cmd','backend/api.py','core/kb/domain.py','web/src/App.tsx','web/dist/index.html','web/dist/assets/site.js','data/raw/holdings.csv','outputs/lab01/benchmark.json','outputs/lab02/experiments/models.json','deliverables/lab01/report.docx'):
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

    def test_missing_build_is_not_silently_packaged(self):
        (self.root/'web/dist/index.html').unlink()
        with self.assertRaises(FileNotFoundError):build_package('lab01',self.root)

    def test_unknown_experiment_rejected_before_packaging(self):
        with self.assertRaises(ValueError):build_package('lab03',self.root)

if __name__=='__main__':unittest.main()
