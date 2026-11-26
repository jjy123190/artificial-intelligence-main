"""Full delivery must include materials alongside source without leaking local state."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from scripts.package_lab04_delivery import CURRENT_REQUIRED,build_delivery,verify_materials,material_files


class LabFourDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='lab04_delivery_fixture_');self.root=Path(self.temp.name).resolve()
        self.assertEqual(self.root.parent,Path(tempfile.gettempdir()).resolve())
        source=self.root/'源码';source.mkdir();self.source=source
        for name in ('main.py','README.md','打开网站.cmd','experiments/lab04/打开网站.cmd','web/dist/index.html','outputs/lab04/results.json'):
            path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture',encoding='utf-8')
        for name in ('.local/private.json','.state/personal.json','web/node_modules/package.json'):
            path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('private',encoding='utf-8')
        self.current=self.root/'deliverables/lab4/当前版本';self.current.mkdir(parents=True)
        for name in CURRENT_REQUIRED:(self.current/name).write_text('fixture '+name,encoding='utf-8')
        self.revision='a'*64
        digest=lambda name:hashlib.sha256((self.current/name).read_bytes()).hexdigest()
        runs=[{'artifact_revision':self.revision,'method':method,'shots':shot,'total':75,'correct':30,'accuracy':.4,
               'support_ids':[f's{i}' for i in range(shot*5)],'query_ids':[f'q{i}' for i in range(75)]}
              for method,shot in [('baseline',1),('calibrated',1),('maml',1),('calibrated',5)]]
        entries={
            'report_evidence.json':{'artifact_revision':self.revision,'report_sha256':digest('实验四_小样本分类与特征校准_当前整合报告.md')},
            'slides_evidence.json':{'artifact_revision':self.revision,'slide_sha256':digest('实验四_小样本分类与特征校准_当前汇报.pptx')},
            'presentation_evidence.json':{'artifact_revision':self.revision,'slides_sha256':digest('实验四_小样本分类与特征校准_当前汇报.pptx'),'video_sha256':digest('实验四_当前版本PPT讲解.mp4')},
            'demo_evidence.json':{'artifact_revision':self.revision,'runs':runs,'console_errors':[],'video_sha256':digest('实验四_当前页面与真实推理_Demo.mp4')},
            'media_support_evidence.json':{'artifact_revision':self.revision,'files':{name:digest(name) for name in CURRENT_REQUIRED if name.endswith(('.jpg','.vtt'))}},
        }
        for name,value in entries.items():(self.current/name).write_text(json.dumps(value),encoding='utf-8')

    def tearDown(self):self.temp.cleanup()

    def test_bundle_layout_checksums_captions_and_launcher_are_correct(self):
        with patch('experiments.lab04.artifact_audit.audit_publication',return_value={'artifact_revision':self.revision}):
            result=build_delivery(self.root)
        self.assertTrue(result['all_hashes_match'])
        with zipfile.ZipFile(result['archive']) as z:
            names=set(z.namelist());hashes=json.loads(z.read('SHA256.json'))
            self.assertEqual(names,set(hashes)|{'SHA256.json'})
            self.assertIn('源码/main.py',names);self.assertIn('deliverables/lab4/当前版本/实验四_当前版本PPT讲解.vtt',names)
            self.assertFalse(any('.local/' in name or '.state/' in name or '/node_modules/' in name for name in names))
            launcher=z.read('打开实验四.cmd').decode('utf-8')
            self.assertIn('源码\\experiments\\lab04\\打开网站.cmd',launcher)
            self.assertNotIn('打开网站.cmd" lab04',launcher)
            for name,digest in hashes.items():self.assertEqual(hashlib.sha256(z.read(name)).hexdigest(),digest)

    def test_missing_current_file_is_not_silently_omitted(self):
        path=self.current/CURRENT_REQUIRED[0];path.unlink()
        with self.assertRaises(FileNotFoundError):material_files(self.root)

    def test_wrong_media_revision_and_changed_demo_are_rejected(self):
        file=self.current/'demo_evidence.json';original=file.read_bytes();value=json.loads(original);value['artifact_revision']='b'*64
        file.write_text(json.dumps(value),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'mix publication'):verify_materials(self.root)
        file.write_bytes(original);(self.current/'实验四_当前页面与真实推理_Demo.mp4').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Demo video'):verify_materials(self.root)

    def test_unpaired_real_demo_claim_is_rejected(self):
        file=self.current/'demo_evidence.json';value=json.loads(file.read_text())
        value['runs'][1]['query_ids'].reverse();file.write_text(json.dumps(value),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'not paired'):verify_materials(self.root)

    def test_changed_model_revision_prevents_packaging(self):
        with patch('experiments.lab04.artifact_audit.audit_publication',return_value={'artifact_revision':'other'}):
            with self.assertRaisesRegex(ValueError,'do not describe'):build_delivery(self.root)
        self.assertFalse((self.source/'outputs/submissions').exists())


if __name__=='__main__':unittest.main()
