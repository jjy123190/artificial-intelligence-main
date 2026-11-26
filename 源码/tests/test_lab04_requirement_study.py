import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from experiments.lab04.requirement_study_audit import raw_values,validate_tasks,check_plan
from experiments.lab04.study_literature import write

class RequirementStudyAuditTests(unittest.TestCase):
 def test_raw_values_and_tamper(self):
  tasks=[{'query_labels':[0,1]},{'query_labels':[1,0]}]
  record={'plan_sha256':'p','test_accessed':False,'predictions':[[0,1],[0,0]],'accuracies':[1.,.5]}
  self.assertEqual(raw_values(record,tasks,'p'),[1.,.5])
  for change in ({'accuracies':[1.,1.]},{'plan_sha256':'wrong'},{'test_accessed':True},{'predictions':[[0.,1.],[0.,0.]]}):
   with self.assertRaises(ValueError):raw_values({**record,**change},tasks,'p')
 def task(self):
  ids=[f'val/class{c}/{i}.jpg' for c in range(5) for i in range(16)]
  task={'support_indices':[c*16 for c in range(5)],'query_indices':[c*16+i for c in range(5) for i in range(1,16)],'support_labels':list(range(5)),'query_labels':[c for c in range(5) for _ in range(15)]}
  return task,ids
 def test_task_labels_and_isolation(self):
  t,ids=self.task();validate_tasks([t],ids)
  with self.assertRaises(ValueError):validate_tasks([{**t,'support_indices':[1,16,32,48,64]}],ids)
  with self.assertRaises(ValueError):validate_tasks([{**t,'support_labels':[1,0,2,3,4]}],ids)
  with self.assertRaises(ValueError):validate_tasks([t],[s.replace('val/','test/') for s in ids])
 def test_source_identity_rechecked(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);write(root/'plan.json',{'test_accessed':False,'external_weights':False,'source_hashes':{'requirement_study_audit.py':'wrong'}})
   with self.assertRaises(ValueError):check_plan(root)
 def test_teacher_not_disguised(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);write(root/'plan.json',{'test_accessed':False,'external_weights':True,'source_hashes':{}})
   with self.assertRaises(ValueError):check_plan(root)
