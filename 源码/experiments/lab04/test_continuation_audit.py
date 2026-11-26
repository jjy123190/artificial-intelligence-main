from copy import deepcopy
import unittest
from .continuation_audit import audit_epoch_selection,audit_tasks


class ContinuationAuditTests(unittest.TestCase):
    def test_epoch_zero_and_first_maximum_are_selected_honestly(self):
        plan={'epochs':2,'test_accessed':False};initial={'1shot':.4,'5shot':.6,'mean':.5}
        history=[{'epoch':i,'ce_loss':1.,'metric_loss':1.,'seconds':1.,'training_accuracy':.5,'validation':dict(initial)} for i in (1,2)]
        training={'status':'complete','test_accessed':False,'rows':history,'best_epoch':0,'best_mean':.5}
        self.assertEqual(audit_epoch_selection(plan,training,initial),0)
        history[0]['validation']={'1shot':.5,'5shot':.7,'mean':.6};history[1]['validation']=dict(history[0]['validation'])
        training.update(best_epoch=1,best_mean=.6)
        self.assertEqual(audit_epoch_selection(plan,training,initial),1)
        training['best_epoch']=2
        with self.assertRaisesRegex(ValueError,'first strict'):audit_epoch_selection(plan,training,initial)

    def test_training_incomplete_cannot_be_treated_as_complete(self):
        with self.assertRaisesRegex(ValueError,'training incomplete'):
            audit_epoch_selection({'test_accessed':False},{'status':'running','test_accessed':False},{})

    def fixture(self):
        records=[]
        for shot in (1,5):
            for episode in (0,1):
                labels=list(range(5))*15;pred=[(v+1)%5 for v in labels]
                for i in range(30):pred[i]=labels[i]
                records.append({'shots':shot,'episode':episode,'support_ids':[f'val/s{i}' for i in range(shot*5)],'query_ids':[f'val/q{i}' for i in range(75)],
                    'query_labels':labels,'predictions':pred,'reference_predictions':pred.copy(),'accuracy':.4,'reference_accuracy':.4})
        shots={str(n):{'candidate':{'accuracy':.4,'ci95':0.,'episodes':2},'reference':{'accuracy':.4,'ci95':0.,'episodes':2},
            'paired':{'mean':0.,'ci95':0.,'wins':0,'ties':2,'losses':0},'promising':False} for n in (1,5)}
        return records,shots

    def test_identical_candidate_is_not_test_eligible(self):
        records,shots=self.fixture();result=audit_tasks(records,shots,2)
        self.assertFalse(result['1']['promising']);self.assertFalse(result['5']['promising'])

    def test_modified_prediction_sample_identity_or_flag_fails(self):
        for mutate in (
            lambda r,s:r[0]['predictions'].__setitem__(0,4),
            lambda r,s:r[0]['support_ids'].__setitem__(0,'test/not-val'),
            lambda r,s:r[0]['support_ids'].__setitem__(0,r[0]['query_ids'][0]),
            lambda r,s:s['1'].__setitem__('promising',True),
            lambda r,s:r.append(deepcopy(r[0])),
        ):
            records,shots=self.fixture();mutate(records,shots)
            with self.assertRaises(ValueError):audit_tasks(records,shots,2)


if __name__=='__main__':unittest.main()
