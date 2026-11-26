"""Module effects use within-task differences and disclose changed conditions."""
import json
import math
from pathlib import Path
import statistics
import unittest
from experiments.lab04.module_contrasts import CONTRASTS,summarize_contrasts
from backend.lab_four import LabFourService


class ModuleContrastUnitTests(unittest.TestCase):
    def test_constant_paired_shift_has_zero_interval_despite_varying_scores(self):
        records={(1,i):{'scores':{'full':v+.1,'no_alignment_full':v}} for i,v in enumerate((.2,.4,.6,.7))}
        row=summarize_contrasts(records)['rows'][0]
        self.assertEqual(row['id'],'alignment');self.assertAlmostEqual(row['paired_delta'],.1)
        self.assertLess(row['paired_ci95'],1e-15)
        self.assertEqual((row['wins'],row['ties'],row['losses']),(4,0,0))
        self.assertGreater(statistics.stdev([v['scores']['full'] for v in records.values()]),0)

    def test_zero_mean_mixed_effect_is_not_marked_as_no_variation(self):
        records={(5,i):{'scores':{'full':left,'prototype_only':right}} for i,(left,right) in enumerate(((.5,.4),(.4,.5),(.6,.6)))}
        row=summarize_contrasts(records)['rows'][0]
        self.assertEqual(row['id'],'statistics_on_prototype')
        self.assertAlmostEqual(row['paired_delta'],0);self.assertGreater(row['paired_ci95'],0)
        self.assertEqual((row['wins'],row['ties'],row['losses']),(1,1,1))
        self.assertLess(row['ci95_low'],0);self.assertGreater(row['ci95_high'],0)

    def test_absent_pairs_are_not_invented_and_incomplete_pairs_fail(self):
        self.assertEqual(summarize_contrasts({(1,0):{'scores':{'full':.4}}})['rows'],[])
        with self.assertRaisesRegex(ValueError,'missing method'):
            summarize_contrasts({(1,0):{'scores':{'full':.5,'no_alignment_full':.4}},(1,1):{'scores':{'full':.5}}})

    def test_nonfinite_and_boolean_scores_are_rejected(self):
        for bad in (float('nan'),float('inf'),True,-.1,1.1):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                summarize_contrasts({(1,0):{'scores':{'full':bad,'no_alignment_full':.4}}})

    def test_contrast_names_do_not_claim_single_module_for_combined_change(self):
        row=next(row for row in CONTRASTS if row[0]=='prototype_balanced')
        self.assertIn('均衡',row[1]);self.assertIn('不能全部归因',row[4])
        evidence=summarize_contrasts({});self.assertIn('多重比较',evidence['interval_note'])


class PublishedModuleContrastTests(unittest.TestCase):
    def test_all_web_contrasts_match_frozen_task_recalculation(self):
        root=Path(__file__).resolve().parents[1]/'outputs/lab04'
        snapshot=LabFourService(root).snapshot();rows=snapshot['module_contrasts']['rows']
        self.assertEqual(len(rows),12)
        tasks=json.loads((root/'episodes.json').read_text(encoding='utf-8'))['episodes']
        for row in rows:
            diffs=[r['methods'][row['method']]['accuracy']-r['methods'][row['reference']]['accuracy'] for r in tasks if r['shots']==row['shot']]
            self.assertEqual(len(diffs),row['episodes']);self.assertAlmostEqual(row['paired_delta'],statistics.mean(diffs),places=12)
            self.assertAlmostEqual(row['paired_ci95'],1.96*statistics.stdev(diffs)/math.sqrt(len(diffs)),places=12)
            self.assertEqual(row['wins']+row['ties']+row['losses'],600)
            self.assertAlmostEqual(row['method_accuracy']-row['reference_accuracy'],row['paired_delta'],places=12)


if __name__=='__main__':unittest.main()
