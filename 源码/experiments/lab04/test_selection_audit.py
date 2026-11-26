"""Validation evidence must be recomputed, never trusted through a success flag."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
import tempfile
from .selection_audit import validate_selection

ROOT=Path(__file__).resolve().parents[2]/'outputs/lab04'


class SelectionAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=json.loads((ROOT/'wideview_plan.json').read_text(encoding='utf-8'))
        cls.selection=json.loads((ROOT/'wideview_selection.json').read_text(encoding='utf-8'))
        cls.journal=[json.loads(line) for line in (ROOT/'next_study/wide_views_20261125/scores.jsonl').read_text(encoding='utf-8').splitlines()]

    def test_published_candidate_and_negative_study_recompute(self):
        positive=validate_selection(self.plan,self.selection,self.journal)
        self.assertEqual(positive['candidate_evaluations'],36)
        self.assertEqual(positive['score_series'],48)
        self.assertEqual(positive['shots'],{'1':{'selected_index':4,'promising':True},'5':{'selected_index':4,'promising':True}})
        root=ROOT/'next_study/cached_refinement_20261125'
        negative=validate_selection(json.loads((root/'selection_plan.json').read_text(encoding='utf-8')),
            json.loads((root/'selection.json').read_text(encoding='utf-8')),
            (root/'scores.jsonl').read_text(encoding='utf-8').splitlines())
        self.assertFalse(negative['shots']['1']['promising'])
        self.assertFalse(negative['shots']['5']['promising'])

    def test_tampered_raw_scores_fail_independently_of_summary_flags(self):
        journal=deepcopy(self.journal);journal[0]['values'][0]=1.
        with self.assertRaisesRegex(ValueError,'summary mismatch'):
            validate_selection(self.plan,self.selection,journal)

    def test_duplicate_missing_extra_and_invalid_score_series_fail(self):
        for mutate in (
            lambda j:j.append(deepcopy(j[0])),
            lambda j:j.pop(),
            lambda j:j.append({'key':'not-registered','values':[.5]}),
            lambda j:j[0]['values'].pop(),
            lambda j:j[0]['values'].__setitem__(0,True),
            lambda j:j[0]['values'].__setitem__(0,.501),
            lambda j:j[0]['values'].__setitem__(0,float('nan')),
        ):
            with self.subTest(mutate=mutate):
                journal=deepcopy(self.journal);mutate(journal)
                with self.assertRaises(ValueError):validate_selection(self.plan,self.selection,journal)

    def test_changed_grid_mean_interval_and_gate_fail(self):
        for mutate in (
            lambda p,s:p['candidates'].reverse(),
            lambda p,s:p.__setitem__('search_episodes',159),
            lambda p,s:s['shots']['1'].__setitem__('promising',False),
            lambda p,s:s['shots']['1']['selected']['confirmation'].__setitem__('accuracy',.99),
            lambda p,s:s['shots']['1']['candidates'][0]['search'].__setitem__('ci95',0.),
            lambda p,s:s['shots']['1']['candidates'][0]['search_paired'].__setitem__('wins',999),
        ):
            with self.subTest(mutate=mutate):
                p,s=deepcopy(self.plan),deepcopy(self.selection);mutate(p,s)
                with self.assertRaises(ValueError):validate_selection(p,s,self.journal)

    def test_different_shortlisted_winner_is_not_accepted(self):
        selection=deepcopy(self.selection);branch=selection['shots']['1']
        branch['selected']=next(row for row in branch['candidates'] if 'confirmation' in row and row['index']!=branch['selected']['index'])
        with self.assertRaisesRegex(ValueError,'winner does not follow'):
            validate_selection(self.plan,selection,self.journal)

    def test_audit_cache_observes_journal_and_original_lock_edits(self):
        from .artifact_audit import observations
        with tempfile.TemporaryDirectory(prefix='lab04_selection_stamp_') as name:
            root=Path(name).resolve();folder=root/'next_study/wide_views_20261125';folder.mkdir(parents=True)
            self.assertEqual(root.parent,Path(tempfile.gettempdir()).resolve())
            journal=folder/'scores.jsonl';lock=folder/'selection_lock.json'
            journal.write_text('{}\n',encoding='utf-8');lock.write_text('{}',encoding='utf-8')
            initial=observations(root)
            journal.write_text('{"changed":true}\n',encoding='utf-8')
            updated=observations(root)
            self.assertNotEqual(initial,updated)
            lock.write_text('{"changed":true}',encoding='utf-8')
            self.assertNotEqual(updated,observations(root))

    def test_extra_confirmation_cannot_expand_search_shortlist(self):
        selection=deepcopy(self.selection);branch=selection['shots']['1']
        extra=next(row for row in branch['candidates'] if 'confirmation' not in row)
        extra['confirmation']=deepcopy(branch['selected']['confirmation'])
        extra['confirmation_paired']=deepcopy(branch['selected']['confirmation_paired'])
        with self.assertRaisesRegex(ValueError,'confirmation not exactly'):
            validate_selection(self.plan,selection,self.journal)


if __name__=='__main__':unittest.main()
