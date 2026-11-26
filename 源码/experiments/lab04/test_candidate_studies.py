"""Pure and bounded checks for additional validation-only candidate studies."""
from __future__ import annotations
from argparse import Namespace
import inspect
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from .calibration import unit_features
from .graph_calibration import apply_transform, centered_whitening
from .study_cached_refinement import compose, load_wide, paired_stats
from .study_wide_views import extract_views
from .supervised_whitening import fit


class CandidateStudyTests(unittest.TestCase):
    def test_feature_fusion_is_inductive_normalized_and_weighted(self):
        rng=np.random.default_rng(52);wide=rng.normal(size=(15,16));light=rng.normal(size=(15,8))
        result=compose(wide,light,.25)
        self.assertEqual(result.shape,(15,24))
        np.testing.assert_allclose((result[:,:16]**2).sum(1),.75)
        np.testing.assert_allclose((result[:,16:]**2).sum(1),.25)
        np.testing.assert_allclose(result,np.concatenate([compose(wide[i:i+1],light[i:i+1],.25) for i in range(15)]))
        np.testing.assert_allclose(compose(wide,light,0),unit_features(wide))
        for mix in (-1,1,2):
            with self.assertRaises(ValueError):compose(wide,light,mix)
        with self.assertRaises(ValueError):compose(wide,light[:-1],.5)

    def test_zero_residual_reference_matches_frozen_total_covariance(self):
        rng=np.random.default_rng(41);train=rng.normal(size=(80,12));labels=np.repeat(np.arange(8),10);query=rng.normal(size=(9,12))
        original=centered_whitening(train,strength=1.,exponent=.5)
        residual=fit(train,labels,strength=1.,exponent=.5,residual_mix=0)
        np.testing.assert_allclose(apply_transform(query,original),apply_transform(query,residual),atol=1e-10)

    def test_paired_interval_uses_task_differences_not_independent_errors(self):
        reference=[.2,.4,.6,.8];value=paired_stats(np.array(reference)+.1,reference)
        self.assertAlmostEqual(value['mean'],.1)
        self.assertLess(value['ci95'],1e-15)
        self.assertEqual((value['wins'],value['ties'],value['losses']),(4,0,0))
        tied=paired_stats(reference,reference)
        self.assertEqual((tied['mean'],tied['ci95'],tied['ties']),(0,0,4))

    def test_test_split_rejected_before_file_or_model_access(self):
        with patch('pathlib.Path.read_text',side_effect=AssertionError('Do not touch any cache')):
            with self.assertRaisesRegex(ValueError,'train/validation'):
                load_wide(Path('not-accessed'),'test')
        with patch('experiments.lab04.study_wide_views.load_wide',side_effect=AssertionError('No model/cache access')):
            with self.assertRaisesRegex(ValueError,'Test data forbidden'):
                extract_views(Namespace(),'test')

    def test_fitting_does_not_have_a_query_label_argument(self):
        self.assertEqual(set(inspect.signature(fit).parameters),{'training_features','training_labels','strength','exponent','residual_mix'})
        self.assertNotIn('query_labels',inspect.signature(compose).parameters)


if __name__=='__main__':unittest.main()
