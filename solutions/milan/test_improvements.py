"""Checks that matter for the proposed ML and application changes."""
import hashlib
import json
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PACKAGES = HERE / '.generated/preview_packages'
if PACKAGES.exists():
    sys.path.insert(0, str(PACKAGES))
import numpy as np
import pandas as pd
from improve import Measurements, make_pipeline, OUT
from mushroom_alternative import development_data, scores, ROOT


class LearningChecks(unittest.TestCase):
    def test_geometry_units_zero_division_and_input_preservation(self):
        frame = pd.DataFrame({'cap_diameter': [10., 3.], 'stem_height': [5., 0.], 'stem_width': [2., 0.]})
        before = frame.copy(deep=True)
        result = Measurements(geometry=True).fit_transform(frame)
        self.assertEqual(result.cap_to_stem_height.iloc[0], 2.)
        self.assertEqual(result.stem_slenderness.iloc[0], 25.)
        self.assertTrue(np.isnan(result.cap_to_stem_height.iloc[1]))
        self.assertTrue(np.isnan(result.stem_slenderness.iloc[1]))
        pd.testing.assert_frame_equal(frame, before)

    def test_imputation_learns_only_training_and_unknown_categories_work(self):
        X, y, _, _ = development_data()
        train = X.iloc[:400].copy()
        validation = X.iloc[400:405].copy()
        validation['cap_diameter'] = 10_000_000.
        validation['habitat'] = 'never_seen'
        model = make_pipeline('rf_features', geometry=True, indicator=True)
        model.set_params(classifier__n_estimators=5).fit(train, y.iloc[:400])
        stats = model.named_steps['preprocessor'].named_transformers_['num'].named_steps['impute'].statistics_
        self.assertAlmostEqual(stats[0], train.cap_diameter.median())
        before = stats.copy()
        result = model.predict_proba(validation)
        np.testing.assert_array_equal(stats, before)
        self.assertTrue(np.isfinite(result).all())

    def test_saved_outer_predictions_are_excluded_from_their_training(self):
        X, y, ids, folds = development_data()
        pred = pd.read_csv(OUT / 'outer_oof_predictions.csv', float_precision='round_trip')
        for name, group in pred.groupby('model'):
            self.assertEqual(len(group), 4100, name)
            self.assertTrue(group.source_row.is_unique, name)
            self.assertEqual(set(group.source_row), set(ids), name)
            for number, (tr, val) in enumerate(folds, 1):
                part = group[group.fold == number]
                self.assertEqual(set(part.source_row), set(ids[val]), (name, number))
                self.assertFalse(set(part.source_row) & set(ids[tr]))
                np.testing.assert_array_equal(part.true_class, y.iloc[[int(np.flatnonzero(ids == i)[0]) for i in part.source_row]])


class IntegrationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from preview import create_app
        cls.client = TestClient(create_app())

    def test_original_form_and_candidate_revision_agree(self):
        metadata = self.client.get('/metadata.json').json()
        self.assertEqual(len(metadata['fields']), 11)
        response = self.client.post('/predict/mushroom', json=metadata['example'])
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        health = self.client.get('/health').json()
        self.assertEqual(health['mushroom_model_id'], metadata['model_id'])
        self.assertEqual(health['mushroom_model_revision'], metadata['source_pipeline_sha256'])
        self.assertEqual(result['model_id'], metadata['model_id'])
        self.assertEqual(result['model_revision'], metadata['source_pipeline_sha256'])
        self.assertEqual(result['decision_threshold'], metadata['decision_threshold'])

    def test_missing_values_still_work_and_bad_input_is_rejected(self):
        metadata = self.client.get('/metadata.json').json()
        self.assertEqual(self.client.post('/predict/mushroom', json={}).status_code, 422)
        fields = {f['name']: None for f in metadata['fields']}
        self.assertEqual(self.client.post('/predict/mushroom', json=fields).status_code, 200)
        fields['stem_width'] = -1
        self.assertEqual(self.client.post('/predict/mushroom', json=fields).status_code, 422)
        fields = metadata['example'].copy(); fields['habitat'] = 'wrong_code'
        self.assertEqual(self.client.post('/predict/mushroom', json=fields).status_code, 422)

    def test_preview_text_does_not_claim_a_new_test_or_edibility(self):
        self.assertIn('Geen nieuwe onafhankelijke eindtest', self.client.get('/').text)
        self.assertNotIn('Resultaten op 900 testrecords', self.client.get('/').text)
        self.assertNotIn('oorspronkelijke modelclassificatie met de 50%-grens', self.client.get('/').text)
        m = self.client.get('/metadata.json').json()
        result = self.client.post('/predict/mushroom', json=m['example']).json()
        self.assertNotEqual(result['prediction'], 'eetbaar')
        self.assertIn(result['decision'], ['uncertain', 'suspected_poisonous'])

    def test_export_is_equivalent_and_raw_metrics_are_not_threshold_metrics(self):
        receipt = json.loads((OUT / 'candidate_package.json').read_text())
        self.assertEqual(receipt['labels_disagree'], 0)
        self.assertEqual(receipt['export_rows_checked'], 4100)
        self.assertLess(receipt['export_max_probability_difference'], 1e-12)
        metadata = json.loads((HERE / 'artifacts/mushroom_rf300.json').read_text())
        table = pd.read_csv(OUT / 'summary.csv').set_index('model')
        self.assertAlmostEqual(metadata['metrics']['recall_poisonous'], table.loc['rf_compatible_default', 'recall_p_mean'])
        self.assertAlmostEqual(metadata['proposed_policy_metrics']['recall_p_mean'], table.loc['rf_compatible_f1_threshold', 'recall_p_mean'])
        self.assertEqual(hashlib.sha256((HERE / 'artifacts/current_pipeline.joblib').read_bytes()).hexdigest(), metadata['source_pipeline_sha256'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
