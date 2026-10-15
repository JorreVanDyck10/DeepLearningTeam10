"""Data preparation, voting and saved-prediction integrity checks for study 3."""
import json
import hashlib
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from sklearn.base import clone
from improve_v3 import InputFeatures, experiments, threshold_at_fpr, OUT, FPR_BUDGET
from mushroom_alternative import development_data, scores


class PreparationChecks(unittest.TestCase):
    def test_category_levels_are_training_local_and_missing_stays_explicit(self):
        X, y, _, _ = development_data()
        train = X.iloc[:200].copy()
        val = X.iloc[200:203].copy()
        val['habitat'] = 'never_seen'
        val['spore_print_color'] = None
        before = val.copy(deep=True)
        native = InputFeatures(kind='lightgbm').fit(train)
        output = native.transform(val)
        self.assertTrue(output.habitat.isna().all())
        self.assertNotIn('never_seen', native.categories_['habitat'])
        cat = InputFeatures(kind='catboost').fit(train).transform(val)
        self.assertTrue(cat.spore_print_color.eq('__missing__').all())
        pd.testing.assert_frame_equal(val, before)

    def test_selected_out_fields_cannot_change_random_forest_predictions(self):
        X, y, _, _ = development_data()
        model = clone(experiments()['rf_selection'][0]).set_params(
            inputs__drop_noise=True, inputs__drop_spore=True, classifier__n_estimators=5)
        model.fit(X.iloc[:300], y.iloc[:300])
        val = X.iloc[300:310].copy()
        a = model.predict_proba(val)
        val['jumbled_noise_1'] = 'changed'
        val['spore_print_color'] = 'changed'
        np.testing.assert_array_equal(a, model.predict_proba(val))
        self.assertEqual(len(model.feature_names_in_), 11)

    def test_false_positive_budget_is_selected_on_given_training_scores(self):
        y = np.array([0,0,0,0,0,0,0,1,1,1])
        p = np.array([.1,.2,.3,.4,.5,.6,.7,.65,.8,.9])
        threshold = threshold_at_fpr(y,p)
        metric = scores(y,p,threshold)
        self.assertLessEqual(1-metric['specificity_e'], FPR_BUDGET)
        self.assertEqual(metric['tp'], 2)


class SavedEvidenceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pred = pd.read_csv(OUT/'outer_oof_predictions.csv',float_precision='round_trip')
        cls.metric = pd.read_csv(OUT/'outer_fold_metrics.csv',float_precision='round_trip')
        cls.selections = json.loads((OUT/'selections.json').read_text())

    def test_every_model_has_the_same_validation_rows_and_no_test_rows(self):
        _, y, ids, folds = development_data()
        self.assertEqual(self.pred.model.nunique(), 14)
        for name, frame in self.pred.groupby('model'):
            self.assertEqual(len(frame),4100,name)
            self.assertTrue(frame.source_row.is_unique,name)
            self.assertEqual(set(frame.source_row),set(ids),name)
            np.testing.assert_array_equal(frame.set_index('source_row').loc[ids].true_class, y)
            for f,(tr,val) in enumerate(folds,1):
                group = frame[frame.fold==f]
                self.assertEqual(set(group.source_row),set(ids[val]))
                self.assertFalse(set(group.source_row)&set(ids[tr]))

    def test_policy_labels_use_the_training_selected_thresholds(self):
        self.assertTrue(np.isfinite(self.pred.probability_p).all())
        self.assertTrue(self.pred.probability_p.between(0,1).all())
        for s in self.selections:
            for suffix, key in [('_f1','f1_threshold'),('_fpr_budget','fpr_threshold')]:
                part = self.pred[(self.pred.model==s['family']+suffix)&(self.pred.fold==s['fold'])]
                self.assertEqual(len(part),820)
                np.testing.assert_allclose(part.threshold,s[key],rtol=0,atol=1e-14)
                np.testing.assert_array_equal(part.prediction,(part.probability_p>=s[key]).astype(int))

    def test_protocol_matches_the_code_that_produced_the_study(self):
        protocol=json.loads((OUT/'protocol.json').read_text())
        path=Path(__file__).with_name('improve_v3.py')
        self.assertEqual(hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest(),protocol['code_sha256_lf'])
        for name,digest in protocol['dependencies_sha256_lf'].items():
            self.assertEqual(hashlib.sha256(path.with_name(name).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),digest)

    def test_metrics_are_recomputed_from_the_actual_saved_predictions(self):
        for (name,f),group in self.pred.groupby(['model','fold']):
            calculated = scores(group.true_class,group.probability_p,prediction=group.prediction)
            row = self.metric[(self.metric.model==name)&(self.metric.fold==f)].iloc[0]
            for key,value in calculated.items():
                self.assertAlmostEqual(value,row[key],places=12,msg=(name,f,key))

    def test_voting_probabilities_match_the_declared_training_selected_weights(self):
        for s in self.selections:
            if s['family']!='voting':
                continue
            number=s['fold']
            parts=[]
            for family in ['rf_selection','lightgbm','catboost']:
                p=self.pred[(self.pred.model==family+'_default')&(self.pred.fold==number)].set_index('source_row')
                parts.append(p.probability_p)
            matrix=pd.concat(parts,axis=1)
            expected=matrix.to_numpy()@np.asarray(s['weights'])
            actual=self.pred[(self.pred.model=='voting_default')&(self.pred.fold==number)].set_index('source_row').loc[matrix.index]
            np.testing.assert_allclose(actual.probability_p,expected,rtol=0,atol=1e-12)


if __name__=='__main__':
    unittest.main(verbosity=2)
