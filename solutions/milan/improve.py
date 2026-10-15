"""Second development study: learned model families, fixed searches, same inputs.

All writes stay under solutions/milan. The 900 development-test rows are never
scored. Searches are declared before outer evaluation; this remains research on
previously used project data, not a restored independent final test.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from mushroom_alternative import (development_data, FEATURES, NUMERIC, HERE, ROOT,
    SOURCE_HASH, scores, write_json)

OUT = HERE / 'results_v2'
DERIVED = ['cap_to_stem_height', 'stem_slenderness']


class Measurements(TransformerMixin, BaseEstimator):
    """Deterministic feature engineering; learns nothing from any target."""
    def __init__(self, geometry=False, zero_as_missing=False):
        self.geometry = geometry
        self.zero_as_missing = zero_as_missing

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        result = X.copy()
        if self.zero_as_missing:
            result[['stem_height', 'stem_width']] = result[['stem_height', 'stem_width']].replace(0, np.nan)
        # Zero is not assumed to be an error. Only division by zero is excluded.
        if self.geometry:
            result['cap_to_stem_height'] = result.cap_diameter / result.stem_height.where(result.stem_height > 0)
            # Height is in cm, stem width in mm: convert height before division.
            result['stem_slenderness'] = 10 * result.stem_height / result.stem_width.where(result.stem_width > 0)
        return result


def make_pipeline(family, geometry=False, indicator=False, strategy='constant'):
    numeric = NUMERIC + (DERIVED if geometry else [])
    num_steps = [('impute', SimpleImputer(strategy='median', add_indicator=indicator))]
    if family == 'svm':
        num_steps.append(('scale', StandardScaler()))
    prep = ColumnTransformer([
        ('num', Pipeline(num_steps), numeric),
        ('cat', Pipeline([('impute', SimpleImputer(strategy=strategy, fill_value='missing')),
            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]),
            [c for c in FEATURES if c not in NUMERIC])], sparse_threshold=0)
    if family in ('rf_compatible', 'rf_features'):
        clf = RandomForestClassifier(n_estimators=300, max_depth=20,
            min_samples_split=4, min_samples_leaf=1, max_features=None,
            class_weight='balanced', random_state=42, n_jobs=1)
    elif family == 'gradient_boosting':
        clf = GradientBoostingClassifier(n_estimators=300, subsample=.8,
            random_state=42, min_samples_leaf=3)
    elif family == 'svm':
        clf = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
    else:
        raise ValueError(family)
    return Pipeline([('measurements', Measurements(geometry=geometry)),
        ('preprocessor', prep), ('classifier', clf)])


def experiments():
    # Preserve RF300/max-depth-20 and the three numeric + eight categorical
    # transform blocks for a candidate the existing PortableMushroom can serve.
    compatible_grid = {'preprocessor__cat__impute__strategy': ['most_frequent', 'constant'],
        'classifier__class_weight': ['balanced', None],
        'classifier__min_samples_leaf': [1, 2]}
    # Feature engineering is represented by different complete estimators, so
    # ColumnTransformer always knows the correct number of numeric columns.
    feature_grid = []
    for geometry in [False, True]:
        for indicator in [False, True]:
            feature_grid.append({'preprocessor': [make_pipeline('rf_features', geometry, indicator).named_steps['preprocessor']],
                'measurements__geometry': [geometry],
                'classifier__min_samples_leaf': [1, 3]})
    gb_grid = []
    for geometry in [False, True]:
        gb_grid.append({'preprocessor': [make_pipeline('gradient_boosting', geometry, True).named_steps['preprocessor']],
            'measurements__geometry': [geometry], 'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [.05, .1]})
    return {
        'rf_compatible': (make_pipeline('rf_compatible', strategy='most_frequent'), compatible_grid),
        'rf_features': (make_pipeline('rf_features'), feature_grid),
        'gradient_boosting': (make_pipeline('gradient_boosting', indicator=True), gb_grid),
        'svm': (make_pipeline('svm', indicator=True),
                {'classifier__C': [1, 10, 100], 'classifier__gamma': ['scale', .01]})}


def clean_parameters(values):
    return {k: (repr(v) if isinstance(v, BaseEstimator) else v) for k, v in values.items()}


def f1_threshold(y, probability):
    # A modest fixed grid; no choosing the rule from outer-validation labels.
    rows = [(float(f1_score(y, probability >= t, average='macro', zero_division=0)),
             -abs(t - .5), t) for t in np.linspace(.20, .70, 51)]
    return float(max(rows)[2])


def summarize():
    metric = pd.read_csv(OUT / 'outer_fold_metrics.csv')
    rows = []
    for name, frame in metric.groupby('model', sort=False):
        row = {'model': name, 'folds': len(frame)}
        for m in ['accuracy', 'macro_f1', 'recall_p', 'precision_p', 'specificity_e',
                  'balanced_accuracy', 'roc_auc', 'average_precision']:
            row[m + '_mean'] = float(frame[m].mean())
            row[m + '_sd'] = float(frame[m].std(ddof=1))
        for m in ['tn', 'fp', 'fn', 'tp']:
            row[m + '_total'] = int(frame[m].sum())
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / 'summary.csv', index=False)
    return summary


def run(families=None):
    OUT.mkdir(parents=True, exist_ok=True)
    X, y, ids, folds = development_data()
    declared = experiments()
    selected = list(declared) if families is None else families
    assert set(selected) <= set(declared)
    protocol_path = OUT / 'protocol.json'
    protocol = {'source_sha256_lf': SOURCE_HASH, 'training_rows': len(X),
        'excluded_development_test_rows': 900, 'outer_folds': 5, 'inner_folds': 3,
        'features': FEATURES, 'searches': {name: [clean_parameters(p) for p in ParameterGrid(grid)]
            for name, (_, grid) in declared.items()},
        'inner_seed': '7310 + outer fold', 'search_metric': 'mean inner CV macro F1',
        'threshold_rule': 'max inner OOF macro F1; grid .20 to .70 step .01, tie nearer .5',
        'no_test_scoring': True, 'prospective_final_test': False,
        'limit': 'Previous project development, including prior v1 folds, informed this study. All metrics are exploratory development evidence.',
        'code_sha256_lf': hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        'environment': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
            'numpy': np.__version__, 'pandas': pd.__version__, 'joblib': joblib.__version__}}
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text(encoding='utf-8'))
        if previous != protocol:
            raise RuntimeError('Protocol/code changed: use a new study folder, never mix runs.')
    else:
        write_json(protocol_path, protocol)
    metrics_path = OUT / 'outer_fold_metrics.csv'
    predictions_path = OUT / 'outer_oof_predictions.csv'
    selection_path = OUT / 'selections.json'
    metrics = pd.read_csv(metrics_path).to_dict('records') if metrics_path.exists() else []
    predictions = pd.read_csv(predictions_path, float_precision='round_trip').to_dict('records') if predictions_path.exists() else []
    selections = json.loads(selection_path.read_text(encoding='utf-8')) if selection_path.exists() else []
    # Reference predictions are the real RF300 refits from v1, on these exact folds.
    if not metrics:
        reference = pd.read_csv(HERE / 'results/outer_oof_predictions.csv', float_precision='round_trip')
        reference = reference[reference.model == 'rf300_reference_default'].copy()
        assert reference.source_row.is_unique and set(reference.source_row) == set(ids)
        for fold, part in reference.groupby('fold'):
            metrics.append({'model': 'rf300_reference_default', 'fold': int(fold), 'threshold': .5,
                **scores(part.true_class, part.probability_p, prediction=part.prediction)})
        predictions = reference.to_dict('records')
    for family in selected:
        prototype, grid = declared[family]
        for number, (tr, val) in enumerate(folds, 1):
            if any(s['family'] == family and s['fold'] == number for s in selections):
                print(f'Reuse completed {family} fold {number}', flush=True)
                continue
            started = perf_counter()
            print(f'{family} fold {number}/5; {len(ParameterGrid(grid))} configs x 3 inner folds', flush=True)
            inner = StratifiedKFold(3, shuffle=True, random_state=7310 + number)
            search = GridSearchCV(clone(prototype), grid, scoring='f1_macro', cv=inner,
                n_jobs=2, pre_dispatch=2, error_score='raise', return_train_score=True)
            search.fit(X.iloc[tr], y.iloc[tr])
            table = pd.DataFrame(search.cv_results_)
            table.to_csv(OUT / f'{family}_search_fold_{number}.csv', index=False)
            model = search.best_estimator_
            prob = model.predict_proba(X.iloc[val])[:, 1]
            oof = cross_val_predict(clone(model), X.iloc[tr], y.iloc[tr], cv=inner,
                method='predict_proba', n_jobs=2, pre_dispatch=2)[:, 1]
            threshold = f1_threshold(y.iloc[tr].to_numpy(), oof)
            for policy, t in [('default', .5), ('f1_threshold', threshold)]:
                label = family + '_' + policy
                pred = model.predict(X.iloc[val]) if policy == 'default' else (prob >= t).astype(int)
                row = {'model': label, 'fold': number, 'threshold': t,
                       **scores(y.iloc[val], prob, t, prediction=pred)}
                metrics.append(row)
                predictions.extend({'source_row': int(ids[i]), 'fold': number, 'true_class': int(y.iloc[i]),
                    'model': label, 'probability_p': float(p), 'threshold': t, 'prediction': int(v)}
                    for i, p, v in zip(val, prob, pred))
            selection = {'family': family, 'fold': number, 'best_parameters': clean_parameters(search.best_params_),
                'inner_cv_macro_f1': float(search.best_score_), 'f1_threshold': threshold,
                'seconds': perf_counter() - started}
            selections.append(selection)
            # Save the selected model only locally for subsequent artifact audits.
            cache = HERE / '.generated/v2'; cache.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, cache / f'{family}_fold_{number}.joblib', compress=3)
            pd.DataFrame(metrics).to_csv(metrics_path, index=False)
            pd.DataFrame(predictions).to_csv(predictions_path, index=False)
            write_json(selection_path, selections)
            print(json.dumps(selection), flush=True)
            print(summarize()[['model', 'folds', 'macro_f1_mean', 'recall_p_mean']].to_string(index=False), flush=True)
    summary = summarize()
    complete = len(selections) == 5 * len(declared)
    write_json(OUT / 'run.json', {'finished_at_utc': datetime.now(timezone.utc).isoformat(),
        'completed_family_folds': len(selections), 'expected_family_folds': 20,
        'complete': complete, 'development_test_rows_scored': 0,
        'tracked_writes_only_under': 'solutions/milan', 'deployment_performed': False})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--families', nargs='+', choices=list(experiments()))
    args = parser.parse_args()
    with joblib.parallel_config(backend='threading'):
        run(args.families)
