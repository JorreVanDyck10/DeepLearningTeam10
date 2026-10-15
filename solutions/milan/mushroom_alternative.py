"""Course-based RF experiment. Never score the 900 development-test records.

Run from the repository: python solutions/milan/mushroom_alternative.py
Only this directory receives output. No serving model is replaced.
"""
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
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score,
    balanced_accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / 'results'
SOURCE = ROOT / 'MushroomFinal/mushroom_project_dataset.csv'
SOURCE_HASH = '1f1a25f2f330458ed95ce9f6ffe3241582312a42cb21797a2b7bd5e5ad281114'
NUMERIC = ['cap_diameter', 'stem_height', 'stem_width']
FEATURES = NUMERIC + ['spore_print_color', 'gill_color', 'habitat', 'season',
    'ring_type', 'cap_shape', 'stem_surface', 'jumbled_noise_1']
# Declared before execution: 8 configurations, no adaptive extension of the grid.
GRID = {'classifier__max_features': ['sqrt', None],
        'classifier__min_samples_leaf': [1, 3],
        'preprocessor__cat__impute__strategy': ['most_frequent', 'constant']}
RECALL_TARGET = .90  # Demonstration objective, not a proven safety requirement.


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False) + '\n', encoding='utf-8')


def development_data():
    """Use saved membership: only the 4,100 historical training rows for learning."""
    assert hashlib.sha256(SOURCE.read_bytes().replace(b'\r\n', b'\n')).hexdigest() == SOURCE_HASH
    raw = pd.read_csv(SOURCE)
    raw.columns = raw.columns.str.strip().str.replace('-', '_', regex=False)
    assert raw.shape == (5000, 13) and not raw.duplicated().any()
    manifest = pd.read_csv(ROOT / 'MushroomFinal/comparison_correct/split_manifest.csv')
    assert manifest.source_row.is_unique and set(manifest.source_row) == set(range(5000))
    rows = manifest[manifest.split == 'train'].sort_values('training_order')
    ids = rows.source_row.to_numpy()
    forbidden = set(manifest.loc[manifest.split == 'test', 'source_row'])
    assert len(ids) == 4100 and len(forbidden) == 900 and not set(ids) & forbidden
    X = raw.iloc[ids][FEATURES].reset_index(drop=True)
    y = raw.iloc[ids]['class'].map({'e': 0, 'p': 1}).reset_index(drop=True)
    assert y.notna().all()
    assert np.array_equal(raw.iloc[ids]['class'].to_numpy(), rows['class'].to_numpy())
    folds = [(np.flatnonzero(rows.cv_validation_fold.to_numpy() != f),
              np.flatnonzero(rows.cv_validation_fold.to_numpy() == f))
             for f in range(1, 6)]
    assert sorted(np.concatenate([val for _, val in folds]).tolist()) == list(range(4100))
    return X, y, ids, folds


def pipeline(trees=150):
    cats = [c for c in FEATURES if c not in NUMERIC]
    pre = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), NUMERIC),
        ('cat', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent', fill_value='missing')),
            ('encode', OneHotEncoder(handle_unknown='ignore'))]), cats)])
    return Pipeline([('preprocessor', pre), ('classifier', RandomForestClassifier(
        n_estimators=trees, max_depth=20, min_samples_split=4,
        max_features=None, class_weight='balanced', random_state=42, n_jobs=1))])


def scores(y, probability, threshold=.5, prediction=None):
    pred = (np.asarray(probability) >= threshold).astype(int) if prediction is None else prediction
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'accuracy': float(accuracy_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'recall_p': float(recall_score(y, pred, zero_division=0)),
        'precision_p': float(precision_score(y, pred, zero_division=0)),
        'specificity_e': float(tn / (tn + fp)),
        'roc_auc': float(roc_auc_score(y, probability)),
        'average_precision': float(average_precision_score(y, probability)),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


def choose_threshold(y, probability):
    """Max specificity with >=90% recall on INNER OOF predictions only."""
    fpr, tpr, thresholds = roc_curve(y, probability, drop_intermediate=False)
    eligible = np.flatnonzero(np.isfinite(thresholds) & (tpr >= RECALL_TARGET))
    # Break equal-specificity ties by higher recall, then higher threshold.
    best = min(eligible, key=lambda i: (fpr[i], -tpr[i], -thresholds[i]))
    return float(thresholds[best])


def run():
    started = perf_counter()
    X, y, ids, folds = development_data()
    RESULTS.mkdir(parents=True, exist_ok=True)
    protocol = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256_lf': SOURCE_HASH, 'features': FEATURES,
        'development_rows': 4100, 'excluded_development_test_rows': 900,
        'outer_folds': '5 original shared stratified folds',
        'inner_folds': '3 stratified folds; seed 20261015 + outer fold',
        'search': GRID, 'search_trees': 150, 'reference_trees': 300,
        'selection_metric': 'mean inner CV macro F1',
        'threshold_rule': 'maximize inner-OOF specificity subject to recall_p >= 0.90',
        'threshold_is_demonstration_only': True,
        'limitation': 'All data previously used in project development. Nested CV does not restore an independent final test.',
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'environment': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
            'numpy': np.__version__, 'pandas': pd.__version__, 'joblib': joblib.__version__}}
    write_json(RESULTS / 'protocol.json', protocol)
    fold_rows, predictions, selections = [], [], []
    for number, (tr, val) in enumerate(folds, 1):
        print(f'Outer fold {number}/5: tune 8 configurations on inner training folds', flush=True)
        assert not set(ids[tr]) & set(ids[val])
        inner = StratifiedKFold(3, shuffle=True, random_state=20261015 + number)
        search = GridSearchCV(pipeline(), GRID, scoring='f1_macro', cv=inner,
            n_jobs=2, pre_dispatch=2, refit=True, return_train_score=True, error_score='raise')
        search.fit(X.iloc[tr], y.iloc[tr])
        pd.DataFrame(search.cv_results_).to_csv(RESULTS / f'inner_search_fold_{number}.csv', index=False)
        chosen = search.best_estimator_
        reference = pipeline(300).fit(X.iloc[tr], y.iloc[tr])
        selection = {'outer_fold': number, 'best_parameters': search.best_params_,
                     'inner_cv_macro_f1': float(search.best_score_)}
        for name, model in [('rf300_reference', reference), ('rf_tuned', chosen)]:
            # The fitted outer model only sees outer-training rows.
            outer_prob = model.predict_proba(X.iloc[val])[:, 1]
            # Inner OOF scores choose the threshold without outer-validation labels.
            inner_prob = cross_val_predict(clone(model), X.iloc[tr], y.iloc[tr],
                cv=inner, method='predict_proba', n_jobs=2, pre_dispatch=2)[:, 1]
            threshold = choose_threshold(y.iloc[tr], inner_prob)
            selection[name + '_threshold'] = threshold
            for policy, t in [('default', .5), ('recall_target', threshold)]:
                label = name + '_' + policy
                # sklearn resolves exactly equal probabilities towards class 0.
                labels = model.predict(X.iloc[val]) if policy == 'default' else (outer_prob >= t).astype(int)
                row = {'model': label, 'fold': number, 'threshold': t,
                       **scores(y.iloc[val], outer_prob, t, prediction=labels)}
                fold_rows.append(row)
                predictions.extend({'source_row': int(ids[i]), 'fold': number,
                    'true_class': int(y.iloc[i]), 'model': label,
                    'probability_p': float(p), 'threshold': t, 'prediction': int(v)}
                    for i, p, v in zip(val, outer_prob, labels))
            selection[name + '_train_accuracy'] = float(accuracy_score(
                y.iloc[tr], model.predict(X.iloc[tr])))
        selections.append(selection)
        write_json(RESULTS / 'selections.json', selections)
        pd.DataFrame(fold_rows).to_csv(RESULTS / 'outer_fold_metrics.csv', index=False)
        pd.DataFrame(predictions).to_csv(RESULTS / 'outer_oof_predictions.csv', index=False)
        print(json.dumps(selection), flush=True)
    pred = pd.DataFrame(predictions)
    pred.to_csv(RESULTS / 'outer_oof_predictions.csv', index=False)
    fold_metrics = pd.DataFrame(fold_rows)
    summary = []
    for name, frame in fold_metrics.groupby('model', sort=False):
        record = {'model': name}
        for metric in ['accuracy', 'macro_f1', 'balanced_accuracy', 'recall_p',
                       'precision_p', 'specificity_e', 'roc_auc', 'average_precision']:
            record[metric + '_mean'] = float(frame[metric].mean())
            record[metric + '_sd'] = float(frame[metric].std(ddof=1))
        for metric in ['tn', 'fp', 'fn', 'tp']:
            record[metric + '_total'] = int(frame[metric].sum())
        subset = pred[pred.model == name]
        assert subset.source_row.is_unique and set(subset.source_row) == set(ids)
        summary.append(record)
    pd.DataFrame(summary).to_csv(RESULTS / 'summary.csv', index=False)
    write_json(RESULTS / 'run.json', {'finished_at_utc': datetime.now(timezone.utc).isoformat(),
        'seconds': perf_counter() - started, 'outer_evaluations': len(fold_rows),
        'oof_rows': len(pred), 'test_rows_scored': 0,
        'new_deployment_artifact_created': False,
        'source_sha256_lf_after': hashlib.sha256(SOURCE.read_bytes().replace(b'\r\n', b'\n')).hexdigest()})
    print(pd.DataFrame(summary)[['model', 'macro_f1_mean', 'recall_p_mean', 'fn_total', 'fp_total']].to_string(index=False), flush=True)
    print('Done. No independent final-test or deployment claim.', flush=True)


if __name__ == '__main__':
    # Threads also work in restricted Windows environments without process spawning.
    with joblib.parallel_config(backend='threading'):
        run()
