"""Reproduce a model, or train/validate/export a candidate for automatic publication."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from MushroomFinal.compare_correct_dataset import candidates_for, evaluate
from Final.tools.data import prepare, SOURCE_SHA256_LF
from backend.mushroom_model import PortableMushroom, FEATURES, MODEL_ID


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def fit_model(key, parameters=None):
    if sklearn.__version__ != '1.4.2':
        raise RuntimeError('Training requires the pinned Python 3.11 environment')
    raw, X, y, train, test, folds = prepare()
    candidate = candidates_for(X)[key].set_params(memory=None)
    if parameters: candidate.set_params(**{'classifier__'+k: v for k, v in parameters.items()})
    fold_rows = []
    started = perf_counter()
    for number, (tr, val) in enumerate(folds, 1):
        print(key, 'fold', number, '/5', flush=True)
        fitted = clone(candidate).fit(X.iloc[train[tr]], y.iloc[train[tr]])
        scores = fitted.predict_proba(X.iloc[train[val]])[:, 1]
        fold_rows.append({'fold': number, **evaluate(y.iloc[train[val]], fitted.predict(X.iloc[train[val]]), scores)})
    cv = {'cv_macro_f1_mean': float(np.mean([r['macro_f1'] for r in fold_rows])),
          'cv_macro_f1_std': float(np.std([r['macro_f1'] for r in fold_rows], ddof=1)),
          'cv_recall_p_mean': float(np.mean([r['recall_p'] for r in fold_rows]))}
    model = clone(candidate).fit(X.iloc[train], y.iloc[train])
    return model, cv, fold_rows, perf_counter()-started, (raw, X, y, train, test)


def reproduce(key):
    model, cv, folds, seconds, (_, X, y, train, test) = fit_model(key)
    saved = pd.read_csv(ROOT/'MushroomFinal/comparison_correct/test_predictions.csv', float_precision='round_trip')
    assert saved.source_row.tolist() == test.tolist()
    scores = model.predict_proba(X.iloc[test])[:, 1]
    labels = model.predict(X.iloc[test])
    np.testing.assert_array_equal(labels, saved[key+'_prediction'])
    np.testing.assert_allclose(scores, saved[key+'_probability_p'], rtol=0, atol=1e-12)
    row = {'model': key, **cv, **evaluate(y.iloc[test], labels, scores),
           'fit_and_cv_seconds': seconds, 'training_rows': len(train), 'test_rows': len(test),
           'labels_match_original': True, 'max_probability_difference': float(np.max(np.abs(scores-saved[key+'_probability_p']))),
           'folds': folds, 'sklearn': sklearn.__version__}
    dest = ROOT/'Final/.generated'/key
    dest.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, dest/'pipeline.joblib', compress=3)
    write_json(dest/'reproduction.json', row)
    return row


def export_model(model, metadata, destination):
    """Same original splits and leaf values; no approximating or new fitting here."""
    destination.mkdir(parents=True, exist_ok=True)
    forest = model.named_steps['classifier']
    assert forest.n_estimators == 300 and forest.max_depth == 20
    transforms = {name: (transformer, columns) for name, transformer, columns in model.named_steps['preprocessor'].transformers_ if name != 'remainder'}
    numeric, numeric_cols = transforms['num']; cat, cat_cols = transforms['cat']; encoder = cat.named_steps['encode']
    assert encoder.drop is None and encoder.handle_unknown == 'ignore'
    trees = [tree.tree_ for tree in forest.estimators_]
    target = destination/'mushroom_rf300.npz'
    np.savez_compressed(target,
        left=np.concatenate([t.children_left for t in trees]).astype(np.int32),
        right=np.concatenate([t.children_right for t in trees]).astype(np.int32),
        feature=np.concatenate([t.feature for t in trees]).astype(np.int16),
        threshold=np.concatenate([t.threshold for t in trees]),
        probabilities=np.concatenate([t.value[:, 0, :] for t in trees]),
        offsets=np.r_[0, np.cumsum([t.node_count for t in trees])])
    metadata.update(forest_file=target.name, forest_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        numeric_features=list(numeric_cols), numeric_medians=numeric.named_steps['impute'].statistics_.tolist(),
        categorical=[{'name': str(name), 'imputation': str(mode), 'categories': [str(c) for c in codes]}
            for name, mode, codes in zip(cat_cols, cat.named_steps['impute'].statistics_, encoder.categories_)])
    write_json(destination/'mushroom_rf300.json', metadata)
    return PortableMushroom(destination/'mushroom_rf300.json')


def train(output):
    config = json.loads((ROOT/'Final/config/mushroom.json').read_text())
    model, cv, folds, seconds, (_, X, y, train_rows, test) = fit_model('rf_final', config['classifier_parameters'])
    # Acceptance is determined exclusively from out-of-fold training predictions.
    for metric, minimum in [('cv_macro_f1_mean', config['minimum_cv_macro_f1']), ('cv_recall_p_mean', config['minimum_cv_recall_p'])]:
        if cv[metric] + config['tolerance'] < minimum:
            raise ValueError(f'Candidate rejected: {metric}={cv[metric]} below {minimum}')
    print('Training-fold quality gate passed; now record the already inspected development test.', flush=True)
    scores = model.predict_proba(X.iloc[test])[:, 1]; labels = model.predict(X.iloc[test])
    metrics = {'model': 'rf_final', **cv, **evaluate(y.iloc[test], labels, scores)}
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    source_relative = 'Final/Mushroom/artifacts/current_pipeline.joblib'
    source = output/source_relative; source.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, source, compress=3)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = json.loads((ROOT/'backend/artifacts/mushroom_rf300.json').read_text(encoding='utf-8'))
    metadata.update(source_pipeline=source_relative, source_pipeline_sha256=source_hash,
        metrics={**metrics, 'recall_poisonous': metrics['recall_p'], 'poisonous_predicted_edible': metrics['fn']},
        dataset_sha256_lf=SOURCE_SHA256_LF, validation_note='Ontwikkeltest; eerder bekeken. Geen onafhankelijke eindtest.')
    exported = export_model(model, metadata, output/'backend/artifacts')
    expected = model.predict_proba(X); actual = exported.predict_proba(X)
    np.testing.assert_array_equal(np.argmax(actual, axis=1), model.predict(X))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    tests = pd.DataFrame({'source_row': test, 'true_class': y.iloc[test].to_numpy(),
                          'prediction': labels, 'probability_p': scores})
    tests.to_csv(output/'backend/artifacts/mushroom_reference_predictions.csv', index=False)
    receipt = {'source_pipeline_sha256': source_hash, 'model_id': MODEL_ID, 'test_rows_verified': len(test),
        'additional_rows_verified': len(X), 'label_disagreements': 0,
        'max_probability_difference': float(np.max(np.abs(expected-actual))), 'tolerance': 1e-12,
        'export_sklearn': sklearn.__version__, 'forest_sha256': metadata['forest_sha256']}
    write_json(output/'backend/artifacts/mushroom_export_audit.json', receipt)
    run = {'trained_at_utc': datetime.now(timezone.utc).isoformat(), 'source_commit': __import__('os').getenv('GITHUB_SHA'),
        'model': 'rf_final', 'parameters': config['classifier_parameters'], 'dataset_sha256_lf': SOURCE_SHA256_LF,
        'acceptance': 'Passed both training-CV gates before development-test scoring', 'cv': cv, 'folds': folds,
        'development_test_metrics': metrics, 'train_rows': len(train_rows), 'test_rows': len(test),
        'fit_and_cv_seconds': seconds, 'source_pipeline_sha256': source_hash,
        'python': platform.python_version(), 'sklearn': sklearn.__version__, 'numpy': np.__version__}
    write_json(output/'Final/Mushroom/artifacts/training_run.json', run)
    print('Candidate fitted, exported and verified on all 5,000 rows:', source_hash, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['reproduce', 'train'])
    parser.add_argument('--model', default='rf_final', choices=['rf_final', 'gb_final', 'lr_final', 'tree_baseline'])
    parser.add_argument('--output', default='Final/.generated/candidate')
    args = parser.parse_args()
    if args.operation == 'train': train(args.output)
    else: print(json.dumps(reproduce(args.model), indent=2))
