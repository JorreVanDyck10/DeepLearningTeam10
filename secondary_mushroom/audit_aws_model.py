"""Audit the unchanged AWS artifact in scikit-learn 1.7.2, without retraining."""
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter
import warnings

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

from comparison_data import ROOT, DATA, load_data


def run():
    if sklearn.__version__ != '1.7.2':
        raise RuntimeError('Use requirements-aws-audit.txt in a separate environment (scikit-learn 1.7.2).')
    raw, data, X, y, train, test, source_rows = load_data()
    folder = ROOT / 'SolutionJorre/MushroomDataset'
    path = folder / 'mushroom_pipeline.joblib'
    original_bytes = path.read_bytes()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = joblib.load(path)
        assert list(model.feature_names_in_) == X.columns.tolist()
        assert list(model.classes_) == [0, 1]
        start = perf_counter()
        predicted = model.predict(X.iloc[test])
        probabilities = model.predict_proba(X.iloc[test])[:, 1]
        elapsed = perf_counter() - start
    yt = y.iloc[test]
    cm = confusion_matrix(yt, predicted, labels=[0, 1])
    metrics = {'accuracy': float(accuracy_score(yt, predicted)),
        'balanced_accuracy': float(balanced_accuracy_score(yt, predicted)),
        'precision': float(precision_score(yt, predicted)), 'recall': float(recall_score(yt, predicted)),
        'f1': float(f1_score(yt, predicted)), 'roc_auc': float(roc_auc_score(yt, probabilities)),
        'average_precision': float(average_precision_score(yt, probabilities)),
        'tn': int(cm[0, 0]), 'fp': int(cm[0, 1]), 'fn': int(cm[1, 0]), 'tp': int(cm[1, 1])}
    exported = json.loads((folder / 'aws_results.json').read_text(encoding='utf-8'))
    assert cm.tolist() == exported['test_confusion_matrix']
    assert all(abs(metrics[key] - value) < 1e-12 for key, value in exported['test_metrics'].items())
    training_matrix = model.named_steps['preprocessing'].transform(X.iloc[train])
    test_matrix = model.named_steps['preprocessing'].transform(X.iloc[test])
    trees = model.named_steps['model']
    params = trees.get_params()
    assert all(params[key.removeprefix('model__')] == value for key, value in exported['best_parameters'].items())
    result = {'source_commit': '7fc26c2', 'artifact_unchanged': True,
        'artifact_sha256': hashlib.sha256(original_bytes).hexdigest(),
        'artifact_bytes': len(original_bytes), 'input_features': X.columns.tolist(),
        'source_url': 'https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset',
        'csv_sha256_lf': hashlib.sha256(DATA.read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        'raw_rows': len(raw), 'unique_rows': len(data), 'train_rows': len(train), 'test_rows': len(test),
        'split': {'test_size': .2, 'random_state': 42, 'stratified': True},
        'recomputed_metrics': metrics, 'matches_exported_metrics': True,
        'training_membership_independently_proven': False,
        'feature_matrix': {'input_features': X.shape[1], 'encoded_features': training_matrix.shape[1],
            'training_rows': training_matrix.shape[0], 'test_rows': test_matrix.shape[0],
            'target_excluded': 'class' not in model.feature_names_in_},
        'tree_depth': {'minimum': min(t.get_depth() for t in trees.estimators_),
            'maximum': max(t.get_depth() for t in trees.estimators_)},
        'classifier_parameters': params, 'predict_and_proba_seconds': elapsed,
        'warnings': [str(w.message) for w in caught],
        'environment': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
            'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__, 'joblib': joblib.__version__}}
    out = ROOT / 'secondary_mushroom/comparison_team'
    out.mkdir(exist_ok=True)
    (out / 'aws_artifact_audit.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    pd.DataFrame({'unique_row': test, 'source_row': source_rows[test],
        'true_class': data.iloc[test]['class'].to_numpy(),
        'prediction': np.where(predicted == 1, 'p', 'e'), 'probability_p': probabilities}).to_csv(out / 'aws_artifact_predictions.csv', index=False)
    assert path.read_bytes() == original_bytes
    print('Unchanged AWS artifact:', metrics)
    print('Matching saved results; warnings:', result['warnings'])
    return result


if __name__ == '__main__':
    run()
