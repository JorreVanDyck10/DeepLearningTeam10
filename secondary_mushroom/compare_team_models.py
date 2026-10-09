"""Compare Andrew's configurations and Jorre's AWS configuration on shared UCI data.

These are new local fits, not scores assigned to Andrew's original saved models.
The unchanged AWS artifact is independently audited by audit_aws_model.py.
"""
import io
import json
import os
from pathlib import Path
import platform
from time import perf_counter
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import binomtest
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, precision_recall_curve
import xgboost

from comparison_data import ROOT, DATA, load_data
from compare_andrew_models import build_candidates, classify, scores, wilson, fingerprint, json_safe, warnings_summary

OUT = ROOT / 'secondary_mushroom/comparison_team'
NAMES = {
    'aws_tuned': 'Jorre AWS RF100 (getuned)',
    'aws_baseline': 'Jorre AWS RF100 (baseline)',
    'random_forest_500': 'Andrew RF500',
    'gradient_boosting': 'Andrew Gradient Boosting',
    'xgboost': 'Andrew XGBoost',
    'logistic_polynomial': 'Andrew Logistic + poly',
    'logistic_artifact_config': 'Andrew Logistic v1-config',
    'majority': 'Meerderheidsreferentie',
}


def aws_pipeline(X, tuned):
    preprocessing = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median'), X.select_dtypes(include='number').columns.tolist()),
        ('categorical', Pipeline([
            ('fill', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encode', OneHotEncoder(handle_unknown='ignore')),
        ]), X.select_dtypes(exclude='number').columns.tolist()),
    ])
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=2,
        max_depth=24 if tuned else None, min_samples_leaf=2 if tuned else 1)
    return Pipeline([('preprocessor', preprocessing), ('classifier', model)])


def paired_accuracy(predictions, selected, truth):
    chosen_correct = predictions[selected] == np.asarray(truth)
    comparisons = []
    for key, labels in predictions.items():
        if key in {selected, 'majority'}:
            continue
        other_correct = labels == np.asarray(truth)
        only_chosen = int((chosen_correct & ~other_correct).sum())
        only_other = int((other_correct & ~chosen_correct).sum())
        discordant = only_chosen + only_other
        comparisons.append({'selected': selected, 'other': key, 'only_selected_correct': only_chosen,
            'only_other_correct': only_other,
            'mcnemar_exact_p': float(binomtest(only_chosen, discordant, .5).pvalue) if discordant else 1.})
    return comparisons


def plots(table, yt, predictions, probabilities):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    trained = table[table.model != 'majority']
    fig, ax = plt.subplots(figsize=(11, 6))
    positions = np.arange(len(trained))
    for offset, metric, color, label in [(-.25, 'accuracy', '#4b8062', 'Accuracy'),
            (0, 'f1_poisonous', '#315e91', 'F1 giftige klasse'),
            (.25, 'recall_poisonous', '#b85c4a', 'Recall giftige klasse')]:
        ax.barh(positions + offset, trained[metric], height=.24, color=color, label=label)
    ax.set_yticks(positions, [NAMES[key] for key in trained.model])
    ax.set_xlim(0, 1.08)
    ax.set_xlabel('Score op dezelfde 12.185 UCI-testrecords')
    ax.set_title('Andrew + Jorre: nieuwe lokale fits op dezelfde 20 features')
    ax.legend(ncols=3, loc='lower center', bbox_to_anchor=(.5, -.25))
    fig.tight_layout(); fig.savefig(OUT / 'shared_metrics.png', dpi=180); plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(13.5, 7))
    for ax, key in zip(axes.flat, table.model):
        cm = confusion_matrix(yt, predictions[key], labels=[0, 1])
        ax.imshow(cm, cmap='Blues', vmin=0, vmax=6749)
        for row in range(2):
            for col in range(2):
                ax.text(col, row, str(cm[row, col]), ha='center', va='center',
                    color='white' if cm[row, col] > 4000 else 'black', fontsize=11)
        ax.set_xticks([0, 1], ['e', 'p']); ax.set_yticks([0, 1], ['e', 'p'])
        ax.set_xlabel('Voorspeld'); ax.set_ylabel('Werkelijk')
        ax.set_title(NAMES[key], fontsize=9)
    fig.suptitle('Gemeenschappelijke testset — links onder: giftig als eetbaar', fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / 'shared_confusion_matrices.png', dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for key in trained.model:
        precision, recall, _ = precision_recall_curve(yt, probabilities[key])
        ax.plot(recall, precision, label=NAMES[key])
    ax.axhline(yt.mean(), linestyle='--', color='gray')
    ax.set(xlabel='Recall giftige klasse', ylabel='Precision giftige klasse', title='Precision-recall op dezelfde UCI-records')
    ax.legend(fontsize=8, loc='lower left')
    fig.tight_layout(); fig.savefig(OUT / 'shared_pr_curves.png', dpi=180); plt.close(fig)


def run():
    if sklearn.__version__ != '1.9.1':
        raise RuntimeError('Use comparison-requirements.txt (scikit-learn 1.9.1).')
    OUT.mkdir(exist_ok=True)
    raw, data, X, y, train, test, source_rows = load_data()
    Xt, yt, Xtest, ytest = X.iloc[train], y.iloc[train], X.iloc[test], y.iloc[test]
    saved_lr = joblib.load(ROOT / 'SolutionAndrew/MushroomDataset/logistic_regression_mushroom_v1.pkl')
    andrew = build_candidates(Xt, saved_lr)
    # The saved LR preprocessor names Andrew's original 12 columns, including noise.
    # Retain its transformations but explicitly adapt the input to the shared 20 features.
    saved_transforms = andrew['logistic_artifact_config'].named_steps['preprocessor'].transformers
    andrew['logistic_artifact_config'].set_params(preprocessor=ColumnTransformer([
        (name, clone(transformer), Xt.select_dtypes(include='number').columns.tolist()
            if name == 'num' else Xt.select_dtypes(exclude='number').columns.tolist())
        for name, transformer, _ in saved_transforms
    ]))
    candidates = {'aws_tuned': aws_pipeline(Xt, True), 'aws_baseline': aws_pipeline(Xt, False),
        **{key: value for key, value in andrew.items() if key != 'majority'}, 'majority': andrew['majority']}
    cv = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(Xt, yt))
    membership = pd.DataFrame({'unique_row': np.arange(len(data)), 'source_row': source_rows,
        'class': data['class'], 'split': 'train', 'cv_validation_fold': pd.Series([pd.NA] * len(data), dtype='Int64')})
    membership.loc[test, 'split'] = 'test'
    for number, (_, val) in enumerate(cv, 1): membership.loc[train[val], 'cv_validation_fold'] = number
    membership.to_csv(OUT / 'shared_split_manifest.csv', index=False)
    folds = []; validations = []; warning_log = {}; params = {}
    for key, prototype in candidates.items():
        params[key] = prototype.named_steps['classifier'].get_params()
        start = perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            for number, (tr, val) in enumerate(cv, 1):
                print(f'CV {NAMES[key]} — fold {number}/3', flush=True)
                model = clone(prototype)
                t = perf_counter(); model.fit(Xt.iloc[tr], yt.iloc[tr]); fit_time = perf_counter() - t
                predicted, probability = classify(model, Xt.iloc[val])
                folds.append({'model': key, 'fold': number, 'fit_seconds': fit_time,
                    **scores(yt.iloc[val], predicted, probability)})
        warning_log[key + '_cv'] = warnings_summary(caught)
        model_folds = [r for r in folds if r['model'] == key]
        validations.append({'model': key, 'cv_f1_mean': float(np.mean([r['f1_poisonous'] for r in model_folds])),
            'cv_f1_std': float(np.std([r['f1_poisonous'] for r in model_folds], ddof=1)),
            'cv_recall_mean': float(np.mean([r['recall_poisonous'] for r in model_folds])),
            'cv_accuracy_mean': float(np.mean([r['accuracy'] for r in model_folds])),
            'cv_seconds': perf_counter() - start})
    validation = pd.DataFrame(validations)
    # Stable order prioritizes the available, depth-limited AWS100 artifact if validation ties exactly.
    ranking = validation.sort_values(['cv_f1_mean', 'cv_recall_mean'], ascending=False, kind='stable')
    selected = str(ranking.iloc[0].model)
    tied = ranking[(ranking.cv_f1_mean == ranking.iloc[0].cv_f1_mean)
                   & (ranking.cv_recall_mean == ranking.iloc[0].cv_recall_mean)].model.tolist()
    print('Choice fixed on training CV:', selected, 'exact ties:', tied, flush=True)
    result_rows = []; predictions = {}; probabilities = {}; error_tables = []
    exported = pd.DataFrame({'unique_row': test, 'source_row': source_rows[test],
        'true_class': data.iloc[test]['class'].to_numpy()})
    for key, prototype in candidates.items():
        print('Full-train fit and test audit:', NAMES[key], flush=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model = clone(prototype)
            t = perf_counter(); model.fit(Xt, yt); fit_time = perf_counter() - t
            train_pred, train_proba = classify(model, Xt)
            predicted, probability = classify(model, Xtest)
        warning_log[key + '_full_fit'] = warnings_summary(caught)
        predictions[key] = predicted; probabilities[key] = probability
        timings = []
        for _ in range(5):
            t = perf_counter(); model.predict_proba(Xtest); timings.append(perf_counter() - t)
        buffer = io.BytesIO(); joblib.dump(model, buffer, compress=3)
        result_rows.append({'model': key, **scores(ytest, predicted, probability),
            'train_accuracy': float((train_pred == yt.to_numpy()).mean()),
            'train_f1_poisonous': scores(yt, train_pred, train_proba)['f1_poisonous'],
            'fit_seconds': fit_time, 'predict_12185_ms': float(np.median(timings) * 1000),
            'serialized_bytes_compress3': buffer.tell(),
            'accuracy_ci95': wilson(int((predicted == ytest.to_numpy()).sum()), len(test)),
            'recall_ci95': wilson(int(((predicted == 1) & (ytest.to_numpy() == 1)).sum()), int(ytest.sum()))})
        exported[key + '_prediction'] = np.where(predicted == 1, 'p', 'e')
        exported[key + '_probability_p'] = probability
        wrong = predicted != ytest.to_numpy()
        if wrong.any():
            errors = Xtest.iloc[np.flatnonzero(wrong)].copy()
            errors.insert(0, 'model', key)
            errors.insert(1, 'unique_row', test[wrong]); errors.insert(2, 'source_row', source_rows[test[wrong]])
            errors['true_class'] = data.iloc[test[wrong]]['class'].to_numpy()
            errors['predicted_class'] = np.where(predicted[wrong] == 1, 'p', 'e')
            errors['probability_p'] = probability[wrong]
            errors['missing_features'] = Xtest.iloc[np.flatnonzero(wrong)].isna().sum(axis=1).to_numpy()
            error_tables.append(errors)
    table = pd.DataFrame(result_rows).merge(validation, on='model', validate='one_to_one')
    table.to_csv(OUT / 'shared_model_comparison.csv', index=False)
    pd.DataFrame(folds).to_csv(OUT / 'shared_cv_folds.csv', index=False)
    exported.to_csv(OUT / 'shared_test_predictions.csv', index=False)
    all_errors = pd.concat(error_tables, ignore_index=True)
    all_errors.to_csv(OUT / 'shared_error_records.csv', index=False)
    aws_audit = json.loads((OUT / 'aws_artifact_audit.json').read_text(encoding='utf-8'))
    artifact_predictions = pd.read_csv(OUT / 'aws_artifact_predictions.csv')
    assert artifact_predictions.unique_row.tolist() == test.tolist()
    assert aws_audit['csv_sha256_lf'] == fingerprint(DATA)['sha256_lf']
    disagreements = int((artifact_predictions.prediction.to_numpy() != exported.aws_tuned_prediction.to_numpy()).sum())
    summary = json_safe({'parameter_source_commit': '7fc26c2',
        'scope': 'New local fits on full deduplicated UCI data; adapted Andrew input schema (20 original features)',
        'selection_rule': 'Mean training-CV F1 p; mean recall tie-break; exact ties prioritize existing regularized AWS RF100; test excluded',
        'selected_model': selected, 'exact_cv_ties': tied, 'selection_is_provisional': True,
        'split': {'test_size': .2, 'random_state': 42, 'stratified': True, 'cv_folds': 3},
        'dataset': {'raw_rows': len(raw), 'unique_rows': len(data), 'removed_duplicates': len(raw) - len(data),
            'features': X.columns.tolist(), 'train_rows': len(train), 'test_rows': len(test),
            'test_class_counts': {'e': int((ytest == 0).sum()), 'p': int(ytest.sum())},
            'missing_percent': (X.isna().mean() * 100).to_dict(),
            'exact_feature_overlap_train_test': int(pd.util.hash_pandas_object(Xt, index=False)
                .isin(pd.util.hash_pandas_object(Xtest, index=False)).sum()),
            **fingerprint(DATA)},
        'environment': {'python': platform.python_version(), 'platform': platform.platform(),
            'logical_cpus': os.cpu_count(), 'sklearn': sklearn.__version__, 'numpy': np.__version__,
            'pandas': pd.__version__, 'scipy': scipy.__version__, 'joblib': joblib.__version__, 'xgboost': xgboost.__version__},
        'parameters': params, 'model_comparison': table.to_dict(orient='records'), 'warnings': warning_log,
        'pairwise_accuracy_audit': paired_accuracy(predictions, selected, ytest),
        'aws_local_refit_vs_original_artifact': {'label_disagreements': disagreements,
            'maximum_probability_difference': float(np.max(np.abs(artifact_predictions.probability_p - exported.aws_tuned_probability_p))),
            'original_artifact_sklearn': aws_audit['environment']['sklearn'], 'local_refit_sklearn': sklearn.__version__},
        'source_hashes': {'andrew_notebook': fingerprint(ROOT / 'SolutionAndrew/MushroomDataset/mushroom.ipynb'),
            'aws_notebook': fingerprint(ROOT / 'SolutionJorre/MushroomDataset/03_aws_mushrooms.ipynb'),
            'aws_results': fingerprint(ROOT / 'SolutionJorre/MushroomDataset/aws_results.json')}})
    (OUT / 'shared_comparison_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    plots(table, ytest, predictions, probabilities)
    print(table[['model', 'cv_f1_mean', 'accuracy', 'recall_poisonous', 'f1_poisonous', 'fn']].to_string(index=False), flush=True)
    return summary


if __name__ == '__main__':
    run()
