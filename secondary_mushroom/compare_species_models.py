"""Compare all available Mushroom configurations on identical species-disjoint folds.

No hyperparameter search or threshold tuning. Original exports remain untouched.
Selection uses mean fold balanced accuracy, then F1 p and recall p; all scores are
development cross-validation scores, not an untouched final-test result.
"""
import json
import hashlib
import inspect
import platform
from time import perf_counter
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier

from audit_species_generalization import verify_species
from compare_andrew_models import build_candidates, clean_measurements, classify, fingerprint, json_safe, scores, warnings_summary
from compare_team_models import aws_pipeline, NAMES as ORIGINAL_NAMES
from comparison_data import ROOT, DATA, load_data

OUT = ROOT / 'secondary_mushroom/comparison_species'
NAMES = {**ORIGINAL_NAMES, 'decision_tree_5': 'Beslisboom depth 5 (baseline)',
         'random_forest_200': 'Andrew RF200 (API-configuratie)'}
ORDER = ['decision_tree_5', 'aws_tuned', 'aws_baseline', 'random_forest_200',
         'random_forest_500', 'gradient_boosting', 'xgboost',
         'logistic_polynomial', 'logistic_artifact_config', 'majority']


def candidates_for(X):
    saved = joblib.load(ROOT / 'SolutionAndrew/MushroomDataset/logistic_regression_mushroom_v1.pkl')
    candidates = build_candidates(X, saved)
    transforms = candidates['logistic_artifact_config'].named_steps['preprocessor'].transformers
    candidates['logistic_artifact_config'].set_params(preprocessor=ColumnTransformer([
        (name, clone(transformer), X.select_dtypes(include='number').columns.tolist()
         if name == 'num' else X.select_dtypes(exclude='number').columns.tolist())
        for name, transformer, _ in transforms]))
    candidates.update(aws_tuned=aws_pipeline(X, True), aws_baseline=aws_pipeline(X, False))
    tree = clone(candidates['aws_baseline'])
    tree.set_params(classifier=DecisionTreeClassifier(max_depth=5, random_state=42))
    candidates['decision_tree_5'] = tree
    api = clone(candidates['random_forest_500'])
    api.set_params(classifier=RandomForestClassifier(n_estimators=200, random_state=42,
        class_weight='balanced', n_jobs=2))
    candidates['random_forest_200'] = api
    return {key: candidates[key] for key in ORDER}


def cache_signature(X, candidates, membership):
    definitions = '\n'.join(inspect.getsource(f) for f in
        [candidates_for, build_candidates, aws_pipeline, clean_measurements])
    return json_safe({'dataset_sha256_lf': fingerprint(DATA)['sha256_lf'],
        'feature_columns': X.columns.tolist(),
        'membership_sha256': hashlib.sha256(membership.to_csv(index=False, lineterminator='\n').encode()).hexdigest(),
        'pipeline_definition_sha256': hashlib.sha256(definitions.encode()).hexdigest(),
        'classifier_parameters': {k: v.steps[-1][1].get_params() for k, v in candidates.items()},
        'environment': {'sklearn': sklearn.__version__, 'numpy': np.__version__, 'pandas': pd.__version__, 'joblib': joblib.__version__}})


def plot_results(table, y, predictions):
    rank = table.sort_values('cv_balanced_accuracy_mean', ascending=False, kind='stable')
    fig, ax = plt.subplots(figsize=(11, 6))
    positions = np.arange(len(rank))
    for delta, name, title, color in [(-.25, 'balanced_accuracy', 'Balanced accuracy', '#315e91'),
        (0, 'f1_poisonous', 'F1 p', '#4b8062'), (.25, 'recall_poisonous', 'Recall p', '#b85c4a')]:
        ax.barh(positions + delta, rank[name], height=.23, label=title, color=color)
    ax.set_yticks(positions, [NAMES[k] for k in rank.model]); ax.invert_yaxis()
    ax.set_xlim(0, 1.02); ax.set_xlabel('Gepoolde score uit vijf soorten-folds')
    ax.set_title('Mushroom: alle configuraties op volledig ongeziene soorten', pad=38)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.01), ncols=3); fig.tight_layout()
    fig.savefig(OUT / 'metrics.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for ax, key in zip(axes.flat, ORDER):
        cm = confusion_matrix(y, (predictions[key + '_prediction'] == 'p').astype(int), labels=[0, 1])
        ax.imshow(cm, cmap='Blues', vmin=0, vmax=35000)
        for i in range(2):
            for j in range(2): ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > 17500 else 'black')
        ax.set_xticks([0, 1], ['e', 'p']); ax.set_yticks([0, 1], ['e', 'p'])
        ax.set_title(NAMES[key].replace(' (API-configuratie)', '\n(API-configuratie)'), fontsize=8)
        ax.set_xlabel('Voorspeld'); ax.set_ylabel('Werkelijk')
    fig.suptitle('Voorspellingen buiten training: 60.923 unieke records'); fig.tight_layout()
    fig.savefig(OUT / 'confusion_matrices.png', dpi=150); plt.close(fig)


def run():
    assert sklearn.__version__ == '1.9.1', 'Use comparison-requirements.txt'
    OUT.mkdir(exist_ok=True)
    raw, data, X, y, _, _, source_rows = load_data()
    raw_groups, evidence = verify_species(raw)
    groups = raw_groups[source_rows]
    candidates = candidates_for(X)
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(X, y, groups))
    membership = pd.DataFrame({'unique_row': np.arange(len(data)), 'source_row': source_rows,
        'species_group': groups, 'true_class': data['class'], 'validation_fold': 0})
    fold_membership = []
    for fold, (tr, val) in enumerate(folds, 1):
        assert not set(groups[tr]) & set(groups[val])
        assert membership.loc[val, 'validation_fold'].eq(0).all()
        membership.loc[val, 'validation_fold'] = fold
        fold_membership.append({'fold': fold, 'train_rows': len(tr), 'validation_rows': len(val),
            'train_species': len(set(groups[tr])), 'validation_species': len(set(groups[val])), 'species_overlap': 0})
    assert membership.validation_fold.gt(0).all()
    signature = cache_signature(X, candidates, membership)
    signature_path = OUT / 'cache_inputs.json'
    if any(OUT.glob('*_predictions.csv')):
        assert signature_path.exists(), 'Cached predictions require cache_inputs.json; regenerate caches.'
        assert json.loads(signature_path.read_text()) == signature, 'Inputs/configuration changed; regenerate cached predictions.'
    signature_path.write_text(json.dumps(signature, indent=2, allow_nan=False) + '\n')
    membership.to_csv(OUT / 'split_manifest.csv', index=False)
    # Persist the declared protocol before evaluating additional configurations.
    protocol = {'primary_metric': 'mean fold balanced accuracy',
        'tie_breakers': ['mean fold F1 p', 'mean fold recall p', 'fixed candidate order'],
        'selection_excludes_majority': True, 'folds': 5, 'shuffle': True, 'random_state': 42,
        'target': 'generalization to unseen simulated species', 'new_tuning': False,
        'untouched_final_test': False, 'candidates': ORDER}
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
    old = ROOT / 'secondary_mushroom/comparison_team'
    old_summary = json.loads((old / 'generalization_audit.json').read_text(encoding='utf-8'))
    assert old_summary['dataset']['sha256_lf'] == fingerprint(DATA)['sha256_lf']
    old_predictions = pd.read_csv(old / 'species_oof_predictions.csv', float_precision='round_trip')
    old_folds = pd.read_csv(old / 'species_cv_folds.csv', float_precision='round_trip')
    pd.testing.assert_frame_equal(membership, old_predictions[membership.columns], check_dtype=False)
    results = []; all_folds = []; warning_log = {}; reused = []
    all_predictions = membership.copy()
    for key, prototype in candidates.items():
        cache = OUT / (key + '_predictions.csv')
        fold_cache = OUT / (key + '_folds.csv')
        if key in old_summary['group_protocol']['evaluated_models']:
            assert prototype.steps[-1][1].get_params() == old_summary['classifier_parameters'][key]
            output = old_predictions[['unique_row', key + '_prediction', key + '_probability_p']].copy()
            records = old_folds.loc[old_folds.model == key].to_dict('records')
            warning_log[key] = old_summary['warnings'][key]
            reused.append(key)
            print('Reuse verified species-CV predictions:', key, flush=True)
        elif cache.exists() and fold_cache.exists():
            output = pd.read_csv(cache, float_precision='round_trip')
            records = pd.read_csv(fold_cache, float_precision='round_trip').to_dict('records')
            warning_log[key] = json.loads((OUT / (key + '_warnings.json')).read_text())
            print('Resume completed candidate:', key, flush=True)
        else:
            predicted_all = np.zeros(len(data), dtype=int)
            probability_all = np.zeros(len(data))
            records = []
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                for fold, (tr, val) in enumerate(folds, 1):
                    print(f'{key}: species fold {fold}/5', flush=True)
                    model = clone(prototype)
                    start = perf_counter(); model.fit(X.iloc[tr], y.iloc[tr]); fit_seconds = perf_counter() - start
                    start = perf_counter(); predicted, proba = classify(model, X.iloc[val]); inference = perf_counter() - start
                    train_predicted, train_proba = classify(model, X.iloc[tr])
                    train_scores = scores(y.iloc[tr], train_predicted, train_proba)
                    records.append({'model': key, 'fold': fold, 'fit_seconds': fit_seconds,
                        'predict_validation_seconds': inference, 'train_accuracy': train_scores['accuracy'],
                        'train_f1_poisonous': train_scores['f1_poisonous'],
                        **scores(y.iloc[val], predicted, proba)})
                    predicted_all[val] = predicted; probability_all[val] = proba
            warning_log[key] = warnings_summary(caught)
            output = pd.DataFrame({'unique_row': np.arange(len(data)),
                key + '_prediction': np.where(predicted_all == 1, 'p', 'e'), key + '_probability_p': probability_all})
        assert output.unique_row.tolist() == membership.unique_row.tolist()
        output.to_csv(cache, index=False)
        pd.DataFrame(records).to_csv(fold_cache, index=False)
        (OUT / (key + '_warnings.json')).write_text(json.dumps(warning_log[key], indent=2) + '\n')
        all_predictions = all_predictions.merge(output, on='unique_row', validate='one_to_one', sort=False)
        labels = (output[key + '_prediction'] == 'p').astype(int)
        row = {'model': key, **scores(y, labels, output[key + '_probability_p'])}
        for metric in ['accuracy', 'balanced_accuracy', 'f1_poisonous', 'recall_poisonous']:
            row['cv_' + metric + '_mean'] = float(np.mean([r[metric] for r in records]))
            row['cv_' + metric + '_std'] = float(np.std([r[metric] for r in records], ddof=1))
        row['mean_fold_fit_seconds'] = float(np.mean([r['fit_seconds'] for r in records]))
        results.append(row); all_folds.extend(records)
    table = pd.DataFrame(results)
    rank = table[table.model != 'majority'].sort_values(
        ['cv_balanced_accuracy_mean', 'cv_f1_poisonous_mean', 'cv_recall_poisonous_mean'], ascending=False, kind='stable')
    selected = str(rank.iloc[0].model)
    table.to_csv(OUT / 'model_comparison.csv', index=False)
    pd.DataFrame(all_folds).to_csv(OUT / 'cv_folds.csv', index=False)
    all_predictions.to_csv(OUT / 'oof_predictions.csv', index=False)
    # Error analysis uses only predictions made by models that did not train on the species.
    wrong = (all_predictions[selected + '_prediction'] != data['class']).to_numpy()
    error_records = membership.loc[wrong].copy()
    error_records['predicted_class'] = all_predictions.loc[wrong, selected + '_prediction'].to_numpy()
    error_records['probability_p'] = all_predictions.loc[wrong, selected + '_probability_p'].to_numpy()
    error_records['missing_features'] = X.loc[wrong].isna().sum(axis=1).to_numpy()
    error_records = error_records.join(X.loc[wrong])
    error_records.to_csv(OUT / 'selected_error_records.csv', index=False)
    species_rows = []
    for key in ORDER:
        pred = all_predictions[key + '_prediction']
        for group in np.unique(groups):
            take = groups == group
            species_rows.append({'model': key, 'species_group': int(group), 'records': int(take.sum()),
                'class': str(data.loc[take, 'class'].iloc[0]),
                'correct': int((pred[take] == data.loc[take, 'class']).sum()),
                'fn': int(((pred[take] == 'e') & (data.loc[take, 'class'] == 'p')).sum()),
                'fp': int(((pred[take] == 'p') & (data.loc[take, 'class'] == 'e')).sum())})
    pd.DataFrame(species_rows).to_csv(OUT / 'species_errors.csv', index=False)
    # A full-data refit is a handoff artifact, not another evaluation or deployment.
    print('Full-data refit of CV-selected configuration:', selected, flush=True)
    selected_model = clone(candidates[selected])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always'); selected_model.fit(X, y)
    warning_log['selected_full_refit'] = warnings_summary(caught)
    artifact = OUT / 'selected_pipeline.joblib'
    joblib.dump(selected_model, artifact, compress=3)
    summary = json_safe({'protocol': protocol, 'dataset': {'raw_rows': len(raw), 'unique_rows': len(data),
        'features': X.columns.tolist(), **fingerprint(DATA)}, 'group_verification': evidence,
        'fold_membership': fold_membership, 'selected_model': selected,
        'selection_status': 'Development CV choice for unseen simulated species, not an independent final-test winner',
        'model_comparison': results, 'classifier_parameters': {k: v.steps[-1][1].get_params() for k, v in candidates.items()},
        'source_files': {name: {**fingerprint(ROOT / 'secondary_mushroom' / name),
            'sha256_lf': hashlib.sha256((ROOT / 'secondary_mushroom' / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()} for name in
            ['compare_species_models.py', 'compare_andrew_models.py', 'compare_team_models.py', 'comparison_data.py']},
        'reused_verified_candidates': reused, 'warnings': warning_log,
        'environment': {'python': platform.python_version(), 'platform': platform.platform(),
            'sklearn': sklearn.__version__, 'numpy': np.__version__, 'pandas': pd.__version__, 'joblib': joblib.__version__},
        'selected_full_refit_artifact': {'path': 'selected_pipeline.joblib', **fingerprint(artifact),
            'training_rows': len(data), 'feature_columns': X.columns.tolist(),
            'class_encoding': {'e': 0, 'p': 1},
            'purpose': 'Handoff only; training metrics are not validation metrics'}})
    (OUT / 'comparison_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    plot_results(table, y, all_predictions)
    print(rank[['model', 'cv_balanced_accuracy_mean', 'accuracy', 'f1_poisonous', 'recall_poisonous', 'fn']].to_string(index=False), flush=True)
    print('CV-selected configuration:', selected, flush=True)


if __name__ == '__main__':
    run()
