"""Diagnose the perfect random-split score using verified species groups and a negative control.

This is a diagnostic evaluation of two fixed configurations, not a new final model search.
Species membership is reconstructed only after checking primary-data consistency.
"""
import io
import json
import platform
from pathlib import Path
from time import perf_counter
import urllib.request
import warnings
import zipfile

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from comparison_data import ROOT, URL, DATA, load_data
from compare_team_models import aws_pipeline
from compare_andrew_models import build_candidates, classify, fingerprint, json_safe, scores, warnings_summary

OUT = ROOT / 'secondary_mushroom/comparison_team'


def verify_species(raw):
    # The authoritative UCI archive contains the primary rows underlying the simulation.
    with urllib.request.urlopen(URL, timeout=60) as response:
        archive = response.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as outer:
        inner_bytes = outer.read('MushroomDataset.zip')
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        secondary = inner.read('MushroomDataset/secondary_data.csv')
        primary_bytes = inner.read('MushroomDataset/primary_data.csv')
    assert secondary.replace(b'\r\n', b'\n') == DATA.read_bytes().replace(b'\r\n', b'\n')
    primary_path = DATA.parent / 'primary_data.csv'
    primary_path.write_bytes(primary_bytes)
    primary = pd.read_csv(primary_path, sep=';')
    primary.columns = primary.columns.str.lower()
    assert len(primary) == 173 and primary['name'].nunique() == 173
    groups = np.arange(len(raw)) // 353
    assert pd.Series(groups).value_counts().eq(353).all()
    assert raw.groupby(groups)['class'].nunique().eq(1).all()
    assert raw.groupby(groups)['class'].first().tolist() == primary['class'].tolist()
    categoricals = raw.select_dtypes(exclude='number').columns.drop('class')
    checks = 0
    for group in range(173):
        block = raw.iloc[group * 353:(group + 1) * 353]
        for column in categoricals:
            value = primary.iloc[group][column]
            allowed = set() if pd.isna(value) else {v.strip(' []') for v in value.split(',')}
            observed = set(block[column].dropna())
            assert observed.issubset(allowed), (group, column)
            if pd.isna(value):
                assert block[column].isna().all()
            checks += 1
    primary[['family', 'name', 'class']].assign(species_group=np.arange(173)).to_csv(OUT / 'species_lookup.csv', index=False)
    return groups, {'species_count': 173, 'raw_rows_per_species': 353,
        'constant_class_blocks': 173, 'primary_class_sequence_matches': True,
        'categorical_primary_consistency_checks': checks, 'categorical_mismatches': 0,
        'primary_data': fingerprint(primary_path),
        'grouping_rule': 'original UCI source row // 353, after verifying all blocks against primary data'}


def run():
    assert sklearn.__version__ == '1.9.1', 'Use comparison-requirements.txt'
    OUT.mkdir(exist_ok=True)
    raw, data, X, y, train, test, source_rows = load_data()
    raw_groups, evidence = verify_species(raw)
    groups = raw_groups[source_rows]
    random_overlap = len(set(groups[train]) & set(groups[test]))
    assert random_overlap == 173
    assert 'class' not in X.columns and 'name' not in X.columns and X.shape[1] == 20
    saved_lr = joblib.load(ROOT / 'SolutionAndrew/MushroomDataset/logistic_regression_mushroom_v1.pkl')
    candidates = {'aws_tuned': aws_pipeline(X, True),
                  'random_forest_500': build_candidates(X, saved_lr)['random_forest_500']}
    grouped_folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(X, y, groups))
    assignments = np.zeros(len(data), dtype=int)
    fold_info = []; folds = []; warning_log = {}; pooled = []
    export = pd.DataFrame({'unique_row': np.arange(len(data)), 'source_row': source_rows,
        'species_group': groups, 'true_class': data['class']})
    for fold, (tr, val) in enumerate(grouped_folds, 1):
        assert not set(groups[tr]) & set(groups[val])
        assignments[val] = fold
        fold_info.append({'fold': fold, 'train_rows': len(tr), 'validation_rows': len(val),
            'train_species': len(set(groups[tr])), 'validation_species': len(set(groups[val])),
            'overlapping_species': 0})
    export['validation_fold'] = assignments
    assert (assignments > 0).all()
    for key, prototype in candidates.items():
        all_predicted = np.zeros(len(data), dtype=int)
        all_probability = np.zeros(len(data), dtype=float)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            for fold, (tr, val) in enumerate(grouped_folds, 1):
                print(f'Species-held-out {key}: fold {fold}/5', flush=True)
                model = clone(prototype)
                start = perf_counter(); model.fit(X.iloc[tr], y.iloc[tr]); elapsed = perf_counter() - start
                predicted, probability = classify(model, X.iloc[val])
                all_predicted[val] = predicted; all_probability[val] = probability
                folds.append({'model': key, 'fold': fold, 'fit_seconds': elapsed, **scores(y.iloc[val], predicted, probability)})
        warning_log[key] = warnings_summary(caught)
        rows = [row for row in folds if row['model'] == key]
        pooled.append({'model': key, **scores(y, all_predicted, all_probability),
            'mean_fold_accuracy': float(np.mean([r['accuracy'] for r in rows])),
            'std_fold_accuracy': float(np.std([r['accuracy'] for r in rows], ddof=1)),
            'mean_fold_f1': float(np.mean([r['f1_poisonous'] for r in rows]))})
        export[key + '_prediction'] = np.where(all_predicted == 1, 'p', 'e')
        export[key + '_probability_p'] = all_probability
    # With randomly shuffled labels, the fixed AWS recipe should not retain a perfect score.
    shuffled_y = pd.Series(np.random.default_rng(42).permutation(y.to_numpy()), index=y.index)
    print('Negative control: independently shuffled labels on the original row split', flush=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        control = clone(candidates['aws_tuned']).fit(X.iloc[train], shuffled_y.iloc[train])
        predicted, probability = classify(control, X.iloc[test])
    control_scores = scores(shuffled_y.iloc[test], predicted, probability)
    warning_log['shuffled_label_control'] = warnings_summary(caught)
    assert control_scores['accuracy'] < .7 and control_scores['balanced_accuracy'] < .6
    transformed = control.named_steps['preprocessor'].get_feature_names_out().tolist()
    assert not any('class' in name or 'species' in name or 'source_row' in name for name in transformed)
    pd.DataFrame({'unique_row': test, 'shuffled_target': shuffled_y.iloc[test].to_numpy(),
        'prediction': predicted, 'probability_p': probability}).to_csv(OUT / 'shuffled_label_predictions.csv', index=False)
    pd.DataFrame(folds).to_csv(OUT / 'species_cv_folds.csv', index=False)
    pd.DataFrame(pooled).to_csv(OUT / 'species_model_comparison.csv', index=False)
    export.to_csv(OUT / 'species_oof_predictions.csv', index=False)
    original = pd.read_csv(OUT / 'shared_model_comparison.csv').set_index('model')
    result = json_safe({'source_commit': '7ca674b', 'dataset': fingerprint(DATA),
        'feature_columns': X.columns.tolist(),
        'classifier_parameters': {key: prototype.steps[-1][1].get_params() for key, prototype in candidates.items()},
        'environment': {'python': platform.python_version(), 'platform': platform.platform(),
            'sklearn': sklearn.__version__, 'numpy': np.__version__, 'pandas': pd.__version__,
            'joblib': joblib.__version__},
        'group_verification': evidence, 'original_random_split_species_overlap': random_overlap,
        'target_index_and_species_excluded_from_inputs': True, 'encoded_feature_count': len(transformed),
        'group_protocol': {'folds': 5, 'shuffle': True, 'random_state': 42,
            'evaluated_models': list(candidates), 'purpose': 'diagnostic new-species generalization; not final model selection'},
        'fold_membership': fold_info, 'species_held_out_scores': pooled,
        'original_row_split_scores': {key: original.loc[key].to_dict() for key in candidates},
        'shuffled_label_negative_control': {'seed': 42, 'permutation': 'all unique target labels before fitting',
            'train_rows': len(train), 'test_rows': len(test), 'metrics': control_scores},
        'warnings': warning_log,
        'ranking_status': 'No overall best model established for unseen species; other candidates not group-evaluated',
        'sources': ['https://www.nature.com/articles/s41598-021-87602-3',
            'https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset',
            'https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data']})
    (OUT / 'generalization_audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = []; accuracy = []; recall = []
    for key in candidates:
        name = 'AWS RF100' if key == 'aws_tuned' else 'Andrew RF500'
        group_score = next(r for r in pooled if r['model'] == key)
        for protocol, row in [('Random split', original.loc[key]), ('Nieuwe soorten (5 folds)', group_score)]:
            labels.append(name + '\n' + protocol); accuracy.append(row['accuracy']); recall.append(row['recall_poisonous'])
    positions = np.arange(len(labels))
    for offset, values, label, color in [(-.18, accuracy, 'Accuracy', '#315e91'), (.18, recall, 'Recall giftige klasse', '#b85c4a')]:
        bars = ax.bar(positions + offset, values, width=.36, label=label, color=color)
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    ax.set_xticks(positions, labels); ax.set_ylim(0, 1.15); ax.set_ylabel('Score')
    ax.set_title('Een random split test geen volledig nieuwe soorten')
    ax.legend(loc='upper center', ncols=2); fig.tight_layout()
    fig.savefig(OUT / 'generalization_comparison.png', dpi=180); plt.close(fig)
    print(pd.DataFrame(pooled)[['model', 'accuracy', 'recall_poisonous', 'f1_poisonous', 'fn']].to_string(index=False), flush=True)
    print('Shuffled-label control:', control_scores, flush=True)
    return result


if __name__ == '__main__':
    run()
