"""Fit/package the compatible RF300 candidate, entirely inside this concept.

No install/promote operation exists here. The existing backend reader is reused
to prove the exported arrays match sklearn. Metrics remain training-CV evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from improve import experiments, OUT, clean_parameters, f1_threshold
from mushroom_alternative import development_data, HERE, ROOT, SOURCE_HASH, write_json

ARTIFACTS = HERE / 'artifacts'


def acceptance(family='rf_compatible', policy='default'):
    table = pd.read_csv(OUT / 'summary.csv').set_index('model')
    metrics = pd.read_csv(OUT / 'outer_fold_metrics.csv')
    candidate = table.loc[family + '_' + policy]
    reference = table.loc['rf300_reference_default']
    assert candidate['folds'] == reference['folds'] == 5
    a = metrics[metrics.model == family + '_' + policy].set_index('fold').macro_f1
    b = metrics[metrics.model == 'rf300_reference_default'].set_index('fold').macro_f1
    difference = a - b
    gates = {'macro_f1_gain_at_least_005': bool(candidate.macro_f1_mean >= reference.macro_f1_mean + .005),
        'recall_not_lower': bool(candidate.recall_p_mean >= reference.recall_p_mean),
        'wins_at_least_three_folds': bool((difference > 0).sum() >= 3)}
    return {'family': family, 'policy': policy, 'passes_performance_gates': all(gates.values()),
        'gates': gates, 'macro_f1_gain': float(candidate.macro_f1_mean - reference.macro_f1_mean),
        'recall_gain': float(candidate.recall_p_mean - reference.recall_p_mean),
        'paired_fold_macro_f1_differences': difference.tolist(),
        'candidate_metrics': candidate.to_dict(), 'reference_metrics': reference.to_dict()}


def export_rf(model, metadata, destination):
    destination.mkdir(parents=True, exist_ok=True)
    clf = model.named_steps['classifier']
    pre = model.named_steps['preprocessor']
    assert clf.n_estimators == 300 and clf.max_depth == 20
    numeric = pre.named_transformers_['num'].named_steps['impute']
    cat = pre.named_transformers_['cat']
    assert not numeric.add_indicator
    trees = [t.tree_ for t in clf.estimators_]
    target = destination / 'mushroom_rf300.npz'
    np.savez_compressed(target,
        left=np.concatenate([t.children_left for t in trees]).astype(np.int32),
        right=np.concatenate([t.children_right for t in trees]).astype(np.int32),
        feature=np.concatenate([t.feature for t in trees]).astype(np.int16),
        threshold=np.concatenate([t.threshold for t in trees]),
        probabilities=np.concatenate([t.value[:, 0, :] for t in trees]),
        offsets=np.r_[0, np.cumsum([t.node_count for t in trees])])
    columns = {name: cols for name, _, cols in pre.transformers_ if name != 'remainder'}
    encoder = cat.named_steps['encode']
    assert encoder.drop is None and encoder.handle_unknown == 'ignore'
    metadata.update(forest_file=target.name,
        forest_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        numeric_features=list(columns['num']), numeric_medians=numeric.statistics_.tolist(),
        categorical=[{'name': str(name), 'imputation': str(mode), 'categories': [str(c) for c in codes]}
            for name, mode, codes in zip(columns['cat'], cat.named_steps['impute'].statistics_, encoder.categories_)])
    write_json(destination / 'mushroom_rf300.json', metadata)


def run(policy='f1_threshold'):
    result = acceptance(policy=policy)
    X, y, ids, folds = development_data()
    prototype, grid = experiments()['rf_compatible']
    # A production candidate has one configuration chosen on development CV.
    # Its honest procedure estimate remains the outer metrics, not best_score_.
    search = GridSearchCV(prototype, grid, scoring='f1_macro',
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        n_jobs=2, pre_dispatch=2, error_score='raise')
    print('Fit final compatible configuration using training rows only', flush=True)
    search.fit(X, y)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(search.cv_results_).to_csv(OUT / 'final_compatible_search.csv', index=False)
    original = search.best_estimator_
    inner = StratifiedKFold(5, shuffle=True, random_state=42)
    inner_probability = cross_val_predict(clone(original), X, y, cv=inner,
        method='predict_proba', n_jobs=2, pre_dispatch=2)[:, 1]
    proposed_threshold = f1_threshold(y.to_numpy(), inner_probability) if policy == 'f1_threshold' else .5
    assert not original.named_steps['measurements'].geometry and not original.named_steps['measurements'].zero_as_missing
    # Remove the deterministic no-op to keep the serialized model free of custom classes.
    model = Pipeline(original.steps[1:])
    model_path = ARTIFACTS / 'current_pipeline.joblib'
    joblib.dump(model, model_path, compress=3)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    metadata = json.loads((ROOT / 'backend/artifacts/mushroom_rf300.json').read_text(encoding='utf-8'))
    # The current serving reader still uses .5. Never label its raw predictions
    # with the more favorable proposed-threshold metrics.
    row = acceptance(policy='default')['candidate_metrics']
    metrics = {'accuracy': row['accuracy_mean'], 'macro_f1': row['macro_f1_mean'],
        'cv_macro_f1_mean': row['macro_f1_mean'], 'cv_macro_f1_std': row['macro_f1_sd'],
        'cv_recall_p_mean': row['recall_p_mean'], 'recall_poisonous': row['recall_p_mean'],
        'poisonous_predicted_edible': int(row['fn_total']), 'fn': int(row['fn_total']),
        'fp': int(row['fp_total']), 'metrics_scope': 'outer CV on 4100 development rows'}
    metadata.update(source_pipeline=model_path.relative_to(ROOT).as_posix(),
        source_pipeline_sha256=digest, model_name='RF300 — Milan ontwikkelkandidaat',
        metrics=metrics, test_rows=0, validation_rows=4100, train_rows=4100,
        validation_note='Ontwikkelings-CV: vijf buitenfolds op 4.100 rijen; geen nieuwe onafhankelijke eindtest.',
        dataset_sha256_lf=SOURCE_HASH, concept_only=True,
        promotion_recommended=result['passes_performance_gates'],
        proposed_decision_threshold=proposed_threshold,
        raw_classifier_threshold=.5, reported_metrics_policy='default',
        proposed_policy=policy, proposed_policy_metrics=result['candidate_metrics'],
        policy_note='New threshold is metadata only; the original runtime still uses its original raw classifier and website policy. Preview applies the proposal explicitly.')
    export_rf(model, metadata, ARTIFACTS)
    sys.path.insert(0, str(ROOT))
    from backend.mushroom_model import PortableMushroom, FEATURES
    portable = PortableMushroom(ARTIFACTS / 'mushroom_rf300.json')
    expected = model.predict_proba(X[FEATURES])
    actual = portable.predict_proba(X[FEATURES])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(np.argmax(actual, axis=1), model.predict(X[FEATURES]))
    # These are export-equivalence checks on training inputs, NOT validation scores.
    pd.DataFrame({'source_row': ids, 'prediction': model.predict(X),
        'probability_p': expected[:, 1]}).to_csv(ARTIFACTS / 'training_input_export_references.csv', index=False)
    example = json.loads(X.iloc[[0]].to_json(orient='records'))[0]
    metadata['example'] = example
    write_json(ARTIFACTS / 'mushroom_rf300.json', metadata)
    write_json(OUT / 'candidate_package.json', {'acceptance': result,
        'parameters': clean_parameters(search.best_params_),
        'selection_cv_macro_f1_not_independent': float(search.best_score_),
        'pipeline_sha256': digest, 'export_rows_checked': len(X),
        'proposed_decision_threshold': proposed_threshold,
        'original_runtime_raw_threshold': .5,
        'export_max_probability_difference': float(np.max(np.abs(actual - expected))),
        'labels_disagree': int((np.argmax(actual, axis=1) != model.predict(X)).sum()),
        'input_features': FEATURES, 'custom_pickle_classes': False,
        'deployment_performed': False, 'development_test_rows_scored': 0})
    print(json.dumps(result, indent=2), flush=True)
    print('Candidate packaged; no installation or promotion performed.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', choices=['default', 'f1_threshold'], default='f1_threshold')
    args = parser.parse_args()
    with joblib.parallel_config(backend='threading'):
        run(args.policy)
