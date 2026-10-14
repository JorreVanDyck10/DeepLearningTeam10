"""Export the chosen fitted pipeline; never train a replacement for deployment."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
import sklearn
from backend.mushroom_model import PortableMushroom, MODEL_ID, FEATURES


def run():
    assert sklearn.__version__ == '1.4.2', 'Use the comparison environment for export'
    folder = ROOT / 'MushroomFinal/comparison_correct'
    summary = json.loads((folder/'comparison_summary.json').read_text(encoding='utf-8'))
    source = folder/'selected_pipeline.joblib'
    assert summary['selected_model'] == 'rf_final'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == summary['artifact']['sha256']
    model = joblib.load(source)
    assert model.feature_names_in_.tolist() == FEATURES and model.classes_.tolist() == [0, 1]
    forest = model.named_steps['classifier']
    assert forest.n_estimators == 300 and forest.max_depth == 20
    transforms = {name: (transformer, columns) for name, transformer, columns in model.named_steps['preprocessor'].transformers_ if name != 'remainder'}
    numeric, numeric_cols = transforms['num']; categorical, categorical_cols = transforms['cat']
    encoder = categorical.named_steps['encode']
    assert encoder.drop is None and encoder.handle_unknown == 'ignore'
    dest = ROOT/'backend/artifacts'; dest.mkdir(exist_ok=True)
    output = dest/'mushroom_rf300.npz'
    trees = [tree.tree_ for tree in forest.estimators_]
    np.savez_compressed(output,
        left=np.concatenate([t.children_left for t in trees]).astype(np.int32),
        right=np.concatenate([t.children_right for t in trees]).astype(np.int32),
        feature=np.concatenate([t.feature for t in trees]).astype(np.int16),
        threshold=np.concatenate([t.threshold for t in trees]),
        probabilities=np.concatenate([t.value[:, 0, :] for t in trees]),
        offsets=np.r_[0, np.cumsum([t.node_count for t in trees])])
    row = next(r for r in summary['model_comparison'] if r['model'] == 'rf_final')
    metadata = {'schema_version': 1, 'model_id': MODEL_ID, 'model_name': 'Random Forest · 300 bomen',
        'features': FEATURES, 'classes': ['e', 'p'], 'n_estimators': 300,
        'source_pipeline': 'MushroomFinal/comparison_correct/selected_pipeline.joblib',
        'source_pipeline_sha256': summary['artifact']['sha256'],
        'forest_file': output.name, 'forest_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'numeric_features': list(numeric_cols), 'numeric_medians': numeric.named_steps['impute'].statistics_.tolist(),
        'categorical': [{'name': str(name), 'imputation': str(mode), 'categories': [str(c) for c in codes]}
            for name, mode, codes in zip(categorical_cols, categorical.named_steps['impute'].statistics_, encoder.categories_)],
        'dataset_rows': 5000, 'train_rows': 4100, 'test_rows': 900,
        'dataset_sha256_lf': summary['protocol']['dataset']['sha256_lf'],
        'metrics': {**row, 'recall_poisonous': row['recall_p'], 'poisonous_predicted_edible': row['fn']},
        'validation_note': 'Ontwikkeltest; deze testset is eerder gebruikt tijdens het onderzoek.'}
    manifest = dest/'mushroom_rf300.json'
    manifest.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    portable = PortableMushroom(manifest)
    data = pd.read_csv(ROOT/'MushroomFinal/mushroom_project_dataset.csv')
    data.columns = data.columns.str.replace('-', '_', regex=False)
    X = data[FEATURES]
    predictions = pd.read_csv(folder/'test_predictions.csv', float_precision='round_trip')
    test_frame = X.iloc[predictions.source_row]
    np.testing.assert_array_equal(portable.predict(test_frame), predictions.rf_final_prediction)
    np.testing.assert_allclose(portable.predict_proba(test_frame)[:, 1], predictions.rf_final_probability_p, rtol=0, atol=1e-12)
    cases = pd.concat([X, X.iloc[:4].assign(stem_height=0, stem_width=0),
        pd.DataFrame([{c: np.nan for c in FEATURES}]), X.iloc[:1].assign(cap_shape='unseen')], ignore_index=True)
    expected = model.predict_proba(cases); actual = portable.predict_proba(cases)
    np.testing.assert_array_equal(np.argmax(actual, axis=1), model.predict(cases))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    metadata['example'] = json.loads(test_frame.iloc[[0]].to_json(orient='records'))[0]
    manifest.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    receipt = {'source_pipeline_sha256': summary['artifact']['sha256'], 'model_id': MODEL_ID,
        'test_rows_verified': 900, 'additional_rows_verified': len(cases),
        'label_disagreements': 0, 'max_probability_difference': float(np.max(np.abs(expected-actual))),
        'tolerance': 1e-12, 'export_sklearn': sklearn.__version__, 'forest_sha256': metadata['forest_sha256']}
    (dest/'mushroom_export_audit.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('Lossless export verified:', receipt)


if __name__ == '__main__': run()
