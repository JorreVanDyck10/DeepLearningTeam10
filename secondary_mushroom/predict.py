"""Predict simulated mushroom labels from a JSON object or list of objects."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DEFAULT_MODEL = Path(__file__).resolve().parent / 'models/baseline_decision_tree.joblib'


def predict_records(records, model_path=DEFAULT_MODEL, *, model=None):
    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, list) or not records or not all(isinstance(row, dict) for row in records):
        raise ValueError('Input must be a JSON object or a non-empty list of objects.')
    # The API can pass a pipeline loaded once at startup; the CLI still loads from disk.
    if model is None:
        if not Path(model_path).is_file():
            raise FileNotFoundError('Run 04_predict_baseline.ipynb first to save the model.')
        model = joblib.load(model_path)
    columns = list(model.feature_names_in_)
    for index, row in enumerate(records):
        missing = set(columns) - set(row)
        extra = set(row) - set(columns)
        if missing or extra:
            raise ValueError(f'Row {index}: missing fields {sorted(missing)}; extra fields {sorted(extra)}. '
                             'Use null for missing values.')
    # Object dtype and np.nan make JSON null work with sklearn imputers.
    frame = pd.DataFrame(records, columns=columns, dtype=object).where(
        lambda data: data.notna(), np.nan)
    numeric = ['cap-diameter', 'stem-height', 'stem-width']
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors='raise')
        if np.isinf(frame[column].to_numpy(dtype=float)).any():
            raise ValueError(f'{column} must be finite or null.')
    for column in set(columns) - set(numeric):
        if not frame[column].map(lambda value: isinstance(value, str) or pd.isna(value)).all():
            raise ValueError(f'{column} must contain a dataset code (string) or null.')
    predicted = model.predict(frame)
    positive_index = list(model.classes_).index('p')
    probabilities = model.predict_proba(frame)[:, positive_index]
    return [
        {'label': str(label), 'prediction': 'giftig' if label == 'p' else 'eetbaar',
         'probability_poisonous': float(probability)}
        for label, probability in zip(predicted, probabilities)
    ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='JSON file with all 20 original feature names.')
    args = parser.parse_args()
    try:
        records = json.loads(args.input.read_text(encoding='utf-8'))
        print(json.dumps(predict_records(records), indent=2))
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f'{error}\n')
