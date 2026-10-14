"""Inspect fixed score cutoffs; never fit or choose a cutoff on the test set."""
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'MushroomFinal/comparison_correct'


def audit():
    result = {'policy_id': 'mushroom_no_edibility_confirmation_v1',
              'conclusion': 'No confirmation of edibility: 95% edible scores still include poisonous validation records.',
              'calibrated': False, 'training_performed': False, 'test_used_for_threshold_selection': False,
              'sources': {}, 'thresholds': []}
    for filename, kind in [('oof_predictions.csv', 'training_oof'), ('test_predictions.csv', 'development_test_descriptive')]:
        path = FOLDER / filename
        result['sources'][filename] = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        data = pd.read_csv(path, float_precision='round_trip')
        score = data.rf_final_probability_p
        actual = data.true_class
        for cutoff in [.5, .1, .05, .025, .01, .005, 0]:
            predicted_edible = score <= cutoff
            result['thresholds'].append({'split': kind, 'rows': len(data), 'max_poison_score': cutoff,
                'min_edible_score': 1-cutoff, 'edible_calls': int(predicted_edible.sum()),
                'poisonous_among_edible_calls': int((predicted_edible & (actual == 1)).sum()),
                'edible_rejected': int((~predicted_edible & (actual == 0)).sum())})
    (FOLDER/'decision_threshold_audit.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result['thresholds'], indent=2))


if __name__ == '__main__': audit()
