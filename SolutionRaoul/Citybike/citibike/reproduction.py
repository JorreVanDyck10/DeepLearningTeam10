"""Controle van notebookreproducties zonder oorspronkelijke studieresultaten te vervangen."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import REPORTS, atomic_json, now, sha256_file
from .features import FEATURES
from .modeling import metrics


def save_model_reproduction(name, training, validation, train_prediction, validation_prediction, fit_seconds, prepared):
    scores = metrics(validation.rides, validation_prediction)
    scores.update({'train_'+k:v for k,v in metrics(training.rides, train_prediction).items()})
    original = next(r for r in json.loads((REPORTS/'model_comparison.json').read_text(encoding='utf-8')) if r['name']==name)
    for key,value in scores.items():
        np.testing.assert_allclose(value, original[key], rtol=1e-10, atol=1e-8, err_msg=name+' '+key)
    result = {'name':name, **scores, 'fit_seconds':fit_seconds, 'reproduced_at':now(),
              'prepared_sha256':prepared['sha256'], 'fold_fingerprint':prepared['fold_fingerprint'],
              'test_used':False, 'model_selection_changed':False,
              'original_metrics_equal':True}
    atomic_json(REPORTS/'notebook_reproduction'/f'{name}.json',result)
    return result


def save_pycaret_reproduction(experiment, ranked, training, validation, prepared):
    from .modeling import candidates
    for stage,expected in [('train',training),('test',validation)]:
        actual=experiment.get_config('X_'+stage)
        if list(actual.columns)!=FEATURES: raise AssertionError('Andere PyCaret-features')
        np.testing.assert_array_equal(actual.to_numpy(dtype=np.float32),expected[FEATURES].to_numpy(dtype=np.float32))
        np.testing.assert_array_equal(np.asarray(experiment.get_config('y_'+stage),dtype=np.float32),expected.rides.to_numpy(dtype=np.float32))
    labels=[next(name for name,model in candidates().items() if type(model.estimator) is type(fitted.estimator)) for fitted in ranked]
    table=experiment.pull().copy()
    table.insert(0,'candidate',labels)
    previous=pd.read_csv(REPORTS/'pycaret_comparison.csv').set_index('candidate')
    for row in table.to_dict('records'):
        for key in ['RMSE','MAE','R2']:
            np.testing.assert_allclose(row[key],previous.loc[row['candidate'],key],rtol=1e-7,atol=1e-4)
    directory=REPORTS/'notebook_reproduction';directory.mkdir(parents=True,exist_ok=True)
    table.to_csv(directory/'pycaret_comparison.csv',index=False)
    atomic_json(directory/'pycaret_audit.json',{'reproduced_at':now(),'actual_internal_rows_checked':True,
        'fold_fingerprint':prepared['fold_fingerprint'],'folds':prepared['folds'],
        'train_rows':len(training),'validation_rows':len(validation),'test_used':False,
        'shuffle':False,'original_metrics_equal':True})
    return table


def check_protected_files(root):
    root=Path(root)
    receipt=json.loads((root/'reports/improvement_protection.json').read_text(encoding='utf-8'))
    for name,digest in receipt['protected'].items():
        if sha256_file(root/name)!=digest:
            raise AssertionError('Beschermd bestand gewijzigd: '+name)
    return {'checked_at':now(),'files_checked':len(receipt['protected']), 'all_equal':True}
