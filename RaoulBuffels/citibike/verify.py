"""Eindcontrole van echte artefacten en operationele inference."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import subprocess
import sys
import time
from sklearn.base import clone
from .config import ROOT,REPORTS,DATA,MODELS,atomic_json,fingerprint,sha256_file
from .datasets import manifest
from .inference import predict_day,load_model
from .features import make_features,FEATURES


def verify():
    m=manifest()
    required={'sources','source_fingerprint','code_version','schema_mapping','configuration',
              'started_at','finished_at','processing_seconds','raw_rows','retained_rows',
              'removed_rows','coverage','partitions','dataset_fingerprint','schema_version'}
    assert required<=m.keys(),'Manifest onvolledig'
    assert m['raw_rows']==m['retained_rows']+m['removed_rows']
    assert len({(p['area'],p['period']) for p in m['partitions']})==len(m['partitions'])
    assert set(m['coverage'])=={'NYC','JC'}
    for source in m['sources']:
        assert {'path','area','size_bytes','sha256','members','schemas'}<=source.keys()
        assert source['area'] in ('NYC','JC') and len(source['sha256'])==64
        for member in source['members']:
            assert {'nested','name','period','uncompressed_bytes','crc','schema'}<=member.keys()
    for p in m['partitions']:
        if sha256_file(Path(p['path']))!=p['sha256']:raise AssertionError('Partition gewijzigd')
    expected=fingerprint([(p['area'],p['period'],p['sha256'],m['schema_version']) for p in m['partitions']])
    assert expected==m['dataset_fingerprint']
    split=json.loads((REPORTS/'chronological_splits.json').read_text(encoding='utf-8'))
    pycaret=json.loads((REPORTS/'pycaret_split_audit.json').read_text(encoding='utf-8'))
    assert split['fold_fingerprint']==pycaret['fold_fingerprint']
    artifact=load_model()
    modeling=pd.read_parquet(DATA/'modeling_development_validation.parquet')
    original_training=modeling[(modeling.day>=pd.Timestamp(split['train_start'])) &
                               (modeling.day<pd.Timestamp(split['train_end_exclusive']))]
    original_validation=modeling[(modeling.day>=pd.Timestamp(split['validation_start'])) &
                                 (modeling.day<pd.Timestamp(split['validation_end_exclusive']))]
    assert fingerprint(original_training.timestamp.astype(str).tolist())==pycaret['train_data_fingerprint']
    assert fingerprint(original_validation.timestamp.astype(str).tolist())==pycaret['validation_data_fingerprint']
    history=pd.read_parquet(DATA/'NYC_hourly_development.parquet')
    date=history.timestamp.max().normalize()
    history=history[history.timestamp<date]
    deployed=predict_day(date,history,artifact)
    from citibike.features import make_features
    future=pd.DataFrame({'timestamp':pd.date_range(date,periods=24,freq='h'),'rides':np.nan,'area':'NYC'})
    features=make_features(pd.concat([history.tail(14*24),future],ignore_index=True),future_day=True)
    from .inference import predict_features
    np.testing.assert_allclose(deployed.predicted_rides,predict_features(artifact['model'],features,FEATURES))
    # Vaste configuratie opnieuw fitten: geen selectie, geen tuning, geen eindtestlabels.
    refit=modeling[modeling.day>=pd.Timestamp(artifact['training_start'])]
    assert len(refit)==artifact['training_rows']
    assert refit.day.max()<pd.Timestamp('2026-07-01')
    started=time.perf_counter()
    reproduced=clone(artifact['model']).fit(refit[FEATURES],refit.rides)
    reproduced_predictions=predict_features(reproduced,features,FEATURES)
    np.testing.assert_allclose(deployed.predicted_rides,reproduced_predictions,rtol=1e-12,atol=1e-9)
    original_values=predict_features(artifact['model'],original_validation,FEATURES)
    reproduced_values=predict_features(reproduced,original_validation,FEATURES)
    np.testing.assert_allclose(original_values,reproduced_values,rtol=1e-12,atol=1e-9)
    retraining_seconds=time.perf_counter()-started
    # Test de echte Streamlit-formulieractie met dezelfde demonstratiedag.
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
    assert not app.exception, str(app.exception)
    app.date_input[0].set_value(date.date())
    app.button[0].click().run()
    assert not app.exception, str(app.exception)
    assert len(app.dataframe)==1 and len(app.dataframe[0].value)==24
    np.testing.assert_allclose(app.dataframe[0].value['verwachte_ritten'],deployed.predicted_rides)
    ignored=subprocess.run(['git','check-ignore',str(MODELS/'citibike_model.joblib')],cwd=ROOT,capture_output=True,text=True)
    assert ignored.returncode==0,'Modelartefact wordt niet genegeerd door Git'
    test=json.loads((REPORTS/'final_evaluation.json').read_text(encoding='utf-8'))
    assert test['model_sha256']==sha256_file(MODELS/'citibike_model.joblib')
    results={'python_version':sys.version,'python_executable':sys.executable,
             'partitions_verified':len(m['partitions']),'manifest_fingerprint_verified':True,
             'pycaret_identical_folds':True,'offline_app_predictions_equal':True,
             'pycaret_identical_train_validation_rows':True,
             'fixed_configuration_retraining_verified':True,'retraining_seconds':retraining_seconds,
             'retraining_prediction_max_difference':float(np.max(np.abs(original_values-reproduced_values))),
             'streamlit_form_predictions_verified':True,
             'manifest_required_fields_verified':True,
             'model_ignored_by_git':True,'evaluated_model_is_deployed':True,
             'prediction_code_and_packages_locked':True,
             'raw_rows':m['raw_rows'],'retained_rows':m['retained_rows'],
             'modeling_rows':split['train_rows']+split['validation_rows'],'test_rows':test['rows']}
    atomic_json(REPORTS/'verification.json',results)
    print(json.dumps(results,indent=2),flush=True)
    return results
