"""Audit the user's real corrected SageMaker export without refitting its model."""
import json
from pathlib import Path
import platform
import sys
import warnings

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import pandas as pd
import sklearn
from MushroomFinal.compare_correct_dataset import load_data,evaluate,digest,write_json,OUT

EXPORTS=ROOT/'SolutionJorre/MushroomDataset/correct_dataset'


def run():
    assert sklearn.__version__=='1.4.2', 'Use the comparison environment; do not migrate the saved artifact implicitly.'
    raw,X,y,train,test,folds=load_data()
    results=json.loads((EXPORTS/'correct_aws_results.json').read_text(encoding='utf-8'))
    tuning=pd.read_csv(EXPORTS/'correct_aws_tuning_results.csv',float_precision='round_trip')
    notebook=json.loads((EXPORTS/'04_aws_correct_dataset.ipynb').read_text(encoding='utf-8'))
    assert results['source_sha256_lf']==digest(ROOT/'MushroomFinal/mushroom_project_dataset.csv')['sha256_lf']
    assert results['rows']==5000 and results['train_rows']==4100 and results['test_rows']==900
    assert results['features']==X.columns.tolist() and results['removed']==['jumbled_noise_0']
    assert results['class_encoding']=={'e':0,'p':1}
    assert not any(o.get('output_type')=='error' for c in notebook['cells'] for o in c.get('outputs',[]))
    executed=[c for c in notebook['cells'] if c['cell_type']=='code' and c.get('execution_count') is not None]
    assert any('search.fit' in ''.join(c['source']) for c in executed)
    assert any('joblib.dump' in ''.join(c['source']) for c in executed)
    assert len(tuning)==4 and set(tuning.rank_test_score)=={1,2,3,4}
    best=tuning.loc[tuning.rank_test_score==1].iloc[0]
    fold_columns=[f'split{i}_test_score' for i in range(5)]
    for _,r in tuning.iterrows():
        values=r[fold_columns].astype(float).to_numpy()
        assert abs(values.mean()-r.mean_test_score)<1e-12
        assert abs(values.std(ddof=0)-r.std_test_score)<1e-12
    assert abs(best.mean_test_score-results['cv_macro_f1'])<1e-12
    assert pd.isna(best.param_classifier__max_depth) and int(best.param_classifier__n_estimators)==100
    assert int(best.param_classifier__min_samples_leaf)==1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model=joblib.load(EXPORTS/'correct_mushroom_pipeline.joblib')
        assert model.feature_names_in_.tolist()==X.columns.tolist() and model.classes_.tolist()==[0,1]
        params=model.named_steps['classifier'].get_params()
        for name,value in results['best_parameters'].items():
            assert params[name.removeprefix('classifier__')]==value
        predicted=model.predict(X.iloc[test]);prob=model.predict_proba(X.iloc[test])[:,1]
    metrics=evaluate(y.iloc[test],predicted,prob)
    for key in ['accuracy','balanced_accuracy','macro_f1','precision_p','recall_p','f1_p','roc_auc']:
        assert abs(metrics[key]-results[key])<1e-12,(key,metrics[key],results[key])
    assert [[metrics['tn'],metrics['fp']],[metrics['fn'],metrics['tp']]]==results['confusion_matrix']
    previous=pd.read_csv(OUT/'test_predictions.csv',float_precision='round_trip')
    local_cv=pd.read_csv(OUT/'cv_folds.csv',float_precision='round_trip')
    assert previous.source_row.tolist()==test.tolist()
    local_baseline=local_cv[local_cv.model=='aws_baseline_local'].sort_values('fold')
    assert np.allclose(best[fold_columns].astype(float),local_baseline.macro_f1,rtol=0,atol=1e-12)
    export=pd.DataFrame({'source_row':test,'true_class':y.iloc[test].to_numpy(),
        'prediction':predicted,'probability_p':prob})
    export.to_csv(OUT/'aws_artifact_predictions.csv',index=False)
    audit={'origin':'User-provided executed SageMaker notebook and corrected exports',
        'dataset':results['source_sha256_lf'],'source_rows':5000,'train_rows':4100,'test_rows':900,
        'features':X.columns.tolist(),'artifact':digest(EXPORTS/'correct_mushroom_pipeline.joblib'),
        'executed_notebook':digest(EXPORTS/'04_aws_correct_dataset.ipynb'),
        'aws_versions':results['versions'],'audit_versions':{'python':platform.python_version(),
            'sklearn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__,'joblib':joblib.__version__},
        'best_parameters':results['best_parameters'],'cv_macro_f1':results['cv_macro_f1'],
        'cv_std_population':float(best.std_test_score),
        'cv_std_sample':float(best[fold_columns].astype(float).std(ddof=1)),
        'recomputed_metrics':metrics,'matches_exported_metrics':True,
        'local_baseline_label_disagreements':int((predicted!=previous.aws_baseline_local_prediction).sum()),
        'local_baseline_max_probability_difference':float(np.max(np.abs(prob-previous.aws_baseline_local_probability_p))),
        'selected_cv_scores_match_local_baseline':True,
        'split_membership_evidence':'Reconstructed from the executed notebook code; no separate original training-row log was exported.',
        'warnings':[str(w.message) for w in caught]}
    write_json(OUT/'aws_artifact_audit.json',audit)
    print('AWS artifact verified on the same 900 rows:',metrics,flush=True)
    print('Label disagreements with prior local AWS baseline:',audit['local_baseline_label_disagreements'],flush=True)
    return audit


if __name__=='__main__':run()
