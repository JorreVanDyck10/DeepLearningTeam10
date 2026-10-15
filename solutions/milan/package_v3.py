"""Package an RF challenger with explicit gains and losses against v2.

The existing serving reader and eleven-field input are retained. No deployment.
Boosters are research comparisons, not an implicit new serving dependency.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from improve_v3 import experiments, threshold_at_fpr, OUT
from improve import f1_threshold
from mushroom_alternative import development_data, HERE, ROOT, SOURCE_HASH, FEATURES, write_json

DEST = HERE/'artifacts_v3'


def run():
    assert json.loads((OUT/'run.json').read_text())['complete']
    table = pd.read_csv(OUT/'summary.csv').set_index('model')
    baseline = table.loc['rf_compatible_f1_threshold']
    candidates = table.loc[['rf_selection_default','rf_selection_f1','rf_selection_fpr_budget']]
    eligible = candidates[(candidates.macro_f1_mean>baseline.macro_f1_mean)
        &(candidates.fn_total<baseline.fn_total)&(candidates.fp_total<=baseline.fp_total)]
    goal_met=bool(len(eligible))
    # If the full goal fails, a simpler alternative may still be useful, but
    # qualify it explicitly against the original model and disclose the v2 cost.
    selected=eligible.macro_f1_mean.idxmax() if goal_met else candidates.macro_f1_mean.idxmax()
    original_reference=table.loc['rf300_reference_default']
    candidate=candidates.loc[selected]
    dominates_original=bool(candidate.macro_f1_mean>original_reference.macro_f1_mean
        and candidate.fn_total<original_reference.fn_total and candidate.fp_total<=original_reference.fp_total)
    if not goal_met and not dominates_original:
        write_json(OUT/'candidate_package.json', {'packaged':False,'goal_met':False,
            'reason':'RF neither meets the complete v2 goal nor reduces both error counts against the original.',
            'previous_candidate_retained':True,'deployment_performed':False})
        print('No useful RF challenger; v2 retained.')
        return
    policy = selected.removeprefix('rf_selection_')
    X,y,ids,_ = development_data()
    prototype,grid = experiments()['rf_selection']
    cv = StratifiedKFold(5,shuffle=True,random_state=42)
    search = GridSearchCV(prototype,grid,scoring='f1_macro',cv=cv,n_jobs=2,pre_dispatch=2,error_score='raise')
    search.fit(X,y)
    original = search.best_estimator_
    inside = cross_val_predict(clone(original),X,y,cv=cv,n_jobs=2,pre_dispatch=2,method='predict_proba')[:,1]
    threshold = .5 if policy=='default' else (f1_threshold(y,inside) if policy=='f1' else threshold_at_fpr(y,inside))
    # ColumnTransformer selects its learned columns by name even when the input
    # contains all eleven fields. This removes the custom selector from pickle.
    model = Pipeline(original.steps[1:])
    expected = original.predict_proba(X)
    np.testing.assert_array_equal(model.predict_proba(X),expected)
    DEST.mkdir(parents=True,exist_ok=True)
    path = DEST/'current_pipeline.joblib'
    joblib.dump(model,path,compress=3)
    clf=model.named_steps['classifier']; pre=model.named_steps['preprocessor']
    trees=[t.tree_ for t in clf.estimators_]
    forest=DEST/'mushroom_rf300.npz'
    np.savez_compressed(forest,left=np.concatenate([t.children_left for t in trees]).astype(np.int32),
        right=np.concatenate([t.children_right for t in trees]).astype(np.int32),
        feature=np.concatenate([t.feature for t in trees]).astype(np.int16),
        threshold=np.concatenate([t.threshold for t in trees]),
        probabilities=np.concatenate([t.value[:,0,:] for t in trees]),
        offsets=np.r_[0,np.cumsum([t.node_count for t in trees])])
    columns={name:cols for name,_,cols in pre.transformers_ if name!='remainder'}
    cats=pre.named_transformers_['cat']
    specs=[{'name':name,'imputation':str(mode),'categories':[str(c) for c in codes]}
        for name,mode,codes in zip(columns['cat'],cats.named_steps['impute'].statistics_,cats.named_steps['encode'].categories_)]
    metadata=json.loads((HERE/'artifacts/mushroom_rf300.json').read_text(encoding='utf-8'))
    used=original.named_steps['inputs'].columns_
    ignored=[f for f in FEATURES if f not in used]
    # Unused public fields remain accepted by the existing form/reader. Append
    # their original, allowed category specs AFTER all learned columns, so no
    # trained split index changes. No tree uses these appended encodings.
    original_specs={s['name']:s for s in metadata['categorical']}
    specs += [original_specs[f] for f in ignored]
    raw=table.loc['rf_selection_default']; chosen=table.loc[selected]
    metadata.update(model_name='RF300 — Milan featureselectie v3',
        source_pipeline=path.relative_to(ROOT).as_posix(),source_pipeline_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        forest_sha256=hashlib.sha256(forest.read_bytes()).hexdigest(),forest_file=forest.name,
        numeric_features=list(columns['num']),numeric_medians=pre.named_transformers_['num'].statistics_.tolist(),
        categorical=specs,ignored_features=ignored,
        metrics={'accuracy':raw.accuracy_mean,'macro_f1':raw.macro_f1_mean,
            'recall_poisonous':raw.recall_p_mean,'fn':int(raw.fn_total),'fp':int(raw.fp_total),
            'metrics_scope':'outer CV, 4100 development rows, raw classifier'},
        proposed_policy=policy,proposed_decision_threshold=threshold,proposed_policy_metrics=chosen.to_dict(),
        reported_metrics_policy='default',raw_classifier_threshold=.5,
        validation_note='Derde ontwikkelstudie op eerder gebruikte 4100 trainingsrijen; geen onafhankelijke eindtest.',
        dataset_sha256_lf=SOURCE_HASH,promotion_recommended=goal_met,concept_only=True,
        policy_note='RF challenger. The complete improvement goal against v2 is '+str(goal_met)+
            '; see both error counts. Public input stays eleven fields; ignored fields are appended as unused encodings.')
    write_json(DEST/'mushroom_rf300.json',metadata)
    sys.path.insert(0,str(ROOT))
    from backend.mushroom_model import PortableMushroom
    portable=PortableMushroom(DEST/'mushroom_rf300.json')
    actual=portable.predict_proba(X)
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
    np.testing.assert_array_equal(portable.predict(X),original.predict(X))
    edge=X.iloc[:10].copy()
    edge.iloc[0,:]=np.nan
    for f in columns['cat']:
        edge.loc[1,f]='never_seen'
    np.testing.assert_allclose(portable.predict_proba(edge),original.predict_proba(edge),rtol=0,atol=1e-12)
    # Record-level API validation still rejects truly unknown category codes.
    example=metadata['example']
    assert isinstance(portable.predict_record(example)['probability_poisonous'],float)
    changed=X.copy()
    for f in ignored:
        changed[f]='never_seen'
    np.testing.assert_array_equal(actual,portable.predict_proba(changed))
    pd.DataFrame(search.cv_results_).to_csv(OUT/'final_rf_search.csv',index=False)
    pd.DataFrame({'source_row':ids,'prediction':original.predict(X),'probability_p':expected[:,1]}).to_csv(DEST/'training_input_export_references.csv',index=False)
    write_json(OUT/'candidate_package.json',{'packaged':True,'goal_met':goal_met,
        'dominates_original_error_counts':dominates_original,
        'extra_fn_against_v2':int(candidate.fn_total-baseline.fn_total),
        'fp_change_against_v2':int(candidate.fp_total-baseline.fp_total),
        'family':'rf_selection','policy':policy,
        'parameters':search.best_params_,'ignored_features':ignored,'input_features':FEATURES,
        'fulltrain_threshold':threshold,'pipeline_sha256':metadata['source_pipeline_sha256'],
        'export_rows_checked':len(X),'export_max_probability_difference':float(np.max(np.abs(expected-actual))),
        'labels_disagree':int((portable.predict(X)!=original.predict(X)).sum()),
        'null_and_unknown_category_export_check':True,'ignored_field_invariance_check':True,
        'custom_pickle_classes':False,'new_serving_dependencies':False,
        'validation_scope':'The selected PROCEDURE is evaluated in outer CV, the fulltrain artifact is not independently tested.',
        'deployment_performed':False,'development_test_rows_scored':0})
    print('RF challenger packaged, exact existing-reader equivalence verified. Complete v2 goal met:',goal_met)


if __name__=='__main__':
    with joblib.parallel_config(backend='threading'):
        run()
