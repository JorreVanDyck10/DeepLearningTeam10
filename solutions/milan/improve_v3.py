"""Feature selection, the lecture's optimized boosters, and weighted voting.

Only solutions/milan receives output. The 900 development-test rows are excluded.
This is continued exploratory development, not an independent final test.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
OPTIONAL = HERE / '.generated/boosting_packages'
if OPTIONAL.exists():
    sys.path.insert(0, str(OPTIONAL))
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from mushroom_alternative import development_data, FEATURES, NUMERIC, SOURCE_HASH, scores, write_json
from improve import f1_threshold

OUT = HERE / 'results_v3'
FPR_BUDGET = .14
WEIGHTS = [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.),
           (.5, .5, 0.), (.5, 0., .5), (0., .5, .5), (1/3, 1/3, 1/3)]


class InputFeatures(TransformerMixin, BaseEstimator):
    """Keep the 11-field public input; optionally ignore fields inside the model.

    Category vocabularies are learned separately within each training fold.
    LightGBM gets explicit pandas categorical types; integer codes are never
    silently treated as continuous measurements.
    """
    def __init__(self, kind='rf', drop_noise=False, drop_spore=False):
        self.kind = kind
        self.drop_noise = drop_noise
        self.drop_spore = drop_spore

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.columns_ = [c for c in FEATURES
            if not (self.drop_noise and c == 'jumbled_noise_1')
            and not (self.drop_spore and c == 'spore_print_color')]
        self.categories_ = {c: sorted(X[c].dropna().astype(str).unique())
                            for c in self.columns_ if c not in NUMERIC}
        return self

    def transform(self, X):
        result = X[self.columns_].copy()
        if self.kind == 'catboost':
            for c in self.categories_:
                result[c] = result[c].fillna('__missing__').astype(str)
        elif self.kind == 'lightgbm':
            for c, values in self.categories_.items():
                result[c] = pd.Categorical(result[c], categories=values)
        return result


def experiments():
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    prep = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), make_column_selector(dtype_include=np.number)),
        ('cat', Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]),
            make_column_selector(dtype_exclude=np.number))], sparse_threshold=0)
    rf = Pipeline([('inputs', InputFeatures()), ('preprocessor', prep),
        ('classifier', RandomForestClassifier(n_estimators=300, max_depth=20,
            min_samples_split=4, max_features=None, class_weight='balanced', random_state=42, n_jobs=1))])
    lgb = Pipeline([('inputs', InputFeatures(kind='lightgbm')),
        ('classifier', LGBMClassifier(n_estimators=600, learning_rate=.04,
            reg_lambda=3., class_weight='balanced', random_state=42,
            n_jobs=1, verbosity=-1, deterministic=True, force_col_wise=True))])
    cat = Pipeline([('inputs', InputFeatures(kind='catboost')),
        ('classifier', CatBoostClassifier(iterations=800, learning_rate=.04,
            random_seed=42, thread_count=1, verbose=False, allow_writing_files=False,
            auto_class_weights='Balanced', max_ctr_complexity=2,
            cat_features=[c for c in FEATURES if c not in NUMERIC and c != 'jumbled_noise_1']))])
    # CatBoost's categorical column names must match each selected feature subset.
    cat_grids = []
    for drop in [False, True]:
        cats = [c for c in FEATURES if c not in NUMERIC and not (drop and c == 'jumbled_noise_1')]
        cat_grids.append({'inputs__drop_noise': [drop], 'classifier__cat_features': [cats],
            'classifier__depth': [4, 6], 'classifier__l2_leaf_reg': [3, 10]})
    return {
        'rf_selection': (rf, {'inputs__drop_noise': [False, True],
            'inputs__drop_spore': [False, True], 'classifier__min_samples_leaf': [1, 3]}),
        'lightgbm': (lgb, {'inputs__drop_noise': [False, True],
            'classifier__num_leaves': [15, 31], 'classifier__min_child_samples': [20, 60]}),
        'catboost': (cat, cat_grids)}


def serializable(params):
    return {k: repr(v) if isinstance(v, BaseEstimator) else v for k, v in params.items()}


def threshold_at_fpr(y, probability):
    fpr, tpr, thresholds = roc_curve(y, probability, drop_intermediate=False)
    eligible = [i for i in range(len(thresholds))
                if np.isfinite(thresholds[i]) and fpr[i] <= FPR_BUDGET]
    if not eligible:
        return float(np.nextafter(np.max(probability), np.inf))
    i = max(eligible, key=lambda i: (tpr[i], -fpr[i], thresholds[i]))
    return float(thresholds[i])


def metrics_summary():
    table = pd.read_csv(OUT/'outer_fold_metrics.csv')
    rows = []
    for name, part in table.groupby('model', sort=False):
        row = {'model': name, 'folds': len(part)}
        for m in ['accuracy', 'macro_f1', 'recall_p', 'precision_p', 'specificity_e',
                  'balanced_accuracy', 'roc_auc', 'average_precision']:
            row[m+'_mean'] = float(part[m].mean())
            row[m+'_sd'] = float(part[m].std(ddof=1))
        for m in ['tn', 'fp', 'fn', 'tp']:
            row[m+'_total'] = int(part[m].sum())
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(OUT/'summary.csv', index=False)
    return result


def run():
    import catboost, lightgbm
    X, y, ids, folds = development_data()
    declared = experiments()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = HERE/'.generated/v3'; cache.mkdir(parents=True, exist_ok=True)
    protocol = {'source_sha256_lf': SOURCE_HASH, 'source_rows': 4100,
        'excluded_development_test_rows': 900, 'outer_folds': 5, 'inner_folds': 3,
        'inner_seeds': '9410 + outer fold',
        'families': list(declared), 'grids': {n: [serializable(p) for p in ParameterGrid(g)] for n, (_,g) in declared.items()},
        'search_metric': 'inner CV macro F1', 'fpr_budget': FPR_BUDGET,
        'voting_weights': WEIGHTS, 'threshold_grid_macro_f1': '.20-.70 step .01, inner OOF only',
        'reference': 'v2 RF300 + tuned policy, macro F1 .769125 / FN515 / FP356',
        'desired_improvement': 'Higher macro F1 and fewer FN, without exceeding the previous FP count; report all failures honestly.',
        'course': '4 - Models, slides 36, 38, 42, 49; model-quality ROC and data-augmentation preparation',
        'limitation': 'Third exploratory study on previously used development data; no independent final-test claim.',
        'code_sha256_lf': hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        'dependencies_sha256_lf': {n:hashlib.sha256((HERE/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            for n in ['improve.py','mushroom_alternative.py']},
        'environment': {'python':platform.python_version(),'sklearn':sklearn.__version__,
            'numpy':np.__version__,'pandas':pd.__version__,'catboost':catboost.__version__,'lightgbm':lightgbm.__version__}}
    path = OUT/'protocol.json'
    # JSON roundtrip gives a stable comparison for tuple/list representation.
    protocol = json.loads(json.dumps(protocol))
    if path.exists():
        assert json.loads(path.read_text()) == protocol, 'Protocol changed: start another study folder.'
    else:
        write_json(path, protocol)
    mp = OUT/'outer_fold_metrics.csv'; pp=OUT/'outer_oof_predictions.csv'; sp=OUT/'selections.json'
    metrics = pd.read_csv(mp).to_dict('records') if mp.exists() else []
    predictions = pd.read_csv(pp,float_precision='round_trip').to_dict('records') if pp.exists() else []
    selections = json.loads(sp.read_text()) if sp.exists() else []
    if not metrics:
        ref = pd.read_csv(HERE/'results_v2/outer_oof_predictions.csv',float_precision='round_trip')
        for name in ['rf300_reference_default','rf_compatible_f1_threshold']:
            part=ref[ref.model==name].copy()
            assert part.source_row.is_unique and set(part.source_row)==set(ids)
            predictions.extend(part.to_dict('records'))
            for f, group in part.groupby('fold'):
                metrics.append({'model':name,'fold':int(f),'threshold':float(group.threshold.iloc[0]),
                    **scores(group.true_class,group.probability_p,prediction=group.prediction)})

    def save():
        pd.DataFrame(metrics).to_csv(mp,index=False)
        pd.DataFrame(predictions).to_csv(pp,index=False)
        write_json(sp,selections)
        metrics_summary()

    def record(name, number, val, probability, threshold, labels=None):
        labels=(probability>=threshold).astype(int) if labels is None else labels
        metrics.append({'model':name,'fold':number,'threshold':threshold,
            **scores(y.iloc[val],probability,threshold,prediction=labels)})
        predictions.extend({'source_row':int(ids[i]),'fold':number,'true_class':int(y.iloc[i]),
            'model':name,'probability_p':float(p),'threshold':threshold,'prediction':int(v)}
            for i,p,v in zip(val,probability,labels))

    for number,(tr,val) in enumerate(folds,1):
        inner = StratifiedKFold(3,shuffle=True,random_state=9410+number)
        probability_inner=[]; probability_outer=[]
        for family,(prototype,grid) in declared.items():
            key=f'{family}_fold_{number}'
            checkpoint=cache/(key+'_scores.npz')
            done=any(s['family']==family and s['fold']==number for s in selections)
            if done:
                with np.load(checkpoint) as saved:
                    probability_inner.append(saved['inner']); probability_outer.append(saved['outer'])
                print('Reuse',key,flush=True)
                continue
            started=perf_counter()
            print(f'{family}, outer fold {number}/5: 8 configs x 3 inner folds',flush=True)
            search=GridSearchCV(clone(prototype),grid,scoring='f1_macro',cv=inner,
                n_jobs=2,pre_dispatch=2,error_score='raise',return_train_score=True)
            search.fit(X.iloc[tr],y.iloc[tr])
            pd.DataFrame(search.cv_results_).to_csv(OUT/(key+'_search.csv'),index=False)
            model=search.best_estimator_
            outer=model.predict_proba(X.iloc[val])[:,1]
            inside=cross_val_predict(clone(model),X.iloc[tr],y.iloc[tr],cv=inner,
                n_jobs=2,pre_dispatch=2,method='predict_proba')[:,1]
            tf=f1_threshold(y.iloc[tr].to_numpy(),inside)
            tb=threshold_at_fpr(y.iloc[tr].to_numpy(),inside)
            record(family+'_default',number,val,outer,.5,model.predict(X.iloc[val]).astype(int).ravel())
            record(family+'_f1',number,val,outer,tf)
            record(family+'_fpr_budget',number,val,outer,tb)
            selections.append({'family':family,'fold':number,'parameters':serializable(search.best_params_),
                'inner_cv_macro_f1':float(search.best_score_),'f1_threshold':tf,'fpr_threshold':tb,
                'seconds':perf_counter()-started})
            np.savez_compressed(checkpoint,inner=inside,outer=outer)
            joblib.dump(model,cache/(key+'.joblib'),compress=3)
            probability_inner.append(inside); probability_outer.append(outer)
            save()
            print(json.dumps(selections[-1]),flush=True)
        if any(s['family']=='voting' and s['fold']==number for s in selections):
            continue
        matrix=np.column_stack(probability_inner)
        options=[]
        for weights in WEIGHTS:
            combined=matrix@np.asarray(weights)
            threshold=f1_threshold(y.iloc[tr].to_numpy(),combined)
            score=float(f1_score(y.iloc[tr],combined>=threshold,average='macro',zero_division=0))
            options.append((score,weights,threshold))
        # Stable fixed order is the tie breaker; no outer scores choose weights.
        chosen=max(enumerate(options),key=lambda item:(item[1][0],-item[0]))[1]
        score,weights,threshold=chosen
        inside=matrix@np.asarray(weights)
        outer=np.column_stack(probability_outer)@np.asarray(weights)
        budget_threshold=threshold_at_fpr(y.iloc[tr].to_numpy(),inside)
        record('voting_default',number,val,outer,.5)
        record('voting_f1',number,val,outer,threshold)
        record('voting_fpr_budget',number,val,outer,budget_threshold)
        selections.append({'family':'voting','fold':number,'weights':weights,
            'inner_oof_macro_f1':score,'f1_threshold':threshold,'fpr_threshold':budget_threshold})
        save()
        print(metrics_summary()[['model','folds','macro_f1_mean','recall_p_mean','fn_total','fp_total']].to_string(index=False),flush=True)
    assert len(selections)==20
    assert pd.read_csv(pp).groupby('model').size().eq(4100).all()
    write_json(OUT/'run.json',{'finished_at_utc':datetime.now(timezone.utc).isoformat(),
        'complete':True,'outer_family_evaluations':20,'grid_fits':360,
        'development_test_rows_scored':0,'deployment_performed':False})


if __name__=='__main__':
    with joblib.parallel_config(backend='threading'):
        run()
