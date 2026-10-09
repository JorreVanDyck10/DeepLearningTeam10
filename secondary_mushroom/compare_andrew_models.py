"""Reproduce Andrew's candidates and audit their saved artifacts on one split.

Run from the repository root: python secondary_mushroom/compare_andrew_models.py
The existing model artifacts and source notebook are never overwritten.
"""
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import os
import re
from time import perf_counter
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import binomtest, norm
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve, precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, PolynomialFeatures, StandardScaler
import xgboost
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "SolutionAndrew/MushroomDataset"
OUT = ROOT / "secondary_mushroom/comparison_andrew"
SEED = 42
NAMES = {
    "majority": "Meerderheidsreferentie",
    "random_forest_500": "Random Forest 500",
    "gradient_boosting": "Gradient Boosting",
    "xgboost": "XGBoost",
    "logistic_polynomial": "Logistic Regression + poly",
    "logistic_artifact_config": "Logistic Regression v1-config",
}


def clean_measurements(frame):
    cleaned = frame.copy()
    cols = ["stem-height", "stem-width"]
    cleaned[cols] = cleaned[cols].replace(0, np.nan)
    return cleaned


def fingerprint(path):
    content = path.read_bytes()
    result = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    if path.suffix in {".csv", ".ipynb"}:
        result["sha256_lf"] = hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()
    return result


def json_safe(value):
    """Represent non-finite estimator defaults as null in standards-compliant JSON."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build_candidates(X, saved_lr):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    def preprocessor(polynomial=False):
        numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
        if polynomial:
            numeric_steps += [("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
                              ("scaler", StandardScaler())]
        return ColumnTransformer([
            ("num", Pipeline(numeric_steps), numeric),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
        ])
    classifiers = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "random_forest_500": RandomForestClassifier(n_estimators=500, max_depth=20,
            min_samples_split=3, min_samples_leaf=1, max_features="sqrt",
            random_state=SEED, class_weight="balanced", n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=350, learning_rate=.08,
            max_depth=8, min_samples_split=4, min_samples_leaf=2, subsample=1,
            random_state=SEED),
        "xgboost": XGBClassifier(n_estimators=500, learning_rate=.05, max_depth=5,
            min_child_weight=2, subsample=.8, colsample_bytree=.8,
            objective="binary:logistic", eval_metric="logloss", random_state=SEED, n_jobs=-1),
        "logistic_polynomial": LogisticRegression(C=1., class_weight="balanced",
            max_iter=3000, random_state=SEED),
    }
    candidates = {key: Pipeline([
        ("clean_measurements", FunctionTransformer(clean_measurements)),
        ("preprocessor", preprocessor(key == "logistic_polynomial")),
        ("classifier", classifier),
    ]) for key, classifier in classifiers.items()}
    # The saved v1 has different preprocessing/settings; clone resets all fitted state.
    copied = clone(saved_lr)
    candidates["logistic_artifact_config"] = Pipeline([
        ("clean_measurements", FunctionTransformer(clean_measurements)),
        *copied.steps,
    ])
    return candidates


def scores(y, predicted, probabilities):
    cm = confusion_matrix(y, predicted, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "precision_poisonous": float(precision_score(y, predicted, zero_division=0)),
        "recall_poisonous": float(recall_score(y, predicted, zero_division=0)),
        "f1_poisonous": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "average_precision": float(average_precision_score(y, probabilities)),
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    }


def wilson(successes, total):
    z = norm.ppf(.975)
    p = successes / total
    den = 1 + z*z/total
    center = (p + z*z/(2*total)) / den
    radius = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / den
    return [float(center-radius), float(center+radius)]


def warnings_summary(caught):
    return dict(Counter(f"{type(w.message).__name__}: {str(w.message)}" for w in caught))


def classify(model, X):
    raw = np.asarray(model.predict(X))
    predicted = np.where(raw == "p", 1, 0) if raw.dtype.kind in "OUS" else raw.astype(int)
    classes = list(model.classes_)
    positive = classes.index("p") if "p" in classes else classes.index(1)
    return predicted, model.predict_proba(X)[:, positive]


def extract_notebook_scores():
    notebook = json.loads((SOURCE / "mushroom.ipynb").read_text(encoding="utf-8"))
    found = {}
    for cell in notebook["cells"]:
        code = "".join(cell.get("source", []))
        texts = ["".join(o.get("text", [])) for o in cell.get("outputs", [])]
        text = "".join(texts)
        key = None
        for needle, candidate in [("y_pred_lr_poly", "logistic_polynomial"),
                                  ("y_pred_xgb", "xgboost"), ("y_pred_gb", "gradient_boosting")]:
            if needle in code and "classification_report" in code: key = candidate
        if key is None and "accuracy_score(y_test, y_pred)" in code: key = "random_forest_500"
        if key:
            match = re.search(r"Accuracy:\s*([\d.]+)", text, re.I)
            row = re.search(r"^\s*(?:p|Poisonous)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$", text, re.M)
            if match and row:
                found[key] = {"accuracy_rounded": float(match.group(1)),
                    "precision_poisonous_rounded": float(row.group(1)),
                    "recall_poisonous_rounded": float(row.group(2)),
                    "f1_poisonous_rounded": float(row.group(3)), "poisonous_support": int(row.group(4))}
    return found


def generate_plots(table, y_test, predictions, probabilities, importance):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    main = table[table.model != "majority"]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    x = np.arange(len(main)); width = .24
    for offset, metric, label, color in [(-1, "accuracy", "Accuracy", "#4b8062"),
        (0, "f1_poisonous", "F1 giftige klasse", "#315e91"),
        (1, "recall_poisonous", "Recall giftige klasse", "#b85c4a")]:
        bars = ax.bar(x+offset*width, main[metric], width, label=label, color=color)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xticks(x, [NAMES[k].replace("Logistic Regression", "Logistic\nRegression") for k in main.model])
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score op bestaande testset")
    ax.set_title("Andrew's modellen — dezelfde 900 testrecords")
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(.5, 1.02))
    fig.tight_layout(); fig.savefig(OUT/"metrics_comparison.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7))
    for ax, key in zip(axes.flat, table.model):
        cm = confusion_matrix(y_test, predictions[key], labels=[0, 1])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=int((y_test==0).sum()))
        for r in range(2):
            for c in range(2):ax.text(c,r,str(cm[r,c]),ha="center",va="center",color="white" if cm[r,c]>300 else "black",fontsize=13)
        ax.set_xticks([0,1],["e", "p"]); ax.set_yticks([0,1],["e", "p"])
        ax.set_xlabel("Voorspeld"); ax.set_ylabel("Werkelijk")
        ax.set_title(NAMES[key],fontsize=10)
    fig.suptitle("Confusion matrices — links onder: giftig als eetbaar", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT/"confusion_matrices.png",dpi=180);plt.close(fig)

    fig, axes = plt.subplots(1,2,figsize=(11,4.8))
    for key in main.model:
        fpr,tpr,_=roc_curve(y_test,probabilities[key]); axes[0].plot(fpr,tpr,label=NAMES[key])
        precision,recall,_=precision_recall_curve(y_test,probabilities[key]);axes[1].plot(recall,precision,label=NAMES[key])
    axes[0].plot([0,1],[0,1],"--",color="gray");axes[0].set(xlabel="False positive rate",ylabel="Recall giftige klasse",title="ROC-curves")
    axes[1].axhline(y_test.mean(),linestyle="--",color="gray");axes[1].set(xlabel="Recall giftige klasse",ylabel="Precision giftige klasse",title="Precision-recall-curves")
    axes[1].legend(fontsize=8,loc="lower left")
    fig.tight_layout();fig.savefig(OUT/"roc_pr_curves.png",dpi=180);plt.close(fig)

    fig, ax=plt.subplots(figsize=(9,5.2))
    ranked=importance.sort_values("f1_decrease")
    ax.barh(ranked.feature,ranked.f1_decrease,xerr=ranked["std"],color="#315e91")
    ax.axvline(0,color="gray",linewidth=.8)
    ax.set(xlabel="Daling F1 bij permuteren (gemiddelde van 8 herhalingen)",title="Verklarende featureanalyse op één CV-validatiefold")
    fig.tight_layout();fig.savefig(OUT/"permutation_importance.png",dpi=180);plt.close(fig)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(SOURCE/"mushroom_project_dataset.csv")
    assert data.shape==(5000,13) and data["class"].isin(["e","p"]).all()
    X=data.drop(columns="class"); y=data["class"].map({"e":0,"p":1})
    train_ids,test_ids=train_test_split(np.arange(len(data)),test_size=.18,stratify=y,random_state=SEED)
    assert not set(train_ids)&set(test_ids)
    X_train,y_train=X.iloc[train_ids],y.iloc[train_ids]
    X_test,y_test=X.iloc[test_ids],y.iloc[test_ids]
    cv=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED).split(X_train,y_train))
    split=pd.DataFrame({"source_row":np.arange(len(data)),"class":data["class"],"split":"train"})
    split.loc[test_ids,"split"]="test";split["cv_validation_fold"]=pd.Series([pd.NA]*len(data),dtype="Int64")
    for i,(_,val) in enumerate(cv,1):split.loc[train_ids[val],"cv_validation_fold"]=i
    split.to_csv(OUT/"split_manifest.csv",index=False)

    artifact_paths={"gradient_boosting":"gradient_boosting_mushroom_v1.pkl",
                    "logistic_artifact_config":"logistic_regression_mushroom_v1.pkl"}
    artifacts={key:joblib.load(SOURCE/name) for key,name in artifact_paths.items()}
    candidates=build_candidates(X_train,artifacts["logistic_artifact_config"])
    params={};folds=[];cv_summary=[]; warning_log={}
    # Complete validation-based selection before computing any comparison test score.
    for key,prototype in candidates.items():
        print("Cross-validation:",key,flush=True)
        params[key]=prototype.named_steps["classifier"].get_params()
        start=perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for number,(tr,val) in enumerate(cv,1):
                model=clone(prototype);t=perf_counter();model.fit(X_train.iloc[tr],y_train.iloc[tr])
                fit_seconds=perf_counter()-t
                predicted,probability=classify(model,X_train.iloc[val])
                folds.append({"model":key,"fold":number,"fit_seconds":fit_seconds,
                              **scores(y_train.iloc[val],predicted,probability)})
        warning_log[key]=warnings_summary(caught)
        rows=[r for r in folds if r["model"]==key]
        cv_summary.append({"model":key,"cv_f1_mean":float(np.mean([r["f1_poisonous"] for r in rows])),
            "cv_f1_std":float(np.std([r["f1_poisonous"] for r in rows],ddof=1)),
            "cv_recall_mean":float(np.mean([r["recall_poisonous"] for r in rows])),
            "cv_accuracy_mean":float(np.mean([r["accuracy"] for r in rows])),
            "cv_seconds":perf_counter()-start})
    validation=pd.DataFrame(cv_summary).sort_values(["cv_f1_mean","cv_recall_mean"],ascending=False)
    selected=str(validation[validation.model!="majority"].iloc[0].model)
    print("Selected on CV F1 (recall tie-break):",selected,flush=True)

    test_rows=[];predictions={};probabilities={}
    export=pd.DataFrame({"source_row":test_ids,"true_class":data.iloc[test_ids]["class"].to_numpy()})
    for key,prototype in candidates.items():
        print("Full fit and test audit:",key,flush=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model=clone(prototype);start=perf_counter();model.fit(X_train,y_train);fit_seconds=perf_counter()-start
            train_pred,train_prob=classify(model,X_train)
            pred,prob=classify(model,X_test)
        predictions[key]=pred;probabilities[key]=prob
        warning_log[key+"_full_fit"]=warnings_summary(caught)
        timings=[]
        for _ in range(7):
            start=perf_counter();model.predict_proba(X_test);timings.append(perf_counter()-start)
        buffer=io.BytesIO();joblib.dump(model,buffer,compress=3)
        test_rows.append({"model":key,**scores(y_test,pred,prob),
            "train_accuracy":float(accuracy_score(y_train,train_pred)),
            "train_f1_poisonous":float(f1_score(y_train,train_pred,zero_division=0)),
            "fit_seconds":fit_seconds,"predict_900_ms":float(np.median(timings)*1000),
            "serialized_bytes_compress3":buffer.tell(),
            "accuracy_ci95":wilson(int((pred==y_test.to_numpy()).sum()),len(y_test)),
            "recall_ci95":wilson(int(((pred==1)&(y_test.to_numpy()==1)).sum()),int((y_test==1).sum()))})
        export[key+"_prediction"]=np.where(pred==1,"p","e");export[key+"_probability_p"]=prob
    table=pd.DataFrame(test_rows).merge(validation,on="model",validate="one_to_one")
    table.to_csv(OUT/"model_comparison.csv",index=False)
    pd.DataFrame(folds).to_csv(OUT/"cv_folds.csv",index=False)
    export.to_csv(OUT/"test_predictions.csv",index=False)

    artifact_audit={}
    for key,model in artifacts.items():
        pred,prob=classify(model,clean_measurements(X_test))
        steps=model.named_steps["preprocessor"].transformers_
        artifact_audit[key]={"path":str((SOURCE/artifact_paths[key]).relative_to(ROOT)),
            **fingerprint(SOURCE/artifact_paths[key]),
            "classes":model.classes_.tolist(),"classifier_parameters":model.named_steps["classifier"].get_params(),
            "preprocessing_steps":{name:[type(step).__name__ for _,step in t.steps] if hasattr(t,"steps") else [type(t).__name__]
                                   for name,t,_ in steps if name!="remainder"},
            "test_audit":scores(y_test,pred,prob),
            "matches_refitted_labels":bool(np.array_equal(pred,predictions[key])),
            "maximum_probability_difference":float(np.max(np.abs(prob-probabilities[key]))),
            "training_membership_verified":False}

    first_train,first_val=cv[0]
    interpretation=clone(candidates[selected]).fit(X_train.iloc[first_train],y_train.iloc[first_train])
    importance=permutation_importance(interpretation,X_train.iloc[first_val],y_train.iloc[first_val],
        scoring="f1",n_repeats=8,random_state=SEED,n_jobs=1)
    important=pd.DataFrame({"feature":X.columns,"f1_decrease":importance.importances_mean,"std":importance.importances_std})
    important.to_csv(OUT/"permutation_importance.csv",index=False)

    errors=X_test.copy();errors.insert(0,"source_row",test_ids)
    errors["true_class"]=data.iloc[test_ids]["class"].to_numpy()
    errors["predicted_class"]=np.where(predictions[selected]==1,"p","e")
    errors["probability_p"]=probabilities[selected]
    errors["missing_features_after_cleaning"]=clean_measurements(X_test).isna().sum(axis=1)
    errors[errors.true_class!=errors.predicted_class].to_csv(OUT/"selected_model_errors.csv",index=False)
    poisonous=errors[errors.true_class=="p"]
    subgroup=[]
    for level in [0,1,2,3,4,5,6,7,8,9,10,11,12]:
        group=poisonous[poisonous.missing_features_after_cleaning==level]
        if len(group):subgroup.append({"missing_features":level,"poisonous_count":len(group),
            "false_negatives":int((group.predicted_class=="e").sum()),
            "recall":float((group.predicted_class=="p").mean())})
    pairwise=[]
    for other in candidates:
        if other in [selected,"majority"]:continue
        a=predictions[selected]==y_test.to_numpy();b=predictions[other]==y_test.to_numpy()
        only_a=int((a&~b).sum());only_b=int((~a&b).sum());discordant=only_a+only_b
        pairwise.append({"selected":selected,"other":other,"only_selected_correct":only_a,
            "only_other_correct":only_b,"accuracy_difference":float(a.mean()-b.mean()),
            "mcnemar_exact_p":float(binomtest(only_a,discordant,.5).pvalue) if discordant else 1.})
    dataset_audit={"rows":len(data),"features":list(X.columns),"classes":data["class"].value_counts().to_dict(),
        "train_rows":len(train_ids),"test_rows":len(test_ids),
        "test_class_counts":data.iloc[test_ids]["class"].value_counts().to_dict(),
        "exact_duplicate_rows":int(data.duplicated().sum()),
        "overlapping_exact_feature_rows_train_test":int(pd.util.hash_pandas_object(X_train,index=False).isin(pd.util.hash_pandas_object(X_test,index=False)).sum()),
        "zero_stem_rows":int(((X['stem-height']==0)|(X['stem-width']==0)).sum()),
        "missing_percent_after_cleaning":(clean_measurements(X).isna().mean()*100).to_dict(),
        **fingerprint(SOURCE/'mushroom_project_dataset.csv')}
    summary={"parameter_source_commit":"17c4a61","scope":"Andrew's 5000-row project dataset; development-test audit",
        "selection_rule":"Mean five-fold training-CV F1 for p; mean recall as tie-break; no test metrics in selection",
        "selected_model":selected,"selection_is_provisional":True,
        "split":{"test_size":.18,"random_state":SEED,"stratify":True,"cv_folds":5},
        "environment":{"python":platform.python_version(),"platform":platform.platform(),"logical_cpus":os.cpu_count(),
            "sklearn":sklearn.__version__,"pandas":pd.__version__,"numpy":np.__version__,
            "scipy":scipy.__version__,"joblib":joblib.__version__,"xgboost":xgboost.__version__},
        "dataset":dataset_audit,"parameters":params,"notebook_recorded_scores":extract_notebook_scores(),
        "model_comparison":table.to_dict(orient="records"),"artifact_audit":artifact_audit,
        "warnings":warning_log,"pairwise_accuracy_audit":pairwise,
        "selected_false_negative_missingness":subgroup,
        "source_notebook":fingerprint(SOURCE/'mushroom.ipynb')}
    summary = json_safe(summary)
    (OUT/"comparison_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    generate_plots(table,y_test,predictions,probabilities,important)
    print(table[['model','cv_f1_mean','accuracy','recall_poisonous','f1_poisonous','fn']].to_string(index=False),flush=True)
    return summary


if __name__ == "__main__":
    run()
