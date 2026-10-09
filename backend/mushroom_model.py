"""Export Andrew's notebook model, including its deterministic measurement cleaning."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "SolutionAndrew/MushroomDataset"
DEFAULT_MODEL = SOURCE / "models/random_forest.joblib"
METRICS = SOURCE / "metrics/random_forest.json"
MODEL_ID = "andrew_random_forest_200"
FEATURES = [
    "cap-diameter", "stem-height", "stem-width", "spore-print-color",
    "gill-color", "habitat", "season", "ring-type", "cap-shape",
    "stem-surface", "jumbled_noise_0", "jumbled_noise_1",
]


def clean_measurements(frame):
    """Match notebook cell 8 for training and every subsequent prediction."""
    cleaned = frame.copy()
    columns = ["stem-height", "stem-width"]
    cleaned[columns] = cleaned[columns].replace(0, np.nan)
    return cleaned


def export_model():
    source_file = SOURCE / "mushroom_project_dataset.csv"
    data = pd.read_csv(source_file)
    X, y = data.drop(columns="class"), data["class"]
    if list(X.columns) != FEATURES or not y.isin(["e", "p"]).all():
        raise ValueError("Andrew's dataset does not match the expected features/classes.")

    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    model = Pipeline([
        ("clean_measurements", FunctionTransformer(clean_measurements)),
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced", n_jobs=2,
        )),
    ])
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, list(model.classes_).index("p")]
    cm = confusion_matrix(y_test, predicted, labels=["e", "p"])
    metrics = {
        "model_id": MODEL_ID,
        "model_name": "Andrew — Random Forest (200 bomen)",
        "source_notebook": "SolutionAndrew/MushroomDataset/mushroom.ipynb",
        "dataset": "SolutionAndrew/MushroomDataset/mushroom_project_dataset.csv",
        "dataset_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
        "dataset_rows": len(data), "train_rows": len(X_train), "test_rows": len(X_test),
        "features": FEATURES,
        "parameters": {"n_estimators": 200, "random_state": 42, "class_weight": "balanced"},
        "test_metrics": {
            "accuracy": float(accuracy_score(y_test, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predicted)),
            "precision_poisonous": float(precision_score(y_test, predicted, pos_label="p")),
            "recall_poisonous": float(recall_score(y_test, predicted, pos_label="p")),
            "f1_poisonous": float(f1_score(y_test, predicted, pos_label="p")),
            "roc_auc_poisonous": float(roc_auc_score(y_test == "p", probabilities)),
            "poisonous_predicted_edible": int(cm[1, 0]),
            "confusion_matrix_labels": ["e", "p"],
            "confusion_matrix": cm.tolist(),
        },
    }
    example = json.loads(X_test.iloc[[0]].to_json(orient="records"))[0]
    DEFAULT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, DEFAULT_MODEL, compress=3)
    METRICS.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (SOURCE / "example_mushroom.json").write_text(
        json.dumps(example, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(metrics["test_metrics"], indent=2))
    print(f"Saved {DEFAULT_MODEL.relative_to(ROOT)} ({DEFAULT_MODEL.stat().st_size:,} bytes)")


if __name__ == "__main__":
    # Keep pickle's function reference stable when run with python -m.
    from backend.mushroom_model import export_model as export
    export()
