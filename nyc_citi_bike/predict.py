"""Inference needs only the saved tree and metadata, never the trip CSV."""
from datetime import date
from pathlib import Path
import pandas as pd

DEFAULT_MODEL = Path(__file__).resolve().parent / "models/baseline_hourly_tree.joblib"


def predict_day(artifact, selected_date: date, hour: int):
    frame = pd.DataFrame({"hour": range(24), "weekday": [selected_date.weekday()] * 24})
    values = artifact["model"].predict(frame)
    metadata = artifact["metadata"]
    trained_month = metadata["source_month"]
    warning = "Eenvoudige baseline zonder weer, feestdagen of jaarlijkse seizoenen."
    if selected_date.strftime("%Y%m") != trained_month:
        warning += " Deze datum valt buiten de beschikbare dataperiode; de voorspelling gebruikt alleen het uur en de weekdag uit die periode."
    return {
        "date": selected_date.isoformat(), "hour": hour, "timezone": "America/New_York",
        "predicted_ride_starts": round(float(values[hour]), 1),
        "daily_predictions": [{"hour": h, "predicted_ride_starts": round(float(v), 1)} for h, v in enumerate(values)],
        "training_period": f"{metadata['train_start']} – {metadata['train_end']}",
        "source_month": trained_month, "test_mae": metadata["results"]["decision_tree"]["mae"],
        "warning": warning,
    }
