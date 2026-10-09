"""Chunked preparation, train-only hypothesis test and chronological baseline evaluation."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import permutation_test
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent


def features(timestamps):
    dates = pd.DatetimeIndex(timestamps)
    return pd.DataFrame({"hour": dates.hour, "weekday": dates.dayofweek})


def prepare(csv_path, month="202501"):
    """Count rows with valid start times in the selected complete calendar month."""
    start = pd.Timestamp(f"{month[:4]}-{month[4:]}-01")
    end = start + pd.offsets.MonthBegin(1)
    counts = pd.Series(0, index=pd.date_range(start, end, freq="h", inclusive="left"), dtype="int64")
    stats = {"raw_rows": 0, "invalid_start_times": 0, "outside_month": 0}
    for chunk in pd.read_csv(csv_path, usecols=["started_at"], chunksize=200_000):
        dates = pd.to_datetime(chunk.started_at, format="mixed", errors="coerce")
        inside = dates.notna() & (dates >= start) & (dates < end)
        stats["raw_rows"] += len(chunk)
        stats["invalid_start_times"] += int(dates.isna().sum())
        stats["outside_month"] += int((dates.notna() & ~inside).sum())
        grouped = dates[inside].dt.floor("h").value_counts()
        counts = counts.add(grouped, fill_value=0).astype("int64")
    if counts.sum() == 0:
        raise ValueError("No valid ride starts in the selected month.")
    hourly = counts.rename("ride_starts").rename_axis("timestamp").reset_index()
    stats["retained_ride_starts"] = int(counts.sum())
    # Counts reflect source rows; ID deduplication and further EDA remain separate work.
    return hourly, stats


def hypothesis(train):
    """Predefined: daily share of starts at 07-09 is higher on weekdays than weekends."""
    data = train.copy()
    data["day"] = data.timestamp.dt.normalize()
    data["morning"] = data.ride_starts.where(data.timestamp.dt.hour.isin([7, 8]), 0)
    daily = data.groupby("day")[["ride_starts", "morning"]].sum()
    daily = daily[daily.ride_starts > 0]
    shares = daily.morning / daily.ride_starts
    weekdays = shares[daily.index.dayofweek < 5].to_numpy()
    weekends = shares[daily.index.dayofweek >= 5].to_numpy()
    result = permutation_test(
        (weekdays, weekends), lambda a, b: a.mean() - b.mean(),
        alternative="greater", n_resamples=9999, random_state=42,
    )
    return {
        "statement": "Daily share of ride starts between 07:00 and 09:00 is higher on weekdays than weekends.",
        "data": "Training days only; one observation per day, not per ride.",
        "weekday_days": len(weekdays), "weekend_days": len(weekends),
        "weekday_mean_share": float(weekdays.mean()), "weekend_mean_share": float(weekends.mean()),
        "effect_percentage_points": float(result.statistic * 100), "p_value": float(result.pvalue),
        "limitations": "Small single-month sample; holiday, weather and temporal dependence can confound this comparison. No causal conclusion.",
    }


def train_baseline(csv_path=ROOT / "data/raw/citibike_trips.csv", month="202501"):
    csv_path = Path(csv_path)
    manifest_path = csv_path.with_suffix(".manifest.json")
    if not manifest_path.is_file() or month not in json.loads(manifest_path.read_text())["months"]:
        raise ValueError("A download manifest confirming the selected complete month is required.")
    hourly, cleaning = prepare(csv_path, month)
    test_start = hourly.timestamp.max().normalize() - pd.Timedelta(days=6)
    train = hourly[hourly.timestamp < test_start]
    test = hourly[hourly.timestamp >= test_start]
    # State and test the hypothesis before fitting. The held-out week is not used here.
    evidence = hypothesis(train)
    print("Hypothesis:", json.dumps(evidence, indent=2), flush=True)
    x_train, x_test = features(train.timestamp), features(test.timestamp)
    y_train, y_test = train.ride_starts, test.ride_starts
    # Fixed settings: no tuning on the test week.
    model = DecisionTreeRegressor(max_depth=5, min_samples_leaf=8, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    reference = DummyRegressor(strategy="mean").fit(x_train, y_train).predict(x_test)
    # Stronger reference: training mean for the same weekday and hour.
    slot_means = pd.DataFrame({"hour": x_train.hour, "weekday": x_train.weekday, "target": y_train.to_numpy()}).groupby(["weekday", "hour"]).target.mean()
    weekly = np.array([slot_means.get((row.weekday, row.hour), y_train.mean()) for row in x_test.itertuples()])
    def scores(values):
        return {"mae": float(mean_absolute_error(y_test, values)), "rmse": float(root_mean_squared_error(y_test, values))}
    with csv_path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    metrics = {
        "model": "decision_tree_depth_5", "features": ["hour", "weekday"],
        "target": "NYC network ride starts per local hour", "timezone": "America/New_York",
        "source_sha256": digest, "source_month": month, "cleaning": cleaning,
        "train_start": str(train.timestamp.min()), "train_end": str(train.timestamp.max()),
        "test_start": str(test.timestamp.min()), "test_end": str(test.timestamp.max()),
        "train_hours": len(train), "test_hours": len(test), "hypothesis": evidence,
        "results": {"decision_tree": scores(predictions), "global_mean": scores(reference), "weekday_hour_mean": scores(weekly)},
        "limitations": "January-only baseline. No weather, holidays, station data or annual seasonality. Counts include source rows without ride-ID deduplication. Test set is reserved for this fixed baseline evaluation.",
    }
    for folder in ["data/processed", "metrics", "models"]:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    hourly.to_csv(ROOT / "data/processed/hourly_rides.csv", index=False)
    test.assign(predicted=predictions, reference_mean=reference, reference_weekday_hour=weekly).to_csv(ROOT / "metrics/baseline_test_predictions.csv", index=False)
    (ROOT / "metrics/baseline_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (ROOT.parent / "frontend/citibike_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    joblib.dump({"model": model, "metadata": metrics}, ROOT / "models/baseline_hourly_tree.joblib")
    print(json.dumps(metrics["results"], indent=2), flush=True)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/citibike_trips.csv")
    parser.add_argument("--month", default="202501")
    args = parser.parse_args()
    train_baseline(args.input, args.month)
