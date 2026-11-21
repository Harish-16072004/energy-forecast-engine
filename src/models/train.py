import json
import os
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from src.features.engineering import FEATURE_COLUMNS, build_features, merge_meter_and_weather

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://energy_user:energy_pass@localhost:5433/energy_db"
)
ARTIFACTS_DIR = Path("artifacts")


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def load_training_frame(engine) -> pd.DataFrame:
    meters_sql = text("SELECT site_id, ts, kwh FROM meter_readings ORDER BY site_id, ts")
    weather_sql = text(
        "SELECT site_id, ts, temp_c, humidity, solar_rad, hdd, cdd FROM weather_hourly ORDER BY site_id, ts"
    )
    meters_df = pd.read_sql(meters_sql, engine)
    weather_df = pd.read_sql(weather_sql, engine)

    merged = merge_meter_and_weather(meters_df, weather_df)
    featured = build_features(merged, drop_na=True)
    return featured.sort_values("ts").reset_index(drop=True)


def chronological_split(df: pd.DataFrame):
    unique_ts = np.sort(df["ts"].unique())
    n = len(unique_ts)
    train_end_ts = unique_ts[int(n * 0.70)]
    val_end_ts = unique_ts[int(n * 0.85)]

    train_df = df[df["ts"] < train_end_ts].copy()
    val_df = df[(df["ts"] >= train_end_ts) & (df["ts"] < val_end_ts)].copy()
    test_df = df[df["ts"] >= val_end_ts].copy()
    return train_df, val_df, test_df


def train_and_evaluate() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    engine = create_engine(DATABASE_URL)

    df = load_training_frame(engine)
    train_df, val_df, test_df = chronological_split(df)

    # 1. Seasonal naive baseline (same hour 7 days prior)
    baseline_val_pred = val_df["lag_168h"].to_numpy()
    baseline_test_pred = test_df["lag_168h"].to_numpy()

    metrics = {
        "baseline_val_wape": wape(val_df["kwh"].to_numpy(), baseline_val_pred),
        "baseline_test_wape": wape(test_df["kwh"].to_numpy(), baseline_test_pred),
        "baseline_test_mae": mae(test_df["kwh"].to_numpy(), baseline_test_pred),
    }

    # 2. Train p10, p50, p90 LightGBM quantile regressors
    quantiles = {
        "p10": 0.10,
        "p50": 0.50,
        "p90": 0.90,
    }
    test_preds = {}

    for name, alpha in quantiles.items():
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample_for_bin=50_000,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(
            train_df[FEATURE_COLUMNS],
            train_df["kwh"],
            eval_set=[(val_df[FEATURE_COLUMNS], val_df["kwh"])],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        joblib.dump(model, ARTIFACTS_DIR / f"lgb_{name}.joblib")
        test_preds[name] = model.predict(test_df[FEATURE_COLUMNS])

    # Ensure non-crossing quantiles (p10 <= p50 <= p90)
    p10 = np.minimum(test_preds["p10"], test_preds["p50"])
    p90 = np.maximum(test_preds["p90"], test_preds["p50"])
    p50 = test_preds["p50"]

    y_test = test_df["kwh"].to_numpy()
    in_interval = (y_test >= p10) & (y_test <= p90)

    metrics["lgb_p50_test_wape"] = wape(y_test, p50)
    metrics["lgb_p50_test_mae"] = mae(y_test, p50)
    metrics["wape_improvement_vs_baseline_pct"] = round(
        (metrics["baseline_test_wape"] - metrics["lgb_p50_test_wape"])
        / metrics["baseline_test_wape"]
        * 100,
        2,
    )
    metrics["interval_80_coverage_p10_p90"] = round(float(np.mean(in_interval)), 4)

    metrics_path = ARTIFACTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print("Training complete. Metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train_and_evaluate()