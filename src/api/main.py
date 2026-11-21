import json
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
import redis
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import create_engine, text

from src.features.engineering import FEATURE_COLUMNS, build_features, merge_meter_and_weather
from src.simulation.tariff_carbon import TariffPlan, recommend_best_tariff

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://energy_user:energy_pass@localhost:5432/energy_db"
)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
ARTIFACTS_DIR = Path("artifacts")
CACHE_TTL_SECONDS = 900

app = FastAPI(title="Commercial Energy Load & Tariff API", version="1.0.0")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2)


@lru_cache(maxsize=1)
def load_quantile_models():
    paths = {
        "p10": ARTIFACTS_DIR / "lgb_p10.joblib",
        "p50": ARTIFACTS_DIR / "lgb_p50.joblib",
        "p90": ARTIFACTS_DIR / "lgb_p90.joblib",
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing model artifact '{name}' at {path}. Run src/models/train.py first.")
    return {name: joblib.load(path) for name, path in paths.items()}


class TariffRequest(BaseModel):
    hourly_kwh: List[float] = Field(..., min_length=1)
    hours_of_day: List[int] = Field(..., min_length=1)
    grid_kg_co2_per_kwh: float = Field(default=0.40, ge=0.0)

    @model_validator(mode="after")
    def validate_series(self):
        if len(self.hourly_kwh) != len(self.hours_of_day):
            raise ValueError("hourly_kwh and hours_of_day must have the same length.")
        if any(h < 0 or h > 23 for h in self.hours_of_day):
            raise ValueError("All hours_of_day values must be between 0 and 23.")
        return self


@app.get("/")
def root():
    return {
        "service": "Commercial Energy Load & Tariff API",
        "docs": "/docs",
        "endpoints": ["/health", "/forecast/{site_id}", "/simulate-tariff"],
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/forecast/{site_id}")
def get_site_forecast(
    site_id: str,
    horizon_rows: int = Query(default=24, ge=1, le=168),
):
    cache_key = f"forecast:{site_id}:{horizon_rows}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            payload = json.loads(cached)
            payload["cache_hit"] = True
            return payload
    except redis.RedisError:
        pass

    lookback_limit = horizon_rows + 250
    meters_sql = text("""
        SELECT site_id, ts, kwh
        FROM meter_readings
        WHERE site_id = :site_id
        ORDER BY ts DESC
        LIMIT :limit
    """)
    weather_sql = text("""
        SELECT site_id, ts, temp_c, humidity, solar_rad, hdd, cdd
        FROM weather_hourly
        WHERE site_id = :site_id
        ORDER BY ts DESC
        LIMIT :limit
    """)

    with engine.connect() as conn:
        meters_df = pd.read_sql(meters_sql, conn, params={"site_id": site_id, "limit": lookback_limit})
        weather_df = pd.read_sql(weather_sql, conn, params={"site_id": site_id, "limit": lookback_limit})

    if meters_df.empty or weather_df.empty:
        raise HTTPException(status_code=404, detail=f"No meter/weather data found for site '{site_id}'")

    merged = merge_meter_and_weather(meters_df, weather_df)
    featured = build_features(merged, drop_na=True)
    if featured.empty:
        raise HTTPException(status_code=422, detail="Not enough historical rows to compute lag features")

    target_slice = featured.tail(horizon_rows).copy()

    try:
        models = load_quantile_models()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    raw_p10 = models["p10"].predict(target_slice[FEATURE_COLUMNS])
    raw_p50 = models["p50"].predict(target_slice[FEATURE_COLUMNS])
    raw_p90 = models["p90"].predict(target_slice[FEATURE_COLUMNS])

    p10 = np.clip(np.minimum(raw_p10, raw_p50), 0.0, None)
    p50 = np.clip(raw_p50, 0.0, None)
    p90 = np.clip(np.maximum(raw_p90, raw_p50), 0.0, None)

    forecast_items = []
    log_rows = []
    for ts_val, q10, q50, q90, actual in zip(
        target_slice["ts"], p10, p50, p90, target_slice["kwh"]
    ):
        ts_iso = pd.Timestamp(ts_val).isoformat()
        forecast_items.append({
            "ts": ts_iso,
            "p10_kwh": round(float(q10), 3),
            "p50_kwh": round(float(q50), 3),
            "p90_kwh": round(float(q90), 3),
            "actual_kwh": round(float(actual), 3),
        })
        log_rows.append({
            "site_id": site_id,
            "forecast_ts": pd.Timestamp(ts_val).to_pydatetime(),
            "p10": float(q10),
            "p50": float(q50),
            "p90": float(q90),
            "actual_kwh": float(actual),
        })

    insert_sql = text("""
        INSERT INTO forecast_logs (site_id, forecast_ts, p10, p50, p90, actual_kwh)
        VALUES (:site_id, :forecast_ts, :p10, :p50, :p90, :actual_kwh)
    """)
    with engine.begin() as conn:
        conn.execute(insert_sql, log_rows)

    response_payload = {
        "site_id": site_id,
        "count": len(forecast_items),
        "cache_hit": False,
        "forecasts": forecast_items,
    }
    try:
        redis_client.setex(cache_key, CACHE_TTL_SECONDS, json.dumps(response_payload))
    except redis.RedisError:
        pass

    return response_payload


@app.post("/simulate-tariff")
def simulate_tariff(req: TariffRequest):
    try:
        plans = [
            TariffPlan(
                name="Commercial Flat Rate",
                off_peak_rate_per_kwh=0.14,
                peak_rate_per_kwh=0.14,
                peak_hours=(),
                demand_charge_per_kw=5.0,
                fixed_monthly_fee=25.0,
            ),
            TariffPlan(
                name="Commercial Time-of-Use (ToU)",
                off_peak_rate_per_kwh=0.09,
                peak_rate_per_kwh=0.24,
                peak_hours=(13, 14, 15, 16, 17, 18, 19, 20),
                demand_charge_per_kw=12.0,
                fixed_monthly_fee=35.0,
            ),
        ]
        return recommend_best_tariff(
            hourly_kwh=req.hourly_kwh,
            hours_of_day=req.hours_of_day,
            plans=plans,
            grid_kg_co2_per_kwh=req.grid_kg_co2_per_kwh,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc