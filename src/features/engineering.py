import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "lag_24h",
    "lag_48h",
    "lag_168h",
    "rolling_mean_24h",
    "rolling_mean_168h",
    "rolling_std_168h",
    "temp_c",
    "humidity",
    "solar_rad",
    "hdd",
    "cdd",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
]


def merge_meter_and_weather(meters_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    meters = meters_df.copy()
    weather = weather_df.copy()

    meters["ts"] = pd.to_datetime(meters["ts"], utc=True)
    weather["ts"] = pd.to_datetime(weather["ts"], utc=True)

    merged = pd.merge(
        meters.sort_values(["site_id", "ts"]),
        weather.sort_values(["site_id", "ts"]),
        on=["site_id", "ts"],
        how="inner",
    )
    return merged.sort_values(["site_id", "ts"]).reset_index(drop=True)


def build_features(df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
    out = df.sort_values(["site_id", "ts"]).copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True)

    # Ensure HDD/CDD exist if caller passes raw temp_c
    if "hdd" not in out.columns and "temp_c" in out.columns:
        out["hdd"] = (18.3 - out["temp_c"]).clip(lower=0.0)
    if "cdd" not in out.columns and "temp_c" in out.columns:
        out["cdd"] = (out["temp_c"] - 18.3).clip(lower=0.0)

    grouped_kwh = out.groupby("site_id")["kwh"]

    # Lags strictly in the past relative to row timestamp t
    out["lag_24h"] = grouped_kwh.shift(24)
    out["lag_48h"] = grouped_kwh.shift(48)
    out["lag_168h"] = grouped_kwh.shift(168)

    # Rolling stats shifted by 24h so no data within the 24h target window leaks
    out["rolling_mean_24h"] = grouped_kwh.transform(
        lambda s: s.shift(24).rolling(window=24, min_periods=12).mean()
    )
    out["rolling_mean_168h"] = grouped_kwh.transform(
        lambda s: s.shift(24).rolling(window=168, min_periods=72).mean()
    )
    out["rolling_std_168h"] = grouped_kwh.transform(
        lambda s: s.shift(24).rolling(window=168, min_periods=72).std()
    )

    # Cyclical calendar encodings
    hour = out["ts"].dt.hour
    dow = out["ts"].dt.dayofweek
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    out["is_weekend"] = (dow >= 5).astype(int)

    if drop_na:
        out = out.dropna(subset=FEATURE_COLUMNS + ["kwh"]).reset_index(drop=True)

    return out