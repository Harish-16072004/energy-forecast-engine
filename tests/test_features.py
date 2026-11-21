import pandas as pd
from src.features.engineering import build_features


def _sample_series(future_spike: float = 10.0) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=200, freq="h", tz="UTC")
    kwh = [100.0] * 200
    kwh[-1] = future_spike  # Mutate the very last hour
    return pd.DataFrame({
        "site_id": ["site_1"] * 200,
        "ts": ts,
        "kwh": kwh,
        "temp_c": [22.0] * 200,
        "humidity": [55.0] * 200,
        "solar_rad": [200.0] * 200,
    })


def test_no_future_leakage():
    df_a = build_features(_sample_series(future_spike=10.0))
    df_b = build_features(_sample_series(future_spike=99999.0))

    # All rows except the final one must be identical
    pd.testing.assert_frame_equal(df_a.iloc[:-1], df_b.iloc[:-1])