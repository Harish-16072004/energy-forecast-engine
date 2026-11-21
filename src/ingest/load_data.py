
import os
import time
from pathlib import Path
from datetime import date

import pandas as pd
import requests
from sqlalchemy import create_engine, text


# ============================================================
# CONFIGURATION
# ============================================================

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://energy_user:energy_pass@localhost:5433/energy_db"
)

# Resolve project paths independently of the working directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

MAX_SITES = 50
BASE_TEMP_C = 18.3
CHUNK_SIZE = 10_000


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_input_files():
    """Validate the required CSV datasets."""

    required_files = [
        DATA_DIR / "metadata.csv",
        DATA_DIR / "electricity_cleaned.csv",
    ]

    for file_path in required_files:
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Required dataset not found: {file_path}"
            )

    print("Input datasets validated successfully.")


# ============================================================
# ELECTRICITY METER DATA INGESTION
# ============================================================

def load_bdg2_meters(engine, max_sites=MAX_SITES):

    meta_path = DATA_DIR / "metadata.csv"
    elec_path = DATA_DIR / "electricity_cleaned.csv"

    print("\nLoading building metadata...")

    meta = pd.read_csv(meta_path)

    required_metadata_columns = {
        "building_id",
        "lat",
        "lng",
    }

    missing_columns = required_metadata_columns - set(meta.columns)

    if missing_columns:
        raise ValueError(
            f"Missing metadata columns: {missing_columns}"
        )

    meta["lat"] = pd.to_numeric(meta["lat"], errors="coerce")
    meta["lng"] = pd.to_numeric(meta["lng"], errors="coerce")

    meta = meta.dropna(subset=["lat", "lng"]).copy()

    print(f"Valid buildings with coordinates: {len(meta):,}")

    print("\nLoading electricity data...")

    elec = pd.read_csv(elec_path)

    if elec.empty or len(elec.columns) < 2:
        raise ValueError(
            "Electricity dataset must contain a timestamp "
            "column and at least one meter column."
        )

    ts_col = elec.columns[0]

    elec[ts_col] = pd.to_datetime(
        elec[ts_col],
        utc=True,
        errors="coerce"
    )

    elec = elec.dropna(subset=[ts_col]).copy()

    metadata_sites = set(meta["building_id"])

    available_sites = [
        col for col in elec.columns[1:]
        if col in metadata_sites
    ]

    if not available_sites:
        raise ValueError(
            "No matching building IDs found between metadata "
            "and electricity datasets."
        )

    # Select buildings with the highest meter-data coverage
    coverage = (
        elec[available_sites]
        .notna()
        .sum()
        .sort_values(ascending=False)
    )

    selected_sites = list(
        coverage.head(max_sites).index
    )

    if not selected_sites:
        raise ValueError(
            "No buildings available for ingestion."
        )

    print(
        f"\nSelected {len(selected_sites)} buildings "
        "based on meter coverage."
    )

    # Convert wide-format data into long-format data
    melted = elec[
        [ts_col] + selected_sites
    ].melt(
        id_vars=[ts_col],
        value_vars=selected_sites,
        var_name="site_id",
        value_name="kwh"
    )

    melted = melted.rename(
        columns={ts_col: "ts"}
    )

    melted["kwh"] = pd.to_numeric(
        melted["kwh"],
        errors="coerce"
    )

    melted = melted.dropna(
        subset=["site_id", "ts", "kwh"]
    ).copy()

    if melted.empty:
        raise ValueError(
            "No valid electricity readings found."
        )

    # Prepare database upsert
    upsert_sql = text("""
        INSERT INTO meter_readings (
            site_id,
            ts,
            kwh
        )
        VALUES (
            :site_id,
            :ts,
            :kwh
        )
        ON CONFLICT (site_id, ts)
        DO UPDATE SET
            kwh = EXCLUDED.kwh;
    """)

    print("\nInserting electricity readings...")

    total_rows = len(melted)

    with engine.begin() as conn:

        for start in range(0, total_rows, CHUNK_SIZE):

            batch = melted.iloc[
                start:start + CHUNK_SIZE
            ].to_dict(orient="records")

            conn.execute(
                upsert_sql,
                batch
            )

            print(
                f"Inserted {min(start + CHUNK_SIZE, total_rows):,}"
                f" / {total_rows:,} rows"
            )

    sites_df = (
        meta[
            meta["building_id"].isin(selected_sites)
        ][["building_id", "lat", "lng"]]
        .drop_duplicates(subset=["building_id"])
        .rename(
            columns={"building_id": "site_id"}
        )
    )

    min_ts = melted["ts"].min()
    max_ts = melted["ts"].max()

    print("\nElectricity ingestion completed.")
    print(f"Total meter readings: {total_rows:,}")
    print(f"Selected buildings: {len(sites_df)}")
    print(f"Start timestamp: {min_ts}")
    print(f"End timestamp: {max_ts}")

    return sites_df, min_ts, max_ts


# ============================================================
# OPEN-METEO WEATHER EXTRACTION
# ============================================================

def fetch_open_meteo_weather(
    lat,
    lng,
    start_date,
    end_date
):

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": float(lat),
        "longitude": float(lng),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "shortwave_radiation"
        ),
        "timezone": "UTC",
    }

    response = requests.get(
        url,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    result = response.json()

    if "hourly" not in result:
        raise ValueError(
            f"Weather API returned an unexpected response: {result}"
        )

    hourly = result["hourly"]

    df = pd.DataFrame({
        "ts": pd.to_datetime(
            hourly["time"],
            utc=True
        ),
        "temp_c": hourly["temperature_2m"],
        "humidity": hourly["relative_humidity_2m"],
        "solar_rad": hourly["shortwave_radiation"],
    })

    for column in ["temp_c", "humidity", "solar_rad"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Heating Degree Days and Cooling Degree Days
    df["hdd"] = (
        BASE_TEMP_C - df["temp_c"]
    ).clip(lower=0.0)

    df["cdd"] = (
        df["temp_c"] - BASE_TEMP_C
    ).clip(lower=0.0)

    return df


# ============================================================
# WEATHER DATA INGESTION
# ============================================================

def load_weather(engine, sites_df, min_ts, max_ts):

    if sites_df.empty:
        raise ValueError(
            "No buildings available for weather ingestion."
        )

    if pd.isna(min_ts) or pd.isna(max_ts):
        raise ValueError(
            "Invalid electricity timestamp range."
        )

    start_date = min_ts.date()
    end_date = max_ts.date()

    upsert_sql = text("""
        INSERT INTO weather_hourly (
            site_id,
            ts,
            temp_c,
            humidity,
            solar_rad,
            hdd,
            cdd
        )
        VALUES (
            :site_id,
            :ts,
            :temp_c,
            :humidity,
            :solar_rad,
            :hdd,
            :cdd
        )
        ON CONFLICT (site_id, ts)
        DO UPDATE SET
            temp_c = EXCLUDED.temp_c,
            humidity = EXCLUDED.humidity,
            solar_rad = EXCLUDED.solar_rad,
            hdd = EXCLUDED.hdd,
            cdd = EXCLUDED.cdd;
    """)

    total_weather_rows = 0

    print("\nStarting weather ingestion...")

    for index, row in sites_df.iterrows():

        site_id = row["site_id"]

        print(
            f"\n[{index + 1}/{len(sites_df)}] "
            f"Fetching weather for building {site_id}"
        )

        weather_df = fetch_open_meteo_weather(
            row["lat"],
            row["lng"],
            start_date,
            end_date
        )

        weather_df["site_id"] = site_id

        records = weather_df[
            [
                "site_id",
                "ts",
                "temp_c",
                "humidity",
                "solar_rad",
                "hdd",
                "cdd"
            ]
        ].to_dict(orient="records")

        if records:

            with engine.begin() as conn:
                conn.execute(
                    upsert_sql,
                    records
                )

        total_weather_rows += len(records)

        print(
            f"Weather rows inserted: {len(records):,}"
        )

        # Respect the free Open-Meteo API
        time.sleep(0.25)

    print("\nWeather ingestion completed.")
    print(
        f"Total weather rows: {total_weather_rows:,}"
    )


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    print("=" * 65)
    print("ENERGY FORECAST ENGINE - DATA INGESTION")
    print("=" * 65)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")

    # Step 1: Validate datasets
    validate_input_files()

    # Step 2: Initialize database connection
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True
    )

    try:

        # Step 3: Load electricity meter readings
        sites_df, min_ts, max_ts = load_bdg2_meters(
            engine,
            max_sites=MAX_SITES
        )

        # Step 4: Load corresponding weather data
        load_weather(
            engine,
            sites_df,
            min_ts,
            max_ts
        )

        print("\n" + "=" * 65)
        print("DATA INGESTION COMPLETED SUCCESSFULLY")
        print("=" * 65)

    finally:

        engine.dispose()


if __name__ == "__main__":
    main()
