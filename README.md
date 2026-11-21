# Commercial Energy Load Forecasting, Tariff Optimization & Carbon Engine

An end-to-end energy data engineering, machine learning, and tariff optimization pipeline designed for commercial buildings. The system ingests multi-site smart-meter interval data and historical hourly weather into PostgreSQL, engineers leakage-free temporal and thermodynamic features, trains LightGBM quantile regression models (`p10`, `p50`, `p90`), simulates Time-of-Use (ToU) utility tariffs and Scope 2 carbon emissions, serves low-latency forecasts via FastAPI with Redis caching, and monitors production drift using Evidently AI.

---

## Table of Contents
1. [Key Results & Benchmarks](#key-results--benchmarks)
2. [System Architecture](#system-architecture)
3. [Repository Structure](#repository-structure)
4. [Tech Stack](#tech-stack)
5. [Local Setup & Quickstart](#local-setup--quickstart)
6. [Docker Compose Full-Stack Deployment](#docker-compose-full-stack-deployment)
7. [API Reference & Example Payloads](#api-reference--example-payloads)
8. [Testing & MLOps Drift Monitoring](#testing--mlops-drift-monitoring)

---

## Key Results & Benchmarks

Evaluated using a strict chronological split (70% train / 15% validation / 15% test) across **608,150** training rows from Building Data Genome Project 2 commercial sites paired with Open-Meteo hourly weather:

| Model / Method | Test WAPE | Test MAE | 80% Interval Coverage (`[p10, p90]`) | Notes |
| --- | --- | --- | --- | --- |
| **Seasonal Naive Baseline (`lag_168h`)** | `0.2091` (20.91%) | `28.28 kWh` | — | Same hour 7 days prior |
| **LightGBM Quantile (`p10`, `p50`, `p90`)** | **`0.1257` (12.57%)** | **`17.00 kWh`** | **`78.86%`** | **39.88% relative WAPE reduction** |

### Production Monitoring Snapshot (`src/monitoring/drift.py`)
* **Logged Evaluation Window:** 48 scored hourly intervals (`Hog_assembly_Jasmine`)
* **Rolling Live WAPE:** `0.1510` (15.10%) vs. `0.1800` (18.00%) alert threshold (`retrain_recommended: false`)

---

## System Architecture

```text
+-----------------------------------+     +-----------------------------------+
|  Building Data Genome 2 (CSV)     |     |  Open-Meteo Historical Weather    |
|  (Hourly kWh Interval Meters)     |     |  (Temp, Humidity, Solar Rad)      |
+-----------------+-----------------+     +-----------------+-----------------+
                  |                                         |
                  +--------------------+--------------------+
                                       |
                                       v
                     +-----------------------------------+
                     |   Ingestion & ETL Pipeline        |
                     |   (src/ingest/load_data.py)       |
                     +-----------------+-----------------+
                                       |
                                       v
                     +-----------------------------------+
                     |   PostgreSQL (Time-Series Store)  |
                     |   - meter_readings                |
                     |   - weather_hourly                |
                     |   - forecast_logs                 |
                     +-----------------+-----------------+
                                       |
                                       v
                     +-----------------------------------+
                     |   Shared Feature Engineering      |
                     |   (src/features/engineering.py)   |
                     |   - HDD & CDD (18.3°C base)       |
                     |   - Cyclical hour/day/month       |
                     |   - Lags (24h, 48h, 168h) &       |
                     |     shifted rolling statistics    |
                     +--------+-----------------+--------+
                              |                 |
                (Training)    |                 |   (Online Serving)
                              v                 v
        +-----------------------------+   +-----------------------------+
        | LightGBM Quantile Trainer   |   | FastAPI Inference Service   |
        | (src/models/train.py)       |   | (src/api/main.py)           |
        | - p10, p50, p90 regressors  |   | - Redis Cache (TTL: 900s)   |
        | - Chronological backtest    |   | - Non-crossing quantiles    |
        +-----------------------------+   | - Logs to forecast_logs     |
                                          +--------------+--------------+
                                                         |
                           +-----------------------------+-----------------------------+
                           |                                                           |
                           v                                                           v
        +-----------------------------------+                       +-----------------------------------+
        | Tariff & Scope 2 Carbon Engine    |                       | Evidently AI Drift & WAPE Monitor |
        | (src/simulation/tariff_carbon.py) |                       | (src/monitoring/drift.py)         |
        | - Flat vs. Time-of-Use (ToU)      |                       | - Rolling WAPE threshold check    |
        | - Peak demand ($/kW) & kg CO2     |                       | - HTML drift report export        |
        +-----------------------------------+                       +-----------------------------------+
```

---

## Repository Structure

```text
energy-forecast-engine/
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI running pytest
├── artifacts/                      # Trained LightGBM .joblib files & metrics.json (gitignored)
├── data/                           # Raw BDG2 CSV files (gitignored)
├── db/
│   └── schema.sql                  # PostgreSQL DDL for meters, weather, and forecast logs
├── reports/                        # Evidently AI HTML/JSON drift reports (gitignored)
├── src/
│   ├── api/
│   │   └── main.py                 # FastAPI app (/health, /forecast/{site_id}, /simulate-tariff)
│   ├── features/
│   │   └── engineering.py          # Leakage-free feature pipeline shared by train & inference
│   ├── ingest/
│   │   └── load_data.py            # BDG2 meter + Open-Meteo weather loader
│   ├── models/
│   │   └── train.py                # Chronological splitter, baseline, and LightGBM quantile trainer
│   ├── monitoring/
│   │   └── drift.py                # Live WAPE evaluation & Evidently AI DataDriftPreset report
│   └── simulation/
│       └── tariff_carbon.py        # Commercial tariff optimizer & Scope 2 emissions calculator
├── tests/
│   └── test_features.py            # Unit tests for future-leakage prevention & tariff calculations
├── .gitignore
├── docker-compose.yml              # Multi-container setup (PostgreSQL, Redis, FastAPI)
├── Dockerfile                      # Container definition for the FastAPI service
└── requirements.txt                # Pinned Python dependencies
```

---

## Tech Stack

* **Data & ML:** Python 3.12, pandas, NumPy, scikit-learn, LightGBM (Quantile Regression)
* **Storage & Caching:** PostgreSQL 16, SQLAlchemy, psycopg, Redis 7
* **API & Validation:** FastAPI, Uvicorn, Pydantic v2
* **MLOps & Quality:** Evidently AI, pytest, GitHub Actions, Docker & Docker Compose

---

## Local Setup & Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/energy-forecast-engine.git
cd energy-forecast-engine
```

### 2. Start PostgreSQL & Redis
```bash
docker compose up -d postgres redis
```

Create a `.env` file in the root directory:
```env
DATABASE_URL=postgresql://energy_user:energy_pass@localhost:5432/energy_db
REDIS_URL=redis://localhost:6379/0
```

### 3. Create a Virtual Environment & Install Dependencies
```bash
python -m venv .venv

# Windows (PowerShell):
.venv\Scripts\activate

# macOS / Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download Data & Run Ingestion
Place `metadata.csv` and `electricity_cleaned.csv` from the [Building Data Genome Project 2](https://github.com/buds-lab/building-data-genome-project-2) into `data/`, then run:
```bash
python -m src.ingest.load_data
```

### 5. Train Quantile Models
```bash
python -m src.models.train
```
This saves `lgb_p10.joblib`, `lgb_p50.joblib`, `lgb_p90.joblib`, and `metrics.json` inside `artifacts/`.

### 6. Start the FastAPI Server
```bash
uvicorn src.api.main:app --reload --port 8000
```
Interactive API documentation is available at `http://localhost:8000/docs`.

---

## Docker Compose Full-Stack Deployment

To build and run PostgreSQL, Redis, and the FastAPI service together in containers:

```bash
docker compose up --build -d
```

Verify running containers:
```bash
docker compose ps
```

---

## API Reference & Example Payloads

### 1. Health Check
`GET /health`
```json
{
  "status": "ok",
  "timestamp": "2026-11-21T02:00:00.000000+00:00"
}
```

### 2. Site Load Forecast (with Redis Caching)
`GET /forecast/{site_id}?horizon_rows=48`

Example request:
```bash
curl "http://localhost:8000/forecast/Hog_assembly_Jasmine?horizon_rows=48"
```

Example response:
```json
{
  "site_id": "Hog_assembly_Jasmine",
  "count": 48,
  "cache_hit": false,
  "forecasts": [
    {
      "ts": "2017-12-30T00:00:00+00:00",
      "p10_kwh": 45.354,
      "p50_kwh": 55.982,
      "p90_kwh": 62.572,
      "actual_kwh": 42.139
    }
  ]
}
```
*(Subsequent requests within 15 minutes return `"cache_hit": true` directly from Redis.)*

### 3. Tariff & Scope 2 Carbon Simulation
`POST /simulate-tariff`

Example request body:
```json
{
  "hourly_kwh": [45.0, 44.0, 43.5, 43.0, 44.5, 50.0, 62.0, 78.0, 85.0, 88.0, 90.0, 91.0, 89.0, 87.0, 86.0, 84.0, 80.0, 70.0, 60.0, 54.0, 50.0, 48.0, 46.0, 45.0],
  "hours_of_day": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
  "grid_kg_co2_per_kwh": 0.40
}
```

---

## Testing & MLOps Drift Monitoring

### Run Unit Tests
Verifies that lag and rolling features contain zero future data leakage and validates tariff/emissions math:
```bash
pytest -v
```

### Run Drift & Live WAPE Check
Evaluates logged predictions against actuals in `forecast_logs` and writes an HTML report to `reports/drift_report.html`:
```bash
python -m src.monitoring.drift
```