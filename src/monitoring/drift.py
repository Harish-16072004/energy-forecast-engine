import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from evidently import Report
from evidently.presets import DataDriftPreset
from sqlalchemy import create_engine, text

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://energy_user:energy_pass@localhost:5432/energy_db"
)
REPORTS_DIR = PROJECT_ROOT / "reports"
WAPE_ALERT_THRESHOLD = 0.18  # Flag if rolling WAPE exceeds 18%


def compute_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def run_drift_check(min_rows: int = 20) -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    logs_sql = text("""
        SELECT site_id, forecast_ts, p10, p50, p90, actual_kwh
        FROM forecast_logs
        WHERE actual_kwh IS NOT NULL
        ORDER BY forecast_ts ASC, id ASC
    """)
    with engine.connect() as conn:
        logs_df = pd.read_sql(logs_sql, conn)

    if len(logs_df) < min_rows:
        summary = {
            "status": "insufficient_logged_data",
            "rows_found": int(len(logs_df)),
            "minimum_required": min_rows,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        print(json.dumps(summary, indent=2))
        return summary

    split_idx = max(1, int(len(logs_df) * 0.5))
    cols = ["p10", "p50", "p90", "actual_kwh"]
    reference_df = logs_df.iloc[:split_idx][cols].copy()
    current_df = logs_df.iloc[split_idx:][cols].copy()

    current_wape = compute_wape(
        current_df["actual_kwh"].to_numpy(),
        current_df["p50"].to_numpy(),
    )

    eval_result = Report([DataDriftPreset()]).run(
        current_data=current_df,
        reference_data=reference_df,
    )

    report_path = REPORTS_DIR / "drift_report.html"
    eval_result.save_html(str(report_path))

    retrain_recommended = bool(current_wape > WAPE_ALERT_THRESHOLD)
    summary = {
        "status": "completed",
        "total_logged_rows": int(len(logs_df)),
        "current_wape": round(current_wape, 4),
        "wape_threshold": WAPE_ALERT_THRESHOLD,
        "retrain_recommended": retrain_recommended,
        "report_path": str(report_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (REPORTS_DIR / "drift_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_drift_check()