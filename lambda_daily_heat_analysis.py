"""
Daily Heat Analysis and Data Loss Tool
======================================
Purpose: Isolated calculation of heat metrics for a single day to analyze
data completeness and compare RAW vs FILLED performance side-by-side.

Constants:
    - Expected DPs per hour: 4,800  (80 DPs/min)
    - Expected DPs per day: 115,200

Outputs two scores:
    1. Raw Score (calculated exactly with gaps left as NaNs/Zeros)
    2. Filled Score (calculated after 100% forward-filling the hourly brackets)
"""

import json
import os
import sys
import logging
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

class Config:
    PG_CONNECTION_STRING = os.environ.get(
        "PG_CONNECTION_STRING",
        "postgresql://vanixuser:vanix%244567%23@10.98.36.21:6432/vanixdb"
    )

    
    EXPECTED_DPS_PER_HOUR = 4800
    EXPECTED_DPS_PER_DAY = 115200

    RES_THRESHOLD = 0.35
    FEED_THRESHOLD = 0.15
    STRESS_TEMP = 40.5
    SCORE_W_SPIKE = 15
    SCORE_W_PERSIST = 40

    # Hours for calculating the Daily Anchor (11 PM - 4 AM)
    ANCHOR_HOURS = [23, 0, 1, 2, 3]
    # Night hours typically used to find the max spike (same range)
    NIGHT_HOURS = [23, 0, 1, 2, 3]
    SOLAR_HOURS = [11, 12, 13, 14, 15, 16]

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger("DailyHeatAnalyzer")

logger = setup_logging()

# ============================================================================
# 2. DATABASE MANAGER
# ============================================================================

class DatabaseManager:
    def __init__(self, connection_string: str):
        self.conn_str = connection_string

    @staticmethod
    def ist_to_utc(dt: datetime) -> datetime:
        return dt - timedelta(hours=5, minutes=30)

    @staticmethod
    def utc_to_ist(dt: datetime) -> datetime:
        return dt + timedelta(hours=5, minutes=30)

    def fetch_single_day(self, node_id: int, target_date: date) -> pd.DataFrame:
        """Fetch exactly one day from 00:00 to 23:59 IST."""
        # Convert the bounds of the IST day to UTC
        start_ist = datetime.combine(target_date, datetime.min.time())
        end_ist = datetime.combine(target_date, datetime.max.time().replace(microsecond=0))
        
        start_utc = self.ist_to_utc(start_ist)
        end_utc = self.ist_to_utc(end_ist)

        logger.info(f"DB: Querying node {node_id} for {target_date} IST")
        
        query = """
            SELECT
                node_id,
                timestamp               AS timestamp_utc,
                timestamp + INTERVAL '5 hours 30 minutes' AS timestamp_ist,
                x, y, z,
                temperature_value
            FROM device_data
            WHERE node_id::text = %(node_id)s::text
              AND timestamp >= %(start_utc)s
              AND timestamp <= %(end_utc)s
              AND x IS NOT NULL AND y IS NOT NULL AND z IS NOT NULL
            ORDER BY timestamp ASC
        """
        
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module='pandas.io.sql')
            with psycopg2.connect(self.conn_str) as conn:
                df = pd.read_sql_query(
                    query, conn,
                    params={"node_id": str(node_id), "start_utc": start_utc, "end_utc": end_utc}
                )
        return df

# ============================================================================
# 3. ANALYSIS ENGINE
# ============================================================================

class DailyAnalysisEngine:
    
    @staticmethod
    def extract_features_and_activity(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df["mag"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
        df["vedba"] = np.abs(df["mag"] - df["mag"].rolling(window=50, center=True).mean()).fillna(0)
        
        conditions = [
            df["vedba"] > Config.RES_THRESHOLD, 
            df["vedba"] > Config.FEED_THRESHOLD
        ]
        df["activity_class"] = np.select(conditions, ["RES", "FEED"], default="STANDING")
        return df

    @staticmethod
    def resample_to_hourly(df: pd.DataFrame, target_date: date) -> pd.DataFrame:
        """Convert raw per-second data into hourly buckets spanning precisely 24 hours."""
        if df.empty:
            full_index = pd.date_range(
                start=pd.Timestamp(datetime.combine(target_date, datetime.min.time())),
                end=pd.Timestamp(datetime.combine(target_date, datetime.max.time().replace(second=0))),
                freq="h"
            )
            empty_df = pd.DataFrame(index=full_index, columns=["temp_mean", "res_ratio"])
            return empty_df

        idx = df.set_index("timestamp_ist")
        hourly = idx["temperature_value"].resample("h").mean().to_frame(name="temp_mean")
        hourly["res_ratio"] = (
            idx["activity_class"]
            .apply(lambda x: 1 if x == "RES" else 0)
            .resample("h").mean()
        )
        
        # Enforce exact 24-hour index for the specific day
        full_index = pd.date_range(
            start=pd.Timestamp(datetime.combine(target_date, datetime.min.time())),
            end=pd.Timestamp(datetime.combine(target_date, datetime.max.time().replace(second=0))),
            freq="h"
        )
        hourly = hourly.reindex(full_index)
        return hourly

    @staticmethod
    def calculate_data_loss(df: pd.DataFrame) -> Tuple[float, dict]:
        """Calculate daily and per-hour data loss based on new DP counts."""
        if df.empty:
            return 100.0, {str(h): {"loss_pct": 100.0, "count": 0} for h in range(24)}

        # Ensure index is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp_ist"]):
            df["timestamp_ist"] = pd.to_datetime(df["timestamp_ist"])
            
        total_dps = len(df)
        daily_loss_pct = round(100.0 * (1 - (total_dps / Config.EXPECTED_DPS_PER_DAY)), 2)
        
        hourly_counts = df.groupby(df["timestamp_ist"].dt.hour).size().to_dict()
        hourly_data_dict = {}
        for h in range(24):
            count = int(hourly_counts.get(h, 0))
            loss_pct = round(100.0 * (1 - (count / Config.EXPECTED_DPS_PER_HOUR)), 2)
            # NO CLAMPING (allows negative loss for data gain)
            hourly_data_dict[str(h)] = {"loss_pct": loss_pct, "count": count}
            
        return daily_loss_pct, hourly_data_dict

    @staticmethod
    def calculate_daily_anchor(hourly_df: pd.DataFrame) -> float:
        """Find the minimum temperature between 11 PM and 4 AM (Isolated for this specific day)."""
        anchor_data = hourly_df[hourly_df.index.hour.isin(Config.ANCHOR_HOURS)]
        temps = anchor_data["temp_mean"].dropna()
        if temps.empty:
            return 0.0
        return float(temps.min())

    @staticmethod
    def calculate_metrics(hourly_df: pd.DataFrame, daily_anchor: float) -> dict:
        """Calculate spike, persistence, and score given an hourly dataframe and anchor."""
        night_data = hourly_df[hourly_df.index.hour.isin(Config.NIGHT_HOURS)]
        night_temps = night_data["temp_mean"].dropna()
        night_max = night_temps.max() if not night_temps.empty else np.nan
        
        spike = max(0.0, night_max - daily_anchor) if not np.isnan(night_max) else 0.0
        
        persistence = hourly_df["res_ratio"].rolling(3, min_periods=1).mean().max()
        persistence = 0.0 if np.isnan(persistence) else float(persistence)
        
        score = (spike * Config.SCORE_W_SPIKE) + (persistence * Config.SCORE_W_PERSIST)
        
        return {
            "daily_anchor_C": round(daily_anchor, 4),
            "night_spike_C": round(spike, 4),
            "persistence_pct": round(persistence * 100, 2),
            "score": round(score, 4)
        }

# ============================================================================
# 4. LAMBDA ORCHESTRATOR
# ============================================================================

class DailyAnalysisHandler:
    def __init__(self):
        self.db = DatabaseManager(Config.PG_CONNECTION_STRING)
        self.engine = DailyAnalysisEngine()

    def process(self, event: dict) -> dict:
        node_id = int(event.get("node_id", 0))
        target_date_str = event.get("target_date", "")

        if not node_id or not target_date_str:
            return self._response(400, {"error": "Missing node_id or target_date"})

        target_date = date.fromisoformat(target_date_str)
        
        # 1. Fetch raw data
        raw_df = self.db.fetch_single_day(node_id, target_date)
        raw_df["timestamp_ist"] = pd.to_datetime(raw_df["timestamp_ist"])
        
        # 2. Calculate data loss metrics
        daily_loss_pct, hourly_loss_dict = self.engine.calculate_data_loss(raw_df)

        if raw_df.empty:
            return self._response(200, {
                "node_id": node_id,
                "target_date": target_date_str,
                "data_loss": {
                    "daily_pct": 100.0,
                    "hourly_stats": {str(h): {"loss_pct": 100.0, "count": 0} for h in range(24)}
                },
                "results": {
                    "raw": None,
                    "filled": None
                },
                "message": "No data found for this node on the target date."
            })

        # 3. Process raw data
        processed_df = self.engine.extract_features_and_activity(raw_df)
        
        # 4. Create RAW hourly buckets (with np.nan for missing gaps)
        hourly_raw = self.engine.resample_to_hourly(processed_df, target_date)
        
        # 5. Create FILLED hourly buckets (100% forward fill)
        hourly_filled = hourly_raw.copy().ffill()

        # 6. Calculate Anchors
        anchor_raw = self.engine.calculate_daily_anchor(hourly_raw)
        anchor_filled = self.engine.calculate_daily_anchor(hourly_filled)

        # 7. Calculate Metrics
        metrics_raw = self.engine.calculate_metrics(hourly_raw, anchor_raw)
        metrics_filled = self.engine.calculate_metrics(hourly_filled, anchor_filled)

        res = {
            "node_id": node_id,
            "target_date": target_date_str,
            "data_loss": {
                "daily_pct": daily_loss_pct,
                "expected_dps_day": Config.EXPECTED_DPS_PER_DAY,
                "expected_dps_hour": Config.EXPECTED_DPS_PER_HOUR,
                "hourly_stats": hourly_loss_dict
            },
            "results": {
                "raw": metrics_raw,
                "filled": metrics_filled
            }
        }
        
        return self._response(200, res)

    def _response(self, status: int, body: dict) -> dict:
        return {
            "statusCode": status,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(body, indent=2, default=str)
        }

# ============================================================================
# ENTRY POINT
# ============================================================================

def handler(event, context):
    logger.info(f"DAILY ANALYSIS START: {json.dumps(event)}")
    try:
        app = DailyAnalysisHandler()
        return app.process(event)
    except Exception as e:
        logger.error(f"FATAL: {e}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }

if __name__ == "__main__":
    test_event = {
        "node_id": 124,
        "target_date": "2026-02-26"
    }
    
    print("\n" + "="*60)
    print("RUNNING ISOLATED DAILY HEAT ANALYSIS")
    print("="*60)
    
    resp = handler(test_event, None)
    
    if resp["statusCode"] == 200:
        body = json.loads(resp["body"])
        
        print(f"\nNode: {body['node_id']} | Date: {body['target_date']}")
        print("\n--- DATA LOSS ---")
        print(f"Daily Loss:  {body['data_loss']['daily_pct']}%")
        
        print("\nHourly Loss & Counts:")
        stats_dict = body['data_loss']['hourly_stats']
        # Format the 24 hours into 3 columns of 8 for neatness
        for i in range(0, 24, 8):
            row = [f"H{h:02d}: {stats_dict[str(h)]['loss_pct']:>6.2f}% ({stats_dict[str(h)]['count']})" for h in range(i, i+8)]
            print("  |  ".join(row))

        if body["results"]["raw"]:
            print("\n--- PERFORMANCE METRICS ---")
            print(f"{'Metric':<20} | {'RAW (Unfilled)':<18} | {'FILLED (ffill)':<18}")
            print("-" * 62)
            
            r_raw = body["results"]["raw"]
            r_fil = body["results"]["filled"]
            
            print(f"{'Daily Anchor':<20} | {r_raw['daily_anchor_C']:>6.2f} C            | {r_fil['daily_anchor_C']:>6.2f} C")
            print(f"{'Night Spike':<20} | {r_raw['night_spike_C']:>6.2f} C            | {r_fil['night_spike_C']:>6.2f} C")
            print(f"{'Persistence':<20} | {r_raw['persistence_pct']:>6.2f} %            | {r_fil['persistence_pct']:>6.2f} %")
            print(f"{'Final Score':<20} | {r_raw['score']:>8.2f}              | {r_fil['score']:>8.2f}")
    else:
        print("\nError:")
        print(resp)
        
    print("\n" + "="*60)
