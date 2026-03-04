"""
Heat Cycle Detection - Modular Benchmark Lambda Function
=========================================================
Purpose: Benchmarking and logic validation ONLY. This version is modularized for better maintainability.

Structure:
1. Configuration (Constants & Envs)
2. DatabaseManager (DB interaction logic)
3. HeatLogicEngine (Core data processing and scoring logic)
4. HeatCyclePredictor (Cycle calculation logic)
5. LambdaHandler (Orchestration and entry point)
"""

import json
import os
import sys
import logging
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from collections import deque
from typing import Dict, List, Tuple, Optional, Any

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

class Config:
    """Centralized configuration for the heat detection system."""
    PG_CONNECTION_STRING = os.environ.get(
        "PG_CONNECTION_STRING",
        "postgresql://vanixuser:vanix%244567%23@10.98.36.21:6432/vanixdb"
    )

    # Heat cycle parameters
    CYCLE_DAYS          = 21    # average bovine estrus cycle length
    CYCLE_TOLERANCE     = 3     # +/- days around expected date
    MIN_BUFFER_DAYS     = 2     # minimum days in buffer before we score
    HEAT_COOL_DOWN      = 15    # days to suppress new alerts after a confirmed heat

    # Detection algorithm parameters
    RES_THRESHOLD       = 0.35   # VeDBA -> Restlessness
    FEED_THRESHOLD      = 0.15   # VeDBA -> Feeding
    NIGHT_HOURS         = [23, 0, 1, 2, 3]
    SOLAR_HOURS         = [11, 12, 13, 14, 15, 16]
    STRESS_TEMP         = 40.5   # degC - environmental heat stress cutoff
    SCORE_W_SPIKE       = 15     # weight for night temperature spike
    SCORE_W_PERSIST     = 40     # weight for behavioral persistence
    PROESTRUS_SCORE     = 25     # score threshold to flag as Proestrus
    MIN_HEAT_SCORE      = 20     # baseline score needed to trigger a "Confirmed Heat"
    
    # Data stats
    EXPECTED_ROWS_PER_DAY = 172800  # 2 data points per second × 86400 seconds/day
    FFILL_MIN_COMPLETENESS = 0.60   # only forward-fill days with ≥60% raw data

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger("HeatBenchmark")

logger = setup_logging()

# ============================================================================
# 2. DATABASE MANAGER
# ============================================================================

class DatabaseManager:
    """Handles all database interactions (connection and data retrieval)."""
    
    def __init__(self, connection_string: str):
        self.conn_str = connection_string

    @staticmethod
    def ist_to_utc(dt: datetime) -> datetime:
        """Convert IST datetime to UTC."""
        return dt - timedelta(hours=5, minutes=30)

    @staticmethod
    def utc_to_ist(dt: datetime) -> datetime:
        """Convert UTC datetime to IST."""
        return dt + timedelta(hours=5, minutes=30)

    def fetch_node_data(self, node_id: int, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Fetch sensor readings for a node between start_date and end_date (IST).
        Returns a DataFrame with raw sensor data.
        """
        start_utc = self.ist_to_utc(datetime(start_date.year, start_date.month, start_date.day, 0, 0, 0))
        end_utc   = self.ist_to_utc(datetime(end_date.year,   end_date.month,   end_date.day,   23, 59, 59))

        logger.info(f"DB: Querying node {node_id} ({start_date} to {end_date} IST)")

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

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module='pandas.io.sql')
                with psycopg2.connect(self.conn_str) as conn:
                    df = pd.read_sql_query(
                        query, conn,
                        params={"node_id": str(node_id), "start_utc": start_utc, "end_utc": end_utc}
                    )
            logger.info(f"DB: Fetched {len(df):,} rows")
            return df
        except Exception as e:
            logger.error(f"DB: Fetch failed: {e}")
            raise

# ============================================================================
# 3. HEAT LOGIC ENGINE
# ============================================================================

class HeatLogicEngine:
    """Encapsulates the core algorithmic logic for processing data and detecting heat cycles."""

    @staticmethod
    def extract_features(df: pd.DataFrame) -> pd.DataFrame:
        """Compute Vector Dynamic Body Acceleration (VeDBA)."""
        df = df.copy()
        df["mag"]   = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
        # 50-sample window corresponds to ~25 seconds of data (at 2Hz)
        df["vedba"] = np.abs(df["mag"] - df["mag"].rolling(window=50, center=True).mean())
        return df.fillna(0)

    @staticmethod
    def predict_activity(df: pd.DataFrame) -> pd.DataFrame:
        """Classify binary activity: Restlessness (RES), Feeding (FEED), or Standing."""
        conditions = [
            df["vedba"] > Config.RES_THRESHOLD, 
            df["vedba"] > Config.FEED_THRESHOLD
        ]
        df["activity_class"] = np.select(conditions, ["RES", "FEED"], default="STANDING")
        return df

    def resample_to_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert raw per-second data into hourly buckets for easier trend analysis."""
        df = self.predict_activity(self.extract_features(df))
        idx = df.set_index("timestamp_ist")

        hourly = idx["temperature_value"].resample("h").mean().to_frame(name="temp_mean")
        hourly["res_ratio"] = (
            idx["activity_class"]
            .apply(lambda x: 1 if x == "RES" else 0)
            .resample("h").mean()
        )
        return hourly

    def check_heat_stress(self, hourly_df: pd.DataFrame) -> bool:
        """Assess if the animal is under environmental heat stress based on solar-hour temperatures."""
        solar = hourly_df[hourly_df.index.hour.isin(Config.SOLAR_HOURS)]["temp_mean"].dropna()
        return bool(solar.mean() > Config.STRESS_TEMP) if not solar.empty else False

    def score_day(self, day_hourly: pd.DataFrame, global_night_base: float) -> dict:
        """
        Calculate a heat score for a single day.
        Combines temperature spike (relative to anchor) and behavioral persistence.
        """
        night_data  = day_hourly[day_hourly.index.hour.isin(Config.NIGHT_HOURS)]
        night_temps = night_data["temp_mean"].dropna()
        night_max   = night_temps.max() if not night_temps.empty else np.nan
        
        # Spike = Today's Night Max - Rolling Window Anchor
        spike       = max(0.0, night_max - global_night_base) if not np.isnan(night_max) else 0.0

        # Persistence = Maximum 3-hour rolling average of restlessness ratio
        persistence = day_hourly["res_ratio"].rolling(3, min_periods=1).mean().max()
        persistence = 0.0 if np.isnan(persistence) else float(persistence)

        score = (spike * Config.SCORE_W_SPIKE) + (persistence * Config.SCORE_W_PERSIST)
        return {"spike": spike, "persistence": persistence, "score": score}

    def run_detection(self, 
                      all_hourly: pd.DataFrame, 
                      window_days: int, 
                      next_cycle_info: dict, 
                      daily_raw_counts: Dict[date, int],
                      filled_days: set) -> List[dict]:
        """
        Orchestrates the rolling window detection across the entire duration.
        Implements winner-take-all peak detection and cool-down logic.
        """
        window_start_str = next_cycle_info["window_start"]
        window_end_str   = next_cycle_info["window_end"]

        daily_groups = all_hourly.groupby(all_hourly.index.date)
        sorted_days  = sorted(daily_groups.groups.keys())

        buffer: deque = deque(maxlen=window_days)
        results = []
        last_alert_date = None

        for day in sorted_days:
            day_hourly = daily_groups.get_group(day)
            buffer.append((day, day_hourly))

            if len(buffer) < Config.MIN_BUFFER_DAYS:
                continue

            # Concatenate buffer to calculate global anchor for this window
            window_df = pd.concat([df for _, df in buffer])
            
            # Global Anchor: Min temperature in midnight buckets (00:00 - 03:00)
            night_mask = window_df.index.hour.isin([0, 1, 2])
            night_vals = window_df.loc[night_mask, "temp_mean"].dropna()
            global_anchor = float(night_vals.min()) if not night_vals.empty else 0.0

            env_stress = self.check_heat_stress(window_df)

            # Score all days in the current sliding window
            scored = []
            for d, d_df in buffer:
                s = self.score_day(d_df, global_anchor)
                s["date"] = d
                scored.append(s)

            # Find the peak (highest score) in the current window
            scored.sort(key=lambda x: x["score"], reverse=True)
            peak_day = scored[0]["date"]

            # Evaluate today's status
            today_score_dict = next((s for s in scored if s["date"] == day), scored[0])
            score_val = today_score_dict["score"]
            in_window = (window_start_str <= day.isoformat() <= window_end_str)
            
            status = "NORMAL"
            
            # HEAT CONFIRMATION LOGIC:
            # 1. Today is the peak of the current rolling window
            # 2. Score exceeds min threshold
            if day == peak_day and score_val >= Config.MIN_HEAT_SCORE:
                # 3. Not in cool-down from previous alert
                if last_alert_date is None or (day - last_alert_date).days >= Config.HEAT_COOL_DOWN:
                    if env_stress:
                        status = "SUPPRESSED (ENV STRESS)"
                    else:
                        status = "ALERT: CONFIRMED HEAT"
                        last_alert_date = day
                else:
                    status = "NORMAL (COOL DOWN)"
            elif score_val >= Config.PROESTRUS_SCORE:
                status = "LOG: PROESTRUS"

            actual_rows = daily_raw_counts.get(day, 0)
            completeness = round((actual_rows / Config.EXPECTED_ROWS_PER_DAY) * 100, 2)
            data_loss = round(100.0 - completeness, 2)

            results.append({
                "date":              day.isoformat(),
                "status":            status,
                "night_spike_C":     round(today_score_dict["spike"], 4),
                "persistence_pct":   round(today_score_dict["persistence"] * 100, 1),
                "score":             round(score_val, 4),
                "global_anchor_C":   round(global_anchor, 4),
                "env_stress":        env_stress,
                "in_expected_window": in_window,
                "completeness_pct":  completeness,
                "data_loss_pct":     data_loss,
                "was_filled":        day in filled_days
            })

        return results

# ============================================================================
# 4. HEAT CYCLE PREDICTOR
# ============================================================================

class HeatCyclePredictor:
    """Predicts future heat windows based on historical events."""
    
    @staticmethod
    def predict_next(last_heat: date) -> dict:
        expected     = last_heat + timedelta(days=Config.CYCLE_DAYS)
        window_start = expected - timedelta(days=Config.CYCLE_TOLERANCE)
        window_end   = expected + timedelta(days=Config.CYCLE_TOLERANCE)
        return {
            "expected_next_heat": expected.isoformat(),
            "window_start":       window_start.isoformat(),
            "window_end":         window_end.isoformat(),
        }

    @staticmethod
    def days_until(window_start_str: str, reference: date) -> int:
        ws    = date.fromisoformat(window_start_str)
        delta = (ws - reference).days
        return max(0, delta)

# ============================================================================
# 5. LAMBDA HANDLER (Orchestrator)
# ============================================================================

class HeatBenchmarkHandler:
    """Main orchestrator for the Benchmark Lambda."""
    
    def __init__(self):
        self.db = DatabaseManager(Config.PG_CONNECTION_STRING)
        self.engine = HeatLogicEngine()
        self.predictor = HeatCyclePredictor()

    def process(self, event: dict) -> dict:
        """Main execution flow."""
        # 1. Parse Input
        node_id         = int(event.get("node_id") or event.get("nodeId") or 0)
        last_heat_str   = event.get("last_heat_date") or event.get("lastHeatDate") or ""
        window_days     = int(event.get("window_days", 7))

        if not node_id or not last_heat_str:
            return self._response(400, {"error": "Missing node_id or last_heat_date"})

        last_heat_date = date.fromisoformat(last_heat_str)
        today_ist = self.db.utc_to_ist(datetime.utcnow()).date()

        # 2. Predict Cycle
        next_cycle = self.predictor.predict_next(last_heat_date)
        days_left  = self.predictor.days_until(next_cycle["window_start"], today_ist)

        # 3. Fetch Data (with lookback window for context)
        fetch_start = last_heat_date - timedelta(days=window_days)
        fetch_end   = today_ist
        raw_df      = self.db.fetch_node_data(node_id, fetch_start, fetch_end)

        if raw_df.empty:
            return self._response(404, {"error": "No data found for given range"})

        # 4. Pre-processing (Timeseries prep + Hourly Resampling)
        raw_df["timestamp_ist"] = pd.to_datetime(raw_df["timestamp_ist"])
        daily_raw_counts = raw_df.groupby(raw_df["timestamp_ist"].dt.date).size().to_dict()
        
        hourly_all = self.engine.resample_to_hourly(raw_df)

        # 5. Data Imputation (Forward Fill)
        # Reindex to ensure no missing hours in the timeseries
        full_index = pd.date_range(
            start=pd.Timestamp(fetch_start), 
            end=pd.Timestamp(fetch_end) + timedelta(hours=23), 
            freq="h"
        )
        hourly_all = hourly_all.reindex(full_index)

        filled_days = set()
        day_groups = hourly_all.groupby(hourly_all.index.date)
        processed_parts = []
        
        for day_date, day_df in day_groups:
            completeness = daily_raw_counts.get(day_date, 0) / Config.EXPECTED_ROWS_PER_DAY
            if completeness >= Config.FFILL_MIN_COMPLETENESS:
                processed_parts.append(day_df.ffill())
                filled_days.add(day_date)
            else:
                processed_parts.append(day_df)
        
        hourly_all = pd.concat(processed_parts)

        # 6. Logic Execution
        daily_results = self.engine.run_detection(
            hourly_all, window_days, next_cycle, daily_raw_counts, filled_days
        )

        # Filter out the context days (return only days after last_heat_date)
        output_results = [r for r in daily_results if r["date"] > last_heat_str]

        # 7. Final Response assembly
        summary = self._build_summary(output_results)
        
        return self._response(200, {
            "node_id": node_id,
            "last_heat_date": last_heat_str,
            "expected_next_heat": next_cycle["expected_next_heat"],
            "expected_window": {
                "start": next_cycle["window_start"],
                "end": next_cycle["window_end"],
            },
            "days_until_next_window": days_left,
            "daily_results": output_results,
            "summary": summary
        })

    def _build_summary(self, results: List[dict]) -> dict:
        confirmed = [r["date"] for r in results if r["status"] == "ALERT: CONFIRMED HEAT"]
        proestrus = [r["date"] for r in results if r["status"] == "LOG: PROESTRUS"]
        suppressed = [r["date"] for r in results if "SUPPRESSED" in r["status"]]
        
        return {
            "total_days_processed": len(results),
            "confirmed_heat_days": confirmed,
            "proestrus_days": proestrus,
            "suppressed_days": suppressed,
            "normal_days": len(results) - len(confirmed) - len(proestrus) - len(suppressed)
        }

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
    """AWS Lambda entry point."""
    logger.info(f"BENCHMARK START: {json.dumps(event)}")
    try:
        app = HeatBenchmarkHandler()
        return app.process(event)
    except Exception as e:
        logger.error(f"FATAL: {e}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }

# ============================================================================
# LOCAL TESTING
# ============================================================================

if __name__ == "__main__":
    test_event = {
        "node_id": 124,
        "last_heat_date": "2026-02-26",
        "window_days": 7
    }
    print(f"Running local test for Node {test_event['node_id']}...")
    resp = handler(test_event, None)
    print(resp["body"])
