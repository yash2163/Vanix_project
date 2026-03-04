"""
Heat Cycle Detection - Benchmark Lambda Function
=================================================
Purpose: Benchmarking and logic validation ONLY (not part of the live scoring pipeline).

Input Payload:
    {
        "node_id":        124,
        "last_heat_date": "2026-02-26",   # ISO date: YYYY-MM-DD
        "window_days":    7               # optional, default 7
    }

Output:
    {
        "node_id": 124,
        "last_heat_date": "2026-02-26",
        "expected_next_heat": "2026-03-19",
        "expected_window": { "start": "2026-03-16", "end": "2026-03-22" },
        "days_until_next_window": 18,
        "daily_results": [
            {
                "date": "2026-02-27",
                "status": "NORMAL",
                "night_spike_C": 0.08,
                "persistence_pct": 32.1,
                "score": 14.05,
                "global_anchor_C": 34.50,
                "buffer_days": 2,
                "in_expected_window": false,
                "actual": 131620,
                "completeness_pct": 76.17
            },
            ...
        ],
        "summary": {
            "total_days_processed": 10,
            "confirmed_heat_days": [],
            "proestrus_days": [],
            "normal_days": 10
        }
    }

DB Reference: Uses same device_data table and PG_CONNECTION_STRING as pipline.py
Schedule: Triggered ON-DEMAND for benchmarking (not on a cron schedule)
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
from typing import Dict, List, Tuple, Optional

# ============================================================================
# LOGGING
# ============================================================================

def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================================
# CONFIGURATION  (mirrors pipline.py)
# ============================================================================

PG_CONNECTION_STRING = os.environ.get(
    "PG_CONNECTION_STRING",
    "postgresql://vanixuser:vanix%244567%23@10.98.36.21:6432/vanixdb"
)

# Heat cycle parameters
CYCLE_DAYS          = 21    # average bovine estrus cycle length
CYCLE_TOLERANCE     = 3     # +/- days around expected date
MIN_BUFFER_DAYS     = 2     # minimum days in buffer before we score

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
HEAT_COOL_DOWN      = 15     # days to suppress new alerts after a confirmed heat
EXPECTED_ROWS_PER_DAY = 172800  # 2 data points per second × 86400 seconds/day
FFILL_MIN_COMPLETENESS = 0.60   # only forward-fill days with ≥60% raw data

# ============================================================================
# TIMEZONE UTILITIES  (mirrors pipline.py)
# ============================================================================

def ist_to_utc(dt: datetime) -> datetime:
    return dt - timedelta(hours=5, minutes=30)

def utc_to_ist(dt: datetime) -> datetime:
    return dt + timedelta(hours=5, minutes=30)

# ============================================================================
# DATABASE: FETCH DATA FOR A DATE RANGE  (mirrors pipline.py pattern)
# ============================================================================

def fetch_node_data(node_id: int,
                    start_date: date,
                    end_date: date) -> pd.DataFrame:
    """
    Fetch all sensor readings for a node between start_date and end_date (IST).
    Mirrors the fetch_device_data_from_db() pattern from pipline.py.

    Returns DataFrame with columns:
        node_id, timestamp_ist, x, y, z, temperature_value
    """
    # Convert IST date range to UTC timestamps for the DB query
    start_utc = ist_to_utc(datetime(start_date.year, start_date.month, start_date.day, 0, 0, 0))
    end_utc   = ist_to_utc(datetime(end_date.year,   end_date.month,   end_date.day,   23, 59, 59))

    logger.info(f"Querying node {node_id}: {start_date} -> {end_date} IST "
                f"({start_utc} -> {end_utc} UTC)")

    query = """
        SELECT
            node_id,
            timestamp               AS timestamp_utc,
            timestamp + INTERVAL '5 hours 30 minutes' AS timestamp_ist,
            x,
            y,
            z,
            temperature_value
        FROM device_data
        WHERE node_id::text = %(node_id)s::text
          AND timestamp >= %(start_utc)s
          AND timestamp <= %(end_utc)s
          AND x IS NOT NULL
          AND y IS NOT NULL
          AND z IS NOT NULL
        ORDER BY timestamp ASC
    """

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module='pandas.io.sql')
            with psycopg2.connect(PG_CONNECTION_STRING) as conn:
                df = pd.read_sql_query(
                    query, conn,
                    params={"node_id": str(node_id), "start_utc": start_utc, "end_utc": end_utc}
                )
        logger.info(f"  Fetched {len(df):,} rows")
        return df

    except Exception as e:
        logger.error(f"DB fetch failed: {e}", exc_info=True)
        raise

# ============================================================================
# BLOCK 1 + 2: FEATURE EXTRACTION + ACTIVITY RECOGNITION
# ============================================================================

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute VeDBA from x, y, z accelerometer axes."""
    df = df.copy()
    df["mag"]   = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
    df["vedba"] = np.abs(df["mag"] - df["mag"].rolling(window=50, center=True).mean())
    return df.fillna(0)

def predict_activity(df: pd.DataFrame) -> pd.DataFrame:
    """Classify each row as RES / FEED / STANDING based on VeDBA thresholds."""
    conditions = [df["vedba"] > RES_THRESHOLD, df["vedba"] > FEED_THRESHOLD]
    df["activity_class"] = np.select(conditions, ["RES", "FEED"], default="STANDING")
    return df

# ============================================================================
# HOURLY RESAMPLING
# ============================================================================

def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample raw rows to hourly buckets.
    Returns DataFrame indexed by hour with temp_mean and res_ratio columns.
    """
    df = predict_activity(extract_features(df))
    idx = df.set_index("timestamp_ist")

    hourly = idx["temperature_value"].resample("h").mean().to_frame(name="temp_mean")
    hourly["res_ratio"] = (
        idx["activity_class"]
        .apply(lambda x: 1 if x == "RES" else 0)
        .resample("h").mean()
    )
    return hourly

# ============================================================================
# BLOCK 3: ENVIRONMENTAL HEAT STRESS
# ============================================================================

def check_heat_stress(hourly_df: pd.DataFrame) -> bool:
    solar = hourly_df[hourly_df.index.hour.isin(SOLAR_HOURS)]["temp_mean"].dropna()
    return bool(solar.mean() > STRESS_TEMP) if not solar.empty else False

# ============================================================================
# BLOCK 4: SCORE A SINGLE DAY  (given full window for global anchor)
# ============================================================================

def score_day(day_hourly: pd.DataFrame, global_night_base: float) -> dict:
    night_data  = day_hourly[day_hourly.index.hour.isin(NIGHT_HOURS)]
    night_temps = night_data["temp_mean"].dropna()
    night_max   = night_temps.max() if not night_temps.empty else np.nan
    spike       = max(0.0, night_max - global_night_base) if not np.isnan(night_max) else 0.0

    persistence = day_hourly["res_ratio"].rolling(3, min_periods=1).mean().max()
    persistence = 0.0 if np.isnan(persistence) else float(persistence)

    score = (spike * SCORE_W_SPIKE) + (persistence * SCORE_W_PERSIST)
    return {"spike": spike, "persistence": persistence, "score": score}

# ============================================================================
# CYCLE PREDICTION UTILITIES
# ============================================================================

def predict_next_cycle(last_heat: date) -> dict:
    expected     = last_heat + timedelta(days=CYCLE_DAYS)
    window_start = expected - timedelta(days=CYCLE_TOLERANCE)
    window_end   = expected + timedelta(days=CYCLE_TOLERANCE)
    return {
        "expected_next_heat": expected.isoformat(),
        "window_start":       window_start.isoformat(),
        "window_end":         window_end.isoformat(),
    }

def days_until_window(window_start: str, reference: date = None) -> int:
    ref   = reference or date.today()
    ws    = date.fromisoformat(window_start)
    delta = (ws - ref).days
    return max(0, delta)

# ============================================================================
# ROLLING WINDOW DETECTION ENGINE
# ============================================================================

def run_rolling_detection(
    all_hourly: pd.DataFrame,
    node_id: int,
    last_heat_date: date,
    window_days: int,
    next_cycle: dict,
    daily_raw_counts: Dict[date, int] = None,
    filled_days: set = None
) -> List[dict]:
    """
    Step through each day in all_hourly one at a time.
    Maintain a rolling buffer of window_days days.
    Score today against the full buffer.
    Returns list of per-day result dicts.
    """
    window_start_str = next_cycle["window_start"]
    window_end_str   = next_cycle["window_end"]

    # Group hourly data by calendar date (IST)
    daily_groups = all_hourly.groupby(all_hourly.index.date)
    sorted_days  = sorted(daily_groups.groups.keys())

    buffer: deque = deque(maxlen=window_days)   # (date, hourly_df) tuples
    results = []
    last_alert_date = None

    for day in sorted_days:
        day_hourly = daily_groups.get_group(day)
        buffer.append((day, day_hourly))

        # Skip until we have at least MIN_BUFFER_DAYS of context
        if len(buffer) < MIN_BUFFER_DAYS:
            logger.info(f"  {day} | BUFFERING ({len(buffer)}/{window_days})")
            continue

        # Build full window
        window_df = pd.concat([df for _, df in buffer])

        # Global anchor: minimum night temp across the whole window
        night_mask = window_df.index.hour.isin([0, 1, 2])
        night_vals = window_df.loc[night_mask, "temp_mean"].dropna()
        global_anchor = float(night_vals.min()) if not night_vals.empty else 0.0

        # Env stress across the window
        env_stress = check_heat_stress(window_df)

        # Score all days in buffer to do winner-take-all comparison
        scored = []
        for d, d_df in buffer:
            s = score_day(d_df, global_anchor)
            s["date"] = d
            scored.append(s)

        scored.sort(key=lambda x: x["score"], reverse=True)
        peak_day = scored[0]["date"]

        # Get today's own score
        today_score = next((s for s in scored if s["date"] == day), scored[0])
        score_val   = today_score["score"]

        # Determine if today is inside the expected heat window
        in_window = (window_start_str <= day.isoformat() <= window_end_str)

        # Determine verdict for today
        status = "NORMAL"
        
        # 1. Check if today is the peak of the current rolling window and meets min score
        if day == peak_day and score_val >= MIN_HEAT_SCORE:
            # 2. Check cool-down period from last alert
            if last_alert_date is None or (day - last_alert_date).days >= HEAT_COOL_DOWN:
                if env_stress:
                    status = "SUPPRESSED (ENV STRESS)"
                else:
                    status = "ALERT: CONFIRMED HEAT"
                    last_alert_date = day
            else:
                # Peak of window but inside cool-down period
                status = "NORMAL (COOL DOWN)"
        elif score_val >= PROESTRUS_SCORE:
            status = "LOG: PROESTRUS"

        # Day-wise data completeness
        actual_rows = daily_raw_counts.get(day, 0) if daily_raw_counts else 0
        completeness = round((actual_rows / EXPECTED_ROWS_PER_DAY) * 100, 2) if EXPECTED_ROWS_PER_DAY > 0 else 0.0
        was_filled = day in filled_days if filled_days else False

        result = {
            "date":              day.isoformat(),
            "status":            status,
            "night_spike_C":     round(today_score["spike"], 4),
            "persistence_pct":   round(today_score["persistence"] * 100, 1),
            "score":             round(score_val, 4),
            "global_anchor_C":   round(global_anchor, 4),
            "env_stress":        env_stress,
            "buffer_days":       len(buffer),
            "in_expected_window": in_window,
            "actual":            actual_rows,
            "completeness_pct":  completeness,
            "filled":            was_filled,
        }
        results.append(result)

        logger.info(
            f"  [{day}] {status:<26} | spike={today_score['spike']:.3f}C "
            f"| persist={round(today_score['persistence']*100)}% "
            f"| score={score_val:.2f} "
            f"| anchor={global_anchor:.2f}C"
            + (" [IN WINDOW]" if in_window else "")
        )

    return results

# ============================================================================
# SUMMARISE RESULTS
# ============================================================================

def build_summary(results: List[dict]) -> dict:
    confirmed = [r["date"] for r in results if r["status"] == "ALERT: CONFIRMED HEAT"]
    proestrus = [r["date"] for r in results if r["status"] == "LOG: PROESTRUS"]
    suppressed = [r["date"] for r in results if "SUPPRESSED" in r["status"]]
    normal    = len(results) - len(confirmed) - len(proestrus) - len(suppressed)

    return {
        "total_days_processed": len(results),
        "confirmed_heat_days":  confirmed,
        "proestrus_days":       proestrus,
        "suppressed_days":      suppressed,
        "normal_days":          normal,
    }

# ============================================================================
# MAIN HANDLER (Lambda entry point)
# ============================================================================

def handler(event, context):
    """
    AWS Lambda handler.

    Expected event payload:
        {
            "node_id":        124,
            "last_heat_date": "2026-02-26",
            "window_days":    7              (optional)
        }
    """
    logger.info("=" * 70)
    logger.info("HEAT CYCLE BENCHMARK LAMBDA - STARTED")
    logger.info("=" * 70)
    logger.info(f"Event: {json.dumps(event)}")

    try:
        # ------------------------------------------------------------------
        # 1. Parse and validate payload
        # ------------------------------------------------------------------
        node_id         = int(event.get("node_id") or event.get("nodeId") or 0)
        last_heat_str   = event.get("last_heat_date") or event.get("lastHeatDate") or ""
        window_days     = int(event.get("window_days", 7))

        if not node_id:
            raise ValueError("Missing required field: node_id")
        if not last_heat_str:
            raise ValueError("Missing required field: last_heat_date (format: YYYY-MM-DD)")

        last_heat_date = date.fromisoformat(last_heat_str)
        today_ist      = utc_to_ist(datetime.utcnow()).date()

        logger.info(f"Node ID         : {node_id}")
        logger.info(f"Last Heat Date  : {last_heat_date}")
        logger.info(f"Today (IST)     : {today_ist}")
        logger.info(f"Rolling Window  : {window_days} days")

        # ------------------------------------------------------------------
        # 2. Cycle prediction
        # ------------------------------------------------------------------
        next_cycle = predict_next_cycle(last_heat_date)
        days_left  = days_until_window(next_cycle["window_start"], today_ist)

        logger.info(f"Next Expected Heat  : {next_cycle['expected_next_heat']}")
        logger.info(f"Watch Window        : {next_cycle['window_start']} to {next_cycle['window_end']}")

        # ------------------------------------------------------------------
        # 3. Query DB: from last_heat_date - 7 days to today
        #    (extra lookback so day-1 always has a full buffer worth of context)
        # ------------------------------------------------------------------
        fetch_start = last_heat_date - timedelta(days=window_days)
        fetch_end   = today_ist

        logger.info(f"\nFetching data: {fetch_start} -> {fetch_end}")
        raw_df = fetch_node_data(node_id, fetch_start, fetch_end)

        if raw_df.empty:
            return _response(404, {
                "error":   "No data found in database for this node/date range",
                "node_id": node_id,
                "range":   f"{fetch_start} to {fetch_end}"
            })

        # ------------------------------------------------------------------
        # 4. Resample to hourly & compute day-wise raw counts
        # ------------------------------------------------------------------
        raw_df["timestamp_ist"] = pd.to_datetime(raw_df["timestamp_ist"])

        # Count raw rows per calendar day (IST) before resampling
        daily_raw_counts = (
            raw_df.groupby(raw_df["timestamp_ist"].dt.date)
            .size()
            .to_dict()
        )

        hourly_all = to_hourly(raw_df)

        # ------------------------------------------------------------------
        # 4b. Forward-fill missing hourly buckets (only for days ≥60% data)
        # ------------------------------------------------------------------
        full_index = pd.date_range(
            start=pd.Timestamp(fetch_start),
            end=pd.Timestamp(fetch_end) + timedelta(hours=23),
            freq="h"
        )
        hourly_all = hourly_all.reindex(full_index)

        # Determine which days qualify for forward-fill
        filled_days = set()
        for day_date, pct in [
            (d, daily_raw_counts.get(d, 0) / EXPECTED_ROWS_PER_DAY)
            for d in sorted(set(full_index.date))
        ]:
            if pct >= FFILL_MIN_COMPLETENESS:
                filled_days.add(day_date)

        # Forward-fill only qualifying days; leave others with NaN gaps
        day_groups = hourly_all.groupby(hourly_all.index.date)
        filled_parts = []
        for day_date, day_df in day_groups:
            if day_date in filled_days:
                filled_parts.append(day_df.ffill())
            else:
                filled_parts.append(day_df)
        hourly_all = pd.concat(filled_parts)

        logger.info(f"Forward-filled {len(filled_days)} day(s) with ≥60% data")
        logger.info(f"Hourly buckets: {len(hourly_all)} across "
                    f"{hourly_all.index.date.min()} to {hourly_all.index.date.max()}")

        # Only score days AFTER last_heat_date (we already know that one)
        hourly_scoring = hourly_all[hourly_all.index.date > last_heat_date]

        if hourly_scoring.empty:
            return _response(200, {
                "node_id":           node_id,
                "last_heat_date":    last_heat_str,
                "expected_next_heat": next_cycle["expected_next_heat"],
                "expected_window":   next_cycle,
                "days_until_next_window": days_left,
                "message": "No new days after last_heat_date yet - check back later.",
                "daily_results": [],
                "summary": {"total_days_processed": 0}
            })

        # ------------------------------------------------------------------
        # 5. Run rolling window detection
        # ------------------------------------------------------------------
        logger.info(f"\nRunning rolling window detection on "
                    f"{len(np.unique(hourly_scoring.index.date))} day(s)...\n")

        daily_results = run_rolling_detection(
            hourly_all,           # full hourly (includes pre-heat context)
            node_id,
            last_heat_date,
            window_days,
            next_cycle,
            daily_raw_counts,
            filled_days
        )

        # Only return results for days AFTER last_heat_date
        daily_results = [r for r in daily_results if r["date"] > last_heat_str]

        # ------------------------------------------------------------------
        # 6. Build response
        # ------------------------------------------------------------------
        summary = build_summary(daily_results)

        logger.info("\n" + "=" * 70)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 70)
        logger.info(f"  Total days processed : {summary['total_days_processed']}")
        logger.info(f"  Confirmed heat       : {summary['confirmed_heat_days'] or 'None'}")
        logger.info(f"  Proestrus flags      : {summary['proestrus_days'] or 'None'}")
        logger.info(f"  Suppressed           : {summary['suppressed_days'] or 'None'}")
        logger.info(f"  Normal               : {summary['normal_days']}")
        logger.info("=" * 70)

        body = {
            "node_id":                node_id,
            "last_heat_date":         last_heat_str,
            "expected_next_heat":     next_cycle["expected_next_heat"],
            "expected_window": {
                "start": next_cycle["window_start"],
                "end":   next_cycle["window_end"],
            },
            "days_until_next_window": days_left,
            "cycle_days":             CYCLE_DAYS,
            "window_days_used":       window_days,
            "data_fetched": {
                "from": fetch_start.isoformat(),
                "to":   fetch_end.isoformat(),
                "total_rows_received": int(len(raw_df)),
                "expected_rows_per_day": EXPECTED_ROWS_PER_DAY,
            },
            "daily_results": daily_results,
            "summary":       summary,
        }

        return _response(200, body)

    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        return _response(400, {"error": str(ve)})

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return _response(500, {"error": str(e), "error_type": type(e).__name__})


def _response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers":    {"Content-Type": "application/json"},
        "body":       json.dumps(body, indent=2, default=str)
    }


# ============================================================================
# LOCAL TEST  (run directly: python lambda_heat_benchmark.py)
# ============================================================================

if __name__ == "__main__":
    test_event = {
        "node_id":        124,
        "last_heat_date": "2026-02-26",
        "window_days":    7
    }

    print("\n" + "=" * 70)
    print("LOCAL TEST - HEAT CYCLE BENCHMARK")
    print("=" * 70)
    print(f"Payload: {json.dumps(test_event, indent=2)}")
    print("=" * 70 + "\n")

    response = handler(test_event, None)
    print("\n" + "=" * 70)
    print("LAMBDA RESPONSE")
    print("=" * 70)
    body = json.loads(response["body"])

    # Pretty print the summary and daily table
    print(f"\nNode            : {body['node_id']}")
    print(f"Last Heat       : {body['last_heat_date']}")
    print(f"Expected Next   : {body['expected_next_heat']}")
    print(f"Watch Window    : {body['expected_window']['start']}  to  {body['expected_window']['end']}")
    print(f"Days to Window  : {body['days_until_next_window']}")

    results = body.get("daily_results", [])
    if results:
        print(f"\n{'Date':<14} {'Status':<26} {'Spike':>8}  {'Persist':>8}  {'Score':>8}  {'InWindow':>10}")
        print("-" * 78)
        for r in results:
            win_flag = "YES **" if r["in_expected_window"] else "-"
            print(
                f"{r['date']:<14} {r['status']:<26}"
                f" {r['night_spike_C']:>7.3f}C"
                f"  {r['persistence_pct']:>6.1f}%"
                f"  {r['score']:>8.2f}"
                f"  {win_flag:>10}"
            )

    summary = body.get("summary", {})
    print(f"\nSummary:")
    print(f"  Total processed : {summary.get('total_days_processed', 0)}")
    print(f"  Confirmed Heat  : {summary.get('confirmed_heat_days') or 'None'}")
    print(f"  Proestrus       : {summary.get('proestrus_days') or 'None'}")
    print(f"  Normal days     : {summary.get('normal_days', 0)}")
