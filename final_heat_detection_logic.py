"""
Daily Heat Analysis and Data Loss Tool
======================================
Purpose: Isolated calculation of heat metrics for a rolling 24-hour window to
analyse data completeness and compare RAW vs FILLED performance side-by-side.

Run Schedule
------------
This script is intended to be triggered TWICE per day, each time using a strict
rolling 24-hour lookback anchored to the scheduled run time (IST):

    Run 1  — midnight   (00:00 IST)  → window: previous day 00:00 → today 00:00
    Run 2  — noon       (12:00 IST)  → window: yesterday 12:00   → today 12:00

The caller passes ONE of these two inputs:

    Option A  (explicit window):
        { "node_id": 124,
          "window_start_ist": "2026-03-04 00:00:00",
          "window_end_ist":   "2026-03-05 00:00:00" }

    Option B  (auto-derive from run time — recommended for scheduled triggers):
        { "node_id": 124 }
        Script calls datetime.utcnow(), converts to IST, snaps to the nearest
        scheduled hour (00 or 12), and sets the window automatically.

Overlap is intentional — every data point is used in two consecutive runs,
giving both a midnight-anchored view and a noon-anchored view of each 24-hour
period.  The results table records the exact window so the two runs are always
distinguishable.

Constants
---------
    Expected DPs per 10-min bucket : 800      (80 DPs/min x 10 min)
    Expected DPs per 30-min slot   : 2,400    (80 DPs/min x 30 min)
    Expected DPs per 24-h window   : 115,200

Change Log
----------
    [BUG-1] calculate_daily_anchor returned 0.0 on empty data -> now returns
            np.nan, preventing a phantom night_spike.
    [BUG-2] Anchor computed as min(night-window temps) with explicit NaN guard.
    [BUG-3] res_ratio is now filled using the same block-copy backfill logic as
            temp_mean (not left as NaN for missing windows).
    [CHG-1] apply_10min_backfill fills BOTH temp_mean AND res_ratio.
    [CHG-2] stress_temp derived as max temp during SOLAR_HOURS in the window,
            not as the most-recent DB row.
    [CHG-3] Output payload simplified: heat_detected, data_completeness, score
            (filled pipeline). Persistence/anchor/max_temp/spike are logged
            separately.
    [CHG-4] Results persisted to heat_analysis_results (auto-created). UTC ts.
    [CHG-5] Calendar-day fetch replaced with rolling 24-hour window fetch.
            target_date parameter removed; window_start_ist / window_end_ist
            used throughout. All slot/bucket indices are anchored to
            window_start, not midnight.
"""

import json
import os
import sys
import logging
import warnings
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, Tuple, Optional

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

class Config:
    PG_CONNECTION_STRING = os.environ.get(
        "PG_CONNECTION_STRING",
        "postgresql://vanixuser:vanix%244567%23@10.98.36.21:6432/vanixdb"
    )

    # -- POC Control (boolean toggle) ----------------------------------------
    IS_POC        = True
    POC_NODES     = [124, 184]

    # -- Sampling constants --------------------------------------------------
    DP_RATE_PER_MIN       = 80
    BACKFILL_WINDOW_MIN   = 10
    SLOT_DURATION_MIN     = 30
    SLOTS_PER_WINDOW      = 48          # 24 h / 0.5 h

    EXPECTED_DPS_PER_MIN  = DP_RATE_PER_MIN                         # 80
    EXPECTED_DPS_PER_SLOT = DP_RATE_PER_MIN * SLOT_DURATION_MIN     # 2 400
    EXPECTED_DPS_PER_DAY  = DP_RATE_PER_MIN * 60 * 24               # 115 200

    # -- Activity thresholds -------------------------------------------------
    RES_THRESHOLD  = 0.35
    FEED_THRESHOLD = 0.15

    # -- Scoring weights (overridden from DB) --------------------------------
    STRESS_TEMP              = 40.5
    SCORE_W_SPIKE            = 15
    SCORE_W_PERSIST          = 40
    HEAT_DETECTION_THRESHOLD = 50

    # -- Hour windows --------------------------------------------------------
    # These refer to IST hours-of-day, evaluated against the full 24-h window
    # regardless of which calendar day those hours fall on.
    ANCHOR_HOURS = [23, 0, 1, 2, 3]
    NIGHT_HOURS  = [23, 0, 1, 2, 3]
    SOLAR_HOURS  = [11, 12, 13, 14, 15, 16]

    # -- Scheduled run hours (IST) -------------------------------------------
    # Used when auto-deriving the window from the current time.
    SCHEDULED_HOURS_IST = [0, 12]


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("DailyHeatAnalyzer")

logger = setup_logging()


# ============================================================================
# 2. WINDOW HELPERS
# ============================================================================

IST_OFFSET = timedelta(hours=5, minutes=30)

def utc_now_ist() -> datetime:
    return datetime.utcnow() + IST_OFFSET

def ist_to_utc(dt: datetime) -> datetime:
    return dt - IST_OFFSET

def utc_to_ist(dt: datetime) -> datetime:
    return dt + IST_OFFSET


def resolve_window(event: dict) -> Tuple[datetime, datetime]:
    """
    Determine the 24-hour analysis window (both endpoints as naive IST datetimes).
    Window is half-open: [window_start, window_end).

    Priority
    --------
    1. Explicit "window_start_ist" + "window_end_ist" strings in the event.
    2. Auto-derive: snap current IST time to nearest scheduled hour (0 or 12)
       and look back exactly 24 hours.

    Examples
    --------
    Midnight run triggered at 00:00 IST 2026-03-05:
        window_start = 2026-03-04 00:00:00 IST
        window_end   = 2026-03-05 00:00:00 IST

    Noon run triggered at 12:00 IST 2026-03-05:
        window_start = 2026-03-04 12:00:00 IST
        window_end   = 2026-03-05 12:00:00 IST
    """
    if "window_start_ist" in event and "window_end_ist" in event:
        ws = datetime.fromisoformat(event["window_start_ist"])
        we = datetime.fromisoformat(event["window_end_ist"])
        if (we - ws) != timedelta(hours=24):
            logger.warning(
                f"Window duration is {we - ws}, not exactly 24 h. Proceeding anyway."
            )
        logger.info(f"Window (explicit): {ws}  ->  {we}  IST")
        return ws, we

    # Auto-derive: snap to nearest scheduled hour
    now_ist   = utc_now_ist().replace(second=0, microsecond=0)
    snap_hour = min(Config.SCHEDULED_HOURS_IST, key=lambda h: abs(now_ist.hour - h))
    window_end_ist   = now_ist.replace(hour=snap_hour, minute=0)
    window_start_ist = window_end_ist - timedelta(hours=24)

    logger.info(
        f"Window (auto, snapped to {snap_hour:02d}:00 IST): "
        f"{window_start_ist}  ->  {window_end_ist}  IST"
    )
    return window_start_ist, window_end_ist


# ============================================================================
# 3. DATABASE MANAGER
# ============================================================================

class DatabaseManager:
    def __init__(self, connection_string: str):
        self.conn_str = connection_string

    def fetch_active_nodes_in_window(self, window_start_ist: datetime, window_end_ist: datetime) -> list:
        start_utc = ist_to_utc(window_start_ist)
        end_utc   = ist_to_utc(window_end_ist)
        query = """
            SELECT DISTINCT node_id
            FROM device_data
            WHERE timestamp >= %(start_utc)s AND timestamp < %(end_utc)s
        """
        try:
            with psycopg2.connect(self.conn_str) as conn:
                with conn.cursor() as cur:
                    cur.execute(query, {"start_utc": start_utc, "end_utc": end_utc})
                    rows = cur.fetchall()
                    return sorted([int(r[0]) for r in rows if r[0] is not None])
        except Exception as e:
            logger.error(f"Failed to fetch active nodes: {e}")
            return []

    def fetch_window(
        self,
        node_id: int,
        window_start_ist: datetime,
        window_end_ist: datetime,
    ) -> pd.DataFrame:
        """
        [CHG-5] Fetch all raw data points inside the 24-h IST window.
        DB stores UTC; both endpoints are converted before querying.
        Window is half-open: [window_start, window_end).
        """
        start_utc = ist_to_utc(window_start_ist)
        end_utc   = ist_to_utc(window_end_ist)

        logger.info(
            f"DB: node={node_id}  UTC [{start_utc}  ->  {end_utc})"
        )

        query = """
            SELECT
                node_id,
                timestamp                                   AS timestamp_utc,
                timestamp + INTERVAL '5 hours 30 minutes'  AS timestamp_ist,
                x, y, z,
                temperature_value
            FROM device_data
            WHERE node_id::text = %(node_id)s::text
              AND timestamp >= %(start_utc)s
              AND timestamp <  %(end_utc)s
              AND x IS NOT NULL AND y IS NOT NULL AND z IS NOT NULL
            ORDER BY timestamp ASC
        """

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="pandas.io.sql")
            with psycopg2.connect(self.conn_str) as conn:
                df = pd.read_sql_query(
                    query, conn,
                    params={
                        "node_id":   str(node_id),
                        "start_utc": start_utc,
                        "end_utc":   end_utc,
                    },
                )
        return df

    def fetch_config_parameters(self) -> dict:
        """Fetch configurable weights and thresholds from heat_analysis_config."""
        query = """
            SELECT *
            FROM heat_analysis_config
            ORDER BY id DESC LIMIT 1;
        """
        try:
            with psycopg2.connect(self.conn_str) as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    row = cur.fetchone()
                    if row:
                        col_names = [desc[0] for desc in cur.description]
                        row_dict = dict(zip(col_names, row))
                        
                        spike_w = float(row_dict["score_w_spike"]) if row_dict.get("score_w_spike") is not None else None
                        persist_w = float(row_dict["score_w_persist"]) if row_dict.get("score_w_persist") is not None else None
                        
                        heat_thresh = None
                        if "heat_detection_threshold" in row_dict and row_dict["heat_detection_threshold"] is not None:
                            heat_thresh = float(row_dict["heat_detection_threshold"])
                        
                        logger.info(
                            f"DB CONFIG: Fetched score_w_spike={spike_w}, "
                            f"score_w_persist={persist_w}, heat_detection_threshold={heat_thresh}"
                        )
                        return {
                            "score_w_spike":   spike_w,
                            "score_w_persist": persist_w,
                            "heat_detection_threshold": heat_thresh,
                        }
        except Exception as e:
            logger.warning(f"Could not fetch config from DB (using defaults): {e}")
        return {}

    def derive_stress_temp_from_solar(self, processed_df: pd.DataFrame) -> float:
        """
        [CHG-2] Peak environmental temperature = max(temperature_value) during
        SOLAR_HOURS (11-16 IST) within the analysis window.
        Falls back to Config.STRESS_TEMP (40.5 C) when no solar-hour data exists.
        """
        if processed_df.empty or "temperature_value" not in processed_df.columns:
            logger.info(f"DB STRESS: Using code default STRESS_TEMP={Config.STRESS_TEMP} C (empty dataframe)")
            return Config.STRESS_TEMP

        df = processed_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp_ist"]):
            df["timestamp_ist"] = pd.to_datetime(df["timestamp_ist"])

        solar_mask  = df["timestamp_ist"].dt.hour.isin(Config.SOLAR_HOURS)
        solar_temps = df.loc[solar_mask, "temperature_value"].dropna()

        if solar_temps.empty:
            logger.info(
                f"DB STRESS: Using code default STRESS_TEMP={Config.STRESS_TEMP} C "
                f"(No solar-hour readings between {min(Config.SOLAR_HOURS)}:00-{max(Config.SOLAR_HOURS)}:59)"
            )
            return Config.STRESS_TEMP

        peak = float(solar_temps.max())
        logger.info(
            f"DB STRESS: Dynamically derived STRESS_TEMP={peak:.4f} C "
            f"from {len(solar_temps):,} solar-hour readings"
        )
        return peak

    # -- Results table -------------------------------------------------------

    def ensure_results_table(self) -> None:
        """
        [CHG-4] Create heat_analysis_results if absent.

        Schema
        ------
        id                SERIAL PK
        node_id           INTEGER
        window_start_utc  TIMESTAMPTZ  -- inclusive start of 24-h window (UTC)
        window_end_utc    TIMESTAMPTZ  -- exclusive end of 24-h window (UTC)
        analysis_run_at   TIMESTAMPTZ  -- when this execution ran (UTC)
        data_completeness NUMERIC(6,2) -- % DPs received, capped at 100
        heat_detected     BOOLEAN
        score             NUMERIC(10,4)-- heat score from filled pipeline

        The triple (node_id, window_start_utc, window_end_utc) identifies a
        logical analysis window. analysis_run_at distinguishes re-runs.
        """
        ddl = """
            CREATE TABLE IF NOT EXISTS heat_analysis_results (
                id                SERIAL PRIMARY KEY,
                node_id           INTEGER        NOT NULL,
                window_start_utc  TIMESTAMPTZ    NOT NULL,
                window_end_utc    TIMESTAMPTZ    NOT NULL,
                analysis_run_at   TIMESTAMPTZ    NOT NULL
                                  DEFAULT (NOW() AT TIME ZONE 'UTC'),
                data_completeness NUMERIC(6,2),
                heat_detected     BOOLEAN,
                score             NUMERIC(10,4)
            );
        """
        try:
            with psycopg2.connect(self.conn_str) as conn:
                with conn.cursor() as cur:
                    cur.execute(ddl)
                conn.commit()
            logger.info("heat_analysis_results table verified / created.")
        except Exception as e:
            logger.error(f"Could not create heat_analysis_results: {e}")
            raise

    def store_result(
        self,
        node_id: int,
        window_start_ist: datetime,
        window_end_ist: datetime,
        data_completeness: float,
        heat_detected: bool,
        score: Optional[float],
    ) -> None:
        """
        [CHG-4] Insert one result row. IST window endpoints converted to UTC.
        Plain INSERT — each call produces a new row, safe for multiple runs.
        """
        sql = """
            INSERT INTO heat_analysis_results
                (node_id, window_start_utc, window_end_utc,
                 analysis_run_at, data_completeness, heat_detected, score)
            VALUES
                (%(node_id)s,
                 %(window_start_utc)s,
                 %(window_end_utc)s,
                 NOW() AT TIME ZONE 'UTC',
                 %(data_completeness)s,
                 %(heat_detected)s,
                 %(score)s);
        """
        try:
            with psycopg2.connect(self.conn_str) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, {
                        "node_id":           node_id,
                        "window_start_utc":  ist_to_utc(window_start_ist),
                        "window_end_utc":    ist_to_utc(window_end_ist),
                        "data_completeness": data_completeness,
                        "heat_detected":     heat_detected,
                        "score":             score,
                    })
                conn.commit()
            logger.info(
                f"Stored -> node={node_id}  "
                f"window [{window_start_ist} -> {window_end_ist}] IST  "
                f"completeness={data_completeness}%  "
                f"heat={heat_detected}  score={score}"
            )
        except Exception as e:
            logger.error(f"Failed to store result: {e}")
            raise


# ============================================================================
# 4. ANALYSIS ENGINE
# ============================================================================

class DailyAnalysisEngine:

    # -- Feature extraction --------------------------------------------------

    @staticmethod
    def extract_features_and_activity(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()

        if "temperature_value" in df.columns:
            df.loc[
                (df["temperature_value"] < 30.0) | (df["temperature_value"] > 45.0),
                "temperature_value",
            ] = np.nan

        df["mag"]   = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)
        df["vedba"] = (
            np.abs(df["mag"] - df["mag"].rolling(window=50, center=True).mean())
            .fillna(0)
        )
        conditions = [
            df["vedba"] > Config.RES_THRESHOLD,
            df["vedba"] > Config.FEED_THRESHOLD,
        ]
        df["activity_class"] = np.select(conditions, ["RES", "FEED"], default="STANDING")
        return df

    # -- Data-loss across the 24-h window ------------------------------------

    @staticmethod
    def calculate_data_loss(
        df: pd.DataFrame,
        window_start_ist: datetime,
        window_end_ist: datetime,
    ) -> Tuple[float, float, dict]:
        """
        [CHG-5] Data-loss anchored to the rolling window, not a calendar day.

        Expected total DPs = 115,200 (always, because window is always 24 h).
        48 x 30-min slots are indexed from window_start_ist.

        Slot labels include the date component when the window crosses midnight,
        so keys remain unique (e.g. "03-04 23:00" vs "03-05 00:00").

        Returns
        -------
        daily_loss_pct    : float
        data_completeness : float  (capped at 100%)
        slot_stats        : dict { label: {loss_pct, count} }
        """
        slot_starts = pd.date_range(
            start   = pd.Timestamp(window_start_ist),
            periods = Config.SLOTS_PER_WINDOW,
            freq    = f"{Config.SLOT_DURATION_MIN}min",
        )

        # Use date-prefixed labels whenever the window spans two calendar days
        crosses_midnight = window_start_ist.date() != (window_end_ist - timedelta(seconds=1)).date()

        def slot_label(ts: pd.Timestamp) -> str:
            return ts.strftime("%m-%d %H:%M") if crosses_midnight else ts.strftime("%H:%M")

        if df.empty:
            slot_stats = {
                slot_label(ts): {"loss_pct": 100.0, "count": 0}
                for ts in slot_starts
            }
            return 100.0, 0.0, slot_stats

        if not pd.api.types.is_datetime64_any_dtype(df["timestamp_ist"]):
            df = df.copy()
            df["timestamp_ist"] = pd.to_datetime(df["timestamp_ist"])

        total_dps          = len(df)
        expected_total_dps = Config.EXPECTED_DPS_PER_DAY   # 115,200 — window is always 24 h

        daily_loss_pct    = round(100.0 * (1 - total_dps / expected_total_dps), 2)
        data_completeness = min(
            100.0, max(0.0, round(100.0 * total_dps / expected_total_dps, 2))
        )

        if daily_loss_pct < 0:
            logger.warning(
                f"Negative data-loss ({daily_loss_pct}%) — possible duplicate rows."
            )

        df          = df.copy()
        df["slot"]  = df["timestamp_ist"].dt.floor(f"{Config.SLOT_DURATION_MIN}min")
        slot_counts = df.groupby("slot").size()

        slot_stats: dict = {}
        for ts in slot_starts:
            count    = int(slot_counts.get(ts, 0))
            loss_pct = round(100.0 * (1 - count / Config.EXPECTED_DPS_PER_SLOT), 2)
            if loss_pct < 0:
                logger.warning(
                    f"Negative slot loss at {slot_label(ts)} ({loss_pct}%) — "
                    f"possible duplicate rows."
                )
            slot_stats[slot_label(ts)] = {"loss_pct": loss_pct, "count": count}

        return daily_loss_pct, data_completeness, slot_stats

    # -- Resample to 10-min buckets ------------------------------------------

    @staticmethod
    def resample_to_10min(
        df: pd.DataFrame,
        window_start_ist: datetime,
        window_end_ist: datetime,
    ) -> pd.DataFrame:
        """
        [CHG-5] 144 x 10-min buckets anchored to window_start_ist.
        Future buckets are trimmed (edge case: script runs slightly early).
        Missing buckets are NaN for both temp_mean and res_ratio.
        """
        full_index = pd.date_range(
            start   = pd.Timestamp(window_start_ist),
            periods = 144,
            freq    = "10min",
        )

        # Trim buckets that haven't happened yet
        now_ist    = utc_now_ist()
        full_index = full_index[full_index < pd.Timestamp(now_ist)]

        if df.empty:
            return pd.DataFrame(
                index   = full_index,
                columns = ["temp_mean", "res_ratio"],
                dtype   = float,
            )

        idx      = df.set_index("timestamp_ist")
        temp_10  = idx["temperature_value"].resample("10min").mean().rename("temp_mean")
        res_flag = idx["activity_class"].apply(lambda x: 1 if x == "RES" else 0)
        res_10   = res_flag.resample("10min").mean().rename("res_ratio")

        buckets = pd.concat([temp_10, res_10], axis=1).reindex(full_index)
        return buckets

    # -- 10-min backfill (both columns) --------------------------------------

    @staticmethod
    def apply_10min_backfill(buckets_10min: pd.DataFrame) -> pd.DataFrame:
        """
        [CHG-1 / CHG-3] Fill gaps in BOTH temp_mean AND res_ratio by copying
        the nearest preceding block of equal length. Falls back to the next
        available block when the gap is at the very start of the window.

        Filling res_ratio rationale
        ---------------------------
        A communication blackout tells us nothing about what the animal was
        doing — it was still active in some way. Copying the nearest known
        behaviour block is the most conservative, plausible assumption and is
        consistent with the thermal backfill approach. The raw pipeline (NaNs
        preserved) remains available for comparison.
        """
        filled       = buckets_10min.copy()
        missing_mask = filled.isna().all(axis=1)

        if not missing_mask.any():
            return filled

        gap_id = (missing_mask != missing_mask.shift()).cumsum()

        for _, group in filled[missing_mask].groupby(gap_id):
            gap_length    = len(group)
            start_idx_pos = filled.index.get_loc(group.index[0])
            end_idx_pos   = start_idx_pos + gap_length

            # Primary strategy: look backwards
            copy_start_pos = max(0, start_idx_pos - gap_length)
            source_block   = filled.iloc[copy_start_pos:start_idx_pos].dropna(how="all")

            # Fallback: look forwards (gap at start of window)
            if len(source_block) == 0:
                copy_end_pos = min(len(filled), end_idx_pos + gap_length)
                source_block = filled.iloc[end_idx_pos:copy_end_pos].dropna(how="all")

            if len(source_block) > 0:
                tiles_needed = int(np.ceil(gap_length / len(source_block)))
                tiled_block  = pd.concat([source_block] * tiles_needed).iloc[:gap_length]
                # Fill BOTH columns
                filled.loc[group.index, "temp_mean"] = tiled_block["temp_mean"].values
                filled.loc[group.index, "res_ratio"] = tiled_block["res_ratio"].values

        return filled

    # -- Aggregate to hourly -------------------------------------------------

    @staticmethod
    def aggregate_to_hourly(
        buckets_10min: pd.DataFrame,
        window_start_ist: datetime,
    ) -> pd.DataFrame:
        """
        [CHG-5] 24 hourly buckets anchored to window_start_ist.
        The index spans the full 24-h window regardless of calendar days.
        """
        hourly_index = pd.date_range(
            start   = pd.Timestamp(window_start_ist),
            periods = 24,
            freq    = "h",
        )
        hourly = buckets_10min.resample("h").mean()
        return hourly.reindex(hourly_index)

    # -- Anchor --------------------------------------------------------------

    @staticmethod
    def calculate_daily_anchor(hourly_df: pd.DataFrame) -> float:
        """
        [CHG-5] Anchor = MIN temperature across ANCHOR_HOURS (IST hour-of-day).

        Works correctly even when those hours span two calendar days within the
        window (e.g. midnight-run window 00:00->00:00 contains hour 23 on day 1
        and hours 0-3 on day 2 — both are inside the 24-h window).

        Returns np.nan (not 0.0) when no anchor-hour data exists, which causes
        night_spike to also return None rather than a phantom large value.
        """
        mask  = hourly_df.index.hour.isin(Config.ANCHOR_HOURS)
        temps = hourly_df.loc[mask, "temp_mean"].dropna()
        if temps.empty:
            logger.warning("No temperature data in anchor hours — anchor = NaN.")
            return np.nan

        anchor = float(temps.min())
        logger.info(
            f"Anchor: {anchor:.4f} C  "
            f"({len(temps)} readings across hours {Config.ANCHOR_HOURS})"
        )
        return anchor

    # -- Metrics -------------------------------------------------------------

    @staticmethod
    def calculate_metrics(hourly_df: pd.DataFrame, daily_anchor: float) -> dict:
        """
        Night spike  = max(NIGHT_HOURS temps) - anchor
        Persistence  = max rolling-3h mean of res_ratio across the window
        Score        = spike x W_SPIKE  +  persistence x W_PERSIST
        heat_detected = score >= HEAT_DETECTION_THRESHOLD

        Also records global max_temp for logging (not part of score).
        Returns None values for all metrics when anchor is NaN.
        """
        all_temps = hourly_df["temp_mean"].dropna()
        max_temp  = float(all_temps.max()) if not all_temps.empty else None

        if np.isnan(daily_anchor):
            return {
                "daily_anchor_C":  None,
                "max_temp_C":      max_temp,
                "night_spike_C":   None,
                "persistence_pct": None,
                "score":           None,
                "heat_detected":   False,
                "note":            "Insufficient night-window data to compute anchor/spike.",
            }

        night_mask  = hourly_df.index.hour.isin(Config.NIGHT_HOURS)
        night_temps = hourly_df.loc[night_mask, "temp_mean"].dropna()
        night_spike = (
            max(0.0, float(night_temps.max()) - daily_anchor)
            if not night_temps.empty else 0.0
        )

        persistence_series = hourly_df["res_ratio"].rolling(3, min_periods=1).mean()
        persistence_max    = persistence_series.max()
        persistence        = 0.0 if pd.isna(persistence_max) else float(persistence_max)

        score = (night_spike * Config.SCORE_W_SPIKE) + (persistence * Config.SCORE_W_PERSIST)

        return {
            "daily_anchor_C":  round(daily_anchor, 4),
            "max_temp_C":      round(max_temp, 4) if max_temp is not None else None,
            "night_spike_C":   round(night_spike, 4),
            "persistence_pct": round(persistence * 100, 2),
            "score":           round(score, 4),
            "heat_detected":   score >= Config.HEAT_DETECTION_THRESHOLD,
        }


# ============================================================================
# 5. ORCHESTRATOR
# ============================================================================

class DailyAnalysisHandler:
    def __init__(self):
        self.db     = DatabaseManager(Config.PG_CONNECTION_STRING)
        self.engine = DailyAnalysisEngine()

    def process_node(self, node_id: int, window_start_ist: datetime, window_end_ist: datetime, window_label: str) -> dict:
        """Process a single node. Returns the node's result dict."""
        # 1. Fetch data for the window
        raw_df = self.db.fetch_window(node_id, window_start_ist, window_end_ist)
        raw_df["timestamp_ist"] = pd.to_datetime(raw_df["timestamp_ist"])

        # 2. Data-loss (window-anchored 30-min slots)
        daily_loss_pct, data_completeness, slot_stats = self.engine.calculate_data_loss(
            raw_df, window_start_ist, window_end_ist
        )

        if raw_df.empty:
            self.db.ensure_results_table()
            self.db.store_result(
                node_id=node_id, window_start_ist=window_start_ist,
                window_end_ist=window_end_ist, data_completeness=0.0,
                heat_detected=False, score=None,
            )
            return {
                "node_id":           node_id,
                "heat_detected":     False,
                "data_completeness": 0.0,
                "score":             None,
                "message":           "No data found in the analysis window.",
            }

        # 3. Feature extraction
        processed_df = self.engine.extract_features_and_activity(raw_df)

        # [CHG-2] stress_temp = peak solar-hour temp in this window
        Config.STRESS_TEMP = self.db.derive_stress_temp_from_solar(processed_df)

        # 4. Resample to 10-min buckets (RAW: gaps = NaN)
        buckets_raw = self.engine.resample_to_10min(
            processed_df, window_start_ist, window_end_ist
        )

        # 5. Backfill both columns (FILLED)
        buckets_filled = self.engine.apply_10min_backfill(buckets_raw)

        # 6. Hourly aggregation
        hourly_raw    = self.engine.aggregate_to_hourly(buckets_raw,    window_start_ist)
        hourly_filled = self.engine.aggregate_to_hourly(buckets_filled, window_start_ist)

        # 7. Anchors
        anchor_raw    = self.engine.calculate_daily_anchor(hourly_raw)
        anchor_filled = self.engine.calculate_daily_anchor(hourly_filled)

        # 8. Metrics
        metrics_raw    = self.engine.calculate_metrics(hourly_raw,    anchor_raw)
        metrics_filled = self.engine.calculate_metrics(hourly_filled, anchor_filled)

        # Primary outputs (filled pipeline only)
        heat_detected = metrics_filled.get("heat_detected", False)
        score         = metrics_filled.get("score")

        # [CHG-3] Detail metrics -> log only, not in response body
        self._log_detail_metrics(
            window_label      = window_label,
            node_id           = node_id,
            data_completeness = data_completeness,
            metrics_raw       = metrics_raw,
            metrics_filled    = metrics_filled,
            stress_temp       = Config.STRESS_TEMP,
        )

        # [CHG-4] Persist to DB
        self.db.ensure_results_table()
        self.db.store_result(
            node_id           = node_id,
            window_start_ist  = window_start_ist,
            window_end_ist    = window_end_ist,
            data_completeness = data_completeness,
            heat_detected     = heat_detected,
            score             = score,
        )

        return {
            "node_id":           node_id,
            "heat_detected":     heat_detected,
            "data_completeness": data_completeness,
            "score":             score,
        }

    def process(self, event: dict) -> dict:
        # [CHG-5] Resolve the rolling 24-h window
        window_start_ist, window_end_ist = resolve_window(event)
        window_label = (
            f"{window_start_ist.strftime('%Y-%m-%d %H:%M')} -> "
            f"{window_end_ist.strftime('%Y-%m-%d %H:%M')} IST"
        )

        # 0. Scoring weights from DB
        db_config = self.db.fetch_config_parameters()
        for attr in ("score_w_spike", "score_w_persist", "heat_detection_threshold"):
            val = db_config.get(attr)
            if val is not None:
                setattr(Config, attr.upper(), val)
            else:
                logger.info(f"DB CONFIG: Using code default {attr.upper()}={getattr(Config, attr.upper())}")

        # Determine target nodes
        is_poc = getattr(Config, "IS_POC", False)
        if is_poc:
            target_nodes = getattr(Config, "POC_NODES", [])
            logger.info(f"POC MODE ENABLED. Running for {len(target_nodes)} nodes: {target_nodes}")
        else:
            target_nodes = self.db.fetch_active_nodes_in_window(window_start_ist, window_end_ist)
            logger.info(f"PRODUCTION MODE. Found {len(target_nodes)} active nodes in window.")
            
        results = []
        for node_id in target_nodes:
            logger.info(f"\n--- PROCESSING NODE {node_id} ---")
            try:
                res = self.process_node(node_id, window_start_ist, window_end_ist, window_label)
                results.append(res)
            except Exception as e:
                logger.error(f"Failed to process node {node_id}: {e}", exc_info=True)
                results.append({"node_id": node_id, "error": str(e)})

        return self._response(200, {
            "window":            window_label,
            "is_poc":            is_poc,
            "target_nodes":      len(target_nodes),
            "results":           results,
            "parameters_used": {
                "score_w_spike":   Config.SCORE_W_SPIKE,
                "score_w_persist": Config.SCORE_W_PERSIST,
                "heat_detection_threshold": Config.HEAT_DETECTION_THRESHOLD,
                "stress_temp":     Config.STRESS_TEMP,
                "res_threshold":   Config.RES_THRESHOLD,
                "feed_threshold":  Config.FEED_THRESHOLD,
            }
        })

    # -- Detail logger -------------------------------------------------------

    @staticmethod
    def _log_detail_metrics(
        window_label: str,
        node_id: int,
        data_completeness: float,
        metrics_raw: dict,
        metrics_filled: dict,
        stress_temp: float,
    ) -> None:
        """
        [CHG-3] Log extended diagnostics NOT included in the API response:
            - Persistence (%)
            - Global anchor (°C)
            - Max temperature (°C)
            - Night spike (°C)
            - Peak solar-hour env temp / stress_temp (°C)
        RAW and FILLED values printed side-by-side.
        """
        sep = "-" * 70
        logger.info(sep)
        logger.info(f"DETAIL METRICS  |  Node: {node_id}  |  Window: {window_label}")
        logger.info(f"  Data completeness             : {data_completeness:.2f}%")
        logger.info(f"  Peak solar-hour temp           : {stress_temp:.4f} C")
        logger.info(sep)

        def fv(v):
            return f"{v:.4f}" if v is not None else "       N/A"

        rows = [
            ("Global Anchor (C)",  "daily_anchor_C"),
            ("Max Temp      (C)",  "max_temp_C"),
            ("Night Spike   (C)",  "night_spike_C"),
            ("Persistence   (%)",  "persistence_pct"),
            ("Heat Score",         "score"),
            ("Heat Detected",      "heat_detected"),
        ]
        logger.info(f"  {'Metric':<22} | {'RAW':>12} | {'FILLED':>12}")
        logger.info(f"  {'-'*22}-+-{'-'*12}-+-{'-'*12}")
        for label, key in rows:
            rv    = metrics_raw.get(key)
            fv_r  = str(rv)  if isinstance(rv, bool)  else fv(rv)
            fv_fv = metrics_filled.get(key)
            fv_f  = str(fv_fv) if isinstance(fv_fv, bool) else fv(fv_fv)
            logger.info(f"  {label:<22} | {fv_r:>12} | {fv_f:>12}")

        for pipeline, m in [("RAW", metrics_raw), ("FILLED", metrics_filled)]:
            if m and m.get("note"):
                logger.warning(f"  {pipeline} note: {m['note']}")

        logger.info(sep)

    def _response(self, status: int, body: dict) -> dict:
        return {
            "statusCode": status,
            "headers":    {"Content-Type": "application/json"},
            "body":       json.dumps(body, indent=2, default=str),
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
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}


# ============================================================================
# LOCAL TEST RUNNER
# ============================================================================

if __name__ == "__main__":
    # Option A: explicit midnight run
    test_event_midnight = {
        "node_id":          124,
        "window_start_ist": "2026-03-04 00:00:00",   # previous midnight
        "window_end_ist":   "2026-03-05 00:00:00",   # current midnight
    }

    # Option B: explicit noon run
    test_event_noon = {
        "node_id":          124,
        "window_start_ist": "2026-03-04 12:00:00",   # yesterday noon
        "window_end_ist":   "2026-03-05 12:00:00",   # today noon
    }

    # Option C: auto-derive window from current time (for live/scheduled runs)
    test_event_auto = {
        "node_id": 124,
    }

    test_event = test_event_midnight   # change to test other scenarios

    print("\n" + "=" * 70)
    print("ROLLING 24-HOUR HEAT ANALYSIS")
    print("=" * 70)

    resp = handler(test_event, None)

    if resp["statusCode"] == 200:
        body = json.loads(resp["body"])
        print(f"\nNode   : {body['node_id']}")
        print(f"Window : {body.get('window')}")
        print(f"\n{'-'*70}")
        print("PRIMARY RESULT  (filled pipeline)")
        print(f"{'-'*70}")
        print(f"  Heat Detected     : {body.get('heat_detected')}")
        print(f"  Data Completeness : {body.get('data_completeness'):.2f}%")
        sv = body.get("score")
        print(f"  Score             : {sv:.4f}" if sv is not None else "  Score             : N/A")
        if body.get("message"):
            print(f"\n  NOTE: {body['message']}")
        print(f"\n{'-'*70}")
        print("  (Detail metrics — anchor, spike, persistence — in log output above)")
        print(f"{'-'*70}")
    else:
        print("\nError:", resp)

    print("\n" + "=" * 70)


# ============================================================================
# TWO-RUNS-PER-DAY REFERENCE
# ============================================================================
#
# Schedule (all times IST)
# -----------------------------------------------------------------------------
#  Trigger   | window_start_ist        | window_end_ist          | Covers
# -----------+-------------------------+-------------------------+-------------
#  00:00 IST | prev-day  00:00:00      | today      00:00:00     | full yesterday
#  12:00 IST | yesterday 12:00:00      | today      12:00:00     | noon-to-noon
# -----------------------------------------------------------------------------
#
# The 12-hour block from yesterday 00:00 -> yesterday 12:00 appears in BOTH
# runs. This is intentional — it ensures no heat event in that overlap zone
# is missed if the device had a gap during one of the runs.
#
# Data completeness denominator
# ------------------------------
# Both runs compare against the full 115,200 DP baseline (24 h x 80 DPs/min
# x 60 min). No partial-day adjustment is needed — both windows are always
# exactly 24 hours by design.
#
# Night anchor hours in each run
# --------------------------------
# ANCHOR_HOURS = [23, 0, 1, 2, 3]  (IST hour-of-day)
#
#   Midnight run (window 00:00 -> 00:00 next day):
#     Hours 0,1,2,3  -> fall at the START of the window (early morning)
#     Hour 23        -> falls at the END of the window (late evening)
#     Both ends are within the 24-h window. All anchor hours are present.
#
#   Noon run (window 12:00 -> 12:00 next day):
#     Hours 23, 0, 1, 2, 3 -> all fall in the MIDDLE of the window (overnight)
#     All anchor hours are present.
#
# stress_temp / solar hours
# --------------------------
# SOLAR_HOURS = [11, 12, 13, 14, 15, 16] (IST)
# Both windows always contain at least some solar hours (11:00-16:00 IST).
# The midnight run captures the full solar block (it sits entirely within the
# window). The noon run also captures the full solar block of the second day.
#
# Results table
# -------------
# Each run inserts one row with its own analysis_run_at timestamp (UTC).
# To retrieve both runs for a node on a given day:
#
#   SELECT * FROM heat_analysis_results
#   WHERE node_id = 124
#     AND window_start_utc >= '2026-03-03 18:30:00'  -- 2026-03-04 00:00 IST
#   ORDER BY analysis_run_at;
#
# To get only the latest per window:
#
#   SELECT DISTINCT ON (node_id, window_start_utc, window_end_utc) *
#   FROM heat_analysis_results
#   ORDER BY node_id, window_start_utc, window_end_utc, analysis_run_at DESC;
#
# No structural changes are required to support the two-run-per-day schedule.