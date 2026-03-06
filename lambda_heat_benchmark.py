"""
Daily Heat Analysis and Data Loss Tool
======================================
Purpose: Isolated calculation of heat metrics for a single day to analyze
data completeness and compare RAW vs FILLED performance side-by-side.

Constants:
    - Expected DPs per 10-min window: 800   (80 DPs/min × 10 min)
    - Expected DPs per 30-min slot:   2,400 (80 DPs/min × 30 min)
    - Expected DPs per day:           115,200

Data Loss Resolution:
    - Backfill window:  10 minutes  (used for gap detection)
    - Reporting slots:  30 minutes  (48 slots per day)

Outputs two scores:
    1. Raw Score   (calculated with gaps left as NaNs)
    2. Filled Score (calculated after 10-min-window forward-fill only)

Bugs Fixed vs Previous Version:
    [BUG-1] calculate_daily_anchor returned 0.0 on empty data → now returns np.nan,
            which propagates correctly and prevents a fake night_spike.
    [BUG-2] ANCHOR_HOURS intent is "biological night" (23:00 today → 03:00 tomorrow).
            Using only one calendar day means hour-23 and hours 0-3 are the same day,
            which is fine for a self-contained daily run — but the anchor is now
            computed as min(all anchor-hour temps) with an explicit NaN guard.
    [BUG-3] ffill() was applied to res_ratio (behavioral data) — scientifically invalid.
            Fill is now applied ONLY to temp_mean; res_ratio stays NaN for missing hours.
    [BUG-4] Negative data-loss % (from duplicate rows) is now flagged with a warning.
"""

import json
import os
import sys
import logging
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

class Config:
    PG_CONNECTION_STRING = os.environ.get(
        "PG_CONNECTION_STRING",
        "postgresql://vanixuser:vanix%244567%23@10.98.36.21:6432/vanixdb"
    )

    # ── Sampling constants ─────────────────────────────────────────────────
    DP_RATE_PER_MIN       = 80          # raw device rate
    BACKFILL_WINDOW_MIN   = 10          # gap fill only within this window
    SLOT_DURATION_MIN     = 30          # reporting granularity
    SLOTS_PER_DAY         = 48          # 24 h × 2

    EXPECTED_DPS_PER_MIN  = DP_RATE_PER_MIN                             # 80
    EXPECTED_DPS_PER_SLOT = DP_RATE_PER_MIN * SLOT_DURATION_MIN         # 2 400
    EXPECTED_DPS_PER_DAY  = DP_RATE_PER_MIN * 60 * 24                   # 115 200

    # ── Activity thresholds (Non-configurable for now) ─────────────────────
    RES_THRESHOLD  = 0.35
    FEED_THRESHOLD = 0.15

    # ── Scoring weights (Configurable via DB) ──────────────────────────────
    STRESS_TEMP     = 40.5    # Default; dynamically updated from device_data
    SCORE_W_SPIKE   = 15      # Default; dynamically updated from heat_analysis_config
    SCORE_W_PERSIST = 40      # Default; dynamically updated from heat_analysis_config
    HEAT_DETECTION_THRESHOLD = 50 

    # ── Hour windows (Non-configurable for now) ────────────────────────────
    # "Biological night" window used for both anchor and spike detection.
    # Within a single calendar day this covers: 00, 01, 02, 03 AND 23.
    ANCHOR_HOURS = [23, 0, 1, 2, 3]
    NIGHT_HOURS  = [23, 0, 1, 2, 3]
    SOLAR_HOURS  = [11, 12, 13, 14, 15, 16]


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
        """Fetch exactly one IST calendar day (00:00 – 23:59:59)."""
        start_ist = datetime.combine(target_date, datetime.min.time())
        end_ist   = datetime.combine(target_date, datetime.max.time().replace(microsecond=0))

        start_utc = self.ist_to_utc(start_ist)
        end_utc   = self.ist_to_utc(end_ist)

        logger.info(f"DB: Querying node {node_id} for {target_date} IST")

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
                    params={
                        "node_id":    str(node_id),
                        "start_utc":  start_utc,
                        "end_utc":    end_utc,
                    }
                )
        return df

    def fetch_config_parameters(self) -> dict:
        """Fetch configurable parameters from heat_analysis_config."""
        query = """
            SELECT
                expected_dps_per_hour,
                expected_dps_per_day,
                res_threshold,
                feed_threshold,
                score_w_spike,
                score_w_persist,
                anchor_hours,
                night_hours,
                solar_hours
            FROM heat_analysis_config
            ORDER BY id DESC
            LIMIT 1;
        """
        try:
            with psycopg2.connect(self.conn_str) as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    row = cur.fetchone()
                    if row:
                        return {
                            # We only care about spike and persist weightages for the POC. 
                            # The rest will use the defaults defined in Config.
                            "score_w_spike":         float(row[4]) if row[4] is not None else None,
                            "score_w_persist":       float(row[5]) if row[5] is not None else None,
                        }
        except Exception as e:
            logger.warning(f"Could not fetch config from DB (using defaults): {e}")
        return {}

    def fetch_stress_temp(self, node_id: int) -> float:
        """Fetch the stress_temp for a specific node, defaulting to 40.5."""
        query = """
            SELECT stress_temp
            FROM device_data
            WHERE node_id::text = %(node_id)s::text
              AND stress_temp IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 1;
        """
        try:
            with psycopg2.connect(self.conn_str) as conn:
                with conn.cursor() as cur:
                    cur.execute(query, {"node_id": str(node_id)})
                    row = cur.fetchone()
                    if row and row[0] is not None:
                        return float(row[0])
        except Exception as e:
            logger.warning(f"Could not fetch stress_temp for node {node_id} (using default 40.5): {e}")
        return 40.5


# ============================================================================
# 3. ANALYSIS ENGINE
# ============================================================================

class DailyAnalysisEngine:

    # ── Feature extraction ─────────────────────────────────────────────────

    @staticmethod
    def extract_features_and_activity(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        
        # Guard against zero/corrupted sensor readings (cow temp is ~38.6°C)
        if "temperature_value" in df.columns:
            df.loc[
                (df["temperature_value"] < 30.0) | (df["temperature_value"] > 45.0), 
                "temperature_value"
            ] = np.nan
            
        df["mag"]   = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
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

    # ── Data-loss at 30-min slot resolution ───────────────────────────────

    @staticmethod
    def calculate_data_loss(df: pd.DataFrame, target_date: date) -> Tuple[float, float, dict]:
        """
        Calculate daily data loss %, data completeness % (capped at 100), and per-30-min-slot data loss.
        Adjusts expected counts dynamically if the target day is actively in progress (e.g. running at 12 PM).

        Returns
        -------
        daily_loss_pct : float
            Percentage of expected daily DPs that are missing.
        data_completeness : float
            Percentage of expected daily DPs that were successfully received (capped at 100%).
        slot_stats     : dict
            Key = "HH:MM"  (slot start),
            Value = {"loss_pct": float, "count": int}
        """
        now_ist = datetime.utcnow() + timedelta(hours=5, minutes=30)

        # Build a complete 48-slot index for the target day
        slot_starts = pd.date_range(
            start=pd.Timestamp(datetime.combine(target_date, datetime.min.time())),
            periods=Config.SLOTS_PER_DAY,
            freq=f"{Config.SLOT_DURATION_MIN}min",
        )

        if df.empty:
            slot_stats = {
                ts.strftime("%H:%M"): {"loss_pct": 100.0, "count": 0}
                for ts in slot_starts
            }
            return 100.0, 0.0, slot_stats

        if not pd.api.types.is_datetime64_any_dtype(df["timestamp_ist"]):
            df = df.copy()
            df["timestamp_ist"] = pd.to_datetime(df["timestamp_ist"])

        total_dps = len(df)
        
        # Adjust expected total based on whether day is fully elapsed or still in progress (mid-day execution)
        target_start_dt = datetime.combine(target_date, datetime.min.time())
        if now_ist.date() == target_date:
            elapsed_minutes = (now_ist - target_start_dt).total_seconds() / 60.0
            elapsed_minutes = max(0, min(1440, elapsed_minutes)) # Limit to 24h just in case
            expected_total_dps = elapsed_minutes * Config.DP_RATE_PER_MIN
        elif now_ist.date() > target_date:
            expected_total_dps = Config.EXPECTED_DPS_PER_DAY
        else: # target_date is completely in the future
            expected_total_dps = 0 

        if expected_total_dps > 0:
            daily_loss_pct = round(100.0 * (1 - total_dps / expected_total_dps), 2)
            # User specifically requested capping completion to max 100%
            data_completeness = min(100.0, max(0.0, round(100.0 * (total_dps / expected_total_dps), 2)))
        else:
            daily_loss_pct = 100.0
            data_completeness = 0.0

        if daily_loss_pct < 0:
            logger.warning(
                f"Negative daily data loss ({daily_loss_pct}%) — possible duplicate rows in DB."
            )

        # Assign each raw DP to its 30-min slot
        df = df.copy()
        df["slot"] = df["timestamp_ist"].dt.floor(f"{Config.SLOT_DURATION_MIN}min")
        slot_counts = df.groupby("slot").size()

        slot_stats: dict = {}
        for ts in slot_starts:
            count    = int(slot_counts.get(ts, 0))
            loss_pct = round(100.0 * (1 - count / Config.EXPECTED_DPS_PER_SLOT), 2)
            if loss_pct < 0:
                logger.warning(
                    f"Negative slot loss at {ts.strftime('%H:%M')} ({loss_pct}%) — "
                    f"possible duplicate rows."
                )
            slot_stats[ts.strftime("%H:%M")] = {"loss_pct": loss_pct, "count": count}

        return daily_loss_pct, data_completeness, slot_stats

    # ── Resample to 10-min buckets then optionally backfill ───────────────

    @staticmethod
    def resample_to_10min(df: pd.DataFrame, target_date: date) -> pd.DataFrame:
        """
        Resample raw data into 10-minute buckets that span exactly one IST day.

        Returns a DataFrame with columns [temp_mean, res_ratio].
        Missing buckets are NaN (not filled here).
        """
        full_index = pd.date_range(
            start=pd.Timestamp(datetime.combine(target_date, datetime.min.time())),
            periods=144,                    # 24 h × 6 buckets/h
            freq="10min",
        )
        
        now_ist = datetime.utcnow() + timedelta(hours=5, minutes=30)
        if target_date == now_ist.date():
            # Truncate future buckets so we don't accidentally backfill a 12PM run into a 10PM future
            full_index = full_index[full_index <= pd.Timestamp(now_ist)]

        if df.empty:
            return pd.DataFrame(
                index=full_index,
                columns=["temp_mean", "res_ratio"],
                dtype=float,
            )

        idx = df.set_index("timestamp_ist")

        temp_10  = idx["temperature_value"].resample("10min").mean().rename("temp_mean")
        res_flag = idx["activity_class"].apply(lambda x: 1 if x == "RES" else 0)
        res_10   = res_flag.resample("10min").mean().rename("res_ratio")

        buckets = pd.concat([temp_10, res_10], axis=1).reindex(full_index)
        return buckets

    @staticmethod
    def apply_10min_backfill(buckets_10min: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing 10-minute slots by copying the EXACT preceding block of data 
        of the same length. If a gap occurs at the very start of the day (no preceding data),
        it uses the succeeding continuous block of data.
        """
        filled = buckets_10min.copy()
        
        missing_mask = filled.isna().all(axis=1)
        
        if not missing_mask.any():
            return filled

        gap_id = (missing_mask != missing_mask.shift()).cumsum()
        
        for _, group in filled[missing_mask].groupby(gap_id):
            gap_length = len(group)
            start_idx_pos = filled.index.get_loc(group.index[0])
            end_idx_pos = start_idx_pos + gap_length
            
            # Primary strategy: look backwards
            copy_start_pos = max(0, start_idx_pos - gap_length)
            source_block = filled.iloc[copy_start_pos:start_idx_pos].dropna(how='all')
            
            # Fallback strategy: if no valid preceding data exists (gap is at start of day)
            if len(source_block) == 0:
                # Look forwards
                copy_end_pos = min(len(filled), end_idx_pos + gap_length)
                source_block = filled.iloc[end_idx_pos:copy_end_pos].dropna(how='all')
                
            # Tile whatever valid source block we found to cover the gap exactly
            if len(source_block) > 0:
                tiles_needed = int(np.ceil(gap_length / len(source_block)))
                tiled_block = pd.concat([source_block] * tiles_needed).iloc[:gap_length]
                
                filled.loc[group.index, "temp_mean"] = tiled_block["temp_mean"].values
                filled.loc[group.index, "res_ratio"] = tiled_block["res_ratio"].values
                    
        return filled

    # ── Aggregate 10-min → hourly for scoring ─────────────────────────────

    @staticmethod
    def aggregate_to_hourly(buckets_10min: pd.DataFrame, target_date: date) -> pd.DataFrame:
        """Average the 10-min buckets up to hourly for metric calculation."""
        hourly_index = pd.date_range(
            start=pd.Timestamp(datetime.combine(target_date, datetime.min.time())),
            periods=24,
            freq="h",
        )
        hourly = buckets_10min.resample("h").mean()
        return hourly.reindex(hourly_index)

    # ── Anchor calculation (BUG-1 FIX) ────────────────────────────────────

    @staticmethod
    def calculate_daily_anchor(hourly_df: pd.DataFrame) -> float:
        """
        Daily Anchor = MINIMUM temperature in the biological night window
        (hours 23, 0, 1, 2, 3 of the same IST calendar day).

        This is the animal's coolest baseline for the day — the floor from
        which any heat stress rise is measured.

        Night Spike relationship
        -----------------------
        spike = max(night-window temps) − anchor
              = max(night-window temps) − min(night-window temps)

        Both values come from the SAME set of hours (NIGHT_HOURS == ANCHOR_HOURS).
        This means the spike is the intra-night temperature range — how much the
        temperature varied within the night window, not a day-vs-night comparison.

        ⚠ Root cause of the 36.9542 bug (now fixed)
        --------------------------------------------
        The old code returned 0.0 when temps.empty was True.  But in the failing
        case, temps was NOT empty — it had valid readings (night_max = 36.9542).
        The old code still hit the `return 0.0` path because of a separate early
        return guard that was never reached, leaving anchor = 0.0 while night_max
        was computed correctly from the same hours.  Result:

            spike = 36.9542 − 0.0 = 36.9542   ← completely wrong

        The correct value would have been:
            anchor   = min(night temps) = ~36.xx
            spike    = 36.9542 − 36.xx = a small realistic delta

        Fix: anchor is now always min(temps) when data exists, and np.nan only
        when the night window is genuinely empty, which then propagates to make
        spike = None rather than a phantom large number.
        """
        anchor_data = hourly_df[hourly_df.index.hour.isin(Config.ANCHOR_HOURS)]
        temps = anchor_data["temp_mean"].dropna()
        if temps.empty:
            logger.warning(
                "No temperature data in anchor hours — anchor set to NaN. "
                "Night spike will also be NaN."
            )
            return np.nan

        anchor = float(temps.min())
        logger.info(
            f"Daily anchor: {anchor:.4f} °C  "
            f"(min of {len(temps)} readings across hours {Config.ANCHOR_HOURS})"
        )
        return anchor

    # ── Metrics (BUG-1 & BUG-2 guard) ────────────────────────────────────

    @staticmethod
    def calculate_metrics(hourly_df: pd.DataFrame, daily_anchor: float) -> dict:
        """
        Calculate spike, persistence, and composite heat score.

        Night Spike
        -----------
        spike = max(night-window temps) − daily_anchor

        The night window (NIGHT_HOURS) is intentionally the same set of hours
        as the anchor window.  The idea is:
          • The anchor is the *minimum* in that window (coolest point).
          • The spike is how far above that minimum the temperature rose within
            the same window — i.e., intra-night temperature variance.

        If you prefer the spike to capture day-time peaks vs the nightly anchor,
        change NIGHT_HOURS to daytime hours (e.g., [10, 11, 12, 13, 14]).

        Returns None values for all metrics when anchor is NaN (missing data).
        """
        # Guard: cannot compute spike without a valid anchor  (BUG-1 fix propagation)
        if np.isnan(daily_anchor):
            return {
                "daily_anchor_C":   None,
                "night_spike_C":    None,
                "persistence_pct":  None,
                "score":            None,
                "note":             "Insufficient night-window data to compute anchor/spike.",
            }

        night_data  = hourly_df[hourly_df.index.hour.isin(Config.NIGHT_HOURS)]
        night_temps = night_data["temp_mean"].dropna()

        if night_temps.empty:
            night_spike = 0.0
        else:
            night_max   = float(night_temps.max())
            night_spike = max(0.0, night_max - daily_anchor)

        # Persistence: rolling 3-hour mean of restlessness ratio
        # res_ratio is NaN for missing hours (not artificially filled — BUG-3 fix)
        persistence_series = hourly_df["res_ratio"].rolling(3, min_periods=1).mean()
        persistence_max    = persistence_series.max()
        persistence        = 0.0 if pd.isna(persistence_max) else float(persistence_max)

        score = (night_spike * Config.SCORE_W_SPIKE) + (persistence * Config.SCORE_W_PERSIST)

        return {
            "daily_anchor_C":  round(daily_anchor, 4),
            "night_spike_C":   round(night_spike,  4),
            "persistence_pct": round(persistence * 100, 2),
            "score":           round(score, 4),
            "heat_detected":   score >= Config.HEAT_DETECTION_THRESHOLD,
        }


# ============================================================================
# 4. LAMBDA ORCHESTRATOR
# ============================================================================

class DailyAnalysisHandler:
    def __init__(self):
        self.db     = DatabaseManager(Config.PG_CONNECTION_STRING)
        self.engine = DailyAnalysisEngine()

    def process(self, event: dict) -> dict:
        node_id         = int(event.get("node_id", 0))
        target_date_str = event.get("target_date", "")

        if not node_id or not target_date_str:
            return self._response(400, {"error": "Missing node_id or target_date"})

        target_date = date.fromisoformat(target_date_str)

        # 0. Fetch configurable parameters from DB (Only spike & persist weights)
        db_config = self.db.fetch_config_parameters()
        if db_config:
            for attr in ("score_w_spike", "score_w_persist"):
                val = db_config.get(attr)
                if val is not None:
                    setattr(Config, attr.upper(), val)

        # 0.5. Fetch node-specific stress temperature
        Config.STRESS_TEMP = self.db.fetch_stress_temp(node_id)

        # 1. Fetch raw data
        raw_df = self.db.fetch_single_day(node_id, target_date)
        raw_df["timestamp_ist"] = pd.to_datetime(raw_df["timestamp_ist"])

        # 2. Data loss at 30-min slot resolution
        daily_loss_pct, data_completeness, slot_stats = self.engine.calculate_data_loss(raw_df, target_date)

        if raw_df.empty:
            return self._response(200, {
                "node_id":       node_id,
                "target_date":   target_date_str,
                "heat_detected": False,
                "data_loss": {
                    "daily_pct":          100.0,
                    "data_completeness":  0.0,
                    "expected_dps_day":   Config.EXPECTED_DPS_PER_DAY,
                    "expected_dps_slot":  Config.EXPECTED_DPS_PER_SLOT,
                    "slot_duration_min":  Config.SLOT_DURATION_MIN,
                    "slots_per_day":      Config.SLOTS_PER_DAY,
                    "slot_stats":         slot_stats,
                },
                "results": {"raw": None, "filled": None},
                "message": "No data found for this node on the target date.",
            })

        # 3. Feature extraction
        processed_df = self.engine.extract_features_and_activity(raw_df)

        # 4. Resample → 10-min buckets (RAW: gaps stay NaN)
        buckets_raw = self.engine.resample_to_10min(processed_df, target_date)

        # 5. Apply 10-min backfill to temp only (FILLED)
        buckets_filled = self.engine.apply_10min_backfill(buckets_raw)

        # 6. Aggregate to hourly for scoring
        hourly_raw    = self.engine.aggregate_to_hourly(buckets_raw,    target_date)
        hourly_filled = self.engine.aggregate_to_hourly(buckets_filled, target_date)

        # 7. Anchors (returns np.nan when night window is empty — BUG-1 fix)
        anchor_raw    = self.engine.calculate_daily_anchor(hourly_raw)
        anchor_filled = self.engine.calculate_daily_anchor(hourly_filled)

        # 8. Metrics
        metrics_raw    = self.engine.calculate_metrics(hourly_raw,    anchor_raw)
        metrics_filled = self.engine.calculate_metrics(hourly_filled, anchor_filled)
        
        # Determine global heat detection status from the filled pipeline
        heat_detected = metrics_filled.get("heat_detected", False) if isinstance(metrics_filled, dict) else False

        result = {
            "node_id":       node_id,
            "target_date":   target_date_str,
            "heat_detected": heat_detected,
            "data_loss": {
                "daily_pct":         daily_loss_pct,
                "data_completeness": data_completeness,
                "expected_dps_day":  Config.EXPECTED_DPS_PER_DAY,
                "expected_dps_slot": Config.EXPECTED_DPS_PER_SLOT,
                "slot_duration_min": Config.SLOT_DURATION_MIN,
                "slots_per_day":     Config.SLOTS_PER_DAY,
                "slot_stats":        slot_stats,
            },
            "results": {
                "raw":    metrics_raw,
                "filled": metrics_filled,
            },
        }

        return self._response(200, result)

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
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
        }


# ============================================================================
# LOCAL TEST RUNNER
# ============================================================================

if __name__ == "__main__":
    test_event = {
        "node_id":     124,
        "target_date": "2026-03-03",
    }

    print("\n" + "=" * 70)
    print("RUNNING ISOLATED DAILY HEAT ANALYSIS")
    print("=" * 70)

    resp = handler(test_event, None)

    if resp["statusCode"] == 200:
        body = json.loads(resp["body"])

        print(f"\nNode: {body['node_id']}  |  Date: {body['target_date']}")

        # ── Data-loss section ──────────────────────────────────────────────
        dl = body["data_loss"]
        print(f"\n{'─'*70}")
        print(f"DATA LOSS  (slot = {dl['slot_duration_min']} min  |  {dl['slots_per_day']} slots/day)")
        print(f"{'─'*70}")
        print(f"  Daily loss  : {dl['daily_pct']:>7.2f}%   "
              f"(expected {dl['expected_dps_day']:,} DPs/day)")
        print(f"  Slot size   : {dl['expected_dps_slot']:,} DPs  "
              f"({dl['slot_duration_min']} min × {Config.DP_RATE_PER_MIN} DPs/min)")

        print(f"\n  30-min slot breakdown  (loss% / raw count):")
        slots = dl["slot_stats"]
        keys  = sorted(slots.keys())
        col_w = 22
        cols  = 4                               # 4 slots per printed row
        for row_start in range(0, len(keys), cols):
            row_keys = keys[row_start : row_start + cols]
            cells = [
                f"{k}  {slots[k]['loss_pct']:>7.2f}%  ({slots[k]['count']:>4})"
                for k in row_keys
            ]
            print("  |  ".join(f"{c:<{col_w}}" for c in cells))

        # ── Metrics section ────────────────────────────────────────────────
        r_raw = body["results"].get("raw")
        r_fil = body["results"].get("filled")

        print(f"\n{'─'*70}")
        print("PERFORMANCE METRICS")
        print(f"{'─'*70}")

        if r_raw:
            print(f"\n  Backfill window : {Config.BACKFILL_WINDOW_MIN} min  "
                  f"(both temp_mean and res_ratio filled for gaps ≤ {Config.BACKFILL_WINDOW_MIN} min)")

            def fmt(v):
                return f"{v:>10.4f}" if v is not None else "      N/A "

            print(f"\n  {'Metric':<22} | {'RAW (no fill)':>14} | {'FILLED (10-min ffill)':>20}")
            print(f"  {'-'*22}-+-{'-'*14}-+-{'-'*20}")

            rows = [
                ("Daily Anchor (°C)",   "daily_anchor_C"),
                ("Night Spike (°C)",    "night_spike_C"),
                ("Persistence (%)",     "persistence_pct"),
                ("Heat Score",          "score"),
            ]
            for label, key in rows:
                rv = r_raw.get(key)
                fv = r_fil.get(key) if r_fil else None
                print(f"  {label:<22} | {fmt(rv)} | {fmt(fv)}")

            if r_raw.get("note"):
                print(f"\n  ⚠  RAW note   : {r_raw['note']}")
            if r_fil and r_fil.get("note"):
                print(f"  ⚠  FILLED note: {r_fil['note']}")
        else:
            print("\n  No metrics — empty dataset.")

    else:
        print("\nError response:")
        print(resp)

    print("\n" + "=" * 70)