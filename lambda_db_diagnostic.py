"""
Database Diagnostic Lambda
==========================
Purpose: Understand the PostgreSQL database completely before building any logic.
Covers:
  1. Table schema (columns, types)
  2. Indexes
  3. Total row counts (all nodes + per-node breakdown)
  4. Date range of data
  5. Exact sampling frequency (timestamp deltas between consecutive rows)
  6. Day-wise row counts for a target node
  7. Derived: expected rows/sec, rows/day, and estimated data completeness
"""

import os
import sys
import json
import logging
import time
import psycopg2
from contextlib import closing
from datetime import datetime

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PG_CONNECTION_STRING = os.environ.get(
    "PG_CONNECTION_STRING",
    "postgresql://vanixuser:vanix%244567%23@10.98.36.21:6432/vanixdb"
)
TARGET_NODE = "124"          # Node we want to deep-dive into
DELTA_SAMPLE_SIZE = 100      # How many consecutive rows to sample for frequency

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def handler(event, context):
    start = time.time()
    logger.info("=" * 70)
    logger.info("DATABASE DIAGNOSTIC - FULL ANALYSIS")
    logger.info("=" * 70)

    try:
        with closing(psycopg2.connect(PG_CONNECTION_STRING, connect_timeout=15)) as conn:
            conn.set_session(readonly=True)
            cur = conn.cursor()

            result = {}

            # ------------------------------------------------------------------
            # 1. TABLE SCHEMA
            # ------------------------------------------------------------------
            cur.execute("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name = 'device_data'
                ORDER BY ordinal_position;
            """)
            result["schema"] = [
                {"column": r[0], "type": r[1], "nullable": r[2], "default": str(r[3]) if r[3] else None}
                for r in cur.fetchall()
            ]

            # ------------------------------------------------------------------
            # 2. INDEXES
            # ------------------------------------------------------------------
            cur.execute("""
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE tablename = 'device_data';
            """)
            result["indexes"] = [
                {"name": r[0], "definition": r[1]}
                for r in cur.fetchall()
            ]

            # ------------------------------------------------------------------
            # 3. OVERALL TABLE STATS
            # ------------------------------------------------------------------
            cur.execute("SELECT count(*) FROM device_data;")
            result["total_rows"] = cur.fetchone()[0]

            cur.execute("SELECT MIN(timestamp), MAX(timestamp) FROM device_data;")
            mn, mx = cur.fetchone()
            result["overall_date_range"] = {
                "earliest_utc": str(mn),
                "latest_utc": str(mx),
            }

            # ------------------------------------------------------------------
            # 4. PER-NODE ROW COUNTS (top 20 nodes by volume)
            # ------------------------------------------------------------------
            cur.execute("""
                SELECT node_id, count(*) AS cnt
                FROM device_data
                GROUP BY node_id
                ORDER BY cnt DESC
                LIMIT 20;
            """)
            result["top_nodes_by_volume"] = [
                {"node_id": str(r[0]), "rows": r[1]}
                for r in cur.fetchall()
            ]

            # ------------------------------------------------------------------
            # 5. TARGET NODE: DATE RANGE & TOTAL
            # ------------------------------------------------------------------
            cur.execute(f"""
                SELECT count(*), MIN(timestamp), MAX(timestamp)
                FROM device_data
                WHERE node_id::text = '{TARGET_NODE}';
            """)
            n_count, n_min, n_max = cur.fetchone()
            result["target_node"] = {
                "node_id": TARGET_NODE,
                "total_rows": n_count,
                "first_record_utc": str(n_min),
                "last_record_utc": str(n_max),
            }

            # ------------------------------------------------------------------
            # 6. SAMPLING FREQUENCY (consecutive timestamp deltas)
            #    We sample from the MIDDLE of the dataset to avoid edge effects
            # ------------------------------------------------------------------
            cur.execute(f"""
                SELECT timestamp
                FROM device_data
                WHERE node_id::text = '{TARGET_NODE}'
                ORDER BY timestamp ASC
                OFFSET {n_count // 2}
                LIMIT {DELTA_SAMPLE_SIZE};
            """)
            ts_list = [r[0] for r in cur.fetchall()]

            deltas = []
            for i in range(1, len(ts_list)):
                diff = (ts_list[i] - ts_list[i - 1]).total_seconds()
                deltas.append(round(diff, 4))

            if deltas:
                avg_delta = round(sum(deltas) / len(deltas), 4)
                min_delta = min(deltas)
                max_delta = max(deltas)
                rows_per_sec = round(1.0 / avg_delta, 2) if avg_delta > 0 else 0
                rows_per_min = round(rows_per_sec * 60, 2)
                rows_per_hour = round(rows_per_sec * 3600, 2)
                rows_per_day = round(rows_per_sec * 86400, 2)
            else:
                avg_delta = min_delta = max_delta = 0
                rows_per_sec = rows_per_min = rows_per_hour = rows_per_day = 0

            result["target_node"]["frequency_analysis"] = {
                "sample_size": len(deltas),
                "avg_delta_seconds": avg_delta,
                "min_delta_seconds": min_delta,
                "max_delta_seconds": max_delta,
                "derived_rows_per_second": rows_per_sec,
                "derived_rows_per_minute": rows_per_min,
                "derived_rows_per_hour": rows_per_hour,
                "derived_rows_per_day": rows_per_day,
                "sample_deltas_first_20": deltas[:20],
            }

            # ------------------------------------------------------------------
            # 7. DAY-WISE ROW COUNTS for target node (last 30 days)
            # ------------------------------------------------------------------
            cur.execute(f"""
                SELECT
                    DATE(timestamp + INTERVAL '5 hours 30 minutes') AS day_ist,
                    COUNT(*) AS rows_per_day
                FROM device_data
                WHERE node_id::text = '{TARGET_NODE}'
                GROUP BY day_ist
                ORDER BY day_ist DESC
                LIMIT 30;
            """)
            daily_rows = cur.fetchall()

            expected_per_day = rows_per_day  # from the frequency analysis above
            daily_breakdown = []
            for row in daily_rows:
                day_str = str(row[0])
                actual = row[1]
                pct = round((actual / expected_per_day) * 100, 2) if expected_per_day > 0 else 0
                daily_breakdown.append({
                    "day_ist": day_str,
                    "actual_rows": actual,
                    "expected_rows": int(expected_per_day),
                    "completeness_pct": pct,
                })

            result["target_node"]["daily_breakdown"] = daily_breakdown

            cur.close()

            duration = round(time.time() - start, 2)
            result["diagnostic_duration_seconds"] = duration

            logger.info(f"✅ DIAGNOSTIC COMPLETE in {duration}s")

            return {
                "statusCode": 200,
                "body": json.dumps(result, default=str),
            }

    except Exception as e:
        logger.error(f"❌ DIAGNOSTIC FAILED: {e}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e), "duration": round(time.time() - start, 2)}),
        }
