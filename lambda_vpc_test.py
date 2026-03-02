import os
import sys
import logging
import psycopg2
import time
from contextlib import closing
import json

def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

PG_CONNECTION_STRING = os.environ.get(
    "PG_CONNECTION_STRING",
    "postgresql://vanixuser:vanix%244567%23@10.98.36.21:6432/vanixdb"
)

def handler(event, context):
    logger.info("=" * 70)
    logger.info("HEAT CYCLE BENCHMARK LAMBDA - VPC DIAGNOSTICS")
    logger.info("=" * 70)
    
    start = time.time()
    logger.info(f"Attempting connection to DB at 10.98.36.21:6432...")
    
    try:
        # Give it a 10 second timeout so we don't wait 3 minutes
        with closing(psycopg2.connect(PG_CONNECTION_STRING, connect_timeout=10)) as conn:
            duration = time.time() - start
            logger.info(f"✅ CONNECTION SUCCESSFUL in {duration:.2f} seconds.")
            
            with conn.cursor() as cur:
                # 1. Total rows
                cur.execute("SELECT count(*) FROM device_data;")
                total_count = cur.fetchone()[0]
                
                # 2. Check indexes
                index_query = """
                SELECT indexname, indexdef 
                FROM pg_indexes 
                WHERE tablename = 'device_data';
                """
                cur.execute(index_query)
                indexes = cur.fetchall()
                index_list = [{"name": idx[0], "def": idx[1]} for idx in indexes]

                # 3. Schema Information
                schema_query = """
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'device_data';
                """
                cur.execute(schema_query)
                schema_info = [{"column": row[0], "type": row[1]} for row in cur.fetchall()]

                # 4. Find exact row frequency (rows per second/minute) for Node 124
                # We check the most active minute to find the theoretical max "burst" rate
                frequency_query = """
                SELECT 
                    DATE_TRUNC('minute', timestamp) AS minute_bucket,
                    COUNT(*) as rows_per_minute
                FROM device_data
                WHERE node_id::text = '124'
                GROUP BY minute_bucket
                ORDER BY rows_per_minute DESC
                LIMIT 5;
                """
                cur.execute(frequency_query)
                samples = cur.fetchall()
                freq_data = [{"minute": str(row[0]), "rows_per_minute": row[1], "rows_per_second": round(row[1]/60.0, 2)} for row in samples]
                
                # 5. Day-wise data volume for Node 124
                daily_query = """
                SELECT 
                    DATE(timestamp) AS day_bucket,
                    COUNT(*) as rows_per_day
                FROM device_data
                WHERE node_id::text = '124'
                GROUP BY day_bucket
                ORDER BY day_bucket DESC
                LIMIT 7;
                """
                cur.execute(daily_query)
                daily_samples = cur.fetchall()
                daily_data = [{"day": str(row[0]), "rows_per_day": row[1]} for row in daily_samples]
                
                # 6. Sample exact timestamp deltas
                hz_query = """
                SELECT timestamp 
                FROM device_data 
                WHERE node_id::text = '124' 
                ORDER BY timestamp DESC 
                LIMIT 20;
                """
                cur.execute(hz_query)
                timestamps = [row[0] for row in cur.fetchall()]
                time_diffs = []
                for i in range(len(timestamps)-1):
                    # timestamps are descending, so i is newer than i+1
                    diff = (timestamps[i] - timestamps[i+1]).total_seconds()
                    time_diffs.append(round(diff, 3))

                # 7. Get total data timespan for Node 124
                cur.execute("SELECT MIN(timestamp), MAX(timestamp), count(*) FROM device_data WHERE node_id::text = '124';")
                min_ts, max_ts, node_count = cur.fetchone()
                
                logger.info(f"✅ DIAGNOSTICS SUCCESSFUL")
                
            return {
                "statusCode": 200,
                "body": json.dumps({
                    "message": "Database Deep Diagnostics",
                    "table_total_rows": total_count,
                    "schema": schema_info,
                    "indexes": index_list,
                    "node_124": {
                        "total_rows_found": node_count,
                        "first_record": str(min_ts),
                        "last_record": str(max_ts),
                        "recent_daily_volumes": daily_data,
                        "peak_minute_analysis": freq_data,
                        "last_20_timestamp_deltas_seconds": time_diffs
                    },
                    "duration_seconds": round(time.time() - start, 2)
                }, default=str)
            }
                
    except Exception as e:
        duration = time.time() - start
        logger.error(f"❌ CONNECTION FAILED after {duration:.2f} seconds.")
        logger.error(str(e))
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": "VPC Connection Failed", 
                "details": str(e),
                "duration_seconds": round(duration, 2)
            })
        }
