import os
import psycopg2
import pandas as pd
from contextlib import closing
import time

PG_CONNECTION_STRING = "postgresql://vanixuser:vanix%244567%23@10.98.36.21:6432/vanixdb"

print("="*60)
print("POSTGRES DB DIAGNOSTICS LOG")
print("="*60)

try:
    print(f"Attempting to connect to: {PG_CONNECTION_STRING}")
    start = time.time()
    with closing(psycopg2.connect(PG_CONNECTION_STRING, connect_timeout=10)) as conn:
        print(f"✅ Connection successful in {time.time() - start:.2f} seconds.")
        
        # 1. Total Rows
        print("\n--- 1. TABLE SIZE ---")
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM device_data;")
            total_rows = cur.fetchone()[0]
            print(f"Total rows in device_data: {total_rows:,}")
            
        # 2. Schema Structure
        print("\n--- 2. TABLE SCHEMA ---")
        schema_query = """
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'device_data';
        """
        schema_df = pd.read_sql_query(schema_query, conn)
        print(schema_df.to_string(index=False))
        
        # 3. Time Range
        print("\n--- 3. DATA TIME RANGE ---")
        with conn.cursor() as cur:
            cur.execute("SELECT MIN(timestamp), MAX(timestamp) FROM device_data;")
            min_ts, max_ts = cur.fetchone()
            print(f"Earliest Record : {min_ts}")
            print(f"Latest Record   : {max_ts}")
            
        # 4. Node 124 Details (The test node)
        print("\n--- 4. NODE 124 STATS ---")
        node_query = "SELECT count(*), MIN(timestamp), MAX(timestamp) FROM device_data WHERE node_id::text = '124';"
        with conn.cursor() as cur:
            cur.execute(node_query)
            n_rows, n_min, n_max = cur.fetchone()
            print(f"Node 124 Rows : {n_rows:,}")
            print(f"Node 124 Start: {n_min}")
            print(f"Node 124 End  : {n_max}")
            
            if n_rows > 0:
                # Estimate size of 1 day for Node 124
                days_diff = (n_max - n_min).days or 1
                rows_per_day = n_rows / days_diff
                print(f"Estimated rows per day for Node 124: ~{int(rows_per_day):,}")

except psycopg2.OperationalError as e:
    print(f"\n❌ CONNECTION FAILED: The database is not accessible from here.")
    print("Error Details:")
    print(e)
    if "timeout" in str(e).lower() or "network is unreachable" in str(e).lower():
        print("\nDIAGNOSIS: This looks like a VPC/Networking issue. The IP 10.98.36.21 is private.")
        print("Your script cannot reach the database through the internet.")
except Exception as e:
    print(f"\n❌ UNEXPECTED ERROR: {e}")
