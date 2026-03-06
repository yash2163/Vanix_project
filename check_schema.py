import psycopg2
import os

conn_str = os.environ.get("PG_CONNECTION_STRING", "postgresql://vanixuser:vanix%244567%23@10.98.36.21:6432/vanixdb")
with psycopg2.connect(conn_str) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'heat_analysis_config';")
        rows = cur.fetchall()
        for r in rows:
            print(r)
