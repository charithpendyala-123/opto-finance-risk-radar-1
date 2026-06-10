import os
import sys
import time
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.db as db

import importlib
if_detector = importlib.import_module("src.07_IF_anomaly_detector")
prepare_features = if_detector.prepare_features

def run_precomputation(user_id="system_default"):
    conn = db.get_connection()
    if conn is None:
        return False, 0
        
    query = """
        SELECT 
            transaction_id, customer_id, project_id, unit_id,
            demand_date, payment_date, demand_amount, collected_amount,
            outstanding_amount, discount_amount, refund_amount,
            payment_delay_days, payment_gap_days
        FROM transactions
        WHERE user_id = %s;
    """
    try:
        df = pd.read_sql(query, conn, params=[user_id])
    except Exception as e:
        print(f"[Daemon Error] Failed to query transactions: {e}")
        conn.close()
        return False, 0
    finally:
        conn.close()
        
    total_records = len(df)
    if total_records < 10:
        return False, total_records

    numeric_cols = ['demand_amount', 'collected_amount', 'outstanding_amount', 'discount_amount', 'refund_amount', 'payment_delay_days']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    cache = {
        "zscore": {},
        "iqr": {},
        "isolation_forest_thresholds": {},
        "projects": {}
    }

    # 1. Z-Score parameters
    for col in numeric_cols:
        series = df[col].dropna()
        mean = float(series.mean()) if not series.empty else 0.0
        std = float(series.std()) if not series.empty else 1.0
        std = std if std > 0 else 1.0
        cache["zscore"][col] = {"mean": mean, "std": std}

    # 2. IQR bounds
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            lower, upper, iqr_val = 0.0, 0.0, 0.0
        else:
            Q1 = float(series.quantile(0.25))
            Q3 = float(series.quantile(0.75))
            iqr_val = Q3 - Q1
            lower = Q1 - 1.5 * iqr_val
            upper = Q3 + 1.5 * iqr_val
        cache["iqr"][col] = {"lower_bound": lower, "upper_bound": upper, "iqr": iqr_val}

    # 3. Project-level group aggregates
    print("[Daemon] Pre-computing project aggregates...")
    try:
        conn = db.get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    project_id,
                    COUNT(*) AS size,
                    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY demand_amount) AS demand_q1,
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY demand_amount) AS demand_q3,
                    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY collected_amount) AS collected_q1,
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY collected_amount) AS collected_q3,
                    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY outstanding_amount / NULLIF(demand_amount, 0)) AS ratio_q1,
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY outstanding_amount / NULLIF(demand_amount, 0)) AS ratio_q3,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY demand_amount) AS median_demand
                FROM transactions
                WHERE user_id = %s
                GROUP BY project_id;
            """, (user_id,))
            rows = cur.fetchall()
            for r in rows:
                p_id = r[0]
                cache["projects"][p_id] = {
                    "size": int(r[1]),
                    "demand_q1": float(r[2]) if r[2] is not None else 0.0,
                    "demand_q3": float(r[3]) if r[3] is not None else 0.0,
                    "collected_q1": float(r[4]) if r[4] is not None else 0.0,
                    "collected_q3": float(r[5]) if r[5] is not None else 0.0,
                    "ratio_q1": float(r[6]) if r[6] is not None else 0.0,
                    "ratio_q3": float(r[7]) if r[7] is not None else 0.0,
                    "median_demand": float(r[8]) if r[8] is not None else 0.0
                }
    except Exception as e:
        print(f"[Daemon Error] Failed to precompute project stats: {e}")
    finally:
        conn.close()

    # 4. Isolation Forest Training
    df_engineered = prepare_features(df)
    features = [
        "demand_amount", "collected_amount", "outstanding_amount", 
        "discount_amount", "refund_amount", "payment_delay_days", 
        "payment_gap_days", "demand_dayofweek", "payment_dayofweek"
    ]
    X = df_engineered[features].fillna(0)
    
    n_est = int(min(200, max(50, len(df) // 10)))
    model = IsolationForest(
        n_estimators=n_est,
        max_samples='auto',
        contamination='auto',
        random_state=42
    )
    model.fit(X)
    
    os.makedirs("models", exist_ok=True)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.abspath(os.path.join(base_dir, "models", "isolation_forest.joblib"))
    cache_path = os.path.abspath(os.path.join(base_dir, "models", "stats_cache.json"))
    
    joblib.dump(model, model_path)
    
    for col in ['refund_amount', 'payment_delay_days', 'discount_amount', 'payment_gap_days']:
        series = df_engineered[col].fillna(0)
        mean_val = float(series.mean())
        std_val = float(series.std())
        std_val = std_val if std_val > 0 else 1.0
        cache["isolation_forest_thresholds"][col] = mean_val + 2.5 * std_val

    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=4)
        
    return True, total_records

def main():
    print("=========================================================")
    print("      OPTOxCRM BACKGROUND PRE-COMPUTATION DAEMON        ")
    print("=========================================================")
    print("Monitoring database for transaction count changes...")
    
    last_processed_count = -1
    
    while True:
        conn = db.get_connection()
        if conn is not None:
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM transactions WHERE user_id = 'system_default';")
                    current_count = cur.fetchone()[0]
            except Exception as e:
                print(f"[Daemon] Error fetching count: {e}")
                current_count = -1
            finally:
                conn.close()
                
            if current_count != -1 and current_count != last_processed_count:
                print(f"\n[Daemon] Change detected! DB Count: {current_count} (prev: {last_processed_count})")
                print("[Daemon] Retraining models and updating thresholds in background...")
                start_time = time.time()
                success, count = run_precomputation()
                if success:
                    last_processed_count = count
                    duration = time.time() - start_time
                    print(f"[Daemon] Retraining finished in {duration:.2f} seconds. Cache files updated.")
                else:
                    print("[Daemon] Retraining skipped or failed.")
                    
        time.sleep(5)

if __name__ == "__main__":
    main()