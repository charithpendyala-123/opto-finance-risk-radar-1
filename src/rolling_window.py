# ==============================================================================
# OPTOxCRM FINANCE RISK RADAR - ROLLING WINDOW LAYER (PHASE 7)
# ==============================================================================
import os
import sys
import pandas as pd
import numpy as np
from psycopg2.extras import execute_values

# Dynamically add parent directory for absolute/relative import compatibility
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_rolling_window_pipeline(conn, user_id="system_default", batch_id=None):
    """
    Calculates 30-day rolling window features for all transaction records of the active user.
    Persists outcomes to the database in the 'transaction_rolling_features' table.
    """
    if conn is None:
        print("[Rolling Window] Error: No database connection.")
        return False
        
    print(f"[Rolling Window] Running rolling window feature generation for user: {user_id}...")
    
    # 1. Fetch all transaction records and their risk severity for the user
    query = """
        SELECT 
            t.id AS transaction_row_id, t.transaction_id, t.customer_id, t.project_id, t.unit_id,
            t.demand_date, t.payment_delay_days, t.demand_amount, t.collected_amount,
            t.outstanding_amount, t.discount_amount, t.refund_amount,
            r.severity
        FROM transactions t
        LEFT JOIN risk_results r ON t.id = r.transaction_row_id
        WHERE t.user_id = %s
        ORDER BY t.demand_date ASC, t.id ASC;
    """
    
    try:
        df = pd.read_sql(query, conn, params=[user_id])
    except Exception as e:
        print(f"[Rolling Window Error] Failed to read transactions from database: {e}")
        return False
        
    if df.empty:
        print("[Rolling Window] Warning: No transactions found for calculations.")
        return True
        
    # Ensure dates are datetime objects for rolling time comparisons
    df['demand_date_parsed'] = pd.to_datetime(df['demand_date'])
    
    feature_records = []
    
    # Pre-calculate flags mapping (flagged if severity is HIGH or CRITICAL)
    df['is_flagged'] = df['severity'].isin(['HIGH', 'CRITICAL']).astype(int)
    
    # Loop chronologically through each row to prevent future data leak
    for idx, row in df.iterrows():
        row_id = int(row['transaction_row_id'])
        txn_id = row['transaction_id']
        cust_id = row['customer_id']
        proj_id = row['project_id']
        unit_id = row['unit_id']
        current_date = row['demand_date_parsed']
        
        # 30-day window boundary: [current_date - 30 days, current_date]
        window_start = current_date - pd.Timedelta(days=30)
        
        # ─── CUSTOMER FEATURES ───
        cust_window = df[
            (df['customer_id'] == cust_id) & 
            (df['demand_date_parsed'] >= window_start) & 
            (df['demand_date_parsed'] <= current_date)
        ]
        
        cust_txn_count = len(cust_window)
        cust_flags = int(cust_window['is_flagged'].sum())
        
        cust_avg_delay = float(cust_window['payment_delay_days'].mean()) if cust_window['payment_delay_days'].notna().any() else 0.0
        cust_avg_discount = float(cust_window['discount_amount'].mean()) if cust_window['discount_amount'].notna().any() else 0.0
        cust_avg_refund = float(cust_window['refund_amount'].mean()) if cust_window['refund_amount'].notna().any() else 0.0
        cust_avg_outstanding = float(cust_window['outstanding_amount'].mean()) if cust_window['outstanding_amount'].notna().any() else 0.0
        cust_max_outstanding = float(cust_window['outstanding_amount'].max()) if cust_window['outstanding_amount'].notna().any() else 0.0
        
        # Outstanding ratio = sum(outstanding) / sum(demand)
        sum_outstanding = cust_window['outstanding_amount'].sum()
        sum_demand = cust_window['demand_amount'].sum()
        cust_avg_outstanding_ratio = float(sum_outstanding / sum_demand) if sum_demand > 0 else 0.0
        
        # ─── PROJECT FEATURES ───
        proj_window = df[
            (df['project_id'] == proj_id) & 
            (df['demand_date_parsed'] >= window_start) & 
            (df['demand_date_parsed'] <= current_date)
        ]
        
        proj_txn_count = len(proj_window)
        proj_flags = int(proj_window['is_flagged'].sum())
        proj_avg_demand = float(proj_window['demand_amount'].mean()) if proj_window['demand_amount'].notna().any() else 0.0
        proj_avg_outstanding = float(proj_window['outstanding_amount'].mean()) if proj_window['outstanding_amount'].notna().any() else 0.0
        proj_avg_refund = float(proj_window['refund_amount'].mean()) if proj_window['refund_amount'].notna().any() else 0.0
        
        sum_proj_outstanding = proj_window['outstanding_amount'].sum()
        sum_proj_demand = proj_window['demand_amount'].sum()
        proj_avg_outstanding_ratio = float(sum_proj_outstanding / sum_proj_demand) if sum_proj_demand > 0 else 0.0
        
        # ─── UNIT FEATURES ───
        unit_window = df[
            (df['unit_id'] == unit_id) & 
            (df['demand_date_parsed'] >= window_start) & 
            (df['demand_date_parsed'] <= current_date)
        ]
        
        unit_txn_count = len(unit_window)
        unit_flags = int(unit_window['is_flagged'].sum())
        unit_avg_outstanding = float(unit_window['outstanding_amount'].mean()) if unit_window['outstanding_amount'].notna().any() else 0.0
        unit_avg_demand = float(unit_window['demand_amount'].mean()) if unit_window['demand_amount'].notna().any() else 0.0
        unit_unique_customers = int(unit_window['customer_id'].nunique())
        
        # Ownership shifts inside the 30d window
        unit_sorted = unit_window.sort_values('demand_date_parsed')
        unit_owner_changes = int((unit_sorted['customer_id'].ne(unit_sorted['customer_id'].shift()) & unit_sorted['customer_id'].shift().notna()).sum())
        
        feature_records.append((
            row_id,
            txn_id,
            cust_txn_count,
            cust_flags,
            cust_avg_delay,
            cust_avg_discount,
            cust_avg_refund,
            cust_avg_outstanding,
            cust_max_outstanding,
            cust_avg_outstanding_ratio,
            proj_txn_count,
            proj_flags,
            proj_avg_demand,
            proj_avg_outstanding,
            proj_avg_outstanding_ratio,
            proj_avg_refund,
            unit_txn_count,
            unit_flags,
            unit_owner_changes,
            unit_avg_outstanding,
            unit_avg_demand,
            unit_unique_customers
        ))
        
    # 2. Persist to database in bulk
    insert_query = """
        INSERT INTO transaction_rolling_features (
            transaction_row_id, transaction_id,
            customer_txn_count_30d, customer_flags_30d, customer_avg_delay_30d,
            customer_avg_discount_30d, customer_avg_refund_30d, customer_avg_outstanding_30d,
            customer_max_outstanding_30d, customer_avg_outstanding_ratio_30d,
            project_txn_count_30d, project_flags_30d, project_avg_demand_30d,
            project_avg_outstanding_30d, project_avg_outstanding_ratio_30d, project_avg_refund_30d,
            unit_txn_count_30d, unit_flags_30d, unit_owner_changes_30d,
            unit_avg_outstanding_30d, unit_avg_demand_30d, unit_unique_customer_count_30d
        ) VALUES %s
        ON CONFLICT (transaction_row_id) DO UPDATE SET
            customer_txn_count_30d = EXCLUDED.customer_txn_count_30d,
            customer_flags_30d = EXCLUDED.customer_flags_30d,
            customer_avg_delay_30d = EXCLUDED.customer_avg_delay_30d,
            customer_avg_discount_30d = EXCLUDED.customer_avg_discount_30d,
            customer_avg_refund_30d = EXCLUDED.customer_avg_refund_30d,
            customer_avg_outstanding_30d = EXCLUDED.customer_avg_outstanding_30d,
            customer_max_outstanding_30d = EXCLUDED.customer_max_outstanding_30d,
            customer_avg_outstanding_ratio_30d = EXCLUDED.customer_avg_outstanding_ratio_30d,
            project_txn_count_30d = EXCLUDED.project_txn_count_30d,
            project_flags_30d = EXCLUDED.project_flags_30d,
            project_avg_demand_30d = EXCLUDED.project_avg_demand_30d,
            project_avg_outstanding_30d = EXCLUDED.project_avg_outstanding_30d,
            project_avg_outstanding_ratio_30d = EXCLUDED.project_avg_outstanding_ratio_30d,
            project_avg_refund_30d = EXCLUDED.project_avg_refund_30d,
            unit_txn_count_30d = EXCLUDED.unit_txn_count_30d,
            unit_flags_30d = EXCLUDED.unit_flags_30d,
            unit_owner_changes_30d = EXCLUDED.unit_owner_changes_30d,
            unit_avg_outstanding_30d = EXCLUDED.unit_avg_outstanding_30d,
            unit_avg_demand_30d = EXCLUDED.unit_avg_demand_30d,
            unit_unique_customer_count_30d = EXCLUDED.unit_unique_customer_count_30d;
    """
    
    try:
        with conn.cursor() as cur:
            execute_values(cur, insert_query, feature_records)
        conn.commit()
        print(f"[Rolling Window] Successfully stored {len(feature_records)} rolling window records in Postgres.")
        return True
    except Exception as e:
        conn.rollback()
        print(f"[Rolling Window Error] Failed to persist rolling window features to DB: {e}")
        return False