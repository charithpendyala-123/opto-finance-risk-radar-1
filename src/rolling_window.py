import os
import sys
import pandas as pd
import numpy as np
from psycopg2.extras import execute_values
from bisect import bisect_left

def run_rolling_window_pipeline(conn, user_id="system_default", batch_id=None):
    if conn is None:
        print("[Rolling Window] Error: No database connection.")
        return False
        
    print(f"[Rolling Window] Running single-loop prepopulated rolling window feature generation for user: {user_id}...")
    
    # 1. Fetch all transaction records and their risk severity for the user
    query = """
        SELECT 
            t.id AS transaction_row_id, t.transaction_id, t.customer_id, t.project_id, t.unit_id,
            t.demand_date, t.payment_delay_days, t.demand_amount, t.collected_amount,
            t.outstanding_amount, t.discount_amount, t.refund_amount, t.upload_batch_id,
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
        
    df['demand_date_parsed'] = pd.to_datetime(df['demand_date'])
    df['is_flagged'] = df['severity'].isin(['HIGH', 'CRITICAL']).astype(int)
    
    feature_records = []
    
    cust_hist = {}
    cust_dates = {}
    cust_lf = {}
    
    proj_hist = {}
    proj_dates = {}
    
    unit_hist = {}
    unit_dates = {}
    
    # 2. Split into history and active
    if batch_id:
        df_history = df[df['upload_batch_id'] != batch_id]
        df_active = df[df['upload_batch_id'] == batch_id]
    else:
        df_history = pd.DataFrame(columns=df.columns)
        df_active = df
        
    # 3. Pre-populate histories from df_history using a single fast list of dicts iteration
    if not df_history.empty:
        hist_records = df_history[[
            'customer_id', 'project_id', 'unit_id', 'is_flagged', 
            'payment_delay_days', 'discount_amount', 'refund_amount', 
            'outstanding_amount', 'demand_amount', 'demand_date_parsed'
        ]].replace({np.nan: None}).to_dict('records')
        
        for r in hist_records:
            c_id = r['customer_id']
            p_id = r['project_id']
            u_id = r['unit_id']
            dt = r['demand_date_parsed']
            
            # Project history
            if p_id not in proj_hist:
                proj_hist[p_id] = []
                proj_dates[p_id] = []
            proj_hist[p_id].append({
                'is_flagged': r['is_flagged'], 'demand': r['demand_amount'],
                'outstanding': r['outstanding_amount'], 'refund': r['refund_amount']
            })
            proj_dates[p_id].append(dt)
                
            # Unit history
            if u_id not in unit_hist:
                unit_hist[u_id] = []
                unit_dates[u_id] = []
            unit_hist[u_id].append({
                'is_flagged': r['is_flagged'], 'customer_id': c_id,
                'outstanding': r['outstanding_amount'], 'demand': r['demand_amount']
            })
            unit_dates[u_id].append(dt)

            # Customer history
            if c_id not in cust_hist:
                cust_hist[c_id] = []
                cust_dates[c_id] = []
                cust_lf[c_id] = {
                    'total_refund_lifetime': 0.0,
                    'discount_count_lifetime': 0,
                    'fraud_count_lifetime': 0
                }
            
            cust_hist[c_id].append({
                'is_flagged': r['is_flagged'], 'delay': r['payment_delay_days'],
                'discount': r['discount_amount'], 'refund': r['refund_amount'],
                'outstanding': r['outstanding_amount'], 'demand': r['demand_amount']
            })
            cust_dates[c_id].append(dt)
            
            lf = cust_lf[c_id]
            if r['refund_amount'] is not None:
                lf['total_refund_lifetime'] += r['refund_amount']
            if r['discount_amount'] is not None and r['discount_amount'] > 0:
                lf['discount_count_lifetime'] += 1
            lf['fraud_count_lifetime'] += r['is_flagged']
            
    # 4. Loop ONLY over active batch records
    active_records = df_active.to_dict('records')
    for row in active_records:
        row_id = int(row['transaction_row_id'])
        txn_id = row['transaction_id']
        cust_id = row['customer_id']
        proj_id = row['project_id']
        unit_id = row['unit_id']
        current_date = row['demand_date_parsed']
        
        val_delay = float(row['payment_delay_days']) if pd.notna(row['payment_delay_days']) else None
        val_discount = float(row['discount_amount']) if pd.notna(row['discount_amount']) else None
        val_refund = float(row['refund_amount']) if pd.notna(row['refund_amount']) else None
        val_outstanding = float(row['outstanding_amount']) if pd.notna(row['outstanding_amount']) else None
        val_demand = float(row['demand_amount']) if pd.notna(row['demand_amount']) else None
        is_flagged = int(row['is_flagged'])
        
        # Update Project history
        if proj_id not in proj_hist:
            proj_hist[proj_id] = []
            proj_dates[proj_id] = []
        proj_hist[proj_id].append({
            'is_flagged': is_flagged, 'demand': val_demand,
            'outstanding': val_outstanding, 'refund': val_refund
        })
        proj_dates[proj_id].append(current_date)
            
        # Update Unit history
        if unit_id not in unit_hist:
            unit_hist[unit_id] = []
            unit_dates[unit_id] = []
        unit_hist[unit_id].append({
            'is_flagged': is_flagged, 'customer_id': cust_id,
            'outstanding': val_outstanding, 'demand': val_demand
        })
        unit_dates[unit_id].append(current_date)

        # Update Customer history
        if cust_id not in cust_hist:
            cust_hist[cust_id] = []
            cust_dates[cust_id] = []
            cust_lf[cust_id] = {
                'total_refund_lifetime': 0.0,
                'discount_count_lifetime': 0,
                'fraud_count_lifetime': 0
            }
        
        cust_hist[cust_id].append({
            'is_flagged': is_flagged, 'delay': val_delay,
            'discount': val_discount, 'refund': val_refund, 'outstanding': val_outstanding, 'demand': val_demand
        })
        cust_dates[cust_id].append(current_date)
        
        lf = cust_lf[cust_id]
        if val_refund is not None:
            lf['total_refund_lifetime'] += val_refund
        if val_discount is not None and val_discount > 0:
            lf['discount_count_lifetime'] += 1
        lf['fraud_count_lifetime'] += is_flagged
        
        # ─── CALCULATE CUSTOMER FEATURES USING BINARY SEARCH ───
        c_dates = cust_dates[cust_id]
        c_entries = cust_hist[cust_id]
        
        # 30-day window
        c_30_start = current_date - pd.Timedelta(days=30)
        idx_30 = bisect_left(c_dates, c_30_start)
        c_30_win = c_entries[idx_30:]
        
        cust_txn_count = len(c_30_win)
        cust_flags = sum(r['is_flagged'] for r in c_30_win)
        
        c_delays = [r['delay'] for r in c_30_win if r['delay'] is not None]
        cust_avg_delay = sum(c_delays) / len(c_delays) if c_delays else 0.0
        
        c_discounts = [r['discount'] for r in c_30_win if r['discount'] is not None]
        cust_avg_discount = sum(c_discounts) / len(c_discounts) if c_discounts else 0.0
        
        c_refunds = [r['refund'] for r in c_30_win if r['refund'] is not None]
        cust_avg_refund = sum(c_refunds) / len(c_refunds) if c_refunds else 0.0
        
        c_outstandings = [r['outstanding'] for r in c_30_win if r['outstanding'] is not None]
        cust_avg_outstanding = sum(c_outstandings) / len(c_outstandings) if c_outstandings else 0.0
        cust_max_outstanding = max(c_outstandings) if c_outstandings else 0.0
        
        sum_outstanding = sum(r['outstanding'] for r in c_30_win if r['outstanding'] is not None)
        sum_demand = sum(r['demand'] for r in c_30_win if r['demand'] is not None)
        cust_avg_outstanding_ratio = sum_outstanding / sum_demand if sum_demand > 0 else 0.0
        
        # 90-day window
        c_90_start = current_date - pd.Timedelta(days=90)
        idx_90 = bisect_left(c_dates, c_90_start)
        c_90_win = c_entries[idx_90:]
        cust_refund_count_90d = sum(1 for r in c_90_win if r['refund'] is not None and r['refund'] > 0)
        
        # 180-day window
        c_180_start = current_date - pd.Timedelta(days=180)
        idx_180 = bisect_left(c_dates, c_180_start)
        c_180_win = c_entries[idx_180:]
        cust_refund_count_180d = sum(1 for r in c_180_win if r['refund'] is not None and r['refund'] > 0)
        c_180_refunds = [r['refund'] for r in c_180_win if r['refund'] is not None]
        cust_total_refund_180d = sum(c_180_refunds) if c_180_refunds else 0.0
        
        # Lifetime (O(1) Running Totals)
        cust_total_refund_lifetime = lf['total_refund_lifetime']
        cust_discount_count_lifetime = lf['discount_count_lifetime']
        cust_fraud_count_lifetime = lf['fraud_count_lifetime']
        
        # ─── CALCULATE PROJECT FEATURES USING BINARY SEARCH ───
        p_dates = proj_dates[proj_id]
        p_entries = proj_hist[proj_id]
        p_30_start = current_date - pd.Timedelta(days=30)
        idx_p30 = bisect_left(p_dates, p_30_start)
        p_30_win = p_entries[idx_p30:]
        
        proj_txn_count = len(p_30_win)
        proj_flags = sum(r['is_flagged'] for r in p_30_win)
        
        p_demands = [r['demand'] for r in p_30_win if r['demand'] is not None]
        proj_avg_demand = sum(p_demands) / len(p_demands) if p_demands else 0.0
        
        p_outstandings = [r['outstanding'] for r in p_30_win if r['outstanding'] is not None]
        proj_avg_outstanding = sum(p_outstandings) / len(p_outstandings) if p_outstandings else 0.0
        
        p_refunds = [r['refund'] for r in p_30_win if r['refund'] is not None]
        proj_avg_refund = sum(p_refunds) / len(p_refunds) if p_refunds else 0.0
        
        sum_proj_outstanding = sum(r['outstanding'] for r in p_30_win if r['outstanding'] is not None)
        sum_proj_demand = sum(r['demand'] for r in p_30_win if r['demand'] is not None)
        proj_avg_outstanding_ratio = sum_proj_outstanding / sum_proj_demand if sum_proj_demand > 0 else 0.0
        
        # ─── CALCULATE UNIT FEATURES USING BINARY SEARCH ───
        u_dates = unit_dates[unit_id]
        u_entries = unit_hist[unit_id]
        u_30_start = current_date - pd.Timedelta(days=30)
        idx_u30 = bisect_left(u_dates, u_30_start)
        u_30_win = u_entries[idx_u30:]
        
        unit_txn_count = len(u_30_win)
        unit_flags = sum(r['is_flagged'] for r in u_30_win)
        
        u_outstandings = [r['outstanding'] for r in u_30_win if r['outstanding'] is not None]
        unit_avg_outstanding = sum(u_outstandings) / len(u_outstandings) if u_outstandings else 0.0
        
        u_demands = [r['demand'] for r in u_30_win if r['demand'] is not None]
        unit_avg_demand = sum(u_demands) / len(u_demands) if u_demands else 0.0
        
        unit_unique_customers = len(set(r['customer_id'] for r in u_30_win))
        
        unit_owner_changes = 0
        prev_cust = None
        for r in u_30_win:
            if prev_cust is not None and r['customer_id'] != prev_cust:
                unit_owner_changes += 1
            prev_cust = r['customer_id']
            
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
            unit_unique_customers,
            cust_refund_count_90d,
            cust_refund_count_180d,
            cust_total_refund_180d,
            cust_total_refund_lifetime,
            cust_discount_count_lifetime,
            cust_fraud_count_lifetime
        ))
        
    # 5. Persist to database in bulk
    insert_query = """
        INSERT INTO transaction_rolling_features (
            transaction_row_id, transaction_id,
            customer_txn_count_30d, customer_flags_30d, customer_avg_delay_30d,
            customer_avg_discount_30d, customer_avg_refund_30d, customer_avg_outstanding_30d,
            customer_max_outstanding_30d, customer_avg_outstanding_ratio_30d,
            project_txn_count_30d, project_flags_30d, project_avg_demand_30d,
            project_avg_outstanding_30d, project_avg_outstanding_ratio_30d, project_avg_refund_30d,
            unit_txn_count_30d, unit_flags_30d, unit_owner_changes_30d,
            unit_avg_outstanding_30d, unit_avg_demand_30d, unit_unique_customer_count_30d,
            customer_refund_count_90d, customer_refund_count_180d, customer_total_refund_180d,
            customer_total_refund_lifetime, customer_discount_count_lifetime, customer_fraud_count_lifetime
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
            unit_unique_customer_count_30d = EXCLUDED.unit_unique_customer_count_30d,
            customer_refund_count_90d = EXCLUDED.customer_refund_count_90d,
            customer_refund_count_180d = EXCLUDED.customer_refund_count_180d,
            customer_total_refund_180d = EXCLUDED.customer_total_refund_180d,
            customer_total_refund_lifetime = EXCLUDED.customer_total_refund_lifetime,
            customer_discount_count_lifetime = EXCLUDED.customer_discount_count_lifetime,
            customer_fraud_count_lifetime = EXCLUDED.customer_fraud_count_lifetime;
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