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
    
    # 1. Fetch only active transactions for the specified user and batch
    if batch_id:
        query_active = """
            SELECT 
                t.id AS transaction_row_id, t.transaction_id, t.customer_id, t.project_id, t.unit_id,
                t.demand_date, t.payment_delay_days, t.demand_amount, t.collected_amount,
                t.outstanding_amount, t.discount_amount, t.refund_amount, t.upload_batch_id,
                r.severity
            FROM transactions t
            LEFT JOIN risk_results r ON t.id = r.transaction_row_id
            WHERE t.user_id = %s AND t.upload_batch_id = %s
            ORDER BY t.demand_date ASC, t.id ASC;
        """
        params_active = [user_id, batch_id]
    else:
        query_active = """
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
        params_active = [user_id]
        
    try:
        df_active = pd.read_sql(query_active, conn, params=params_active)
    except Exception as e:
        print(f"[Rolling Window Error] Failed to read active transactions from database: {e}")
        return False
        
    if df_active.empty:
        print("[Rolling Window] Warning: No transactions found in this batch for calculations.")
        return True
        
    df_active['demand_date_parsed'] = pd.to_datetime(df_active['demand_date'])
    df_active['is_flagged'] = df_active['severity'].isin(['HIGH', 'CRITICAL']).astype(int)
    
    # 2. Extract unique customer, project, and unit IDs from the active batch
    active_customers = [c for c in df_active['customer_id'].unique() if pd.notna(c)]
    active_projects = [p for p in df_active['project_id'].unique() if pd.notna(p)]
    active_units = [u for u in df_active['unit_id'].unique() if pd.notna(u)]
    
    # Calculate min active date to set the historical cutoff boundary
    valid_dates = df_active['demand_date_parsed'].dropna()
    if not valid_dates.empty:
        min_active_date = valid_dates.min()
        max_active_date = valid_dates.max()
        history_cutoff_date = min_active_date - pd.Timedelta(days=180)
    else:
        min_active_date = None
        max_active_date = None
        history_cutoff_date = None
        
    # 3. Retrieve pre-aggregated customer lifetime stats for transactions prior to cutoff
    cust_lf = {}
    if active_customers:
        params_lf = [user_id]
        if batch_id:
            conditions_lf = ["t.user_id = %s", "t.upload_batch_id != %s"]
            params_lf.append(batch_id)
        else:
            conditions_lf = ["t.user_id = %s"]
            
        if history_cutoff_date is not None:
            conditions_lf.append("(t.demand_date < %s OR t.demand_date IS NULL)")
            params_lf.append(history_cutoff_date)
            
        conditions_lf.append("t.customer_id IN %s")
        params_lf.append(tuple(active_customers))
        
        query_lf = f"""
            SELECT 
                t.customer_id,
                COALESCE(SUM(t.refund_amount), 0.0) AS total_refund_prior,
                COALESCE(SUM(CASE WHEN t.discount_amount > 0 THEN 1 ELSE 0 END), 0) AS discount_count_prior,
                COALESCE(SUM(CASE WHEN r.severity IN ('HIGH', 'CRITICAL') THEN 1 ELSE 0 END), 0) AS fraud_count_prior
            FROM transactions t
            LEFT JOIN risk_results r ON t.id = r.transaction_row_id
            WHERE {" AND ".join(conditions_lf)}
            GROUP BY t.customer_id;
        """
        
        try:
            with conn.cursor() as cur:
                cur.execute(query_lf, params_lf)
                rows = cur.fetchall()
                for customer_id, total_refund_prior, discount_count_prior, fraud_count_prior in rows:
                    cust_lf[customer_id] = {
                        'total_refund_lifetime': float(total_refund_prior),
                        'discount_count_lifetime': int(discount_count_prior),
                        'fraud_count_lifetime': int(fraud_count_prior)
                    }
        except Exception as e:
            print(f"[Rolling Window Error] Failed to fetch aggregate customer lifetime features: {e}")
            return False

    # 4. Fetch only relevant historical transactions within the last 180 days for active customers and units
    df_history = pd.DataFrame(columns=df_active.columns)
    if history_cutoff_date is not None and (active_customers or active_units):
        conditions_hist = ["t.user_id = %s", "t.demand_date >= %s", "t.demand_date <= %s"]
        params_hist = [user_id, history_cutoff_date, max_active_date]
        
        if batch_id:
            conditions_hist.append("t.upload_batch_id != %s")
            params_hist.append(batch_id)
            
        entity_conds = []
        if active_customers:
            entity_conds.append("t.customer_id IN %s")
            params_hist.append(tuple(active_customers))
        if active_units:
            entity_conds.append("t.unit_id IN %s")
            params_hist.append(tuple(active_units))
            
        conditions_hist.append("(" + " OR ".join(entity_conds) + ")")
            
        query_history = f"""
            SELECT 
                t.id AS transaction_row_id, t.transaction_id, t.customer_id, t.project_id, t.unit_id,
                t.demand_date, t.payment_delay_days, t.demand_amount, t.collected_amount,
                t.outstanding_amount, t.discount_amount, t.refund_amount, t.upload_batch_id,
                r.severity
            FROM transactions t
            LEFT JOIN risk_results r ON t.id = r.transaction_row_id
            WHERE {" AND ".join(conditions_hist)}
            ORDER BY t.demand_date ASC, t.id ASC;
        """
        try:
            df_history = pd.read_sql(query_history, conn, params=params_hist)
        except Exception as e:
            print(f"[Rolling Window Error] Failed to read historical transactions: {e}")
            return False

    df_history['demand_date_parsed'] = pd.to_datetime(df_history['demand_date'])
    df_history['is_flagged'] = df_history['severity'].isin(['HIGH', 'CRITICAL']).astype(int)

    # 4b. Fetch daily project aggregates for active projects in a 30-day lookback window
    proj_daily_stats = {}
    if active_projects and history_cutoff_date is not None:
        history_cutoff_30d = min_active_date - pd.Timedelta(days=30)
        
        conditions_proj = [
            "t.user_id = %s",
            "t.demand_date >= %s",
            "t.demand_date <= %s",
            "t.project_id IN %s"
        ]
        params_proj = [user_id, history_cutoff_30d.date(), max_active_date.date(), tuple(active_projects)]
        
        if batch_id:
            conditions_proj.append("t.upload_batch_id != %s")
            params_proj.append(batch_id)
            
        query_proj = f"""
            SELECT 
                t.project_id,
                t.demand_date,
                COUNT(*) AS txn_count,
                SUM(CASE WHEN r.severity IN ('HIGH', 'CRITICAL') THEN 1 ELSE 0 END) AS flags_count,
                COALESCE(SUM(t.demand_amount), 0.0) AS total_demand,
                COUNT(t.demand_amount) AS count_demand,
                COALESCE(SUM(t.outstanding_amount), 0.0) AS total_outstanding,
                COUNT(t.outstanding_amount) AS count_outstanding,
                COALESCE(SUM(t.refund_amount), 0.0) AS total_refund,
                COUNT(t.refund_amount) AS count_refund
            FROM transactions t
            LEFT JOIN risk_results r ON t.id = r.transaction_row_id
            WHERE {" AND ".join(conditions_proj)}
            GROUP BY t.project_id, t.demand_date
            ORDER BY t.demand_date ASC;
        """
        try:
            with conn.cursor() as cur:
                cur.execute(query_proj, params_proj)
                rows = cur.fetchall()
                for (p_id, d_date, txn_count, flags_count, total_demand, count_demand,
                     total_outstanding, count_outstanding, total_refund, count_refund) in rows:
                    if p_id not in proj_daily_stats:
                        proj_daily_stats[p_id] = []
                    dt = pd.to_datetime(d_date)
                    proj_daily_stats[p_id].append({
                        'date': dt,
                        'txn_count': int(txn_count),
                        'flags_count': int(flags_count),
                        'total_demand': float(total_demand),
                        'count_demand': int(count_demand),
                        'total_outstanding': float(total_outstanding),
                        'count_outstanding': int(count_outstanding),
                        'total_refund': float(total_refund),
                        'count_refund': int(count_refund)
                    })
        except Exception as e:
            print(f"[Rolling Window Error] Failed to fetch project daily stats: {e}")
            return False

    cust_hist = {}
    cust_dates = {}
    
    unit_hist = {}
    unit_dates = {}

    active_proj_records = {} # Holds active batch records processed so far for project calculation

    # 5. Populate history states from df_history (which is now extremely small and excludes project matching)
    if not df_history.empty:
        hist_records = df_history[[
            'customer_id', 'project_id', 'unit_id', 'is_flagged', 
            'payment_delay_days', 'discount_amount', 'refund_amount', 
            'outstanding_amount', 'demand_amount', 'demand_date_parsed'
        ]].to_dict('records')
        
        for r in hist_records:
            c_id = r['customer_id']
            u_id = r['unit_id']
            dt = r['demand_date_parsed']
            
            val_delay = float(r['payment_delay_days']) if pd.notna(r['payment_delay_days']) else None
            val_discount = float(r['discount_amount']) if pd.notna(r['discount_amount']) else None
            val_refund = float(r['refund_amount']) if pd.notna(r['refund_amount']) else None
            val_outstanding = float(r['outstanding_amount']) if pd.notna(r['outstanding_amount']) else None
            val_demand = float(r['demand_amount']) if pd.notna(r['demand_amount']) else None
            is_flagged = int(r['is_flagged'])
            
            # Update lifetime stats
            if c_id not in cust_lf:
                cust_lf[c_id] = {
                    'total_refund_lifetime': 0.0,
                    'discount_count_lifetime': 0,
                    'fraud_count_lifetime': 0
                }
            lf = cust_lf[c_id]
            if val_refund is not None:
                lf['total_refund_lifetime'] += val_refund
            if val_discount is not None and val_discount > 0:
                lf['discount_count_lifetime'] += 1
            lf['fraud_count_lifetime'] += is_flagged
            
            if pd.isna(dt):
                continue
                
            # Unit history
            if u_id not in unit_hist:
                unit_hist[u_id] = []
                unit_dates[u_id] = []
            unit_hist[u_id].append({
                'is_flagged': is_flagged, 'customer_id': c_id,
                'outstanding': val_outstanding, 'demand': val_demand
            })
            unit_dates[u_id].append(dt)

            # Customer history
            if c_id not in cust_hist:
                cust_hist[c_id] = []
                cust_dates[c_id] = []
            cust_hist[c_id].append({
                'is_flagged': is_flagged, 'delay': val_delay,
                'discount': val_discount, 'refund': val_refund,
                'outstanding': val_outstanding, 'demand': val_demand
            })
            cust_dates[c_id].append(dt)

    # 6. Loop ONLY over active batch records
    feature_records = []
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
        
        if cust_id not in cust_lf:
            cust_lf[cust_id] = {
                'total_refund_lifetime': 0.0,
                'discount_count_lifetime': 0,
                'fraud_count_lifetime': 0
            }
        lf = cust_lf[cust_id]
        
        if val_refund is not None:
            lf['total_refund_lifetime'] += val_refund
        if val_discount is not None and val_discount > 0:
            lf['discount_count_lifetime'] += 1
        lf['fraud_count_lifetime'] += is_flagged
        
        cust_total_refund_lifetime = lf['total_refund_lifetime']
        cust_discount_count_lifetime = lf['discount_count_lifetime']
        cust_fraud_count_lifetime = lf['fraud_count_lifetime']
        
        if pd.notna(current_date):
            # Update Project active records list
            if proj_id not in active_proj_records:
                active_proj_records[proj_id] = []
            active_proj_records[proj_id].append({
                'date': current_date,
                'is_flagged': is_flagged,
                'demand': val_demand,
                'outstanding': val_outstanding,
                'refund': val_refund
            })
                
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
            cust_hist[cust_id].append({
                'is_flagged': is_flagged, 'delay': val_delay,
                'discount': val_discount, 'refund': val_refund,
                'outstanding': val_outstanding, 'demand': val_demand
            })
            cust_dates[cust_id].append(current_date)
            
        # Feature calculations
        if pd.isna(current_date):
            cust_txn_count = 0
            cust_flags = 0
            cust_avg_delay = 0.0
            cust_avg_discount = 0.0
            cust_avg_refund = 0.0
            cust_avg_outstanding = 0.0
            cust_max_outstanding = 0.0
            cust_avg_outstanding_ratio = 0.0
            
            cust_refund_count_90d = 0
            cust_refund_count_180d = 0
            cust_total_refund_180d = 0.0
            
            proj_txn_count = 0
            proj_flags = 0
            proj_avg_demand = 0.0
            proj_avg_outstanding = 0.0
            proj_avg_refund = 0.0
            proj_avg_outstanding_ratio = 0.0
            
            unit_txn_count = 0
            unit_flags = 0
            unit_owner_changes = 0
            unit_avg_outstanding = 0.0
            unit_avg_demand = 0.0
            unit_unique_customers = 0
        else:
            c_dates = cust_dates.get(cust_id, [])
            c_entries = cust_hist.get(cust_id, [])
            
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
            
            c_90_start = current_date - pd.Timedelta(days=90)
            idx_90 = bisect_left(c_dates, c_90_start)
            c_90_win = c_entries[idx_90:]
            cust_refund_count_90d = sum(1 for r in c_90_win if r['refund'] is not None and r['refund'] > 0)
            
            c_180_start = current_date - pd.Timedelta(days=180)
            idx_180 = bisect_left(c_dates, c_180_start)
            c_180_win = c_entries[idx_180:]
            cust_refund_count_180d = sum(1 for r in c_180_win if r['refund'] is not None and r['refund'] > 0)
            c_180_refunds = [r['refund'] for r in c_180_win if r['refund'] is not None]
            cust_total_refund_180d = sum(c_180_refunds) if c_180_refunds else 0.0
            
            # Project calculations using daily aggregates + active batch records
            p_30_start = current_date - pd.Timedelta(days=30)
            
            # 1. Sum up from daily project aggregates
            p_stats_list = proj_daily_stats.get(proj_id, [])
            hist_in_win = [s for s in p_stats_list if p_30_start <= s['date'] <= current_date]
            
            proj_txn_count = sum(s['txn_count'] for s in hist_in_win)
            proj_flags = sum(s['flags_count'] for s in hist_in_win)
            sum_proj_demand = sum(s['total_demand'] for s in hist_in_win)
            count_proj_demand = sum(s['count_demand'] for s in hist_in_win)
            sum_proj_outstanding = sum(s['total_outstanding'] for s in hist_in_win)
            count_proj_outstanding = sum(s['count_outstanding'] for s in hist_in_win)
            sum_proj_refund = sum(s['total_refund'] for s in hist_in_win)
            count_proj_refund = sum(s['count_refund'] for s in hist_in_win)
            
            # 2. Add active batch records processed so far
            active_in_win = [r for r in active_proj_records.get(proj_id, []) if p_30_start <= r['date'] <= current_date]
            proj_txn_count += len(active_in_win)
            proj_flags += sum(1 for r in active_in_win if r['is_flagged'])
            sum_proj_demand += sum(r['demand'] for r in active_in_win if r['demand'] is not None)
            count_proj_demand += sum(1 for r in active_in_win if r['demand'] is not None)
            sum_proj_outstanding += sum(r['outstanding'] for r in active_in_win if r['outstanding'] is not None)
            count_proj_outstanding += sum(1 for r in active_in_win if r['outstanding'] is not None)
            sum_proj_refund += sum(r['refund'] for r in active_in_win if r['refund'] is not None)
            count_proj_refund += sum(1 for r in active_in_win if r['refund'] is not None)
            
            # 3. Compute project rolling features
            proj_avg_demand = sum_proj_demand / count_proj_demand if count_proj_demand > 0 else 0.0
            proj_avg_outstanding = sum_proj_outstanding / count_proj_outstanding if count_proj_outstanding > 0 else 0.0
            proj_avg_refund = sum_proj_refund / count_proj_refund if count_proj_refund > 0 else 0.0
            proj_avg_outstanding_ratio = sum_proj_outstanding / sum_proj_demand if sum_proj_demand > 0 else 0.0
            
            # Unit calculations
            u_dates = unit_dates.get(unit_id, [])
            u_entries = unit_hist.get(unit_id, [])
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
            row_id, txn_id, cust_txn_count, cust_flags, cust_avg_delay, cust_avg_discount,
            cust_avg_refund, cust_avg_outstanding, cust_max_outstanding, cust_avg_outstanding_ratio,
            proj_txn_count, proj_flags, proj_avg_demand, proj_avg_outstanding, proj_avg_outstanding_ratio,
            proj_avg_refund, unit_txn_count, unit_flags, unit_owner_changes, unit_avg_outstanding,
            unit_avg_demand, unit_unique_customers, cust_refund_count_90d, cust_refund_count_180d,
            cust_total_refund_180d, cust_total_refund_lifetime, cust_discount_count_lifetime,
            cust_fraud_count_lifetime
        ))
        
    # 7. Persist to database in bulk
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