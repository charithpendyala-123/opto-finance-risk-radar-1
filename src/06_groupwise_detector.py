import sys
import os
import pandas as pd
import numpy as np
import json
import importlib

# Dynamically add parent directory to search path for safe imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

csv_loader = importlib.import_module("src.02_csv_loader")
load_finance_csv = csv_loader.load_finance_csv

iqr_detector = importlib.import_module("src.05_iqr_anomalydetector")
iqr = iqr_detector.iqr

# Replace detect_groupwise_anomalies signature and calculations (lines 17-90) with:
def detect_groupwise_anomalies(df, cache=None):
    """
    Highly optimized, vectorized Group-wise Contextual Outlier Engine.
    Modified to leverage pre-computed project aggregates from stats_cache.json
    and prevent collapsing bounds when IQR equals 0.
    """
    df_clean = df.copy()
    df_clean['original_row_num'] = df_clean.index + 2
    
    numeric_cols = ['demand_amount', 'collected_amount', 'outstanding_amount', 'discount_amount', 'refund_amount', 'payment_delay_days']
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
        
    df_clean['demand_date_parsed'] = pd.to_datetime(df_clean['demand_date'], errors='coerce')
    df_clean['payment_date_parsed'] = pd.to_datetime(df_clean['payment_date'], errors='coerce')

    anomaly_registry = {}
    
    # ── 1. VECTORIZED GROUP SIZES ──
    if cache and "projects" in cache:
        df_clean['project_id_size'] = df_clean['project_id'].map(lambda p: cache['projects'].get(p, {}).get('size', 0)).fillna(0)
    else:
        df_clean['project_id_size'] = df_clean.groupby('project_id')['project_id'].transform('size').fillna(0)
        
    df_clean['project_size'] = df_clean['project_id_size']
    df_clean['customer_id_size'] = df_clean.groupby('customer_id')['customer_id'].transform('size').fillna(0)
    df_clean['customer_size'] = df_clean['customer_id_size']
    df_clean['unit_id_size'] = df_clean.groupby('unit_id')['unit_id'].transform('size').fillna(0)
    df_clean['unit_size'] = df_clean['unit_id_size']

    # ── 1b. OPTIMIZED VECTORIZED STATS ──
    df_clean['customer_unique_projs'] = df_clean.groupby('customer_id')['project_id'].transform('nunique').fillna(0)
    df_clean['customer_proj_unique_units'] = df_clean.groupby(['customer_id', 'project_id'])['unit_id'].transform('nunique').fillna(0)
    df_clean['customer_demand_date_count'] = df_clean.groupby(['customer_id', 'demand_date'])['demand_date'].transform('size').fillna(0)

    # ── 2. PRE-CALCULATE LOCAL IQR BOUNDS (WITH ZERO IQR PREVENTION) ──
    # Project-level demand bounds
    if cache and "projects" in cache:
        def get_proj_demand_upper(p):
            p_stats = cache["projects"].get(p, {})
            if not p_stats: return 0.0
            q1, q3 = p_stats["demand_q1"], p_stats["demand_q3"]
            iqr_val = q3 - q1
            if iqr_val == 0: iqr_val = 0.1 * p_stats["median_demand"]
            return q3 + 1.5 * iqr_val
        df_clean['project_id_demand_upper'] = df_clean['project_id'].map(get_proj_demand_upper).fillna(0)
    else:
        q1 = df_clean.groupby('project_id')['demand_amount'].transform(lambda x: x.quantile(0.25))
        q3 = df_clean.groupby('project_id')['demand_amount'].transform(lambda x: x.quantile(0.75))
        iqr_val = q3 - q1
        # Fallback for 0 IQR
        med_proj = df_clean.groupby('project_id')['demand_amount'].transform('median').replace(0, 1.0)
        iqr_val = np.where(iqr_val == 0, 0.1 * med_proj, iqr_val)
        df_clean['project_id_demand_upper'] = q3 + 1.5 * iqr_val

    # Customer-level demand bounds
    q1_c = df_clean.groupby('customer_id')['demand_amount'].transform(lambda x: x.quantile(0.25))
    q3_c = df_clean.groupby('customer_id')['demand_amount'].transform(lambda x: x.quantile(0.75))
    iqr_c = q3_c - q1_c
    med_cust = df_clean.groupby('customer_id')['demand_amount'].transform('median').replace(0, 1.0)
    iqr_c = np.where(iqr_c == 0, 0.1 * med_cust, iqr_c)
    df_clean['customer_id_demand_upper'] = q3_c + 1.5 * iqr_c

    # Project-level collected bounds
    if cache and "projects" in cache:
        def get_proj_collected_upper(p):
            p_stats = cache["projects"].get(p, {})
            if not p_stats: return 0.0
            q1, q3 = p_stats["collected_q1"], p_stats["collected_q3"]
            iqr_val = q3 - q1
            if iqr_val == 0: iqr_val = 0.1 * p_stats["median_demand"]
            return q3 + 1.5 * iqr_val
        df_clean['project_id_collected_upper'] = df_clean['project_id'].map(get_proj_collected_upper).fillna(0)
    else:
        q1 = df_clean.groupby('project_id')['collected_amount'].transform(lambda x: x.quantile(0.25))
        q3 = df_clean.groupby('project_id')['collected_amount'].transform(lambda x: x.quantile(0.75))
        iqr_val = q3 - q1
        med_proj = df_clean.groupby('project_id')['collected_amount'].transform('median').replace(0, 1.0)
        iqr_val = np.where(iqr_val == 0, 0.1 * med_proj, iqr_val)
        df_clean['project_id_collected_upper'] = q3 + 1.5 * iqr_val

    # Customer-level collected bounds
    q1_cc = df_clean.groupby('customer_id')['collected_amount'].transform(lambda x: x.quantile(0.25))
    q3_cc = df_clean.groupby('customer_id')['collected_amount'].transform(lambda x: x.quantile(0.75))
    iqr_cc = q3_cc - q1_cc
    med_cust_c = df_clean.groupby('customer_id')['collected_amount'].transform('median').replace(0, 1.0)
    iqr_cc = np.where(iqr_cc == 0, 0.1 * med_cust_c, iqr_cc)
    df_clean['customer_id_collected_upper'] = q3_cc + 1.5 * iqr_cc

    # Project-level outstanding ratio bounds
    df_clean['outstanding_ratio'] = np.where(df_clean['demand_amount'] > 0, df_clean['outstanding_amount'] / df_clean['demand_amount'], 0)
    if cache and "projects" in cache:
        def get_proj_ratio_bounds(p):
            p_stats = cache["projects"].get(p, {})
            if not p_stats: return 0.0, 0.0
            q1, q3 = p_stats["ratio_q1"], p_stats["ratio_q3"]
            iqr_val = q3 - q1
            if iqr_val == 0: iqr_val = 0.05
            return q1 - 1.5 * iqr_val, q3 + 1.5 * iqr_val
        bounds = df_clean['project_id'].map(get_proj_ratio_bounds)
        df_clean['project_ratio_lower'] = [b[0] for b in bounds]
        df_clean['project_ratio_upper'] = [b[1] for b in bounds]
    else:
        q1_r = df_clean.groupby('project_id')['outstanding_ratio'].transform(lambda x: x.quantile(0.25))
        q3_r = df_clean.groupby('project_id')['outstanding_ratio'].transform(lambda x: x.quantile(0.75))
        iqr_r = q3_r - q1_r
        iqr_r = np.where(iqr_r == 0, 0.05, iqr_r)
        df_clean['project_ratio_lower'] = q1_r - 1.5 * iqr_r
        df_clean['project_ratio_upper'] = q3_r + 1.5 * iqr_r

    # ── 3. PRE-CALCULATE SYSTEM & VOLUME STATS ──
        # ── 3. PRE-CALCULATE SYSTEM & VOLUME STATS ──
    if '_is_new' in df_clean.columns:
        df_hist_only = df_clean[df_clean['_is_new'] == False]
    else:
        df_hist_only = df_clean

    hist_demand_counts = df_hist_only.groupby('demand_date')['transaction_id'].size()
    hist_payment_counts = df_hist_only.groupby('payment_date')['transaction_id'].size()
    
    def local_iqr(series):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr_val = q3 - q1
        if iqr_val == 0: iqr_val = 1.0
        return q3 + 1.5 * iqr_val
        
    d_upper = local_iqr(hist_demand_counts)
    p_upper = local_iqr(hist_payment_counts)
    
    comb_demand_counts = df_clean.groupby('demand_date')['transaction_id'].size()
    comb_payment_counts = df_clean.groupby('payment_date')['transaction_id'].size()

    df_clean['demand_date_count'] = df_clean['demand_date'].map(comb_demand_counts).fillna(0)
    df_clean['payment_date_count'] = df_clean['payment_date'].map(comb_payment_counts).fillna(0)

    df_clean['daily_proj_avg'] = df_clean.groupby(['project_id', 'demand_date'])['demand_amount'].transform('mean')
    
    if cache and "projects" in cache:
        df_clean['proj_median_30d'] = df_clean['project_id'].map(lambda p: cache["projects"].get(p, {}).get("median_demand", 0.0)).fillna(0)
    else:
        df_clean['proj_median_30d'] = df_clean.groupby('project_id')['demand_amount'].transform('median')

    df_clean = df_clean.sort_values(by=['unit_id', 'customer_id', 'demand_date_parsed']).reset_index(drop=True)
# (Keep the rest of the file loops and checks completely unchanged)

    records = df_clean.to_dict('records')
    current_unit_id = None
    unit_window = []

    def nunique_safe(lst):
        filtered = [x for x in lst if pd.notna(x) and x != 'N/A' and str(x).strip() != '' and str(x).lower() != 'nan']
        return len(set(filtered))

    # ── 4. EXECUTE ALL 15 FORENSIC CHECKS INSTANTLY ──
    for idx, row in enumerate(records):
        row_num = int(row['original_row_num'])
        raw_txn_id = row['transaction_id']
        
        if pd.isna(raw_txn_id) or str(raw_txn_id).strip() == "" or str(raw_txn_id).lower() == 'nan':
            txn_id = f"SYS_MISSING_ROW_{row_num}"
        else:
            txn_id = str(raw_txn_id)
            
        cust_id = row['customer_id']
        proj_id = row['project_id']
        unit_id = row['unit_id']

        if txn_id not in anomaly_registry:
            anomaly_registry[txn_id] = []

        # CATEGORY 1: Dynamic Statistical Checks (IQRs)
        # Case 1: Demand Spikes
        for lens in ['project_id', 'customer_id']:
            if row[f'{lens}_size'] > 1 and row['demand_amount'] > row[f'{lens}_demand_upper']:
                anomaly_registry[txn_id].append({
                    "lens": lens, "case": "Demand Outlier Flag",
                    "reason": f"Demand ({row['demand_amount']:,.0f}) exceeds local fence ({row[f'{lens}_demand_upper']:,.0f})"
                })
        
        # Case 2: Collection Spikes
        for lens in ['project_id', 'customer_id']:
            if row[f'{lens}_size'] > 1 and row['collected_amount'] > row[f'{lens}_collected_upper']:
                anomaly_registry[txn_id].append({
                    "lens": lens, "case": "Collection Outlier Flag",
                    "reason": f"Collection ({row['collected_amount']:,.0f}) exceeds local fence ({row[f'{lens}_collected_upper']:,.0f})"
                })

        # Case 3: Overpayments & Phantom Billing
        if row['project_size'] > 1 and row['demand_amount'] > 0:
            ratio = row['outstanding_ratio']
            if ratio < row['project_ratio_lower']:
                anomaly_registry[txn_id].append({
                    "lens": "project_id", "case": "Laundering Cash Flag",
                    "reason": f"Owed/Demand ratio ({ratio:.2f}) falls below local fence ({row['project_ratio_lower']:.2f}) - Potential Overpayment"
                })
            elif ratio > row['project_ratio_upper']:
                anomaly_registry[txn_id].append({
                    "lens": "project_id", "case": "Phantom Billing Flag",
                    "reason": f"Owed/Demand ratio ({ratio:.2f}) exceeds local fence ({row['project_ratio_upper']:.2f}) - Potential Phantom Billing"
                })

        # CATEGORY 2: Logical Integrity Checks
        # Case 4: Phantom Settlement
        if row['payment_delay_days'] > 2 and row['outstanding_amount'] == 0 and row['collected_amount'] == 0:
            anomaly_registry[txn_id].append({
                "lens": "unit_id", "case": "Critical Fraud Flag: Phantom Settlement",
                "reason": "Payment delay > 2 recorded with zero collections and zero outstanding balance reduction."
            })

        # Case 5: Ghost Collection Pattern
        if idx > 0:
            prev = records[idx - 1]
            if prev['customer_id'] == cust_id:
                collected_diff = row['collected_amount'] - prev['collected_amount']
                outstanding_diff = prev['outstanding_amount'] - row['outstanding_amount']
                if collected_diff > 1000 and outstanding_diff < (0.5 * collected_diff) and (row['demand_amount'] - prev['demand_amount']) <= 0:
                    anomaly_registry[txn_id].append({
                        "lens": "customer_id", "case": "Critical Fraud Flag: Ghost Collection",
                        "reason": f"Collected cash increased by {collected_diff:,.0f}, but outstanding balance only dropped by {outstanding_diff:,.0f}."
                    })

        # CATEGORY 3: Business Policy Checks
        # Case 6: Kickback Discount
        if row['demand_amount'] > 0 and (row['discount_amount'] / row['demand_amount']) > 0.15 and row['payment_delay_days'] > 15:
            anomaly_registry[txn_id].append({
                "lens": "customer_id", "case": "High Policy Outlier Flag: Kickback Discount",
                "reason": f"High discount ({row['discount_amount']/row['demand_amount']*100:.0f}%) approved on payment delayed by {row['payment_delay_days']} days."
            })

        # Case 7: Default Evasion Scheme (Alternating payments)
        if idx >= 3:
            seq = records[idx-3 : idx+1]
            if nunique_safe([r['unit_id'] for r in seq]) == 1:
                vals = [r['collected_amount'] for r in seq]
                if (vals[0] > 0 and vals[1] == 0 and vals[2] > 0 and vals[3] == 0) or \
                   (vals[0] == 0 and vals[1] > 0 and vals[2] == 0 and vals[3] > 0):
                    anomaly_registry[txn_id].append({
                        "lens": "unit_id", "case": "Default Evasion Flag",
                        "reason": "Alternating payment pattern (Value -> 0 -> Value -> 0) detected across 4 consecutive unit logs."
                    })

        # Case 8: Boiling Frog (Engineered billing escalation)
        if idx >= 2:
            seq = records[idx-2 : idx+1]
            if nunique_safe([r['customer_id'] for r in seq]) == 1:
                vals = [r['demand_amount'] for r in seq]
                if vals[0] < vals[1] < vals[2]:
                    anomaly_registry[txn_id].append({
                        "lens": "customer_id", "case": "Billing Escalation Flag",
                        "reason": f"Monotonically increasing demand amounts ({vals[0]:,.0f} -> {vals[1]:,.0f} -> {vals[2]:,.0f}) across 3 sequential invoices."
                    })

        # Case 9: Silent Liability Write-down
        if idx >= 2:
            seq = records[idx-2 : idx+1]
            if nunique_safe([r['unit_id'] for r in seq]) == 1:
                vals = [r['demand_amount'] for r in seq]
                if vals[0] > vals[1] > vals[2] and all(r['collected_amount'] == 0 for r in seq) and all(r['discount_amount'] == 0 for r in seq):
                    anomaly_registry[txn_id].append({
                        "lens": "unit_id", "case": "Liability Write-down Flag",
                        "reason": f"Monotonically decreasing demand amounts ({vals[0]:,.0f} -> {vals[1]:,.0f} -> {vals[2]:,.0f}) with zero collections/discounts."
                    })

        # CATEGORY 4: System & Allocation Outliers
        # Case 10: Multi-Project Customer
        unique_projs = int(row['customer_unique_projs'])
        if unique_projs > 1:
            anomaly_registry[txn_id].append({
                "lens": "customer_id", "case": "High Credit / Speculation Exposure",
                "reason": f"Customer is linked to {unique_projs} different real-estate projects."
            })

        # Case 11: Multi-Unit Customer
        unique_units = int(row['customer_proj_unique_units'])
        if unique_units > 2:
            anomaly_registry[txn_id].append({
                "lens": "customer_id", "case": "High Default Risk",
                "reason": f"Customer booked {unique_units} units within project {proj_id}."
            })

        # Case 12: System Bulk Volume Spikes
        if pd.notna(d_upper) and row['demand_date_count'] > d_upper:
            anomaly_registry[txn_id].append({
                "lens": "demand_date", "case": "Bulk System Glitch Flag",
                "reason": f"System daily demand volume ({row['demand_date_count']:.0f} invoices) exceeds upper global threshold ({d_upper})."
            })
        if pd.notna(p_upper) and row['payment_date_count'] > p_upper:
            anomaly_registry[txn_id].append({
                "lens": "payment_date", "case": "Bulk System Glitch Flag",
                "reason": f"System daily collection volume ({row['payment_date_count']:.0f} payments) exceeds upper global threshold ({p_upper})."
            })

        # Case 13: Project-Level Distribution Shift
        if row['daily_proj_avg'] > (3 * row['proj_median_30d']) and row['proj_median_30d'] > 0:
            anomaly_registry[txn_id].append({
                "lens": "project_id", "case": "Project Distribution Shift Flag",
                "reason": f"Daily project average demand ({row['daily_proj_avg']:,.0f}) shifted beyond 3x historical median ({row['proj_median_30d']:,.0f})."
            })

        # Case 14: Date Compression Burst
        same_day_count = int(row['customer_demand_date_count'])
        if same_day_count >= 4:
            anomaly_registry[txn_id].append({
                "lens": "customer_id", "case": "Compressed Activity Burst Flag",
                "reason": f"Customer registered {same_day_count} invoices on the exact same day ({row['demand_date']})."
            })

        # Case 15: Unit Recycling Pattern
        if unit_id != current_unit_id:
            current_unit_id = unit_id
            unit_window = []
        if pd.notna(cust_id) and cust_id != 'N/A' and str(cust_id).strip() != '' and str(cust_id).lower() != 'nan':
            unit_window.append((row['demand_date_parsed'], cust_id))
        cutoff = row['demand_date_parsed'] - pd.Timedelta(days=15)
        unit_window = [w for w in unit_window if w[0] >= cutoff]
        unique_owners = len(set(w[1] for w in unit_window))
        if unique_owners >= 3:
            anomaly_registry[txn_id].append({
                "lens": "unit_id", "case": "Rapid Unit Reassignment Flag",
                "reason": f"Physical property unit reassigned across {unique_owners} unique customers in a 15-day window."
            })

    return df_clean, anomaly_registry

if __name__ == "__main__":
    df = load_finance_csv("data/sample_finance_data.csv")
    if df is not None:
        df_analyzed, registry = detect_groupwise_anomalies(df)
        
        # Compile only flagged transactions with strict JSON validation
        flagged_transactions = []
        for idx, row in df_analyzed.iterrows():
            # --- THE EXPORT HOOPHOLE PATCH: Generate Virtual ID if blank to match registry ---
            row_num = int(row['original_row_num'])  # Matches actual Excel/CSV row number (1-indexed + header)
            raw_txn_id = row['transaction_id']
            
            if pd.isna(raw_txn_id) or str(raw_txn_id).strip() == "" or str(raw_txn_id).lower() == 'nan':
                txn_id = f"SYS_MISSING_ROW_{row_num}"
            else:
                txn_id = str(raw_txn_id)
                
            if txn_id in registry and len(registry[txn_id]) > 0:
                
                # Helper to map NaN, NaT, and float nan to Python None (JSON null)
                def sanitize(val, is_num=False):
                    if pd.isna(val) or val == 'nan' or val == 'NaN':
                        return None
                    if is_num:
                        try:
                            f_val = float(val)
                            if np.isnan(f_val) or np.isinf(f_val):
                                return None
                            return f_val
                        except:
                            return None
                    return str(val)
                def sanitize_int(val):
                    if pd.isna(val) or (isinstance(val, float) and np.isnan(val)):
                        return None
                    try:
                        return int(float(val))
                    except:
                        return None
                record = {
                    "transaction_id": txn_id, # Uses the correct patched ID (virtual or raw)
                    "customer_id": sanitize(row['customer_id']),
                    "project_id": sanitize(row['project_id']),
                    "unit_id": sanitize(row['unit_id']),
                    "demand_amount": sanitize(row['demand_amount'], is_num=True),
                    "collected_amount": sanitize(row['collected_amount'], is_num=True),
                    "outstanding_amount": sanitize(row['outstanding_amount'], is_num=True),
                    "discount_amount": sanitize(row['discount_amount'], is_num=True),
                    "refund_amount": sanitize(row['refund_amount'], is_num=True),
                    "payment_delay_days": sanitize_int(row['payment_delay_days']),
                    "demand_date": sanitize(row['demand_date']),
                    "payment_date": sanitize(row['payment_date']),
                    "groupwise_anomalies": registry[txn_id]
                }
                flagged_transactions.append(record)
                
        # Ensure reports directory exists and save to reports/groupwise_anomalies.json
        os.makedirs('reports', exist_ok=True)
        output_path = 'reports/groupwise_anomalies.json'
        with open(output_path, 'w') as f:
            json.dump(flagged_transactions, f, indent=4)
            
        print(f"Groupwise detection test run complete. Total transactions: {len(df_analyzed)}")
        print(f"Transactions flagged with contextual anomalies: {len(flagged_transactions)}")
        print(f"Report successfully saved to: {output_path}")