import sys
import os
import pandas as pd
import numpy as np
import json
import importlib
from sklearn.ensemble import IsolationForest

# Dynamically add parent directory to search path for safe imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

csv_loader = importlib.import_module("src.02_csv_loader")
load_finance_csv = csv_loader.load_finance_csv

zscore_detector = importlib.import_module("src.04_zscore_anomalydetector")
z_score = zscore_detector.z_score
get_zscore_reason = zscore_detector.get_anomaly_reason

iqr_detector = importlib.import_module("src.05_iqr_anomalydetector")
iqr = iqr_detector.iqr
get_iqr_reason = iqr_detector.get_anomaly_reason

groupwise_detector = importlib.import_module("src.06_groupwise_detector")
detect_groupwise_anomalies = groupwise_detector.detect_groupwise_anomalies

if_detector = importlib.import_module("src.07_IF_anomaly_detector")
prepare_features = if_detector.prepare_features
generate_if_reason = if_detector.generate_if_reason

# ── SANITIZATION HELPERS ──
# Prevents unquoted float NaN, NaT, or Infinity leaks in JSON exports

def sanitize_float(val):
    if pd.isna(val):
        return None
    try:
        f_val = float(val)
        if np.isnan(f_val) or np.isinf(f_val):
            return None
        return f_val
    except (ValueError, TypeError):
        return None

def sanitize_int(val):
    if pd.isna(val) or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None

def sanitize_str(val):
    if pd.isna(val) or str(val).strip() == "" or str(val).lower() == "nan":
        return None
    return str(val)


# ── UPSTREAM VIRTUAL ID PATCHING ──

def patch_transaction_ids(df):
    """
    Applies the Upstream Virtual ID Patching strategy globally.
    Ensures all missing, blank, or literal 'nan' transaction IDs are permanently
    assigned an identical virtual primary key 'SYS_MISSING_ROW_{idx+2}' at ingestion.
    """
    df_patched = df.copy()
    tx_id_col = df_patched['transaction_id']
    is_missing = tx_id_col.isna() | (tx_id_col.astype(str).str.strip() == "") | (tx_id_col.astype(str).str.lower() == "nan")
    if is_missing.any():
        virtual_ids = "SYS_MISSING_ROW_" + (df_patched.index + 2).astype(str)
        df_patched.loc[is_missing, 'transaction_id'] = virtual_ids[is_missing]
        patched_count = is_missing.sum()
        print(f"[Upstream Patch] Assigned {patched_count} virtual transaction IDs to prevent alignment gaps.")
    return df_patched


# ── MAIN PIPELINE ORCHESTRATION ──

def run_anomaly_pipeline(csv_path="data/sample_finance_data.csv", z_threshold=3.0, contamination='auto', conn=None, user_id="system_default", batch_id=None):
    """
    Executes all four statistical and machine learning engines in parallel,
    aligning them via the upstream patched transaction IDs, and merges results
    into a unified anomaly auditing dataset.
    """
    import joblib
    # ── BULLETPROOF WINDOWS PATH RESOLUTION ──
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(csv_path):
        csv_path = os.path.abspath(os.path.join(base_dir, csv_path))

    # 1. Ingest dataset using CSV Loader
    df_raw = load_finance_csv(csv_path)
    if df_raw is None:
        print("Error: Ingested DataFrame is empty or structural check failed.")
        return None

    # 2. Perform Upstream Virtual ID Patching (Ensures exact primary keys in memory)
    df = patch_transaction_ids(df_raw)

    # 3. Clean and convert numeric columns globally
    numeric_cols = ['demand_amount', 'collected_amount', 'outstanding_amount', 'discount_amount', 'refund_amount', 'payment_delay_days']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Load cache and pre-trained models if they exist
    cache = None
    cache_path = os.path.abspath(os.path.join(base_dir, "models", "stats_cache.json"))
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cache = json.load(f)
            print("[Anomaly Pipeline] Loaded pre-computed historical statistics cache.")
        except Exception as e:
            print(f"Warning: Failed to load stats cache: {e}")

    print("\n[Engine Initialization] Running parallel anomaly audit sub-engines...")

    # ── ENGINE 1: GLOBAL Z-SCORE SUB-ENGINE ──
    df['demand_amount_z_score'] = z_score(df['demand_amount'], 'demand_amount', cache)
    df['collected_amount_z_score'] = z_score(df['collected_amount'], 'collected_amount', cache)
    df['outstanding_amount_z_score'] = z_score(df['outstanding_amount'], 'outstanding_amount', cache)
    df['discount_amount_z_score'] = z_score(df['discount_amount'], 'discount_amount', cache)
    df['refund_amount_z_score'] = z_score(df['refund_amount'], 'refund_amount', cache)
    df['payment_delay_days_z_score'] = z_score(df['payment_delay_days'], 'payment_delay_days', cache)

    # ── ENGINE 2: ROBUST IQR SUB-ENGINE ──
    for col in numeric_cols:
        lower, upper, true_iqr = iqr(df[col], col, cache)
        df[f'{col}_lower_bound'] = lower
        df[f'{col}_upper_bound'] = upper
        df[f'{col}_iqr'] = true_iqr

    # ── ENGINE 3: CONTEXTUAL GROUP-WISE SUB-ENGINE ──
    # Replace the Groupwise Engine step (lines 121-124) with:
    # ── ENGINE 3: CONTEXTUAL GROUP-WISE SUB-ENGINE ──
    groupwise_registry = {}
    if conn and user_id:
        try:
            print("[Groupwise Engine] Fetching historical context for active entities...")
            active_customers = [c for c in df['customer_id'].unique() if pd.notna(c) and c != 'N/A' and str(c).strip() != '']
            active_units = [u for u in df['unit_id'].unique() if pd.notna(u) and u != 'N/A' and str(u).strip() != '']
            
            df['demand_date_parsed'] = pd.to_datetime(df['demand_date'], errors='coerce')
            min_date = df['demand_date_parsed'].min()
            cutoff_date = min_date - pd.Timedelta(days=180) if pd.notna(min_date) else None
            
            conditions = ["t.user_id = %s"]
            params = [user_id]
            if batch_id:
                conditions.append("t.upload_batch_id != %s")
                params.append(batch_id)
            if cutoff_date:
                conditions.append("t.demand_date >= %s")
                params.append(cutoff_date.date())
                
            entity_conds = []
            if active_customers:
                entity_conds.append("t.customer_id IN %s")
                params.append(tuple(active_customers))
            if active_units:
                entity_conds.append("t.unit_id IN %s")
                params.append(tuple(active_units))
                
            if entity_conds:
                conditions.append("(" + " OR ".join(entity_conds) + ")")
                
                query_hist = f"""
                    SELECT 
                        t.transaction_id, t.customer_id, t.project_id, t.unit_id,
                        t.demand_date, t.payment_date, t.demand_amount, t.collected_amount,
                        t.outstanding_amount, t.discount_amount, t.refund_amount,
                        t.payment_delay_days, t.payment_gap_days, t.upload_batch_id
                    FROM transactions t
                    WHERE {" AND ".join(conditions)}
                    ORDER BY t.demand_date ASC, t.id ASC;
                """
                df_hist = pd.read_sql(query_hist, conn, params=params)
                
                # Suffix historical transaction IDs to prevent colliding keys in the registry
                if not df_hist.empty:
                    df_hist['transaction_id'] = df_hist['transaction_id'] + '_HIST_' + df_hist['upload_batch_id'].fillna('UNK')
                
                print(f"[Groupwise Engine] Loaded {len(df_hist)} contextual history transactions.")
                
                df_temp = df.copy()
                df_temp['_is_new'] = True
                df_hist['_is_new'] = False
                
                df_combined = pd.concat([df_hist, df_temp], ignore_index=True)
                _, full_registry = detect_groupwise_anomalies(df_combined, cache=cache)
                
                new_txn_ids = set(df_temp['transaction_id'].unique())
                groupwise_registry = {k: v for k, v in full_registry.items() if k in new_txn_ids}
            else:
                _, groupwise_registry = detect_groupwise_anomalies(df, cache=cache)
        except Exception as e:
            print(f"Warning: Groupwise historical context lookup failed: {e}. Falling back to batch-only.")
            _, groupwise_registry = detect_groupwise_anomalies(df, cache=cache)
    else:
        _, groupwise_registry = detect_groupwise_anomalies(df, cache=cache)

    # ── ENGINE 4: UNSUPERVISED ISOLATION FOREST ML ENGINE ──
    df_engineered = prepare_features(df)
    features = [
        "demand_amount",
        "collected_amount",
        "outstanding_amount",
        "discount_amount",
        "refund_amount",
        "payment_delay_days",
        "payment_gap_days",
        "demand_dayofweek",
        "payment_dayofweek"
    ]
    
    missing_feats = [c for c in features if c not in df_engineered.columns]
    if missing_feats:
        raise ValueError(f"Missing expected engineered machine learning features: {missing_feats}")
        
    X = df_engineered[features].fillna(0)
    
    # Load pre-trained Isolation Forest model if available
    model = None
    model_path = os.path.abspath(os.path.join(base_dir, "models", "isolation_forest.joblib"))
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            print("[Anomaly Pipeline] Loaded pre-trained Isolation Forest model.")
        except Exception as e:
            print(f"Warning: Failed to load Isolation Forest model: {e}")
            
    if model is None:
        print("[Anomaly Pipeline] Training a new Isolation Forest model on the fly...")
        n_est = int(min(200, max(50, len(df) // 10)))
        model = IsolationForest(
            n_estimators=n_est,
            max_samples='auto',
            contamination=contamination,
            random_state=42
        )
        model.fit(X)
        
    df_engineered["if_prediction"] = model.predict(X)
    df_engineered["if_score"] = model.decision_function(X)

    # Compute/Load Isolation Forest thresholds for reason generator
    thresholds = {}
    if cache and "isolation_forest_thresholds" in cache:
        thresholds = cache["isolation_forest_thresholds"]
    else:
        for col in ['refund_amount', 'payment_delay_days', 'discount_amount', 'payment_gap_days']:
            mean_val = df_engineered[col].mean()
            std_val = df_engineered[col].std()
            std_val = std_val if std_val > 0 else 1.0
            thresholds[col] = mean_val + 2.5 * std_val

    # ── CONSOLIDATE SUB-ENGINE OUTPUTS ──
    flagged_records = []
    engineered_dict = df_engineered.drop_duplicates(subset=['transaction_id']).set_index('transaction_id').to_dict('index')

    total_zscore_flags = 0
    total_iqr_flags = 0
    total_groupwise_flags = 0
    total_if_flags = 0

    print("\n[Consolidation] Packaging parallel forensic reports and validating schemas...")

    df_records = df.to_dict('records')
    for idx, row in enumerate(df_records):
        txn_id = row['transaction_id']
        
        # 1. Z-Score evaluation
        zscore_flagged = False
        for col in numeric_cols:
            z_val = row[f'{col}_z_score']
            if pd.notna(z_val) and abs(z_val) >= z_threshold:
                zscore_flagged = True
                
        zscore_reason = get_zscore_reason(row, z_threshold) if zscore_flagged else ""
        if zscore_flagged:
            total_zscore_flags += 1

        # 2. IQR evaluation
        iqr_flagged = False
        for col in numeric_cols:
            val = row[col]
            lower = row[f'{col}_lower_bound']
            upper = row[f'{col}_upper_bound']
            if pd.notna(val) and pd.notna(lower) and pd.notna(upper):
                if val < lower or val > upper:
                    iqr_flagged = True
                    
        iqr_reason = get_iqr_reason(row) if iqr_flagged else ""
        if iqr_flagged:
            total_iqr_flags += 1

        # 3. Group-wise Contextual evaluation
        groupwise_anomalies = groupwise_registry.get(txn_id, [])
        groupwise_flagged = len(groupwise_anomalies) > 0
        if groupwise_flagged:
            total_groupwise_flags += 1

        # 4. Isolation Forest evaluation
        if_flagged = False
        if_score = 0.0
        if_reason = ""
        
        eng_row = engineered_dict.get(txn_id)
        if eng_row:
            if_flagged = eng_row['if_prediction'] == -1
            if_score = float(eng_row['if_score'])
            if_reason = generate_if_reason(eng_row, thresholds) if if_flagged else ""
            
        if if_flagged:
            total_if_flags += 1

        if zscore_flagged or iqr_flagged or groupwise_flagged or if_flagged:
            engines = []
            if zscore_flagged: engines.append("Z-Score")
            if iqr_flagged: engines.append("IQR")
            if groupwise_flagged: engines.append("Group-wise")
            if if_flagged: engines.append("Isolation Forest")

            record = {
                "transaction_id": txn_id,
                "customer_id": sanitize_str(row['customer_id']),
                "project_id": sanitize_str(row['project_id']),
                "unit_id": sanitize_str(row['unit_id']),
                "demand_amount": sanitize_float(row['demand_amount']),
                "collected_amount": sanitize_float(row['collected_amount']),
                "outstanding_amount": sanitize_float(row['outstanding_amount']),
                "discount_amount": sanitize_float(row['discount_amount']),
                "refund_amount": sanitize_float(row['refund_amount']),
                "payment_delay_days": sanitize_int(row['payment_delay_days']),
                "demand_date": sanitize_str(row['demand_date']),
                "payment_date": sanitize_str(row['payment_date']),
                
                "engines_flagged": engines,
                "total_engines_flagged": len(engines),
                
                "zscore_details": {
                    "flagged": bool(zscore_flagged),
                    "reason": zscore_reason if zscore_flagged else None
                },
                "iqr_details": {
                    "flagged": bool(iqr_flagged),
                    "reason": iqr_reason if iqr_flagged else None
                },
                "groupwise_details": {
                    "flagged": bool(groupwise_flagged),
                    "anomalies": groupwise_anomalies if groupwise_flagged else []
                },
                "isolation_forest_details": {
                    "flagged": bool(if_flagged),
                    "score": round(if_score, 4),
                    "reason": if_reason if if_flagged else None
                }
            }
            flagged_records.append(record)

    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.abspath(os.path.join(reports_dir, "anomaly_report.json"))
    
    with open(report_path, "w") as f:
        json.dump(flagged_records, f, indent=4)

    print(f"\n=======================================================")
    print(f"     CONSOLIDATED ANOMALY RADAR AUDIT COMPLETE")
    print(f"=======================================================")
    print(f"Total Transactions Audited: {len(df)}")
    print(f"Flagged Anomalous Transactions: {len(flagged_records)}")
    print(f"Consolidated Report Saved To: {report_path}")
    print(f"\nSub-Engine Outlier Detections:")
    print(f"  - Z-Score Outliers Flagged          : {total_zscore_flags}")
    print(f"  - IQR Bounds Outliers Flagged        : {total_iqr_flags}")
    print(f"  - Contextual Group-wise Outliers     : {total_groupwise_flags}")
    print(f"  - Isolation Forest ML Outliers       : {total_if_flags}")
    print(f"=======================================================\n")

    return flagged_records

if __name__ == "__main__":
    run_anomaly_pipeline(csv_path="data/sample_finance_data.csv")