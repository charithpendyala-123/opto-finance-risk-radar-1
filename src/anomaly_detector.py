import sys
import os
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import IsolationForest

# Dynamically add parent directory to search path for safe imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.csv_loader import load_finance_csv
from src.zscore_anomalydetector import z_score, get_anomaly_reason as get_zscore_reason
from src.iqr_anomalydetector import iqr, get_anomaly_reason as get_iqr_reason
from src.groupwise_detector import detect_groupwise_anomalies
from src.IF_anomaly_detector import prepare_features, generate_if_reason, safe_txn_id

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
    except:
        return None

def sanitize_int(val):
    if pd.isna(val) or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return int(float(val))
    except:
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
    patched_count = 0
    for idx, row in df_patched.iterrows():
        val = row.get('transaction_id')
        row_num = idx + 2  # Physical CSV/Excel row number (0-based index + header offset)
        if pd.isna(val) or str(val).strip() == "" or str(val).lower() == "nan":
            df_patched.at[idx, 'transaction_id'] = f"SYS_MISSING_ROW_{row_num}"
            patched_count += 1
    if patched_count > 0:
        print(f"[Upstream Patch] Assigned {patched_count} virtual transaction IDs to prevent alignment gaps.")
    return df_patched


# ── MAIN PIPELINE ORCHESTRATION ──

def run_anomaly_pipeline(csv_path="data/sample_finance_data.csv", z_threshold=3.0, contamination='auto'):
    """
    Executes all four statistical and machine learning engines in parallel,
    aligning them via the upstream patched transaction IDs, and merges results
    into a unified anomaly auditing dataset.
    """
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

    print("\n[Engine Initialization] Running parallel anomaly audit sub-engines...")

    # ── ENGINE 1: GLOBAL Z-SCORE SUB-ENGINE (Leaves NaNs intact for proper mean/std math) ──
    df['demand_amount_z_score'] = z_score(df['demand_amount'])
    df['collected_amount_z_score'] = z_score(df['collected_amount'])
    df['outstanding_amount_z_score'] = z_score(df['outstanding_amount'])
    df['discount_amount_z_score'] = z_score(df['discount_amount'])
    df['refund_amount_z_score'] = z_score(df['refund_amount'])
    df['payment_delay_days_z_score'] = z_score(df['payment_delay_days'])

    # ── ENGINE 2: ROBUST IQR SUB-ENGINE (Leaves NaNs intact for proper Tukey bounds quantiles) ──
    numeric_cols = ['demand_amount', 'collected_amount', 'outstanding_amount', 'discount_amount', 'refund_amount', 'payment_delay_days']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert, but DO NOT fillna(0) here!
        
    for col in numeric_cols:
        lower, upper, true_iqr = iqr(df[col])
        df[f'{col}_lower_bound'] = lower
        df[f'{col}_upper_bound'] = upper
        df[f'{col}_iqr'] = true_iqr

    # ── ENGINE 3: CONTEXTUAL GROUP-WISE SUB-ENGINE ──
    df_groupwise, groupwise_registry = detect_groupwise_anomalies(df)

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
    X = df_engineered[features].fillna(0)
    
    # Calculate Isolation Forest estimators dynamically based on dataset volume
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

    # Compute dynamic thresholds for Isolation Forest reason generator
    thresholds = {}
    for col in ['refund_amount', 'payment_delay_days', 'discount_amount', 'payment_gap_days']:
        mean_val = df_engineered[col].mean()
        std_val = df_engineered[col].std()
        std_val = std_val if std_val > 0 else 1.0
        thresholds[col] = mean_val + 2.5 * std_val

    # ── CONSOLIDATE SUB-ENGINE OUTPUTS ──
    flagged_records = []
    
    # Create lookup map for engineered features to keep consolidation rapid
    engineered_map = df_engineered.set_index('transaction_id')

    total_zscore_flags = 0
    total_iqr_flags = 0
    total_groupwise_flags = 0
    total_if_flags = 0

    print("\n[Consolidation] Packaging parallel forensic reports and validating schemas...")

    for idx, row in df.iterrows():
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

        # 2. IQR evaluation (NaN values naturally bypass inequality checks and are not flagged)
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
        
        if txn_id in engineered_map.index:
            eng_row = engineered_map.loc[txn_id]
            # Handle duplicate index edge case safely
            if isinstance(eng_row, pd.DataFrame):
                eng_row = eng_row.iloc[0]
            if_flagged = eng_row['if_prediction'] == -1
            if_score = float(eng_row['if_score'])
            if_reason = generate_if_reason(eng_row, thresholds) if if_flagged else ""
            
        if if_flagged:
            total_if_flags += 1

        # A transaction is flagged if ANY of the four sub-engines reports an anomaly
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

    # 5. Save the consolidated anomaly report using a secure absolute path
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.abspath(os.path.join(reports_dir, "anomaly_report.json"))
    
    with open(report_path, "w") as f:
        json.dump(flagged_records, f, indent=4)

    # Print summary metrics to console
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