import sys
import os
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import IsolationForest

# Dynamically add parent directory to search path for safe imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.csv_loader import load_finance_csv

def safe_txn_id(val, row_num):
    """
    Safely resolves a transaction ID, falling back to a virtual physical
    row coordinate if the original ID is missing, blank, or a literal 'nan' string.
    """
    if pd.isna(val) or str(val).strip() == "" or str(val).lower() == "nan":
        return f"SYS_MISSING_ROW_{row_num}"
    return str(val)


# ── STEP 1: Feature Engineering Function (Blueprint Step 5) ──
def prepare_features(df):
    """
    Cleans financial columns, parses dates, and constructs derived
    temporal features required for multivariate anomaly detection.
    """
    # Create a copy to prevent modifying the original DataFrame
    df_feat = df.copy()
    
    # A) Convert numeric columns safely
    numeric_cols = [
        "demand_amount",
        "collected_amount",
        "outstanding_amount",
        "discount_amount",
        "refund_amount",
        "payment_delay_days"
    ]
    
    for col in numeric_cols:
        df_feat[col] = pd.to_numeric(df_feat[col], errors="coerce").fillna(0)
        
    # B) Convert date columns safely
    df_feat["demand_date_parsed"] = pd.to_datetime(df_feat["demand_date"], errors="coerce")
    df_feat["payment_date_parsed"] = pd.to_datetime(df_feat["payment_date"], errors="coerce")
    
    # C) Create derived columns
    # 1. Payment Gap (Clearance time in days)
    df_feat["payment_gap_days"] = (df_feat["payment_date_parsed"] - df_feat["demand_date_parsed"]).dt.days.fillna(-1)
    
    # 2. Demand weekday (0 = Monday, 6 = Sunday)
    df_feat["demand_dayofweek"] = df_feat["demand_date_parsed"].dt.dayofweek.fillna(-1).astype(int)
    
    # 3. Payment weekday (0 = Monday, 6 = Sunday)
    df_feat["payment_dayofweek"] = df_feat["payment_date_parsed"].dt.dayofweek.fillna(-1).astype(int)
    
    return df_feat


# ── STEP 2: Anomaly Reason Generator (Blueprint Step 13 - Now Fully Dynamic!) ──
def generate_if_reason(row, thresholds):
    """
    Generates a context-aware reason based on dataset-specific
    statistical standard deviation thresholds.
    """
    reasons = []
    
    # Check if the transaction's metrics exceed 2.5 standard deviations from the dataset mean
    if abs(row['refund_amount']) > thresholds['refund_amount']:
        reasons.append("abnormal refund volume")
    if abs(row['payment_delay_days']) > thresholds['payment_delay_days']:
        reasons.append("extreme payment latency")
    if abs(row['discount_amount']) > thresholds['discount_amount']:
        reasons.append("high discount exposure")
    if abs(row['payment_gap_days']) > thresholds['payment_gap_days']:
        reasons.append("abnormal invoice-to-payment window")
        
    if reasons:
        return f"Abnormal combination of {', '.join(reasons)} and Temporal Behavior"
    return "Abnormal combination of financial and temporal behavior"


# ── STEP 3: The Main Entrypoint Runner (Blueprint Step 17) ──
if __name__ == "__main__":
    # 1. Ingest sample data using CSV Loader
    df = load_finance_csv("data/sample_finance_data.csv")
    
    # FIX BUG 1: Safe-check. Stop immediately if data is missing before calling len(df)
    if df is None:
        import sys
        sys.exit("Error: Data file is empty or missing. Model training aborted.")
    
    # 2. Engineer features (FIX BUG 2: Safely wrapped inside the main block)
    df_engineered = prepare_features(df)
    
    # 3. Define the final features list (Blueprint Step 6)
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
    
    # 4. Build feature matrix (X) (Blueprint Step 7)
    X = df_engineered[features].fillna(0)
    
    # 5. Initialize Isolation Forest model with dynamic tree calculation (Blueprint Step 8)
    n_est = int(min(200, max(50, len(df) // 10)))
    model = IsolationForest(
        n_estimators=n_est,
        max_samples='auto',
        contamination='auto',
        random_state=42
    )
    
    # 6. Train model (Blueprint Step 9)
    model.fit(X)
    
    # 7. Predict predictions (1 = normal, -1 = anomaly) (Blueprint Step 10)
    df_engineered["if_prediction"] = model.predict(X)
    
    # 8. Generate anomaly scores (Blueprint Step 11)
    df_engineered["if_score"] = model.decision_function(X)
    
    # 9. Extract anomalies (Blueprint Step 12)
    anomalies = df_engineered[df_engineered["if_prediction"] == -1].copy()
    
    # ── CALCULATE DYNAMIC THRESHOLDS ──
    # Calculate standard outlier thresholds dynamically (Mean + 2.5 * Standard Deviation)
    # This scales perfectly whether the data values are in the millions or in the hundreds!
    thresholds = {}
    for col in ['refund_amount', 'payment_delay_days', 'discount_amount', 'payment_gap_days']:
        mean_val = df_engineered[col].mean()
        std_val = df_engineered[col].std()
        
        # Guard against standard deviation being 0 (avoiding division/comparison issues)
        std_val = std_val if std_val > 0 else 1.0
        
        # 2.5 standard deviations represents the top ~1% outliers in a normal curve
        thresholds[col] = mean_val + 2.5 * std_val
    
    # 10. Compile Anomalies matching exact output structure (Blueprint Step 14 & 15)
    flagged_records = []
    for idx, row in anomalies.iterrows():
        row_num = idx + 2
        raw_txn_id = row['transaction_id']
        
        # Virtual Row ID patch for missing entries
        if pd.isna(raw_txn_id) or str(raw_txn_id).strip() == "" or str(raw_txn_id).lower() == 'nan':
            txn_id = f"SYS_MISSING_ROW_{row_num}"
        else:
            txn_id = str(raw_txn_id)
            
        record = {
            "transaction_id": txn_id,
            "if_score": round(float(row['if_score']), 4),
            "detected_by": "IF",
            "reason": generate_if_reason(row, thresholds), # Passed dynamically
            "anomaly_engine": "Isolation Forest"
        }
        flagged_records.append(record)
        
    # 11. Save sanitized report (Blueprint Step 15)
    os.makedirs("reports", exist_ok=True)
    output_path = "reports/isolationforest_anomalies.json"
    with open(output_path, "w") as f:
        json.dump(flagged_records, f, indent=4)
        
    # 12. Print Console Summary Report (Blueprint Step 16)
    avg_score = float(df_engineered["if_score"].mean())
    print("\nIsolation Forest Analysis Complete")
    print("--------------------------------")
    print(f"Total Transactions: {len(df)}")
    print(f"Flagged Anomalies: {len(anomalies)}")
    print(f"Average Score: {avg_score:.3f}")
    
    if not anomalies.empty:
        most_suspicious_idx = anomalies["if_score"].idxmin()
        most_suspicious_row = anomalies.loc[most_suspicious_idx]
        # One simple line does the exact same check!
        most_suspicious_txn = safe_txn_id(
            most_suspicious_row["transaction_id"], 
            most_suspicious_idx + 2
        )

            
        print(f"Most Suspicious: {most_suspicious_txn}")
    else:
        print("Most Suspicious: None")
    print(f"Report saved to: {output_path}\n")