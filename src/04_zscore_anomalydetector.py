# Add this at the very top of src/zscore_anomalydetector.py:
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib

# Now all your imports are perfectly safe!
import pandas as pd
import json
csv_loader = importlib.import_module("src.02_csv_loader")
load_finance_csv = csv_loader.load_finance_csv

# Safe Z-score calculation for a single column/series
def z_score(series):
    # Convert to numeric safely
    numeric_series = pd.to_numeric(series, errors='coerce')
    
    mean = numeric_series.mean()
    std = numeric_series.std()
    
    # Shield against division by zero or NaN standard deviation/mean
    if std == 0 or pd.isna(std) or pd.isna(mean):
        return pd.Series(0.0, index=series.index)
        
    z_scores = (numeric_series - mean) / std
    return z_scores

# Helper to construct reasoning with safety guardrails (pd.notna)
def get_anomaly_reason(row, z_threshold):
    reasons = []
    
    # 1. Demand Amount
    z_demand = row['demand_amount_z_score']
    if pd.notna(z_demand) and abs(z_demand) > z_threshold:
        reasons.append(f"demand_amount (Z={z_demand:.2f})")
        
    # 2. Collected Amount
    z_collected = row['collected_amount_z_score']
    if pd.notna(z_collected) and abs(z_collected) > z_threshold:
        reasons.append(f"collected_amount (Z={z_collected:.2f})")
        
    # 3. Outstanding Amount
    z_outstanding = row['outstanding_amount_z_score']
    if pd.notna(z_outstanding) and abs(z_outstanding) > z_threshold:
        reasons.append(f"outstanding_amount (Z={z_outstanding:.2f})")
        
    # 4. Discount Amount
    z_discount = row['discount_amount_z_score']
    if pd.notna(z_discount) and abs(z_discount) > z_threshold:
        reasons.append(f"discount_amount (Z={z_discount:.2f})")
        
    # 5. Refund Amount
    z_refund = row['refund_amount_z_score']
    if pd.notna(z_refund) and abs(z_refund) > z_threshold:
        reasons.append(f"refund_amount (Z={z_refund:.2f})")
        
    # 6. Payment Delay Days
    z_delay = row['payment_delay_days_z_score']
    if pd.notna(z_delay) and abs(z_delay) > z_threshold:
        reasons.append(f"payment_delay_days (Z={z_delay:.2f})")
        
    # Join them together with a comma if multiple columns are flagged
    return ", ".join(reasons)
if __name__ == "__main__":
    # 1. Load the data using your smart loader
    df = load_finance_csv("data/sample_finance_data.csv")
    
    # 2. Set Z-score threshold for anomaly detection
    z_threshold = 3
    
    # 3. Calculate z-scores for numerical columns
    df['demand_amount_z_score'] = z_score(df['demand_amount'])
    df['collected_amount_z_score'] = z_score(df['collected_amount'])
    df['outstanding_amount_z_score'] = z_score(df['outstanding_amount'])
    df['discount_amount_z_score'] = z_score(df['discount_amount'])
    df['refund_amount_z_score'] = z_score(df['refund_amount'])
    df['payment_delay_days_z_score'] = z_score(df['payment_delay_days'])
    
    # 4. Identify anomalies based on z-score threshold
    anomalies = df[
        (df['demand_amount_z_score'].abs() > z_threshold) |
        (df['collected_amount_z_score'].abs() > z_threshold) |
        (df['outstanding_amount_z_score'].abs() > z_threshold) |
        (df['discount_amount_z_score'].abs() > z_threshold) |
        (df['refund_amount_z_score'].abs() > z_threshold) |
        (df['payment_delay_days_z_score'].abs() > z_threshold)
    ].copy()  # <--- Avoids SettingWithCopyWarning by creating an independent copy

    # 5. Generate and add the reason column to each flagged transaction
    # Note: we pass z_threshold here to make sure it's accessible!
    anomalies['anomaly_reason'] = anomalies.apply(lambda r: get_anomaly_reason(r, z_threshold), axis=1)
    
    # 6. Calculate individual anomaly counts
    demand_count = (df['demand_amount_z_score'].abs() > z_threshold).sum()
    collected_count = (df['collected_amount_z_score'].abs() > z_threshold).sum()
    outstanding_count = (df['outstanding_amount_z_score'].abs() > z_threshold).sum()
    discount_count = (df['discount_amount_z_score'].abs() > z_threshold).sum()
    refund_count = (df['refund_amount_z_score'].abs() > z_threshold).sum()
    delay_count = (df['payment_delay_days_z_score'].abs() > z_threshold).sum()
    
    # 7. Convert anomalies to JSON format
    anomalies_json = anomalies.to_json(orient='records')

    # 8. Make sure the reports directory exists
    os.makedirs('reports', exist_ok=True)

    # 9. Save anomalies to a JSON file inside reports
    with open('reports/zscore_anomalies.json', 'w') as f:
        json.dump(json.loads(anomalies_json), f, indent=4)
        
    # 10. Print summary of anomalies
    print(f"\n=========================================")
    print(f"   Z-SCORE ANOMALY DETECTION COMPLETE")
    print(f"=========================================")
    print(f"Total transactions audited: {len(df)}")
    print(f"Suspicious Z-score anomalies flagged: {len(anomalies)}")
    print(f"Report saved to: reports/zscore_anomalies.json")
    print(f"\nAnomaly Breakdown:")
    print(f"  - Demand Amount Outliers     : {demand_count}")
    print(f"  - Collected Amount Outliers  : {collected_count}")
    print(f"  - Outstanding Amount Outliers: {outstanding_count}")
    print(f"  - Discount Amount Outliers   : {discount_count}")
    print(f"  - Refund Amount Outliers     : {refund_count}")
    print(f"  - Payment Delay Outliers     : {delay_count}")