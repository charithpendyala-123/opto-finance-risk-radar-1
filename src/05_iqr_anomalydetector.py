import sys
import os
# Dynamically add the parent directory to the search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now your imports will work perfectly!
import pandas as pd
import numpy as np
import json
import importlib
csv_loader = importlib.import_module("src.02_csv_loader")
load_finance_csv = csv_loader.load_finance_csv

# Fixed to return the TRUE IQR directly!
def iqr(series, col_name=None, cache=None):
    # Avoid double conversion if already numeric
    numeric_series = pd.to_numeric(series, errors='coerce') if series.dtype == object else series

    if cache and col_name and col_name in cache.get('iqr', {}):
        lower_bound = cache['iqr'][col_name]['lower_bound']
        upper_bound = cache['iqr'][col_name]['upper_bound']
        IQR = cache['iqr'][col_name]['iqr']
        
        # Zero-IQR collapsing bounds prevention
        if IQR == 0:
            mean_val = cache.get('zscore', {}).get(col_name, {}).get('mean', 1.0)
            IQR = 0.1 * mean_val if mean_val > 0 else 1.0
            lower_bound = lower_bound - 1.5 * IQR
            upper_bound = upper_bound + 1.5 * IQR
    else:
        if numeric_series.isna().all():
            return np.nan, np.nan, np.nan
        
        Q1 = numeric_series.quantile(0.25)
        Q3 = numeric_series.quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR == 0:
            median_val = numeric_series.median()
            IQR = 0.1 * median_val if (pd.notna(median_val) and median_val > 0) else 1.0
            
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
    return lower_bound, upper_bound, IQR

# Helper to construct reasoning with safety guardrails (pd.notna)
def get_anomaly_reason(row):
    reasons = []
    cols = ['demand_amount', 'collected_amount', 'outstanding_amount', 'discount_amount', 'refund_amount', 'payment_delay_days']
    
    for col in cols:
        val = row[col]
        lower = row[f'{col}_lower_bound']
        upper = row[f'{col}_upper_bound']
        iqr_val = row[f'{col}_iqr']
        
        # Guardrail check: Make sure all three are valid numbers before comparing!
        if pd.notna(val) and pd.notna(lower) and pd.notna(upper):
            if val < lower or val > upper:
                reasons.append(f"{col} (IQR={iqr_val:.2f})")
                
    return ", ".join(reasons)

if __name__ == "__main__":
    # 1. Load the data using your loader
    df = load_finance_csv("data/sample_finance_data.csv")
    
    # 2. Safely convert audited columns to numeric in-place
    cols = ['demand_amount', 'collected_amount', 'outstanding_amount', 'discount_amount', 'refund_amount', 'payment_delay_days']
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. Calculate bounds and unpack the TRUE IQR directly (No subtraction needed!)
    for col in cols:
        lower, upper, true_iqr = iqr(df[col]) # <-- Unpacks the returned IQR
        df[f'{col}_lower_bound'] = lower
        df[f'{col}_upper_bound'] = upper
        df[f'{col}_iqr'] = true_iqr            # <-- Stores the correct math directly!

    # 4. Identify anomalies based on IQR bounds
    anomalies = df[
        (df['demand_amount'] < df['demand_amount_lower_bound']) |
        (df['demand_amount'] > df['demand_amount_upper_bound']) |
        (df['collected_amount'] < df['collected_amount_lower_bound']) |
        (df['collected_amount'] > df['collected_amount_upper_bound']) |
        (df['outstanding_amount'] < df['outstanding_amount_lower_bound']) |
        (df['outstanding_amount'] > df['outstanding_amount_upper_bound']) |
        (df['discount_amount'] < df['discount_amount_lower_bound']) |
        (df['discount_amount'] > df['discount_amount_upper_bound']) |
        (df['refund_amount'] < df['refund_amount_lower_bound']) |
        (df['refund_amount'] > df['refund_amount_upper_bound']) |
        (df['payment_delay_days'] < df['payment_delay_days_lower_bound']) |
        (df['payment_delay_days'] > df['payment_delay_days_upper_bound'])
    ].copy()
    
    # 5. Apply reason mapping
    anomalies['anomaly_reason'] = anomalies.apply(get_anomaly_reason, axis=1)

    # 6. Count anomalies by column
    demand_count = (anomalies['demand_amount'] < anomalies['demand_amount_lower_bound']).sum() + (anomalies['demand_amount'] > anomalies['demand_amount_upper_bound']).sum()
    collected_count = (anomalies['collected_amount'] < anomalies['collected_amount_lower_bound']).sum() + (anomalies['collected_amount'] > anomalies['collected_amount_upper_bound']).sum()
    outstanding_count = (anomalies['outstanding_amount'] < anomalies['outstanding_amount_lower_bound']).sum() + (anomalies['outstanding_amount'] > anomalies['outstanding_amount_upper_bound']).sum()
    discount_count = (anomalies['discount_amount'] < anomalies['discount_amount_lower_bound']).sum() + (anomalies['discount_amount'] > anomalies['discount_amount_upper_bound']).sum()
    refund_count = (anomalies['refund_amount'] < anomalies['refund_amount_lower_bound']).sum() + (anomalies['refund_amount'] > anomalies['refund_amount_upper_bound']).sum()
    paymentdelay_count = (anomalies['payment_delay_days'] < anomalies['payment_delay_days_lower_bound']).sum() + (anomalies['payment_delay_days'] > anomalies['payment_delay_days_upper_bound']).sum()

    # 7. Convert and save
    anomalies_json = anomalies.to_json(orient='records')
    os.makedirs('reports', exist_ok=True)
    with open('reports/iqr_anomalies.json', 'w') as f:
        json.dump(json.loads(anomalies_json), f, indent=4)

    # 8. Print summary
    print(f"\n=========================================")
    print(f"Total transactions audited: {len(df)}")
    print(f"Suspicious IQR anomalies flagged: {len(anomalies)}")
    print(f"Report saved to: reports/iqr_anomalies.json")
    print(f"\nAnomaly Breakdown:")
    print(f"  - Demand Amount Outliers     : {demand_count}")
    print(f"  - Collected Amount Outliers  : {collected_count}")
    print(f"  - Outstanding Amount Outliers: {outstanding_count}")
    print(f"  - Discount Amount Outliers   : {discount_count}")
    print(f"  - Refund Amount Outliers     : {refund_count}")
    print(f"  - Payment Delay Outliers     : {paymentdelay_count}")