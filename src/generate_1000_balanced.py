# ==============================================================================
# OPTOxCRM - DYNAMIC 10,000 RECORD LEDGER GENERATOR (9,000 CLEAN / 1,000 FRAUD)
# ==============================================================================
import os
import random
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
#  MASTER DATASETS & PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DATE = date(2024, 1, 1)  # Starting date

# Hardcoded holidays to avoid weekend/holiday violations (RV-31/32)
HOLIDAYS = {
    date(2024, 1, 1),
    date(2024, 8, 15),
    date(2024, 10, 2),
    date(2025, 1, 1),
    date(2025, 8, 15),
    date(2025, 10, 2),
    date(2026, 1, 1),
    date(2026, 8, 15),
    date(2026, 10, 2),
}

# Helper to find valid weekday dates that are not holidays
def get_valid_weekday(start_date, offset_days):
    curr = start_date + timedelta(days=offset_days)
    while curr.weekday() >= 5 or curr in HOLIDAYS:
        curr += timedelta(days=1)
    return curr

# ─────────────────────────────────────────────────────────────────────────────
#  1. GENERATE 9,000 PERFECT CLEAN RECORDS (Rows 1 to 9,000)
# ─────────────────────────────────────────────────────────────────────────────
clean_records = []
print("Generating 9,000 perfectly clean records...")

for i in range(1, 9001):
    txn_id = f"TXN{100000 + i}"   # Sequential IDs, no gaps
    cust_id = f"CUST{100000 + i}"  # Unique customers
    unit_id = f"UNIT{100000 + i}"  # Unique units to avoid duplicate alerts
    proj_id = "SKYLINE"            # Isolated project to avoid average-demand shifts
    
    # Evenly distribute demand dates across 300 distinct dates (exactly 30 per day)
    # This prevents daily volume spikes (Bulk System Glitches)
    date_offset = i % 300
    demand_dt = get_valid_weekday(BASE_DATE, date_offset * 3)
    
    # Consistently set payment 10 weekdays later
    payment_dt = get_valid_weekday(demand_dt, 10)
    delay_days = (payment_dt - demand_dt).days
    
    # Financial constants: standard, tightly-clustered amounts to prevent Z-Score/IQR flags
    demand_amt = 500000
    discount_amt = 10000
    collected_amt = 400000
    refund_amt = 0
    outstanding_amt = demand_amt - collected_amt - discount_amt + refund_amt  # 90,000 (perfect math)
    
    clean_records.append({
        "transaction_id": txn_id,
        "customer_id": cust_id,
        "project_id": proj_id,
        "unit_id": unit_id,
        "demand_amount": demand_amt,
        "collected_amount": collected_amt,
        "outstanding_amount": outstanding_amt,
        "discount_amount":   discount_amt,
        "refund_amount":     refund_amt,
        "payment_delay_days": delay_days,
        "demand_date":       demand_dt.strftime("%Y-%m-%d"),
        "payment_date":      payment_dt.strftime("%Y-%m-%d"),
        "record_type":       "CLEAN"
    })

# ─────────────────────────────────────────────────────────────────────────────
#  2. GENERATE 1,000 FRAUD RECORDS (Rows 9,001 to 10,000)
# ─────────────────────────────────────────────────────────────────────────────
fraud_records = []
print("Generating 1,000 fraudulent records...")

for i in range(1, 1001):
    txn_id = f"TXN{200000 + i}"
    cust_id = f"CUST{200000 + i}"
    unit_id = f"UNIT{200000 + i}"
    proj_id = "AQUA"  # Isolated project to prevent average distortion on SKYLINE
    
    # Default valid base dates
    demand_dt = get_valid_weekday(BASE_DATE, i % 100)
    payment_dt = get_valid_weekday(demand_dt, 10)
    delay_days = (payment_dt - demand_dt).days
    
    demand_amt = 200000
    collected_amt = 150000
    discount_amt = 10000
    refund_amt = 0
    outstanding_amt = demand_amt - collected_amt - discount_amt
    
    # 10 types of distinct fraud categories (100 records of each type)
    violation_type = i % 10
    
    if violation_type == 0:
        # 1. Collected amount exceeds demand (RV-14)
        collected_amt = demand_amt + 50000
        outstanding_amt = demand_amt - collected_amt - discount_amt + refund_amt  # Negative outstanding
        
    elif violation_type == 1:
        # 2. Refund amount exceeds collected (RV-17)
        collected_amt = 20000
        refund_amt = 50000
        outstanding_amt = demand_amt - collected_amt - discount_amt + refund_amt
        
    elif violation_type == 2:
        # 3. Refund without any collections (RV-21)
        collected_amt = 0
        refund_amt = 30000
        outstanding_amt = demand_amt - collected_amt - discount_amt + refund_amt
        
    elif violation_type == 3:
        # 4. Payment date is before demand date (RV-24)
        payment_dt = demand_dt - timedelta(days=8)
        delay_days = (payment_dt - demand_dt).days  # -8 delay
        
    elif violation_type == 4:
        # 5. Calendar dates do not match payment_delay_days field (RV-25)
        delay_days = 999
        
    elif violation_type == 5:
        # 6. Outstanding ledger mismatch (RV-19)
        outstanding_amt = 888888
        
    elif violation_type == 6:
        # 7. Invoiced on a weekend Saturday (RV-31)
        sat = demand_dt
        while sat.weekday() != 5:
            sat += timedelta(days=1)
        demand_dt = sat
        payment_dt = sat + timedelta(days=10)
        delay_days = 10
        
    elif violation_type == 7:
        # 8. Negative demand amount (RV-09)
        demand_amt = -40000
        
    elif violation_type == 8:
        # 9. Large gap in transaction ID numbering (RV-08)
        txn_id = f"TXN{i + 900000}"
        
    elif violation_type == 9:
        # 10. Multi-million rupee demand amount (Z-Score & IQR Outlier)
        demand_amt = 15000000
        collected_amt = 14000000
        discount_amt = 1000000
        outstanding_amt = 0

    fraud_records.append({
        "transaction_id": txn_id,
        "customer_id": cust_id,
        "project_id": proj_id,
        "unit_id": unit_id,
        "demand_amount": demand_amt,
        "collected_amount": collected_amt,
        "outstanding_amount": outstanding_amt,
        "discount_amount": discount_amt,
        "refund_amount": refund_amt,
        "payment_delay_days": delay_days,
        "demand_date":       demand_dt.strftime("%Y-%m-%d"),
        "payment_date":      payment_dt.strftime("%Y-%m-%d"),
        "record_type":       f"FRAUD_{violation_type}"
    })

# ─────────────────────────────────────────────────────────────────────────────
#  SHUFFLE, SORT, AND SAVE TO CSV FILES
# ─────────────────────────────────────────────────────────────────────────────
all_records = clean_records + fraud_records
random.shuffle(all_records)

df = pd.DataFrame(all_records)

# Extract numeric part to sort naturally by ID (so TXN100001 comes before TXN100002)
df['sort_key'] = df['transaction_id'].str.extract(r'(\d+)').astype(float)
df = df.sort_values(by='sort_key').drop(columns=['sort_key'])

# Create output directories
os.makedirs("data", exist_ok=True)
os.makedirs("data/uploads", exist_ok=True)

# Drop internal record_type flag from public import sheets
df_export = df.drop(columns=["record_type"])

# Save files
df_export.to_csv("data/sample_finance_data.csv", index=False)
df_export.to_csv("data/uploads/test_10000_balanced.csv", index=False)
df.to_csv("data/sample_finance_data_labelled.csv", index=False)