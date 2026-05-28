import pandas as pd
import numpy as np
from datetime import datetime, date
import sys
import os

# Dynamically add the parent directory to the search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.csv_loader import load_finance_csv


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Expected transaction_id format  e.g. TXN001, TXN1234
TXN_ID_PATTERN = r'^TXN\d+$'

# Expected customer_id format  e.g. CUST001, CUST99
CUST_ID_PATTERN = r'^CUST\d+$'

# Dates beyond this many days in the future are flagged
MAX_FUTURE_DAYS = 30

# If transaction_id numbers jump more than this, flag it
MAX_ID_GAP = 1000

# If one customer books more than this many distinct units in one batch, flag it
MAX_UNITS_PER_CUSTOMER = 20

# Public holidays (add your own)
PUBLIC_HOLIDAYS = {
    date(2025, 1, 1),   # New Year
    date(2025, 8, 15),  # Independence Day
    date(2025, 10, 2),  # Gandhi Jayanti
    date(2026, 1, 1),
    date(2026, 8, 15),
    date(2026, 10, 2),
}


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER — makes a numeric Series from a column, bad values become NaN
# ─────────────────────────────────────────────────────────────────────────────

def to_num(df, col):#Safely convert a column to numeric, returning None if the column doesn't exist
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else None #without errors=coerce it would raise an error if something other than numbers are present

def to_date(df, col):
    return pd.to_datetime(df[col], errors="coerce") if col in df.columns else None #NaT becomes the output if something other than dates are present

def is_blank(series):
    return series.isna() | (series.astype(str).str.strip() == "")

def no_flag(df):
    return pd.Series([False] * len(df), index=df.index)


# ─────────────────────────────────────────────────────────────────────────────
#  40 RULE CHECKERS
#  Each function gets the full dataframe and returns a boolean Series.
#  True  = this row violates the rule so it flags
#  False = this row is fine so it doesn't give a shit
# ─────────────────────────────────────────────────────────────────────────────

# ── IDENTITY ─────────────────────────────────────────────────────────────────

def rv01_missing_transaction_id(df):
    if "transaction_id" not in df.columns: return no_flag(df)
    return is_blank(df["transaction_id"])

def rv02_duplicate_transaction_id(df):
    if "transaction_id" not in df.columns: return no_flag(df)
    non_blank = df["transaction_id"].replace("", np.nan).dropna()
    return df["transaction_id"].isin(non_blank[non_blank.duplicated(keep=False)])

def rv03_blank_customer_id(df):
    if "customer_id" not in df.columns: return no_flag(df)
    return is_blank(df["customer_id"])

def rv04_blank_unit_id(df):
    if "unit_id" not in df.columns: return no_flag(df)
    return is_blank(df["unit_id"])

def rv05_blank_project_id(df):
    if "project_id" not in df.columns: return no_flag(df)
    return is_blank(df["project_id"])

def rv06_invalid_transaction_id_format(df):
    if "transaction_id" not in df.columns: return no_flag(df)
    filled = ~is_blank(df["transaction_id"])
    valid  = df["transaction_id"].str.match(TXN_ID_PATTERN, na=False)
    return filled & ~valid

def rv07_invalid_customer_id_format(df):
    if "customer_id" not in df.columns: return no_flag(df)
    filled = ~is_blank(df["customer_id"])
    valid  = df["customer_id"].str.match(CUST_ID_PATTERN, na=False)
    return filled & ~valid

def rv08_transaction_id_large_gap(df):
    # Extract the numeric part of TXN IDs and check for large sequential jumps
    if "transaction_id" not in df.columns: return no_flag(df)
    extracted = df["transaction_id"].str.extract(r'(\d+)')[0]
    nums = pd.to_numeric(extracted, errors="coerce")
    
    # Calculate difference between consecutive rows in the original CSV order (no sorting!)
    gaps = nums.diff().abs()
    
    # Flag the rows where the jump is greater than MAX_ID_GAP
    return gaps > MAX_ID_GAP


# ── FINANCIAL ────────────────────────────────────────────────────────────────

def rv09_negative_demand(df):
    s = to_num(df, "demand_amount")
    return s < 0 if s is not None else no_flag(df)

def rv10_negative_collected(df):
    s = to_num(df, "collected_amount")
    return s < 0 if s is not None else no_flag(df)

def rv11_negative_refund(df):
    s = to_num(df, "refund_amount")
    return s < 0 if s is not None else no_flag(df)

def rv12_negative_discount(df):
    s = to_num(df, "discount_amount")
    return s < 0 if s is not None else no_flag(df)

def rv13_zero_demand(df):
    s = to_num(df, "demand_amount")
    return s == 0 if s is not None else no_flag(df)

def rv14_collected_exceeds_demand(df):
    demand    = to_num(df, "demand_amount")
    collected = to_num(df, "collected_amount")
    if demand is None or collected is None: return no_flag(df)
    both = demand.notna() & collected.notna()
    return both & (collected > demand)

def rv15_refund_exceeds_demand(df):
    demand = to_num(df, "demand_amount")
    refund = to_num(df, "refund_amount")
    if demand is None or refund is None: return no_flag(df)
    both = demand.notna() & refund.notna()
    return both & (refund > demand)

def rv16_discount_exceeds_demand(df):
    demand   = to_num(df, "demand_amount")
    discount = to_num(df, "discount_amount")
    if demand is None or discount is None: return no_flag(df)
    both = demand.notna() & discount.notna()
    return both & (discount > demand)

def rv17_refund_exceeds_collected(df):
    collected = to_num(df, "collected_amount")
    refund    = to_num(df, "refund_amount")
    if collected is None or refund is None: return no_flag(df)
    both = collected.notna() & refund.notna()
    return both & (refund > collected)

def rv18_discount_plus_refund_exceeds_demand(df):
    demand   = to_num(df, "demand_amount")
    discount = to_num(df, "discount_amount")
    refund   = to_num(df, "refund_amount")
    if any(x is None for x in [demand, discount, refund]): return no_flag(df)
    all_present = demand.notna() & discount.notna() & refund.notna()
    return all_present & ((discount + refund) > demand)

def rv19_outstanding_mismatch(df):
    demand      = to_num(df, "demand_amount")
    collected   = to_num(df, "collected_amount")
    discount    = to_num(df, "discount_amount")
    refund      = to_num(df, "refund_amount")
    outstanding = to_num(df, "outstanding_amount")
    if any(x is None for x in [demand, collected, discount, refund, outstanding]):
        return no_flag(df)
    expected    = demand - collected - discount + refund
    all_present = demand.notna() & collected.notna() & discount.notna() & refund.notna() & outstanding.notna()
    return all_present & ((outstanding - expected).abs() > 1)   # allow ±1 rounding

def rv20_negative_outstanding(df):
    s = to_num(df, "outstanding_amount")
    return s < 0 if s is not None else no_flag(df)

def rv21_refund_without_collection(df):
    collected = to_num(df, "collected_amount")
    refund    = to_num(df, "refund_amount")
    if collected is None or refund is None: return no_flag(df)
    return (refund > 0) & (collected == 0)

def rv22_all_financial_amounts_blank(df):
    financial_cols = ["demand_amount", "collected_amount", "outstanding_amount",
                      "discount_amount", "refund_amount"]
    present = [c for c in financial_cols if c in df.columns]
    if not present: return no_flag(df)
    all_blank = df[present].apply(lambda col: is_blank(col)).all(axis=1)
    return all_blank


# ── DATE & TIME ───────────────────────────────────────────────────────────────

def rv23_invalid_date_format(df):
    mask = no_flag(df)
    for col in ["demand_date", "payment_date"]:
        if col not in df.columns: continue
        has_value = ~is_blank(df[col])
        parsed    = pd.to_datetime(df[col], errors="coerce")
        mask = mask | (has_value & parsed.isna())
    return mask

def rv24_payment_before_demand(df):
    demand  = to_date(df, "demand_date")
    payment = to_date(df, "payment_date")
    if demand is None or payment is None: return no_flag(df)
    both = demand.notna() & payment.notna()
    return both & (payment < demand)

def rv25_delay_days_inconsistent(df):
    demand  = to_date(df, "demand_date")
    payment = to_date(df, "payment_date")
    delay   = to_num(df, "payment_delay_days")
    if any(x is None for x in [demand, payment, delay]): return no_flag(df)
    all_present = demand.notna() & payment.notna() & delay.notna()
    actual_delay = (payment - demand).dt.days
    return all_present & ((delay - actual_delay).abs() > 1)   # allow ±1 day

def rv26_future_demand_date(df):
    demand = to_date(df, "demand_date")
    if demand is None: return no_flag(df)
    cutoff = pd.Timestamp(datetime.today()) + pd.Timedelta(days=MAX_FUTURE_DAYS)
    return demand.notna() & (demand > cutoff)

def rv27_future_payment_date(df):
    payment = to_date(df, "payment_date")
    if payment is None: return no_flag(df)
    cutoff = pd.Timestamp(datetime.today()) + pd.Timedelta(days=MAX_FUTURE_DAYS)
    return payment.notna() & (payment > cutoff)

def rv28_zero_delay_but_different_dates(df):
    demand  = to_date(df, "demand_date")
    payment = to_date(df, "payment_date")
    delay   = to_num(df, "payment_delay_days")
    if any(x is None for x in [demand, payment, delay]): return no_flag(df)
    all_present = demand.notna() & payment.notna() & delay.notna()
    return all_present & (delay == 0) & (payment != demand)

def rv29_negative_delay_but_late_payment(df):
    demand  = to_date(df, "demand_date")
    payment = to_date(df, "payment_date")
    delay   = to_num(df, "payment_delay_days")
    if any(x is None for x in [demand, payment, delay]): return no_flag(df)
    all_present = demand.notna() & payment.notna() & delay.notna()
    return all_present & (delay < 0) & (payment > demand)

def rv30_same_date_but_positive_delay(df):
    demand  = to_date(df, "demand_date")
    payment = to_date(df, "payment_date")
    delay   = to_num(df, "payment_delay_days")
    if any(x is None for x in [demand, payment, delay]): return no_flag(df)
    all_present = demand.notna() & payment.notna() & delay.notna()
    return all_present & (demand == payment) & (delay > 0)

def rv31_demand_on_weekend_or_holiday(df):
    demand = to_date(df, "demand_date")
    if demand is None: return no_flag(df)
    is_weekend = demand.dt.dayofweek >= 5
    is_holiday = demand.dt.date.isin(PUBLIC_HOLIDAYS)
    return demand.notna() & (is_weekend | is_holiday)

def rv32_payment_on_weekend_or_holiday(df):
    payment = to_date(df, "payment_date")
    if payment is None: return no_flag(df)
    is_weekend = payment.dt.dayofweek >= 5
    is_holiday = payment.dt.date.isin(PUBLIC_HOLIDAYS)
    return payment.notna() & (is_weekend | is_holiday)


# ── CROSS-RECORD ──────────────────────────────────────────────────────────────

def rv33_payment_date_without_collection(df):
    collected = to_num(df, "collected_amount")
    payment   = to_date(df, "payment_date")
    if collected is None or payment is None: return no_flag(df)
    return (collected == 0) & payment.notna()

def rv34_same_txn_different_customers(df):
    if "transaction_id" not in df.columns or "customer_id" not in df.columns:
        return no_flag(df)
    # Ignore blank transactions and blank customers for this logical check
    non_blank = df[~is_blank(df["transaction_id"]) & ~is_blank(df["customer_id"])]
    counts = non_blank.groupby("transaction_id")["customer_id"].nunique()
    bad_txns = counts[counts > 1].index
    return df["transaction_id"].isin(bad_txns) & ~is_blank(df["transaction_id"])

def rv35_same_txn_different_demand(df):
    if "transaction_id" not in df.columns or "demand_amount" not in df.columns:
        return no_flag(df)
    # Ignore blank transactions and blank demands for this logical check
    non_blank = df[~is_blank(df["transaction_id"]) & ~is_blank(df["demand_amount"])]
    counts = non_blank.groupby("transaction_id")["demand_amount"].nunique()
    bad_txns = counts[counts > 1].index
    return df["transaction_id"].isin(bad_txns) & ~is_blank(df["transaction_id"])

def rv36_duplicate_full_rows(df):
    return df.duplicated(keep=False)

def rv37_identical_demand_for_same_unit(df):
    if "unit_id" not in df.columns or "demand_amount" not in df.columns:
        return no_flag(df)
    
    # Filter out blanks first
    mask_valid = ~is_blank(df["unit_id"]) & ~is_blank(df["demand_amount"])
    valid_df = df[mask_valid]
    
    if valid_df.empty:
        return no_flag(df)
        
    unit_counts  = valid_df.groupby("unit_id")["demand_amount"].transform("count")
    unit_nunique = valid_df.groupby("unit_id")["demand_amount"].transform("nunique")
    bad_mask = (unit_counts > 1) & (unit_nunique == 1)
    
    # Map back to the original index safely
    return mask_valid & bad_mask.reindex(df.index, fill_value=False)

def rv38_likely_duplicate_submission(df):
    needed = ["customer_id", "unit_id", "demand_date", "transaction_id"]
    if not all(c in df.columns for c in needed): return no_flag(df)
    key = ["customer_id", "unit_id", "demand_date"]
    
    # Drop rows where any of the duplicate check keys is blank
    mask_blank = is_blank(df["customer_id"]) | is_blank(df["unit_id"]) | is_blank(df["demand_date"]) | is_blank(df["transaction_id"])
    
    # Find keys that have more than 1 unique transaction_id
    bad_keys = df[~mask_blank].groupby(key)["transaction_id"].nunique()
    bad_keys = bad_keys[bad_keys > 1].index
    
    if bad_keys.empty:
        return no_flag(df)
        
    # Check if the row's key is in bad_keys, preserving the original index perfectly
    row_keys = pd.MultiIndex.from_frame(df[key])
    return ~mask_blank & row_keys.isin(bad_keys)

def rv39_too_many_units_per_customer(df):
    if "customer_id" not in df.columns or "unit_id" not in df.columns:
        return no_flag(df)
    # Ignore blank customers/units for this behavioral check
    non_blank = df[~is_blank(df["customer_id"]) & ~is_blank(df["unit_id"])]
    unit_counts = non_blank.groupby("customer_id")["unit_id"].nunique()
    bad_customers = unit_counts[unit_counts > MAX_UNITS_PER_CUSTOMER].index
    return df["customer_id"].isin(bad_customers) & ~is_blank(df["customer_id"])

def rv40_all_rows_same_demand_date(df):
    if "demand_date" not in df.columns: return no_flag(df)
    if df["demand_date"].nunique() == 1 and len(df) > 1:
        return pd.Series([True] * len(df), index=df.index)
    return no_flag(df)


# ─────────────────────────────────────────────────────────────────────────────
#  RULES LIST  —  ties each rule ID to its checker function
#  To add a new rule: add a new dict here and write its checker above.
# ─────────────────────────────────────────────────────────────────────────────

RULES = [
    # ── Identity
    {"id": "RV-01", "description": "Missing transaction_id",                                        "fn": rv01_missing_transaction_id},
    {"id": "RV-02", "description": "Duplicate transaction_id",                                      "fn": rv02_duplicate_transaction_id},
    {"id": "RV-03", "description": "Blank customer_id",                                             "fn": rv03_blank_customer_id},
    {"id": "RV-04", "description": "Blank unit_id",                                                 "fn": rv04_blank_unit_id},
    {"id": "RV-05", "description": "Blank project_id",                                              "fn": rv05_blank_project_id},
    {"id": "RV-06", "description": "Invalid transaction_id format",                                 "fn": rv06_invalid_transaction_id_format},
    {"id": "RV-07", "description": "Invalid customer_id format",                                    "fn": rv07_invalid_customer_id_format},
    {"id": "RV-08", "description": "transaction_id sequential gap > 1000",                         "fn": rv08_transaction_id_large_gap},
    # ── Financial
    {"id": "RV-09", "description": "Negative demand_amount",                                        "fn": rv09_negative_demand},
    {"id": "RV-10", "description": "Negative collected_amount",                                     "fn": rv10_negative_collected},
    {"id": "RV-11", "description": "Negative refund_amount",                                        "fn": rv11_negative_refund},
    {"id": "RV-12", "description": "Negative discount_amount",                                      "fn": rv12_negative_discount},
    {"id": "RV-13", "description": "demand_amount = 0",                                             "fn": rv13_zero_demand},
    {"id": "RV-14", "description": "collected_amount > demand_amount",                              "fn": rv14_collected_exceeds_demand},
    {"id": "RV-15", "description": "refund_amount > demand_amount",                                 "fn": rv15_refund_exceeds_demand},
    {"id": "RV-16", "description": "discount_amount > demand_amount",                               "fn": rv16_discount_exceeds_demand},
    {"id": "RV-17", "description": "refund_amount > collected_amount",                              "fn": rv17_refund_exceeds_collected},
    {"id": "RV-18", "description": "discount + refund > demand_amount",                             "fn": rv18_discount_plus_refund_exceeds_demand},
    {"id": "RV-19", "description": "Outstanding amount mismatch",                                   "fn": rv19_outstanding_mismatch},
    {"id": "RV-20", "description": "outstanding_amount is negative",                                "fn": rv20_negative_outstanding},
    {"id": "RV-21", "description": "refund_amount > 0 but collected_amount = 0",                   "fn": rv21_refund_without_collection},
    {"id": "RV-22", "description": "All financial amounts are blank",                               "fn": rv22_all_financial_amounts_blank},
    # ── Date & time
    {"id": "RV-23", "description": "Invalid date format",                                           "fn": rv23_invalid_date_format},
    {"id": "RV-24", "description": "payment_date is before demand_date",                            "fn": rv24_payment_before_demand},
    {"id": "RV-25", "description": "payment_delay_days does not match actual date difference",      "fn": rv25_delay_days_inconsistent},
    {"id": "RV-26", "description": "demand_date is too far in the future",                          "fn": rv26_future_demand_date},
    {"id": "RV-27", "description": "payment_date is too far in the future",                        "fn": rv27_future_payment_date},
    {"id": "RV-28", "description": "delay_days = 0 but payment_date differs from demand_date",     "fn": rv28_zero_delay_but_different_dates},
    {"id": "RV-29", "description": "Negative delay but payment_date is after demand_date",         "fn": rv29_negative_delay_but_late_payment},
    {"id": "RV-30", "description": "Same dates but delay_days > 0",                                "fn": rv30_same_date_but_positive_delay},
    {"id": "RV-31", "description": "demand_date falls on weekend or public holiday",               "fn": rv31_demand_on_weekend_or_holiday},
    {"id": "RV-32", "description": "payment_date falls on weekend or public holiday",              "fn": rv32_payment_on_weekend_or_holiday},
    # ── Cross-record
    {"id": "RV-33", "description": "payment_date exists but collected_amount = 0",                 "fn": rv33_payment_date_without_collection},
    {"id": "RV-34", "description": "Same transaction_id linked to different customer_ids",         "fn": rv34_same_txn_different_customers},
    {"id": "RV-35", "description": "Same transaction_id with different demand_amounts",            "fn": rv35_same_txn_different_demand},
    {"id": "RV-36", "description": "Duplicate full transaction rows",                              "fn": rv36_duplicate_full_rows},
    {"id": "RV-37", "description": "Same unit_id has identical demand_amount across all rows",    "fn": rv37_identical_demand_for_same_unit},
    {"id": "RV-38", "description": "Likely duplicate submission (same customer+unit+date)",       "fn": rv38_likely_duplicate_submission},
    {"id": "RV-39", "description": "Customer linked to too many units in one batch",              "fn": rv39_too_many_units_per_customer},
    {"id": "RV-40", "description": "All rows share the same demand_date",                         "fn": rv40_all_rows_same_demand_date},
]


# ─────────────────────────────────────────────────────────────────────────────
#  ENGINE  —  runs every rule and collects violations
# ─────────────────────────────────────────────────────────────────────────────

def run_validation(df):

    all_violations = []

    for rule in RULES:

        try:
            mask = rule["fn"](df)

        except Exception as e:
            print(f"{rule['id']} crashed: {e}")
            continue

        violated = df[mask].copy()

        if violated.empty:
            continue

        violated.insert(0, "rule_id", rule["id"])
        violated.insert(1, "rule_description", rule["description"])
        violated.insert(2, "source_row", violated.index + 2)

        all_violations.append(violated)

    if all_violations:
        return pd.concat(all_violations, ignore_index=True)

    return pd.DataFrame()



def print_report(df, violations):

    print("\nRULE VALIDATION REPORT")
    print("Run at:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    print("\nTotal records:", len(df))
    print("Rules evaluated:", len(RULES))

    if violations.empty:
        print("\nAll records passed every rule.\n")
        return

    violated_row_count = violations["source_row"].nunique()
    clean_row_count = len(df) - violated_row_count

    print("\nSUMMARY")
    print("Total violations:", len(violations))
    print("Rows with at least 1 error:", violated_row_count)
    print("Clean rows:", clean_row_count)
    print("Pass rate:", round(clean_row_count / len(df) * 100, 1), "%")

    print("\nVIOLATIONS BY RULE")

    summary = (
        violations.groupby(["rule_id", "rule_description"])
        .size()
        .reset_index(name="count")
        .sort_values("rule_id")
    )

    for _, row in summary.iterrows():

        pct = row["count"] / len(df) * 100

        print(
            row["rule_id"],
            "-",
            row["rule_description"],
            ":",
            row["count"],
            "violations",
            f"({pct:.1f}%)"
        )

    print("\nSAMPLE ROWS")

    show = [
        "rule_id",
        "source_row",
        "transaction_id",
        "demand_amount",
        "collected_amount",
        "demand_date",
        "payment_date"
    ]

    show = [c for c in show if c in violations.columns]

    for rule_id, group in violations.groupby("rule_id"):

        print("\n", rule_id, "-", group["rule_description"].iloc[0])

        print(group[show].head(3).to_string(index=False))



def validate_finance_csv(path="data/sample_finance_data.csv"):

    # Load safely using your custom structural loader!
    df = load_finance_csv(path)
    
    # Safety guard: if the file was rejected by the loader, exit gracefully
    if df is None:
        return None, None

    violations = run_validation(df)

    print_report(df, violations)

    violations.to_json(
        "reports/rule_validation_report.json",
        orient="records",
        indent=4
    )

    return df, violations



if __name__ == "__main__":

    df, violations = validate_finance_csv("data/sample_finance_data.csv")