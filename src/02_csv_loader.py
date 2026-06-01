import pandas as pd
from datetime import datetime

EXPECTED_COLUMNS = [
    "transaction_id",
    "customer_id",
    "project_id",
    "unit_id",
    "demand_amount",
    "collected_amount",
    "outstanding_amount",
    "discount_amount",
    "refund_amount",
    "payment_delay_days",
    "demand_date",
    "payment_date"
]

def load_finance_csv(path="data/sample_finance_data.csv"):
    print("\nCSV CHECKPOINT")
    print("Source:", path)

    try: ##included this because if the file is empty, pandas will throw an error instead of returning an empty dataframe, so this allows us to catch that and provide a more user-friendly message.
        df = pd.read_csv(path, dtype=object, keep_default_na=True)
    except pd.errors.EmptyDataError:
        print("\nDATA REJECTED - blank sheet submitted.")
        print("The file has no transaction records (empty file).")
        return None

    # Empty file check (for files that have headers but 0 rows)
    if len(df) == 0:
        print("\nDATA REJECTED - blank sheet submitted.")
        print("The file has no transaction records.")
        return None

        # ── Missing or 100% Empty Columns Check ──
    absent = []

    for col in EXPECTED_COLUMNS:
        # Check 1: Is the column physically missing from the headers?
        if col not in df.columns:
            absent.append(col)
        else:
            # Check 2: Does it exist but contains 100% blank/null/whitespace values?
            if df[col].fillna("").astype(str).str.strip().eq("").all():
                absent.append(col)

    if absent:
        print("\nDATA REJECTED - required columns missing or 100% blank:")
        for c in absent:
            print(f"- {c}")
        print("\nPlease re-submit with all required columns populated.")
        return None

    # Summary

    total_cells = len(df) * len(EXPECTED_COLUMNS)

    null_cells = int(df[EXPECTED_COLUMNS].isnull().sum().sum())

    completeness = (
        1 - null_cells / total_cells
    ) * 100 if total_cells else 0

    print("\nData accepted - passed structural checks.")

    print(
        "Row-level issues will be handled by rule validator and anomaly engine."
    )

    print("\nSUMMARY")

    print("Total records:", len(df))
    print("Total columns:", len(df.columns))
    print("Overall completeness:", completeness)
    print("Total missing cells:", null_cells)
    print("Total cells:", total_cells)

    # Missing values per column

    print("\nMISSING VALUES PER COLUMN")

    any_missing = False

    for col in EXPECTED_COLUMNS:

        n = int(df[col].isnull().sum())

        pct = n / len(df) * 100

        if n > 0:

            any_missing = True

            print(col, ":", n, "missing values", "(", pct, "% )")

    if not any_missing:
        print("No missing values.")

    return df


if __name__ == "__main__":

    df = load_finance_csv("data/sample_finance_data.csv")