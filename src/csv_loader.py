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

    df = pd.read_csv(path, dtype=object, keep_default_na=True)

    print("\nCSV CHECKPOINT")
    print("Source:", path)
    

    # Empty file check

    if len(df) == 0:
        print("\nDATA REJECTED - blank sheet submitted.")
        print("The file has no transaction records.")
        return None

    # Missing columns check

    absent = [c for c in EXPECTED_COLUMNS if c not in df.columns]

    if absent:
        print("\nDATA REJECTED - required columns missing:")

        for c in absent:
            print(c)

        print("\nPlease re-submit with all required columns present.")
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