import pandas as pd
import random
from datetime import datetime, timedelta


# ─────────────────────────────────────────────
#  PER-RUN CHAOS SEEDS
#  Every single run re-rolls these, so the shape
#  of the data is different each time.
# ─────────────────────────────────────────────

# How many records? Anywhere from 200 to 2000.
NUM_RECORDS = random.randint(0,50)

# Missing-field rates: each run picks its own
# probability (0–100%) independently per field.
MISSING_RATE = {
    "transaction_id":     random.random(),        # could be 0% or 100%
    "customer_id":        random.random(),
    "unit_id":            random.random(),
    "project_id":         random.random(),
    "demand_date":        random.random(),
    "payment_date":       random.random(),
    "demand_amount":      random.random(),
    "collected_amount":   random.random(),
    "outstanding_amount": random.random(),
    "discount_amount":    random.random(),
    "refund_amount":      random.random(),
    "payment_delay_days": random.random(),
}
# Financial chaos: this run's "normal" ranges — themselves random
DEMAND_MIN   = random.randint(-10_000_000, 0)
DEMAND_MAX   = random.randint(0, 50_000_000)

COLLECTED_MIN = random.randint(-5_000_000, 0)
COLLECTED_MAX = random.randint(0, 60_000_000)

DISCOUNT_MIN  = random.randint(-1_000_000, 0)
DISCOUNT_MAX  = random.randint(0, 999_999_999)   # sometimes discounts blow up globally

REFUND_MIN    = random.randint(-1_000_000, 0)
REFUND_MAX    = random.randint(0, 999_999_999)

DELAY_MIN     = random.randint(-90, 0)
DELAY_MAX     = random.randint(0, 1500)

# "Extreme outlier" injection rate per field — also random per run
EXTREME_RATE = {
    "discount":     random.random() ,
    "refund":       random.random() ,
    "collected":    random.random() ,
    "outstanding":  random.random() ,
    "demand":       random.random() ,
    "delay":        random.random()
}
# Track the actual number of extreme values written
actual_extreme_counts = {
    "discount": 0,
    "refund": 0,
    "collected": 0,
    "outstanding": 0,
    "demand": 0,
    "delay": 0
}

# Invalid-date rate
INVALID_DATE_RATE = random.random() 

# Date universe: start year/month/day are random
BASE_DATE = datetime(
    random.randint(2018, 2026),
    random.randint(1, 12),
    random.randint(1, 28)
)
DATE_SPREAD_DAYS = random.randint(1, 730)   # how wide is the date range?

# Categorical pools — sizes vary per run
ALL_PROJECTS      = ["AQUA", "SKYLINE", "GREEN", "TERRA", "NOVA", "APEX", "ZENITH"]

# Slice these pools randomly — so some runs have 2 owners, some have 9
projects      = random.sample(ALL_PROJECTS, random.randint(1, len(ALL_PROJECTS)))


# ─────────────────────────────────────────────
#  ID GENERATION  (gaps, dupes, blanks — chaos)
# ─────────────────────────────────────────────

def make_transaction_ids(n):
    """
    Generate n transaction IDs that may have:
    - gaps (some numbers skipped)
    - duplicates (same ID on multiple rows)
    - completely missing (empty string)
    """
    # ── GUARD CLAUSE FOR ZERO RECORDS ──
    if n == 0:
        return []

    pool_size = random.randint(0, int(n * 2.5))
    
    # Safe check: if pool_size is 0, return all empty strings directly!
    if pool_size == 0:
        return [""] * n

    pool = [f"TXN{i}" for i in range(1, pool_size + 1)]
    chosen = [random.choice(pool) for _ in range(n)]
    return chosen


def make_customer_ids(n):
    """Customer IDs: sometimes sequential, sometimes random, sometimes all same, sometimes completely empty."""
    # ── GUARD CLAUSE FOR ZERO RECORDS ──
    if n == 0:
        return []

    style = random.choice(["sequential", "random_pool", "all_same", "alpha_numeric","empty"])
    
    if style == "empty":
        return [""] * n
    elif style == "sequential":
        return [f"CUST{i}" for i in range(1, n + 1)]
    elif style == "random_pool":
        pool_size = random.randint(5, n)
        pool = [f"CUST{i}" for i in range(1, pool_size + 1)]
        return [random.choice(pool) for _ in range(n)]
    elif style == "all_same":
        cid = f"CUST{random.randint(1, 999)}"
        return [cid] * n
    else:
        chars = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"
        return ["C" + "".join(random.choices(chars, k=6)) for _ in range(n)]


def make_unit_ids(n):
    # ── GUARD CLAUSE FOR ZERO RECORDS ──
    if n == 0:
        return []

    style = random.choice(["sequential", "random_pool", "mixed_format", "empty"])
    
    if style == "empty":
        return [""] * n
    elif style == "sequential":
        return [f"UNIT{i}" for i in range(1, n + 1)]
    elif style == "random_pool":
        # Safe guard: ensure pool_size is at least 1
        pool_size = random.randint(1, max(11, n // 2))
        pool = [f"UNIT{i}" for i in range(1, pool_size + 1)]
        return [random.choice(pool) for _ in range(n)]
    else:
        return [
            random.choice([
                f"UNIT{j}",
                f"U-{random.randint(100,999)}",
                f"BLK{random.randint(1,10)}-{random.randint(1,50)}"
            ])
            for j in range(1, n + 1)
        ]

#  HELPERS

def maybe_blank(value, field_name):
    """Blank out a value based on this run's missing rate for that field."""
    if random.random() < MISSING_RATE.get(field_name, 0):
        return ""
    return value


def random_financial(mn, mx):
    """Random int in range, but occasionally inject an extreme outlier."""
    return random.randint(mn, mx)


def extreme_or_normal(value, field_key, extreme_mn, extreme_mx):
    """Replace value with an extreme number based on this run's extreme rate."""
    if random.random() < EXTREME_RATE.get(field_key, 0):
        actual_extreme_counts[field_key] += 1  # <-- Increment our tracker!
        return random.randint(extreme_mn, extreme_mx)
    return value


def random_date():
    return BASE_DATE + timedelta(days=random.randint(0, DATE_SPREAD_DAYS))


# ─────────────────────────────────────────────
#  BUILD RECORDS
# ─────────────────────────────────────────────

txn_ids  = make_transaction_ids(NUM_RECORDS)
cust_ids = make_customer_ids(NUM_RECORDS)
unit_ids = make_unit_ids(NUM_RECORDS)

records = []

for i in range(NUM_RECORDS):

    # ── Financial values ──────────────────────
    demand_amount    = random_financial(DEMAND_MIN, DEMAND_MAX)
    collected_amount = random_financial(COLLECTED_MIN, COLLECTED_MAX)
    discount_amount  = random_financial(DISCOUNT_MIN, DISCOUNT_MAX)
    refund_amount    = random_financial(REFUND_MIN, REFUND_MAX)
    payment_delay_days = random.randint(DELAY_MIN, DELAY_MAX)

    # Inject per-record extremes based on run-level rate
    demand_amount    = extreme_or_normal(demand_amount,    "demand",      1_000_000, 999_999_999)
    discount_amount  = extreme_or_normal(discount_amount,  "discount",    5_000_000, 999_999_999)
    refund_amount    = extreme_or_normal(refund_amount,    "refund",      5_000_000, 999_999_999)
    collected_amount = extreme_or_normal(collected_amount, "collected",   5_000_000, 999_999_999)
    payment_delay_days = extreme_or_normal(payment_delay_days, "delay", -365, 3650)

    outstanding_amount = demand_amount - collected_amount - discount_amount + refund_amount
    outstanding_amount = extreme_or_normal(outstanding_amount, "outstanding", 5_000_000, 999_999_999)

    # ── Dates ──────────────────────────────────
    demand_date  = random_date()
    payment_date = demand_date + timedelta(days=payment_delay_days)

    # Invalid date injection
    if random.random() < INVALID_DATE_RATE:
        payment_date = demand_date - timedelta(days=random.randint(1, 365))

    # ── Categorical ────────────────────────────
    
    project_id     = random.choice(projects)


       # ── Apply missing-field rates ─────────────
    transaction_id = maybe_blank(txn_ids[i],  "transaction_id")
    customer_id    = maybe_blank(cust_ids[i], "customer_id")
    unit_id_val    = maybe_blank(unit_ids[i], "unit_id")
    project_id     = maybe_blank(project_id,  "project_id")

    # Apply to dates
    demand_date_str  = maybe_blank(demand_date.strftime("%Y-%m-%d"),  "demand_date")
    payment_date_str = maybe_blank(payment_date.strftime("%Y-%m-%d"), "payment_date")

    # Apply to numericals
    demand_amount_val      = maybe_blank(demand_amount,      "demand_amount")
    collected_amount_val   = maybe_blank(collected_amount,   "collected_amount")
    outstanding_amount_val = maybe_blank(outstanding_amount, "outstanding_amount")
    discount_amount_val    = maybe_blank(discount_amount,    "discount_amount")
    refund_amount_val      = maybe_blank(refund_amount,      "refund_amount")
    payment_delay_days_val = maybe_blank(payment_delay_days, "payment_delay_days")

    records.append({
        "transaction_id":     transaction_id,
        "customer_id":        customer_id,
        "project_id":         project_id,
        "unit_id":            unit_id_val,
        "demand_amount":      demand_amount_val,
        "collected_amount":   collected_amount_val,
        "outstanding_amount": outstanding_amount_val,
        "discount_amount":    discount_amount_val,
        "refund_amount":      refund_amount_val,
        "payment_delay_days": payment_delay_days_val,
        "demand_date":        demand_date_str,
        "payment_date":       payment_date_str,
    })


# ─────────────────────────────────────────────
#  SAVE & ACTUAL METRICS PRINT
# ─────────────────────────────────────────────

# Explicitly define columns so that even if records is empty, headers are generated!
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

df = pd.DataFrame(records, columns=EXPECTED_COLUMNS)
output_path = "data/sample_finance_data.csv"
df.to_csv(output_path, index=False)

print(f"Generated {NUM_RECORDS} records → {output_path}")
print(f"  Date universe   : {BASE_DATE.date()} + {DATE_SPREAD_DAYS} days")
print(f"  Demand range    : {DEMAND_MIN:,} → {DEMAND_MAX:,}")
print(f"  Discount range  : {DISCOUNT_MIN:,} → {DISCOUNT_MAX:,}")
print(f"  Delay range     : {DELAY_MIN} → {DELAY_MAX} days")

# CALCULATE ACTUAL MISSING RATE FOR EVERY COLUMN
actual_missing_rates = {}
for col in df.columns:
    missing_count = df[col].isnull().sum() + (df[col] == "").sum()
    pct = (missing_count / len(df)) * 100 if len(df) > 0 else 0
    actual_missing_rates[col] = f"{pct:.0f}%"

print(f"  Actual Missing rates : {actual_missing_rates}")

#  CALCULATE ACTUAL EXTREME RATE FOR EVERY TARGET COLUMN
actual_extreme_rates = {}
for key, count in actual_extreme_counts.items():
    # Map key to the actual column name in the DataFrame
    col_name = "payment_delay_days" if key == "delay" else f"{key}_amount"
    
    # If the column is 100% blank, actual extreme rate is physically 0%
    if col_name in df.columns and df[col_name].fillna("").astype(str).str.strip().eq("").all():
        pct = 0
    else:
        pct = (count / len(df)) * 100 if len(df) > 0 else 0
        
    actual_extreme_rates[key] = f"{pct:.0f}%"

print(f"  Actual Extreme rates : {actual_extreme_rates}")
print(f"  Invalid date %  : {INVALID_DATE_RATE*100:.0f}%")
print(f"  Projects pool   : {projects}")