# ==============================================================================
# TRANSACTION LEDGER REPOSITORY (OPTION 2)
# ==============================================================================
import pandas as pd
from psycopg2.extras import execute_values

def clean_value(val, data_type=str):
    if pd.isna(val):
        return None
    try:
        if data_type == float:
            return float(val)
        if data_type == int:
            return int(float(val))
        return str(val).strip()
    except (ValueError, TypeError):
        return None

def save_transactions(conn, df):
    """
    Ingests and stores transaction rows in the database using bulk insert,
    returning the list of auto-generated serial database IDs.
    """
    if conn is None or df is None or len(df) == 0:
        return []

    records_to_insert = []
    for idx, row in df.iterrows():
        txn_id = clean_value(row.get('transaction_id'))
        if not txn_id:
            row_num = idx + 2
            txn_id = f"SYS_MISSING_ROW_{row_num}"

        gap = row.get('payment_gap_days')
        if pd.isna(gap):
            try:
                d_date = pd.to_datetime(row.get('demand_date'), errors='coerce')
                p_date = pd.to_datetime(row.get('payment_date'), errors='coerce')
                if pd.notna(d_date) and pd.notna(p_date):
                    gap = int((p_date - d_date).days)
                else:
                    gap = None
            except Exception:
                gap = None
        else:
            gap = clean_value(gap, int)

        records_to_insert.append((
            txn_id,
            clean_value(row.get('customer_id')),
            clean_value(row.get('project_id')),
            clean_value(row.get('unit_id')),
            clean_value(row.get('demand_date')),
            clean_value(row.get('payment_date')),
            clean_value(row.get('demand_amount'), float),
            clean_value(row.get('collected_amount'), float),
            clean_value(row.get('outstanding_amount'), float),
            clean_value(row.get('discount_amount'), float),
            clean_value(row.get('refund_amount'), float),
            clean_value(row.get('payment_delay_days'), int),
            gap
        ))

    insert_query = """
        INSERT INTO transactions (
            transaction_id, customer_id, project_id, unit_id,
            demand_date, payment_date, demand_amount, collected_amount,
            outstanding_amount, discount_amount, refund_amount,
            payment_delay_days, payment_gap_days
        ) VALUES %s;
    """

    try:
        with conn.cursor() as cur:
            execute_values(cur, insert_query, records_to_insert)
            # Retrieve all newly created database IDs in their exact insertion order
            cur.execute("SELECT id FROM transactions ORDER BY id ASC;")
            ids = [r[0] for r in cur.fetchall()]
        conn.commit()
        return ids
    except Exception as e:
        conn.rollback()
        print(f"[Database Repo Error] Failed to bulk insert transactions: {e}")
        return []

def get_transaction(conn, txn_id):
    if conn is None:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM transactions WHERE transaction_id = %s LIMIT 1;", (txn_id,))
            row = cur.fetchone()
            if row:
                colnames = [desc[0] for desc in cur.description]
                return dict(zip(colnames, row))
    except Exception as e:
        print(f"[Database Repo Error] Failed to fetch transaction {txn_id}: {e}")
    return None

def get_customer_transactions(conn, customer_id):
    if conn is None:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM transactions WHERE customer_id = %s ORDER BY demand_date ASC;", (customer_id,))
            rows = cur.fetchall()
            colnames = [desc[0] for desc in cur.description]
            return [dict(zip(colnames, r)) for r in rows]
    except Exception as e:
        print(f"[Database Repo Error] Failed to fetch transactions for customer {customer_id}: {e}")
    return []