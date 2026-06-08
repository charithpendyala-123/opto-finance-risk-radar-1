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

def save_transactions(conn, df, user_id="system_default", upload_batch_id="BAT_DEFAULT"):
    """
    Ingests and stores transaction rows in the database using bulk upsert (insert or update),
    returning the list of database serial row IDs for the inserted/updated transactions.
    """
    if conn is None or df is None or len(df) == 0:
        return []

    records_to_insert = []
    df_records = df.to_dict('records')
    for idx, row in enumerate(df_records):
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

        # Standardize customer_id and unit_id to 'N/A' to support Unique constraint comparisons
        cust_id = clean_value(row.get('customer_id')) or 'N/A'
        proj_id = clean_value(row.get('project_id')) or 'N/A'
        unit_id = clean_value(row.get('unit_id')) or 'N/A'

        records_to_insert.append((
            user_id,
            upload_batch_id,
            txn_id,
            cust_id,
            proj_id,
            unit_id,
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

    # Deduplicate unique records to prevent ON CONFLICT DO UPDATE duplication error
    seen = {}
    for r in records_to_insert:
        key = (r[0], r[1], r[2], r[3], r[5])  # (user_id, upload_batch_id, transaction_id, customer_id, unit_id)
        seen[key] = r
    unique_records = list(seen.values())

    insert_query = """
        INSERT INTO transactions (
            user_id, upload_batch_id, transaction_id, customer_id, project_id, unit_id,
            demand_date, payment_date, demand_amount, collected_amount,
            outstanding_amount, discount_amount, refund_amount,
            payment_delay_days, payment_gap_days
        ) VALUES %s
        ON CONFLICT (user_id, upload_batch_id, transaction_id, customer_id, unit_id) 
        DO UPDATE SET
            project_id = EXCLUDED.project_id,
            demand_date = EXCLUDED.demand_date,
            payment_date = EXCLUDED.payment_date,
            demand_amount = EXCLUDED.demand_amount,
            collected_amount = EXCLUDED.collected_amount,
            outstanding_amount = EXCLUDED.outstanding_amount,
            discount_amount = EXCLUDED.discount_amount,
            refund_amount = EXCLUDED.refund_amount,
            payment_delay_days = EXCLUDED.payment_delay_days,
            payment_gap_days = EXCLUDED.payment_gap_days,
            uploaded_at = CURRENT_TIMESTAMP
        RETURNING id, user_id, upload_batch_id, transaction_id, customer_id, unit_id;
    """

    try:
        key_to_id = {}
        page_size = 1000
        with conn.cursor() as cur:
            for i in range(0, len(unique_records), page_size):
                chunk = unique_records[i:i+page_size]
                execute_values(cur, insert_query, chunk, page_size=len(chunk))
                # Fetch row ID mapping to align with duplicated rows
                for row_data in cur.fetchall():
                    db_id, u_id, b_id, t_id, c_id, un_id = row_data
                    key_to_id[(u_id, b_id, t_id, c_id, un_id)] = db_id
        conn.commit()
        
        # Build the final aligned list of database IDs matching the original records_to_insert list
        aligned_ids = []
        for r in records_to_insert:
            key = (r[0], r[1], r[2], r[3], r[5])
            aligned_ids.append(key_to_id.get(key))
            
        return aligned_ids
    except Exception as e:
        conn.rollback()
        print(f"[Database Repo Error] Failed to bulk upsert transactions: {e}")
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