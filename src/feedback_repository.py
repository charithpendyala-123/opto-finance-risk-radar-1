# ==============================================================================
# AUDITOR FEEDBACK REPOSITORY
# ==============================================================================
import psycopg2

def save_feedback(conn, transaction_id, fraud_label, auditor_comments=""):
    """
    Saves or updates the auditor's fraud verdict for a transaction.
    """
    if conn is None or not transaction_id:
        return False

    upsert_query = """
        INSERT INTO auditor_feedback (transaction_id, fraud_label, auditor_comments, reviewed_at)
        VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (transaction_id) DO UPDATE SET
            fraud_label = EXCLUDED.fraud_label,
            auditor_comments = EXCLUDED.auditor_comments,
            reviewed_at = CURRENT_TIMESTAMP;
    """

    try:
        with conn.cursor() as cur:
            cur.execute(upsert_query, (transaction_id, fraud_label, auditor_comments))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"[Database Repo Error] Failed to save auditor feedback for {transaction_id}: {e}")
        return False

def get_feedback(conn, transaction_id):
    """
    Fetches the auditor verdict and comments for a transaction.
    """
    if conn is None or not transaction_id:
        return None

    query = "SELECT transaction_id, fraud_label, auditor_comments, reviewed_at FROM auditor_feedback WHERE transaction_id = %s;"
    
    try:
        with conn.cursor() as cur:
            cur.execute(query, (transaction_id,))
            row = cur.fetchone()
            if row:
                colnames = [desc[0] for desc in cur.description]
                return dict(zip(colnames, row))
    except Exception as e:
        print(f"[Database Repo Error] Failed to fetch feedback for {transaction_id}: {e}")
    return None

def get_all_feedback(conn):
    """
    Retrieves all human-labeled transactions (used as training labels for Phase 9 ML model).
    """
    if conn is None:
        return []

    query = "SELECT transaction_id, fraud_label, auditor_comments, reviewed_at FROM auditor_feedback ORDER BY reviewed_at DESC;"
    
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
            colnames = [desc[0] for desc in cur.description]
            return [dict(zip(colnames, r)) for r in rows]
    except Exception as e:
        print(f"[Database Repo Error] Failed to fetch all feedback entries: {e}")
    return []
def initialize_feedback(conn, transaction_ids):
    """
    Pre-populates the auditor_feedback table with True (Fraud) for a list of transaction IDs.
    Updates existing records to True if they are already present.
    """
    if conn is None or not transaction_ids:
        return False

    query = """
        INSERT INTO auditor_feedback (transaction_id, fraud_label, auditor_comments, reviewed_at)
        VALUES (%s, TRUE, 'System initialized as fraud', CURRENT_TIMESTAMP)
        ON CONFLICT (transaction_id) DO UPDATE SET
            fraud_label = EXCLUDED.fraud_label,
            auditor_comments = EXCLUDED.auditor_comments,
            reviewed_at = CURRENT_TIMESTAMP;
    """

    try:
        with conn.cursor() as cur:
            unique_ids = list(set(transaction_ids))
            cur.executemany(query, [(tid,) for tid in unique_ids])
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"[Database Repo Error] Failed to initialize feedback records: {e}")
        return False