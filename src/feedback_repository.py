# ==============================================================================
# AUDITOR FEEDBACK REPOSITORY
# ==============================================================================
import psycopg2

def save_feedback(conn, transaction_row_id, fraud_label, auditor_comments=""):
    """
    Saves or updates the auditor's fraud verdict for a transaction row ID.
    Resolves the logical transaction_id automatically via sub-selection.
    """
    if conn is None or not transaction_row_id:
        return False

    upsert_query = """
        INSERT INTO auditor_feedback (transaction_row_id, transaction_id, fraud_label, auditor_comments, reviewed_at)
        SELECT %s, transaction_id, %s, %s, CURRENT_TIMESTAMP
        FROM transactions WHERE id = %s
        ON CONFLICT (transaction_row_id) DO UPDATE SET
            fraud_label = EXCLUDED.fraud_label,
            auditor_comments = EXCLUDED.auditor_comments,
            reviewed_at = CURRENT_TIMESTAMP;
    """

    try:
        with conn.cursor() as cur:
            cur.execute(upsert_query, (transaction_row_id, fraud_label, auditor_comments, transaction_row_id))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"[Database Repo Error] Failed to save auditor feedback for row {transaction_row_id}: {e}")
        return False

def get_feedback(conn, transaction_row_id):
    """
    Fetches the auditor verdict and comments for a transaction row ID.
    """
    if conn is None or not transaction_row_id:
        return None

    query = "SELECT transaction_row_id, transaction_id, fraud_label, auditor_comments, reviewed_at FROM auditor_feedback WHERE transaction_row_id = %s;"
    
    try:
        with conn.cursor() as cur:
            cur.execute(query, (transaction_row_id,))
            row = cur.fetchone()
            if row:
                colnames = [desc[0] for desc in cur.description]
                return dict(zip(colnames, row))
    except Exception as e:
        print(f"[Database Repo Error] Failed to fetch feedback for row {transaction_row_id}: {e}")
    return None

def get_all_feedback(conn):
    """
    Retrieves all human-labeled transactions (used as training labels for Phase 9 ML model).
    """
    if conn is None:
        return []

    query = """
        SELECT transaction_row_id, transaction_id, fraud_label, auditor_comments, reviewed_at 
        FROM auditor_feedback
        ORDER BY reviewed_at DESC;
    """
    
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
            colnames = [desc[0] for desc in cur.description]
            return [dict(zip(colnames, r)) for r in rows]
    except Exception as e:
        print(f"[Database Repo Error] Failed to fetch all feedback entries: {e}")
    return []

def initialize_feedback(conn, transaction_row_ids):
    """
    Pre-populates the auditor_feedback table with NULL (Pending Review) for a list of transaction row IDs.
    Does not overwrite existing human verdicts (True/False).
    Resolves matching transaction_ids in bulk.
    """
    if conn is None or not transaction_row_ids:
        return False

    query = """
        INSERT INTO auditor_feedback (transaction_row_id, transaction_id, fraud_label, auditor_comments, reviewed_at)
        SELECT id, transaction_id, NULL, 'Pending auditor review', NULL
        FROM transactions
        WHERE id = ANY(%s)
        ON CONFLICT (transaction_row_id) DO NOTHING; -- Preserves existing human reviews
    """

    try:
        with conn.cursor() as cur:
            cur.execute(query, (transaction_row_ids,))
        conn.commit()
        print("[Database] Initialized new transactions as Pending (NULL) feedback records with transaction IDs.")
        return True
    except Exception as e:
        conn.rollback()
        print(f"[Database Repo Error] Failed to initialize feedback records: {e}")
        return False