# ==============================================================================
# RISK ENGINE RESULTS REPOSITORY (OPTION 2)
# ==============================================================================
from psycopg2.extras import execute_values

def save_risk_results(conn, risk_list, db_ids):
    """
    Saves final risk assessment and recommendation metrics to the database.
    """
    if conn is None or not risk_list or not db_ids:
        return 0

    records_to_insert = []
    for r in risk_list:
        s_row = r.get("source_row")
        if s_row is None:
            continue
        idx = int(s_row) - 2
        if idx < 0 or idx >= len(db_ids):
            continue
        db_id = db_ids[idx]

        records_to_insert.append((
            db_id,
            r.get("transaction_id", ""),
            int(r.get("risk_score", 0)),
            str(r.get("risk_severity", "LOW")),
            str(r.get("recommended_action", ""))
        ))

    upsert_query = """
        INSERT INTO risk_results (transaction_row_id, transaction_id, risk_score, severity, recommendation)
        VALUES %s
        ON CONFLICT (transaction_row_id) DO UPDATE SET
            risk_score = EXCLUDED.risk_score,
            severity = EXCLUDED.severity,
            recommendation = EXCLUDED.recommendation;
    """

    try:
        with conn.cursor() as cur:
            execute_values(cur, upsert_query, records_to_insert)
        conn.commit()
        return len(records_to_insert)
    except Exception as e:
        conn.rollback()
        print(f"[Database Repo Error] Failed to bulk save risk results: {e}")
        return 0