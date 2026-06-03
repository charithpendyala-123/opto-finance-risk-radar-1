# ==============================================================================
# ANOMALY RESULTS LOGGER REPOSITORY (OPTION 2)
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

def save_anomaly_results(conn, anomalies, violations_df, db_ids, df):
    """
    Parses and stores anomaly engine details to the anomaly_results table,
    linking them to the auto-generated serial transaction IDs.
    """
    if conn is None or not db_ids:
        return 0

    records_to_insert = []

    # Map transaction keys to their corresponding database IDs to correctly handle duplicates in order
    lookup = {}
    for idx, row in df.iterrows():
        key = (
            str(row.get('transaction_id')).strip() if pd.notna(row.get('transaction_id')) else None,
            str(row.get('customer_id')).strip() if pd.notna(row.get('customer_id')) else None,
            str(row.get('project_id')).strip() if pd.notna(row.get('project_id')) else None,
            str(row.get('unit_id')).strip() if pd.notna(row.get('unit_id')) else None,
            str(row.get('demand_date')).strip() if pd.notna(row.get('demand_date')) else None,
            str(row.get('payment_date')).strip() if pd.notna(row.get('payment_date')) else None
        )
        if key not in lookup:
            lookup[key] = []
        lookup[key].append(db_ids[idx])

    # A) Process Parallel Outlier Detections
    if anomalies:
        for a in anomalies:
            txn_id = a.get("transaction_id")
            if not txn_id:
                continue

            key = (
                str(txn_id).strip() if txn_id else None,
                str(a.get('customer_id')).strip() if a.get('customer_id') else None,
                str(a.get('project_id')).strip() if a.get('project_id') else None,
                str(a.get('unit_id')).strip() if a.get('unit_id') else None,
                str(a.get('demand_date')).strip() if a.get('demand_date') else None,
                str(a.get('payment_date')).strip() if a.get('payment_date') else None
            )

            db_id = None
            if key in lookup and lookup[key]:
                db_id = lookup[key].pop(0)

            if db_id is None:
                continue

            # 1. Z-Score
            z_det = a.get("zscore_details", {})
            if z_det.get("flagged", False):
                records_to_insert.append((
                    db_id,
                    txn_id,
                    "ZScore",
                    True,
                    None,
                    z_det.get("reason", "Z-Score threshold breached")
                ))

            # 2. IQR
            iqr_det = a.get("iqr_details", {})
            if iqr_det.get("flagged", False):
                records_to_insert.append((
                    db_id,
                    txn_id,
                    "IQR",
                    True,
                    None,
                    iqr_det.get("reason", "IQR boundary breached")
                ))

            # 3. Contextual Group-wise Engine
            gw_det = a.get("groupwise_details", {})
            if gw_det.get("flagged", False):
                gw_list = gw_det.get("anomalies", [])
                reasons = [f"[{g.get('case', 'Anomaly')}] {g.get('reason', '')}" for g in gw_list if g]
                records_to_insert.append((
                    db_id,
                    txn_id,
                    "Groupwise",
                    True,
                    None,
                    "; ".join(reasons) if reasons else "Groupwise context criteria breach"
                ))

            # 4. Unsupervised ML Isolation Forest Engine
            if_det = a.get("isolation_forest_details", {})
            if if_det.get("flagged", False):
                records_to_insert.append((
                    db_id,
                    txn_id,
                    "IsolationForest",
                    True,
                    float(if_det.get("score", 0.0)),
                    if_det.get("reason", "Isolation Forest score outlier")
                ))

    # B) Process Rule Engine Violations
    if violations_df is not None and len(violations_df) > 0:
        grouped = violations_df.groupby("source_row")
        for s_row, group in grouped:
            idx = int(s_row) - 2
            if idx < 0 or idx >= len(db_ids):
                continue
            db_id = db_ids[idx]
            txn_id = str(group.iloc[0].get("transaction_id", ""))
            
            reasons = [f"[{row['rule_id']}] {row.get('rule_description', '')}" for _, row in group.iterrows()]
            records_to_insert.append((
                db_id,
                txn_id,
                "RuleEngine",
                True,
                float(len(group)),
                "; ".join(reasons)
            ))

    if not records_to_insert:
        return 0

    upsert_query = """
        INSERT INTO anomaly_results (transaction_row_id, transaction_id, engine_name, anomaly_flag, anomaly_score, reason)
        VALUES %s
        ON CONFLICT (transaction_row_id, engine_name) DO UPDATE SET
            anomaly_flag = EXCLUDED.anomaly_flag,
            anomaly_score = EXCLUDED.anomaly_score,
            reason = EXCLUDED.reason;
    """

    try:
        with conn.cursor() as cur:
            execute_values(cur, upsert_query, records_to_insert)
        conn.commit()
        return len(records_to_insert)
    except Exception as e:
        conn.rollback()
        print(f"[Database Repo Error] Failed to save anomaly results: {e}")
        return 0