import sys
import os
import importlib
import psycopg2
import pandas as pd 

# Dynamically add the parent directory (project root) to search path for safe imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main(user_id="system_default", csv_path="data/sample_finance_data.csv", batch_id=None):
    print("=========================================================================")
    print("                 OPTOxCRM FINANCE RISK RADAR SYSTEM                      ")
    print("=========================================================================")
    
    print("\n[Orchestrator] Importing pipeline sub-engines...")
    try:
        rule_validator = importlib.import_module("src.03_rule_validator")
        importlib.reload(rule_validator)
        anomaly_detector = importlib.import_module("src.08_anomaly_detector")
        importlib.reload(anomaly_detector)
        risk_score = importlib.import_module("src.09_risk_score")
        importlib.reload(risk_score)
        rec_engine = importlib.import_module("src.10_recommendation_engine")
        importlib.reload(rec_engine)
        report_gen = importlib.import_module("src.11_report_generator")
        importlib.reload(report_gen)
        rolling_window = importlib.import_module("src.rolling_window") 
        importlib.reload(rolling_window)
        
        # Ingest database repositories
        import src.db as db
        importlib.reload(db)
        import src.transaction_repository as txn_repo
        importlib.reload(txn_repo)
        import src.anomaly_repository as anom_repo
        importlib.reload(anom_repo)
        import src.risk_repository as risk_repo
        importlib.reload(risk_repo)
    except ModuleNotFoundError as e:
        print(f"Error importing modules: {e}")
        print("Please ensure your working directory is set to the project root.")
        return

    # Initialize batch_id early
    import datetime
    if not batch_id:
        batch_id = f"BAT_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Attempt database connection
    conn = db.get_connection()
    backup_reviews = []
    if conn:
        print(f"[Orchestrator] Connected to PostgreSQL successfully. Initializing schema...")
        db.init_db(conn)
        
        # Backup human auditor reviews for this specific batch before flushing
        print(f"[Database] Backing up existing human reviews for batch {batch_id}...")
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT t.transaction_id, t.customer_id, t.unit_id, af.fraud_label, af.auditor_comments
                    FROM auditor_feedback af
                    JOIN transactions t ON af.transaction_row_id = t.id
                    WHERE t.user_id = %s AND t.upload_batch_id = %s AND af.fraud_label IS NOT NULL;
                """, (user_id, batch_id))
                backup_reviews = cur.fetchall()
                print(f"[Database] Backed up {len(backup_reviews)} human reviews.")
        except Exception as e:
            print(f"Warning: Failed to backup reviews: {e}")
            
        print("[Database] Ingesting in upsert mode. Old reviews will be preserved in-place.")
    else:
        print("[Orchestrator] Warning: PostgreSQL offline. Running in file-only fallback mode.")

    # Step 1: Rule Validation
    print("\n[Step 1/5] Running Hard Rules Validation Engine...")
    df_raw, violations = rule_validator.validate_finance_csv(csv_path)
    if df_raw is None:
        print("Pipeline aborted. Rule validation failed structurally.")
        return

    # Apply Upstream Virtual ID patching
    df = anomaly_detector.patch_transaction_ids(df_raw)

    # Save to Transactions Table and fetch keys
    db_ids = []
    if conn:
        print(f"[Database] Persisting transactions to 'transactions' table (Batch: {batch_id})...")
        db_ids = txn_repo.save_transactions(conn, df, user_id=user_id, upload_batch_id=batch_id)
        print(f"[Database] Successfully stored/upserted {len(db_ids)} transaction rows.")
        
        # Initialize auditor feedback records as Clear (False) by default
        import src.feedback_repository as feedback_repo
        print("[Database] Initializing auditor feedback records...")
        feedback_repo.initialize_feedback(conn, db_ids)

        # Restore backed up human reviews by mapping them to new transaction IDs
        if backup_reviews:
            print(f"[Database] Restoring human reviews for batch {batch_id}...")
            try:
                with conn.cursor() as cur:
                    for txn_id, cust_id, unit_id, label, comments in backup_reviews:
                        cur.execute("""
                            SELECT id FROM transactions 
                            WHERE user_id = %s AND upload_batch_id = %s AND transaction_id = %s AND customer_id = %s AND unit_id = %s;
                        """, (user_id, batch_id, txn_id, cust_id, unit_id))
                        row = cur.fetchone()
                        if row:
                            new_row_id = row[0]
                            cur.execute("""
                                INSERT INTO auditor_feedback (transaction_row_id, transaction_id, fraud_label, auditor_comments, reviewed_at)
                                VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                                ON CONFLICT (transaction_row_id) DO UPDATE SET
                                    fraud_label = EXCLUDED.fraud_label,
                                    auditor_comments = EXCLUDED.auditor_comments,
                                    reviewed_at = CURRENT_TIMESTAMP;
                            """, (new_row_id, txn_id, label, comments))
                conn.commit()
                print(f"[Database] Successfully restored {len(backup_reviews)} auditor reviews.")
            except Exception as e:
                conn.rollback()
                print(f"Warning: Failed to restore reviews: {e}")

    # Step 2: Anomaly Outlier Detection
        # Step 2: Anomaly Outlier Detection
    print("\n[Step 2/5] Running Statistical & ML Anomaly Detection Engines...")
    anomalies = anomaly_detector.run_anomaly_pipeline(csv_path, conn=conn, user_id=user_id, batch_id=batch_id)
    if anomalies is None:
        print("Pipeline aborted. Anomaly detection failed.")
        return

    # Save to Anomalies Table
    if conn and db_ids:
        print("[Database] Persisting detections to 'anomaly_results' table...")
        anoms_saved = anom_repo.save_anomaly_results(conn, anomalies, violations, db_ids, df)
        print(f"[Database] Successfully stored {anoms_saved} anomaly result records.")

    # Step 3: Risk Scoring & Classification
    print("\n[Step 3/5] Running Unified Risk Scoring Engine...")
    scores = risk_score.run_risk_scoring(csv_path)
    if scores is None:
        print("Pipeline aborted. Risk scoring failed.")
        return

    # Step 4: CFO Audit Recommendations
    print("\n[Step 4/5] Running Mitigation Recommendation Engine...")
    enriched_profiles = rec_engine.run_recommendation_engine()

    # Save to Risk Results Table
    if conn and db_ids:
        print("[Database] Persisting assessment outcomes to 'risk_results' table...")
        risk_saved = risk_repo.save_risk_results(conn, enriched_profiles, db_ids)
        print(f"[Database] Successfully stored {risk_saved} risk profile records.")
        
        # Step 4.5: Calculate and Persist Rolling Window Features
        print("\n[Step 4.5] Running Rolling Window Feature Layer...")
        rw_success = rolling_window.run_rolling_window_pipeline(conn, user_id=user_id, batch_id=batch_id)
        if rw_success:
            print("[Orchestrator] Rolling window features generated successfully.")
        else:
            print("[Orchestrator] Warning: Rolling window feature generation encountered errors.")

    # Step 5: Consolidated Exporter
    print("\n[Step 5/5] Packaging Consolidated Corporate Reports...")
    report_gen.run_report_generator()

    # Phase 6: Verify Historical Storage (Verify everything populated before exit)
    if conn:
        print("\n[Phase 6: Verification] Auditing database storage levels...")
        try:
            with conn.cursor() as cur:
                # Query database records specifically for the active batch
                cur.execute("SELECT COUNT(*) FROM transactions WHERE user_id = %s AND upload_batch_id = %s;", (user_id, batch_id))
                t_count = cur.fetchone()[0]
                cur.execute("""
                    SELECT COUNT(*) 
                    FROM anomaly_results ar
                    JOIN transactions t ON ar.transaction_row_id = t.id
                    WHERE t.user_id = %s AND t.upload_batch_id = %s;
                """, (user_id, batch_id))
                a_count = cur.fetchone()[0]
                cur.execute("""
                    SELECT COUNT(*) 
                    FROM risk_results rr
                    JOIN transactions t ON rr.transaction_row_id = t.id
                    WHERE t.user_id = %s AND t.upload_batch_id = %s;
                """, (user_id, batch_id))
                r_count = cur.fetchone()[0]
                
                # Fetch rolling window features count for this batch
                cur.execute("""
                    SELECT COUNT(*) 
                    FROM transaction_rolling_features trf
                    JOIN transactions t ON trf.transaction_row_id = t.id
                    WHERE t.user_id = %s AND t.upload_batch_id = %s;
                """, (user_id, batch_id))
                trf_count = cur.fetchone()[0]
                
                # Also fetch total partition counts for information
                cur.execute("SELECT COUNT(*) FROM transactions WHERE user_id = %s;", (user_id,))
                total_t_count = cur.fetchone()[0]
            
            expected_count = len(set(db_ids)) if db_ids else 0
            
            print("=========================================================================")
            print("                POSTGRESQL DATABASE VERIFICATION                         ")
            print("=========================================================================")
            print(f"  - User partition           : {user_id}")
            print(f"  - Active Batch ID          : {batch_id}")
            print(f"  - Batch 'transactions'     : {t_count} records (Expected: {expected_count})")
            print(f"  - Batch 'anomaly_results'  : {a_count} records")
            print(f"  - Batch 'risk_results'     : {r_count} records (Expected: {expected_count})")
            print(f"  - Batch 'rolling_features' : {trf_count} records (Expected: {expected_count})")
            print(f"  - Cumulative Total History : {total_t_count} records in user partition")
            print("=========================================================================")
            
            if t_count == expected_count and r_count == expected_count:
                print(f"[Verification Result] PASS: All {expected_count} unique records for batch '{batch_id}' stored successfully.")
            else:
                print(f"[Verification Result] WARNING: Stored counts ({t_count}) deviate from expected unique records ({expected_count}).")
        except Exception as e:
            print(f"[Verification Result] Error running checks: {e}")
        finally:
            conn.close()
            print("[Database] Connection closed.")

    print("\n=========================================================================")
    print("            ALL RADAR PIPELINE MODULES EXECUTED SUCCESSFULLY             ")
    print("=========================================================================")
    print("Forensic Outputs Ready for CFO Review:")
    print("  - Unified Ledger JSON      : reports/risk_report.json")
    print("  - Flagged spreadsheet CSV  : reports/high_risk_transactions.csv")
    print("  - Executive ASCII Summary  : reports/summary_report.txt")
    print("=========================================================================\n")

if __name__ == "__main__":
    main()