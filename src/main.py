import sys
import os
import importlib
import psycopg2
import pandas as pd 

# Dynamically add the parent directory (project root) to search path for safe imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("=========================================================================")
    print("                 OPTOxCRM FINANCE RISK RADAR SYSTEM                      ")
    print("=========================================================================")
    
    print("\n[Orchestrator] Importing pipeline sub-engines...")
    try:
        rule_validator = importlib.import_module("src.03_rule_validator")
        anomaly_detector = importlib.import_module("src.08_anomaly_detector")
        risk_score = importlib.import_module("src.09_risk_score")
        rec_engine = importlib.import_module("src.10_recommendation_engine")
        report_gen = importlib.import_module("src.11_report_generator")
        
        # Ingest database repositories
        import src.db as db
        import src.transaction_repository as txn_repo
        import src.anomaly_repository as anom_repo
        import src.risk_repository as risk_repo
    except ModuleNotFoundError as e:
        print(f"Error importing modules: {e}")
        print("Please ensure your working directory is set to the project root.")
        return

    # Attempt database connection
    conn = db.get_connection()
    if conn:
        print("[Orchestrator] Connected to PostgreSQL successfully. Initializing schema...")
        db.init_db(conn)
        
        # Clear transactional tables before reloading to prevent double storage on run
        print("[Database] Flushing previous runs...")
        try:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE transactions CASCADE;")
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"Error flushing database tables: {e}")
    else:
        print("[Orchestrator] Warning: PostgreSQL offline. Running in file-only fallback mode.")

    csv_path = "data/sample_finance_data.csv"

    # Step 1: Rule Validation
    print("\n[Step 1/5] Running Hard Rules Validation Engine...")
    df_raw, violations = rule_validator.validate_finance_csv(csv_path)
    if df_raw is None:
        print("Pipeline aborted. Rule validation failed structurally.")
        return

    # Apply Upstream Virtual ID patching
    df = anomaly_detector.patch_transaction_ids(df_raw)

    # Save to Transactions Table and fetch keys
        # Save to Transactions Table and fetch keys
    db_ids = []
    if conn:
        print("[Database] Persisting transactions to 'transactions' table...")
        db_ids = txn_repo.save_transactions(conn, df)
        print(f"[Database] Successfully stored {len(db_ids)} transaction rows.")
        
        # Initialize auditor feedback records as Clear (False) by default
        import src.feedback_repository as feedback_repo
        print("[Database] Initializing auditor feedback records...")
        feedback_repo.initialize_feedback(conn, df['transaction_id'].tolist())

    # Step 2: Anomaly Outlier Detection
    print("\n[Step 2/5] Running Statistical & ML Anomaly Detection Engines...")
    anomalies = anomaly_detector.run_anomaly_pipeline(csv_path)
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

    # Step 5: Consolidated Exporter
    print("\n[Step 5/5] Packaging Consolidated Corporate Reports...")
    report_gen.run_report_generator()

    # Phase 6: Verify Historical Storage (Verify everything populated before exit)
        # Phase 6: Verify Historical Storage (Verify everything populated before exit)
    if conn:
        print("\n[Phase 6: Verification] Auditing data storage levels...")
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM transactions;")
                t_count = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM anomaly_results;")
                a_count = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM risk_results;")
                r_count = cur.fetchone()[0]
            
            print("=========================================================================")
            print("                POSTGRESQL DATABASE VERIFICATION (OPTION 2)              ")
            print("=========================================================================")
            print(f"  - Table 'transactions'     : {t_count} records (Expected: {len(df)})")
            print(f"  - Table 'anomaly_results'  : {a_count} records")
            print(f"  - Table 'risk_results'     : {r_count} records (Expected: {len(df)})")
            print("=========================================================================")
            
            if t_count == len(df) and r_count == len(df):
                print(f"[Verification Result] PASS: All {len(df)} records stored successfully.")
            else:
                print(f"[Verification Result] WARNING: Stored counts ({t_count}) deviate from raw row dimensions ({len(df)}).")
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