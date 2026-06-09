# ==============================================================================
# OPTOxCRM - DYNAMIC AUDITOR SEEDING & TRAINING TRIGGER (NO-OVERLAP)
# ==============================================================================
import sys
import os

# Ensure project root is in path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.db as db
import src.train_model as trainer

def seed_reviews_and_train():
    # Establish connection using configuration from src/db.py
    conn = db.get_connection()
    if conn is None:
        print("[Error] Could not connect to PostgreSQL database.")
        return

    try:
        with conn.cursor() as cur:
            # 1. Get the total count of transactions present in the active user partition
            cur.execute("""
                SELECT COUNT(*) 
                FROM risk_results rr 
                JOIN transactions t ON rr.transaction_row_id = t.id 
                WHERE t.user_id = 'system_default';
            """)
            total_records = cur.fetchone()[0] or 0
            
            if total_records < 10:
                print(f"[Error] Too few records ({total_records}) to map a balanced training set (need at least 10).")
                return

            # Calculate a dynamic limit that guarantees 0% overlap
            # E.g., if total is 60, limit is 30. If total is 1000+, limit is capped at 50.
            limit = min(50, total_records // 2)
            print(f"Total database records found: {total_records}. Using limit: {limit} per class (no overlap).")

            # 2. Fetch the top N highest-risk transactions (to label as Fraud)
            cur.execute("""
                SELECT rr.transaction_row_id, rr.transaction_id, rr.risk_score
                FROM risk_results rr
                JOIN transactions t ON rr.transaction_row_id = t.id
                WHERE t.user_id = 'system_default'
                ORDER BY rr.risk_score DESC
                LIMIT %s;
            """, (limit,))
            fraud_rows = cur.fetchall()

            # 3. Fetch the bottom N lowest-risk transactions (to label as Clean)
            cur.execute("""
                SELECT rr.transaction_row_id, rr.transaction_id, rr.risk_score
                FROM risk_results rr
                JOIN transactions t ON rr.transaction_row_id = t.id
                WHERE t.user_id = 'system_default'
                ORDER BY rr.risk_score ASC
                LIMIT %s;
            """, (limit,))
            clean_rows = cur.fetchall()

            # 4. Save Fraud reviews (upsert mode to preserve overrides)
            print(f"Writing {len(fraud_rows)} Confirmed Fraud reviews to database...")
            for row_id, txn_id, score in fraud_rows:
                cur.execute("""
                    INSERT INTO auditor_feedback (transaction_row_id, transaction_id, fraud_label, auditor_comments, reviewed_at)
                    VALUES (%s, %s, TRUE, 'Auto-seeded as Confirmed Fraud (Score: ' || %s || ')', CURRENT_TIMESTAMP)
                    ON CONFLICT (transaction_row_id) DO UPDATE SET
                        fraud_label = TRUE,
                        auditor_comments = EXCLUDED.auditor_comments,
                        reviewed_at = CURRENT_TIMESTAMP;
                """, (row_id, txn_id, score))

            # 5. Save Clean reviews (upsert mode to preserve overrides)
            print(f"Writing {len(clean_rows)} Confirmed Clean reviews to database...")
            for row_id, txn_id, score in clean_rows:
                cur.execute("""
                    INSERT INTO auditor_feedback (transaction_row_id, transaction_id, fraud_label, auditor_comments, reviewed_at)
                    VALUES (%s, %s, FALSE, 'Auto-seeded as Confirmed Clean (Score: ' || %s || ')', CURRENT_TIMESTAMP)
                    ON CONFLICT (transaction_row_id) DO UPDATE SET
                        fraud_label = FALSE,
                        auditor_comments = EXCLUDED.auditor_comments,
                        reviewed_at = CURRENT_TIMESTAMP;
                """, (row_id, txn_id, score))

            conn.commit()
            print(f"Successfully seeded {len(fraud_rows)} Fraud and {len(clean_rows)} Clean reviews (Total: {len(fraud_rows) + len(clean_rows)})!")

            # 6. Trigger model training immediately
            print("\nTriggering ML model training pipeline...")
            success = trainer.run_model_training(user_id="system_default", batch_id="All Batches")
            if success:
                print("[Success] Model trained successfully on the new balanced dataset!")
            else:
                print("[Skip/Fail] Model training was skipped or encountered an error.")

    except Exception as e:
        conn.rollback()
        print(f"[Error] Failed during seeding: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    seed_reviews_and_train()