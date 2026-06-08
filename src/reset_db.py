import sys
import os

# Ensure the project root is in the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.db as db

def reset_database():
    conn = db.get_connection()
    if conn is None:
        print("Error: Could not connect to the database.")
        return

    try:
        with conn.cursor() as cur:
            print("Dropping existing tables to apply the new 5-column unique constraint...")
            cur.execute("""
                DROP TABLE IF EXISTS 
                    anomaly_results, 
                    risk_results, 
                    transaction_rolling_features, 
                    transactions, 
                    auditor_feedback, 
                    system_notifications 
                CASCADE;
            """)
        conn.commit()
        print("Existing tables successfully dropped.")
        
        # Re-initialize the database using the updated schema in src/db.py
        if db.init_db(conn):
            print("Database successfully re-initialized with the new constraints!")
        else:
            print("Failed to re-initialize database.")
            
    except Exception as e:
        conn.rollback()
        print(f"Error resetting database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    reset_database()