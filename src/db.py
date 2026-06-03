# ==============================================================================
# OPTOxCRM FINANCE RISK RADAR - DATABASE CONNECTION & INITIALIZATION (OPTION 2)
# ==============================================================================
import os
import psycopg2

def get_connection():
    """
    Establish a connection to the PostgreSQL database.
    Configuration parameters are loaded from environment variables:
      - DB_HOST (default: localhost)
      - DB_PORT (default: 5432)
      - DB_NAME (default: finance_risk_radar)
      - DB_USER (default: postgres)
      - DB_PASSWORD (default: postgres)
    """
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    dbname = os.getenv("DB_NAME", "finance_risk_radar")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "postgres")

    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
        return conn
    except Exception as e:
        print(f"[Database Error] Failed to connect to PostgreSQL: {e}")
        return None

def init_db(conn):
    """
    Initializes core tables and indexes in the database schema.
    """
    if conn is None:
        return False
    
    queries = [
        # Table 1: transactions (Auto-incrementing ID preserves duplicates)
        """
        CREATE TABLE IF NOT EXISTS transactions (
            id SERIAL PRIMARY KEY,
            transaction_id VARCHAR(100),
            customer_id VARCHAR(100),
            project_id VARCHAR(100),
            unit_id VARCHAR(100),
            demand_date DATE,
            payment_date DATE,
            demand_amount NUMERIC,
            collected_amount NUMERIC,
            outstanding_amount NUMERIC,
            discount_amount NUMERIC,
            refund_amount NUMERIC,
            payment_delay_days INT,
            payment_gap_days INT,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        # Table 2: anomaly_results (References transactions.id to handle ID collisions)
        """
        CREATE TABLE IF NOT EXISTS anomaly_results (
            transaction_row_id INT REFERENCES transactions(id) ON DELETE CASCADE,
            transaction_id VARCHAR(100),
            engine_name VARCHAR(100),
            anomaly_flag BOOLEAN,
            anomaly_score NUMERIC,
            reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (transaction_row_id, engine_name)
        );
        """,
        # Table 3: risk_results (References transactions.id)
        """
        CREATE TABLE IF NOT EXISTS risk_results (
            transaction_row_id INT PRIMARY KEY REFERENCES transactions(id) ON DELETE CASCADE,
            transaction_id VARCHAR(100),
            risk_score INT,
            severity VARCHAR(20),
            recommendation TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        # Table 4: auditor_feedback (Logical keys for human label persistence)
        """
        CREATE TABLE IF NOT EXISTS auditor_feedback (
            transaction_id VARCHAR(100) PRIMARY KEY,
            fraud_label BOOLEAN,
            auditor_comments TEXT,
            reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        # Index on transaction_id for rapid lookups since it is no longer the primary key
        "CREATE INDEX IF NOT EXISTS idx_txn_id ON transactions(transaction_id);",
        "CREATE INDEX IF NOT EXISTS idx_customer ON transactions(customer_id);",
        "CREATE INDEX IF NOT EXISTS idx_project ON transactions(project_id);",
        "CREATE INDEX IF NOT EXISTS idx_unit ON transactions(unit_id);",
        "CREATE INDEX IF NOT EXISTS idx_demand_date ON transactions(demand_date);"
    ]

    try:
        with conn.cursor() as cur:
            # Drop the old tables to clean the schema for Option 2
            print("[Database Schema] Resetting legacy table configurations...")
            cur.execute("DROP TABLE IF EXISTS anomaly_results, risk_results, transactions CASCADE;")
            
            for q in queries:
                cur.execute(q)
        conn.commit()
        print("[Database Schema] Successfully initialized core tables and indexes (Option 2).")
        return True
    except Exception as e:
        conn.rollback()
        print(f"[Database Schema Error] Failed to initialize tables/indexes: {e}")
        return False