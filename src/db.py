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
        # Table 1: transactions (Multi-tenant schema with unique constraint)
                # Table 1: transactions (Multi-tenant schema with unique constraint)
        """
        CREATE TABLE IF NOT EXISTS transactions (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(50) NOT NULL,
            upload_batch_id VARCHAR(50) NOT NULL,
            transaction_id VARCHAR(100) NOT NULL,
            customer_id VARCHAR(100) DEFAULT 'N/A',
            project_id VARCHAR(100) DEFAULT 'N/A',
            unit_id VARCHAR(100) DEFAULT 'N/A',
            demand_date DATE,
            payment_date DATE,
            demand_amount NUMERIC,
            collected_amount NUMERIC,
            outstanding_amount NUMERIC,
            discount_amount NUMERIC,
            refund_amount NUMERIC,
            payment_delay_days INT,
            payment_gap_days INT,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT unique_user_transaction UNIQUE (user_id, upload_batch_id, transaction_id, customer_id, unit_id)
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
        # Table 4: auditor_feedback (Row-level human label persistence)
        """
        CREATE TABLE IF NOT EXISTS auditor_feedback (
            transaction_row_id INT PRIMARY KEY REFERENCES transactions(id) ON DELETE CASCADE,
            transaction_id VARCHAR(100),
            fraud_label BOOLEAN,
            auditor_comments TEXT,
            reviewed_at TIMESTAMP
        );
        """,
        # Table 5: system_notifications (Audit events and SLA breach logs)
        """
        CREATE TABLE IF NOT EXISTS system_notifications (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(50) NOT NULL,
            transaction_row_id INT REFERENCES transactions(id) ON DELETE CASCADE,
            transaction_id VARCHAR(100),
            notification_type VARCHAR(50),
            message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            read_status BOOLEAN DEFAULT FALSE
        );
        """,
        # Indexes for rapid lookups and filtering
        "CREATE INDEX IF NOT EXISTS idx_txn_id ON transactions(transaction_id);",
        "CREATE INDEX IF NOT EXISTS idx_customer ON transactions(customer_id);",
        "CREATE INDEX IF NOT EXISTS idx_project ON transactions(project_id);",
        "CREATE INDEX IF NOT EXISTS idx_unit ON transactions(unit_id);",
        "CREATE INDEX IF NOT EXISTS idx_demand_date ON transactions(demand_date);",
        "CREATE INDEX IF NOT EXISTS idx_user_id ON transactions(user_id);",
        "CREATE INDEX IF NOT EXISTS idx_batch_id ON transactions(upload_batch_id);",

        # Trigger Function: Detects actual updates in amounts/dates and logs to system_notifications
        """
        CREATE OR REPLACE FUNCTION log_transaction_update()
        RETURNS TRIGGER AS $$
        BEGIN
            IF (OLD.demand_amount IS DISTINCT FROM NEW.demand_amount) OR
               (OLD.collected_amount IS DISTINCT FROM NEW.collected_amount) OR
               (OLD.outstanding_amount IS DISTINCT FROM NEW.outstanding_amount) OR
               (OLD.payment_date IS DISTINCT FROM NEW.payment_date) THEN
               
               INSERT INTO system_notifications (user_id, transaction_row_id, transaction_id, notification_type, message)
               VALUES (
                   NEW.user_id,
                   NEW.id,
                   NEW.transaction_id,
                   'VALUE_UPDATED',
                   'Transaction values updated during import (Batch: ' || NEW.upload_batch_id || ').'
               );
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """,

            # Bind Trigger to Table
        """
        DROP TRIGGER IF EXISTS trigger_log_transaction_update ON transactions;
        """,
        """
        CREATE TRIGGER trigger_log_transaction_update
        AFTER UPDATE ON transactions
        FOR EACH ROW
        EXECUTE FUNCTION log_transaction_update();
        """,
        # Table 6: transaction_rolling_features (Phase 7 Rolling Window Layer)
                # Table 6: transaction_rolling_features (Phase 7 Rolling Window Layer + Phase 7.5 updates)
        """
        CREATE TABLE IF NOT EXISTS transaction_rolling_features (
            transaction_row_id INT PRIMARY KEY REFERENCES transactions(id) ON DELETE CASCADE,
            transaction_id VARCHAR(100) NOT NULL,
            customer_txn_count_30d INT,
            customer_flags_30d INT,
            customer_avg_delay_30d NUMERIC,
            customer_avg_discount_30d NUMERIC,
            customer_avg_refund_30d NUMERIC,
            customer_avg_outstanding_30d NUMERIC,
            customer_max_outstanding_30d NUMERIC,
            customer_avg_outstanding_ratio_30d NUMERIC,
            project_txn_count_30d INT,
            project_flags_30d INT,
            project_avg_demand_30d NUMERIC,
            project_avg_outstanding_30d NUMERIC,
            project_avg_outstanding_ratio_30d NUMERIC,
            project_avg_refund_30d NUMERIC,
            unit_txn_count_30d INT,
            unit_flags_30d INT,
            unit_owner_changes_30d INT,
            unit_avg_outstanding_30d NUMERIC,
            unit_avg_demand_30d NUMERIC,
            unit_unique_customer_count_30d INT,
            -- Phase 7.5 Long-Term & Lifetime Features
            customer_refund_count_90d INT,
            customer_refund_count_180d INT,
            customer_total_refund_180d NUMERIC,
            customer_total_refund_lifetime NUMERIC,
            customer_discount_count_lifetime INT,
            customer_fraud_count_lifetime INT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    ]

    try:
        with conn.cursor() as cur:
            # Comment out/disable this line to stop dropping your tables on every upload:
            # cur.execute("DROP TABLE IF EXISTS anomaly_results, risk_results, transaction_rolling_features, transactions, auditor_feedback, system_notifications CASCADE;")
            
            for q in queries:
                cur.execute(q)
        conn.commit()
        print("[Database Schema] Successfully initialized core tables, indexes, and triggers.")
        return True
    except Exception as e:
        conn.rollback()
        print(f"[Database Schema Error] Failed to initialize tables/indexes: {e}")
        return False