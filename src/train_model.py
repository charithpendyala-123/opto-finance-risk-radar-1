# ==============================================================================
# OPTOxCRM FINANCE RISK RADAR - SUPERVISED ML TRAINING PIPELINE (PHASE 1-10)
# ==============================================================================
import os
import sys
import json
import datetime
import pandas as pd
import numpy as np
import joblib

# ML imports
try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
except ImportError:
    print("[Error] Missing dependencies. Please run: pip install scikit-learn xgboost catboost joblib")
    sys.exit(1)

# Dynamically add parent directory for absolute/relative import compatibility
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_model_training(user_id="system_default", batch_id="All Batches"):
    import src.db as db
    conn = db.get_connection()
    if conn is None:
        print("[Training Pipeline] Error: No database connection.")
        return False
        
    print("=========================================================================")
    # Phase 1: Training Readiness Validation
    print(f"[Phase 1] Running training readiness validation checks for batch: {batch_id}...")
    
    # 1. Fetch total transactions count
    try:
        with conn.cursor() as cur:
            if batch_id and batch_id != "All Batches":
                cur.execute("SELECT COUNT(*) FROM transactions WHERE user_id = %s AND upload_batch_id = %s;", (user_id, batch_id))
            else:
                cur.execute("SELECT COUNT(*) FROM transactions WHERE user_id = %s;", (user_id,))
            total_tx = cur.fetchone()[0]
            
            # 2. Fetch total reviewed transactions (fraud_label is not null)
            if batch_id and batch_id != "All Batches":
                cur.execute("""
                    SELECT COUNT(*), 
                           SUM(CASE WHEN fraud_label = TRUE THEN 1 ELSE 0 END),
                           SUM(CASE WHEN fraud_label = FALSE THEN 1 ELSE 0 END)
                    FROM auditor_feedback af
                    JOIN transactions t ON af.transaction_row_id = t.id
                    WHERE t.user_id = %s AND t.upload_batch_id = %s AND af.fraud_label IS NOT NULL;
                """, (user_id, batch_id))
            else:
                cur.execute("""
                    SELECT COUNT(*), 
                           SUM(CASE WHEN fraud_label = TRUE THEN 1 ELSE 0 END),
                           SUM(CASE WHEN fraud_label = FALSE THEN 1 ELSE 0 END)
                    FROM auditor_feedback af
                    JOIN transactions t ON af.transaction_row_id = t.id
                    WHERE t.user_id = %s AND af.fraud_label IS NOT NULL;
                """, (user_id,))
            reviewed_row = cur.fetchone()
            total_reviewed = reviewed_row[0] or 0
            fraud_count = reviewed_row[1] or 0
            clean_count = reviewed_row[2] or 0
    except Exception as e:
        print(f"[Training Error] Failed to read validation data: {e}")
        conn.close()
        return False

    if total_tx == 0:
        print("[Training Aborted] Skip: Dataset/Batch contains 0 total transactions.")
        conn.close()
        return False

    coverage = (total_reviewed / total_tx) * 100
    
    print(f"  - Total Transactions  : {total_tx}")
    print(f"  - Reviewed By Auditor : {total_reviewed} ({coverage:.2f}% coverage)")
    print(f"  - Confirmed Fraud (1) : {fraud_count}")
    print(f"  - Confirmed Clean (0) : {clean_count}")
    
    # Determine size-based validation thresholds
    ready = False
    size_category = ""
    req_cov = 0.0
    
    if total_tx < 100:
        size_category = "Small (<100 rows)"
        req_cov = 50.0
        if coverage >= 50.0 and fraud_count > 0 and clean_count > 0:
            ready = True
    elif 100 <= total_tx <= 1000:
        size_category = "Medium (100-1000 rows)"
        req_cov = 20.0
        if coverage >= 20.0 and fraud_count > 0 and clean_count > 0:
            ready = True
    else:
        size_category = "Large (>1000 rows)"
        req_cov = 10.0
        if coverage >= 10.0 and fraud_count > 0 and clean_count > 0:
            ready = True
            
    print(f"  - Dataset Size Category: {size_category}")
    print(f"  - Required Coverage   : >= {req_cov}%")
    print(f"  - Label requirements  : At least 1 Fraud (1) and 1 Clean (0) record")
    
    if not ready:
        print("\n[SKIP TRAINING] Dataset/Batch fails training readiness validation criteria.")
        print("Heuristics (Rule Engine, Z-Score, IQR, Groupwise, Isolation Forest) will continue to run.")
        print("NO mock labels or bootstrapping performed. Only real auditor reviews accepted.")
        conn.close()
        return False
        
    print("\n[Phase 1 Result] PASS: Dataset/Batch meets readiness criteria. Starting training...")
    
    # Phase 2: Dataset Construction
    print("\n[Phase 2] Constructing dataset from PostgreSQL database...")
    
    if batch_id and batch_id != "All Batches":
        query_main = """
            SELECT 
                t.id AS transaction_row_id,
                -- Transaction Features
                t.demand_amount, t.collected_amount, t.outstanding_amount, t.discount_amount, t.refund_amount,
                t.payment_delay_days, t.payment_gap_days,
                
                -- Rolling Features
                trf.customer_txn_count_30d, trf.customer_flags_30d, trf.customer_avg_delay_30d,
                trf.customer_avg_discount_30d, trf.customer_avg_refund_30d, trf.customer_avg_outstanding_30d,
                trf.customer_max_outstanding_30d, trf.customer_avg_outstanding_ratio_30d,
                trf.customer_refund_count_90d, trf.customer_refund_count_180d, trf.customer_total_refund_180d,
                trf.customer_total_refund_lifetime, trf.customer_discount_count_lifetime, trf.customer_fraud_count_lifetime,
                trf.project_txn_count_30d, trf.project_flags_30d, trf.project_avg_demand_30d,
                trf.project_avg_outstanding_30d, trf.project_avg_outstanding_ratio_30d, trf.project_avg_refund_30d,
                trf.unit_txn_count_30d, trf.unit_flags_30d, trf.unit_owner_changes_30d,
                trf.unit_avg_outstanding_30d, trf.unit_avg_demand_30d, trf.unit_unique_customer_count_30d,
                
                -- Target Variable
                af.fraud_label
            FROM transactions t
            JOIN transaction_rolling_features trf ON t.id = trf.transaction_row_id
            JOIN auditor_feedback af ON t.id = af.transaction_row_id
            WHERE t.user_id = %s AND t.upload_batch_id = %s AND af.fraud_label IS NOT NULL;
        """
        params_main = [user_id, batch_id]
        
        query_anom = """
            SELECT ar.transaction_row_id, ar.engine_name, ar.anomaly_flag, ar.anomaly_score 
            FROM anomaly_results ar
            JOIN transactions t ON ar.transaction_row_id = t.id
            WHERE t.user_id = %s AND t.upload_batch_id = %s;
        """
        params_anom = [user_id, batch_id]
    else:
        query_main = """
            SELECT 
                t.id AS transaction_row_id,
                -- Transaction Features
                t.demand_amount, t.collected_amount, t.outstanding_amount, t.discount_amount, t.refund_amount,
                t.payment_delay_days, t.payment_gap_days,
                
                -- Rolling Features
                trf.customer_txn_count_30d, trf.customer_flags_30d, trf.customer_avg_delay_30d,
                trf.customer_avg_discount_30d, trf.customer_avg_refund_30d, trf.customer_avg_outstanding_30d,
                trf.customer_max_outstanding_30d, trf.customer_avg_outstanding_ratio_30d,
                trf.customer_refund_count_90d, trf.customer_refund_count_180d, trf.customer_total_refund_180d,
                trf.customer_total_refund_lifetime, trf.customer_discount_count_lifetime, trf.customer_fraud_count_lifetime,
                trf.project_txn_count_30d, trf.project_flags_30d, trf.project_avg_demand_30d,
                trf.project_avg_outstanding_30d, trf.project_avg_outstanding_ratio_30d, trf.project_avg_refund_30d,
                trf.unit_txn_count_30d, trf.unit_flags_30d, trf.unit_owner_changes_30d,
                trf.unit_avg_outstanding_30d, trf.unit_avg_demand_30d, trf.unit_unique_customer_count_30d,
                
                -- Target Variable
                af.fraud_label
            FROM transactions t
            JOIN transaction_rolling_features trf ON t.id = trf.transaction_row_id
            JOIN auditor_feedback af ON t.id = af.transaction_row_id
            WHERE t.user_id = %s AND af.fraud_label IS NOT NULL;
        """
        params_main = [user_id]
        
        query_anom = """
            SELECT ar.transaction_row_id, ar.engine_name, ar.anomaly_flag, ar.anomaly_score 
            FROM anomaly_results ar
            JOIN transactions t ON ar.transaction_row_id = t.id
            WHERE t.user_id = %s;
        """
        params_anom = [user_id]
        
    try:
        df_main = pd.read_sql(query_main, conn, params=params_main)
        
        # Load and pivot anomaly results in python
        anom_df = pd.read_sql(query_anom, conn, params=params_anom)
    except Exception as e:
        print(f"[Training Error] Failed to read dataset records from PostgreSQL: {e}")
        conn.close()
        return False
        
    conn.close()
    
    # Process and aggregate anomaly features
    print("  - Aggregating anomaly engines output features...")
    anom_features = []
    for txn_row_id, group in anom_df.groupby('transaction_row_id'):
        rule_violations = group[group['engine_name'] == 'RuleEngine']['anomaly_flag'].sum()
        zscore = group[group['engine_name'] == 'ZScore']['anomaly_flag'].sum()
        iqr = group[group['engine_name'] == 'IQR']['anomaly_flag'].sum()
        groupwise = group[group['engine_name'] == 'Groupwise']['anomaly_flag'].sum()
        
        if_row = group[group['engine_name'] == 'IsolationForest']
        if_score = float(if_row['anomaly_score'].iloc[0]) if not if_row.empty and pd.notna(if_row['anomaly_score'].iloc[0]) else 0.0
        
        tot_anom = int(group['anomaly_flag'].sum())
        
        anom_features.append({
            'transaction_row_id': txn_row_id,
            'rule_violation_count': int(rule_violations),
            'zscore_count': int(zscore),
            'iqr_count': int(iqr),
            'groupwise_count': int(groupwise),
            'isolation_forest_score': if_score,
            'total_anomaly_count': tot_anom
        })
        
    anom_feats_df = pd.DataFrame(anom_features)
    
    if anom_feats_df.empty:
        # Create empty placeholder if no anomalies exist
        anom_feats_df = pd.DataFrame(columns=[
            'transaction_row_id', 'rule_violation_count', 'zscore_count', 
            'iqr_count', 'groupwise_count', 'isolation_forest_score', 'total_anomaly_count'
        ])
        
    # Merge datasets
    df = df_main.merge(anom_feats_df, on='transaction_row_id', how='left')
    
    # Fill missing values for anomaly features
    anom_cols = ['rule_violation_count', 'zscore_count', 'iqr_count', 'groupwise_count', 'isolation_forest_score', 'total_anomaly_count']
    df[anom_cols] = df[anom_cols].fillna(0.0)
    
    # Standardize types and fillna for numeric columns
    numeric_cols = [
        'demand_amount', 'collected_amount', 'outstanding_amount', 'discount_amount', 'refund_amount',
        'payment_delay_days', 'payment_gap_days',
        'customer_txn_count_30d', 'customer_flags_30d', 'customer_avg_delay_30d',
        'customer_avg_discount_30d', 'customer_avg_refund_30d', 'customer_avg_outstanding_30d',
        'customer_max_outstanding_30d', 'customer_avg_outstanding_ratio_30d',
        'customer_refund_count_90d', 'customer_refund_count_180d', 'customer_total_refund_180d',
        'customer_total_refund_lifetime', 'customer_discount_count_lifetime', 'customer_fraud_count_lifetime',
        'project_txn_count_30d', 'project_flags_30d', 'project_avg_demand_30d',
        'project_avg_outstanding_30d', 'project_avg_outstanding_ratio_30d', 'project_avg_refund_30d',
        'unit_txn_count_30d', 'unit_flags_30d', 'unit_owner_changes_30d',
        'unit_avg_outstanding_30d', 'unit_avg_demand_30d', 'unit_unique_customer_count_30d'
    ]
    df[numeric_cols] = df[numeric_cols].fillna(0.0)
    
    # Phase 3: Feature Selection
    print("\n[Phase 3] Selecting features (Identity columns excluded)...")
    feature_columns = numeric_cols + anom_cols
    print(f"  - Total feature count: {len(feature_columns)}")
    
    # Phase 4: Target Variable
    print("\n[Phase 4] Formatting target variable ('fraud_label')...")
    X = df[feature_columns]
    y = df['fraud_label'].astype(int)
    
    # Phase 5: Train/Test Split
    class_counts = y.value_counts()
    low_classes = class_counts[class_counts < 2].index.tolist()
    
    if not low_classes:
        print("\n[Phase 5] Splitting dataset (80% Train, 20% Test) with y-stratification...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        print(f"\n[Phase 5] Warning: Classes {low_classes} have fewer than 2 members. Splitting without stratification...")
        # Force low-member class indices into the training set to prevent training failure
        force_train_idx = []
        for cls in low_classes:
            force_train_idx.extend(y[y == cls].index.tolist())
            
        remaining_idx = y.index.difference(force_train_idx)
        train_idx, test_idx = train_test_split(
            remaining_idx, test_size=0.2, random_state=42
        )
        
        final_train_idx = list(train_idx) + force_train_idx
        final_test_idx = list(test_idx)
        
        X_train, X_test = X.loc[final_train_idx], X.loc[final_test_idx]
        y_train, y_test = y.loc[final_train_idx], y.loc[final_test_idx]
    
    # Phase 6: Train XGBoost
    print("\n[Phase 6] Training XGBoost Classifier...")
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    
    # Phase 7: Train CatBoost
    print("\n[Phase 7] Training CatBoost Classifier...")
    cat_model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        verbose=0,
        random_state=42
    )
    cat_model.fit(X_train, y_train)
    
    # Phase 8: Evaluate Models
    print("\n[Phase 8] Running model prediction and metric evaluation...")
    # Predictions
    xgb_pred = xgb_model.predict(X_test)
    cat_pred = cat_model.predict(X_test)
    
    # Probabilities for ROC-AUC
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    cat_proba = cat_model.predict_proba(X_test)[:, 1]
    
    # Metrics
    f1_xgb = f1_score(y_test, xgb_pred)
    roc_xgb = roc_auc_score(y_test, xgb_proba)
    rec_xgb = recall_score(y_test, xgb_pred)
    prec_xgb = precision_score(y_test, xgb_pred)
    
    f1_cat = f1_score(y_test, cat_pred)
    roc_cat = roc_auc_score(y_test, cat_proba)
    rec_cat = recall_score(y_test, cat_pred)
    prec_cat = precision_score(y_test, cat_pred)
    
    # Phase 9: Winner Selection Logic
    print("\n[Phase 9] Selecting winner model (Priority: F1 -> ROC-AUC -> Recall -> Precision)...")
    winner = "CatBoost"
    best_model = cat_model
    best_metrics = {
        "f1_score": f1_cat,
        "roc_auc": roc_cat,
        "recall": rec_cat,
        "precision": prec_cat
    }
    
    # Check if XGBoost beats CatBoost on F1
    if f1_xgb > f1_cat:
        winner = "XGBoost"
        best_model = xgb_model
        best_metrics = {"f1_score": f1_xgb, "roc_auc": roc_xgb, "recall": rec_xgb, "precision": prec_xgb}
    elif f1_xgb == f1_cat:
        # F1 Tie, check ROC-AUC
        if roc_xgb > roc_cat:
            winner = "XGBoost"
            best_model = xgb_model
            best_metrics = {"f1_score": f1_xgb, "roc_auc": roc_xgb, "recall": rec_xgb, "precision": prec_xgb}
        elif roc_xgb == roc_cat:
            # ROC-AUC Tie, check Recall
            if rec_xgb > rec_cat:
                winner = "XGBoost"
                best_model = xgb_model
                best_metrics = {"f1_score": f1_xgb, "roc_auc": roc_xgb, "recall": rec_xgb, "precision": prec_xgb}
            elif rec_xgb == rec_cat:
                # Recall Tie, check Precision
                if prec_xgb > prec_cat:
                    winner = "XGBoost"
                    best_model = xgb_model
                    best_metrics = {"f1_score": f1_xgb, "roc_auc": roc_xgb, "recall": rec_xgb, "precision": prec_xgb}

    # Print Comparison Table
    print("\n=========================================================================")
    print("                   SUPERVISED MODEL PERFORMANCE COMPARISON               ")
    print("=========================================================================")
    print(f"  Metric       XGBoost     CatBoost")
    print(f"  F1 Score     {f1_xgb:.4f}      {f1_cat:.4f}      (Primary Target)")
    print(f"  ROC-AUC      {roc_xgb:.4f}      {roc_cat:.4f}")
    print(f"  Recall       {rec_xgb:.4f}      {rec_cat:.4f}")
    print(f"  Precision    {prec_xgb:.4f}      {prec_cat:.4f}")
    print("=========================================================================")
    print(f"  WINNER SELECTION: {winner}")
    print("=========================================================================")
    
    # Phase 10: Save Winning Model
    print("\n[Phase 10] Saving winning model and training metadata...")
    os.makedirs("models", exist_ok=True)
    
    model_path = os.path.join("models", "best_fraud_model.pkl")
    metadata_path = os.path.join("models", "model_metadata.json")
    
    # Dump model
    joblib.dump(best_model, model_path)
    
    # Save metadata JSON
    metadata = {
        "model_type": winner,
        "f1_score": float(best_metrics["f1_score"]),
        "roc_auc": float(best_metrics["roc_auc"]),
        "recall": float(best_metrics["recall"]),
        "precision": float(best_metrics["precision"]),
        "trained_on": datetime.date.today().isoformat()
    }
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"  - Saved model to   : {model_path}")
    print(f"  - Saved metadata to: {metadata_path}")
    print("Model training pipeline execution complete.")
    return True

if __name__ == "__main__":
    run_model_training()