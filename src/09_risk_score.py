import sys
import json
import os
import numpy as np
import pandas as pd
import importlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

csv_loader = importlib.import_module("src.02_csv_loader")
load_finance_csv = csv_loader.load_finance_csv

anomaly_detector = importlib.import_module("src.08_anomaly_detector")
patch_transaction_ids = anomaly_detector.patch_transaction_ids
sanitize_float= anomaly_detector.sanitize_float
sanitize_int=   anomaly_detector.sanitize_int
sanitize_str=   anomaly_detector.sanitize_str

Z=15
iqr=20
isfo=25
gw=40
critical_rule_ids={"RV-01", "RV-02", "RV-34", "RV-35", "RV-36", "RV-38"}
critical_gwcases={"Phantom Settlement", "Ghost Collection", "Laundering Cash"}
high_gwcases={"Kickback Discount", "Default Evasion", "Billing Escalation", "Liability Write-down"}
medium_gwcases={"Multi-unit", "Date Compression", "Distribution Shift", "Compressed Activity Burst Flag", "Rapid Unit Reassignment Flag"}

def generate_action_recommendation(severity, score, rules_violated, gw_anomalies):
    """
    Generates action-oriented, CFO-level audit instructions based on threat severity and anomaly details.
    """
    if severity == "CRITICAL":
        has_id_issue = any(v.get("rule_id") in {"RV-01", "RV-02"} for v in rules_violated)
        has_phantom_settlement = any(
            "Phantom Settlement" in str(a.get("case", "")) or "Phantom Settlement" in str(a.get("reason", ""))
            for a in gw_anomalies
        )
        
        if has_id_issue:
            return (
                "IMMEDIATE LEDGER SUSPENSION! Critical identity anomalies (missing or duplicate transaction IDs) "
                "detected. Freeze the accounting sub-ledger, halt payment processing, and perform a manual database key reconciliation."
            )
        elif has_phantom_settlement:
            return (
                "HIGH PRIORITY FORENSIC AUDIT! Suspected Phantom Settlement or Cash Laundering scheme detected "
                "(outstanding balance drops to zero without matching bank cash collections). Restrict user access, "
                "retrieve corresponding physical bank SWIFT records, and audit unit ledger balances."
            )
        else:
            return (
                f"CRITICAL COMPLIANCE THREAT (Score {score})! Multiple statistical and machine learning engines "
                "have flagged this transaction. Assign a senior forensic auditor immediately to review documentation."
            )
    elif severity == "HIGH":
        has_discount_issue = any(
            any(case in str(a.get("case", "")) or case in str(a.get("reason", "")) for case in high_gwcases)
            for a in gw_anomalies
        )
        if has_discount_issue:
            return (
                "HIGH RISK COMPLIANCE ALERT! Suspected discount evasion or billing escalation. Review the project "
                "sales schedule, check authorized manager discount approvals, and reconcile demand invoices."
            )
        else:
            return (
                "HIGH RISK WATCHLIST. Multiple rule violations and outlier flags detected. Conduct a standard "
                "forensic ledger audit and require management sign-off before archiving."
            )
    elif severity == "MEDIUM":
        return (
            "Medium Policy Variance. Minor transaction delays, distribution shifts, or isolated statistical anomalies. "
            "Flag for validation check during routine quarterly financial closing procedures."
        )
    else:
        return (
            "Safe Ledger Entry. No material rule violations or anomalous outlier signatures detected. "
            "Approved for standard financial archiving."
        )
def run_risk_scoring(
    csv_path="data/sample_finance_data.csv",
    anomaly_report_path="reports/anomaly_report.json",
    rule_report_path="reports/rule_validation_report.json"
):
        # 1. Retrieve the base directory of your project
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 2. Resolve relative CSV path
    if not os.path.isabs(csv_path):
        csv_path = os.path.abspath(os.path.join(base_dir, csv_path))
        
    # 3. Resolve relative Anomaly Report path
    if not os.path.isabs(anomaly_report_path):
        anomaly_report_path = os.path.abspath(os.path.join(base_dir, anomaly_report_path))
        
    # 4. Resolve relative Rule Report path
    if not os.path.isabs(rule_report_path):
        rule_report_path = os.path.abspath(os.path.join(base_dir, rule_report_path))
    
        # 1. Ingest raw CSV data using the imported CSV loader
    df_raw = load_finance_csv(csv_path)
    if df_raw is None:
        print("Error: Risk engine cannot run. Ingestion dataframe is empty.")
        return None

    # 2. Patch transaction IDs upstream to ensure identical virtual primary keys
    df = patch_transaction_ids(df_raw)

    # 3. Load Rule Violations Report and map by physical Excel row number
    violations_map = {}
    if os.path.exists(rule_report_path):
        try:
            with open(rule_report_path, "r") as f:
                violations = json.load(f)
            for v in violations:
                s_row = int(v.get("source_row", 0))
                if s_row > 0:
                    if s_row not in violations_map:
                        violations_map[s_row] = []
                    violations_map[s_row].append(v)
            print(f"[Risk Engine] Loaded rule violations from: {rule_report_path}")
        except Exception as e:
            print(f"Warning: Failed to load rule violations: {e}")
    else:
        print("Warning: rule_validation_report.json not found. Proceeding with 0 rule weights.")

    # 4. Load Consolidated Anomaly Report and map by patched transaction_id
    anomaly_map = {}
    if os.path.exists(anomaly_report_path):
        try:
            with open(anomaly_report_path, "r") as f:
                anomalies = json.load(f)
            for a in anomalies:
                txn_id = a.get("transaction_id")
                if txn_id:
                    anomaly_map[txn_id] = a
            print(f"[Risk Engine] Loaded parallel anomaly details from: {anomaly_report_path}")
        except Exception as e:
            print(f"Warning: Failed to load anomaly report: {e}")
    else:
        print("Warning: anomaly_report.json not found. Proceeding with 0 outlier weights.")
        # 5. Core Scoring & Consolidation Loop (Runs over 100% of transactions)
    risk_records = []
    
    total_low = 0
    total_medium = 0
    total_high = 0
    total_critical = 0

    print("\n[Risk Scoring] Scoring ledger entries and computing threat indexes...")

    for idx, row in df.iterrows():
        txn_id = row['transaction_id']
        source_row = idx + 2  # Physical CSV/Excel row
        
        # A) Fetch Rule Violations
        rules_violated = violations_map.get(source_row, [])
        
        # B) Fetch Outlier Details
        anom_details = anomaly_map.get(txn_id, {})
        
        # C) Compute Score Points
        score = 0
        
        # I) Rule Violations Scoring
        for v in rules_violated:
            rule_id = v.get("rule_id")
            if rule_id in critical_rule_ids:
                score += 35  # Critical identity/evasion duplication rules
            else:
                score += 25  # Standard policy deviations
                
        # II) Statistical Sub-Engine Scoring
        z_flag = False
        z_reason = None
        if "zscore_details" in anom_details:
            z_flag = bool(anom_details["zscore_details"].get("flagged", False))
            z_reason = anom_details["zscore_details"].get("reason")
            if z_flag:
                score += Z
                
        iqr_flag = False
        iqr_reason = None
        if "iqr_details" in anom_details:
            iqr_flag = bool(anom_details["iqr_details"].get("flagged", False))
            iqr_reason = anom_details["iqr_details"].get("reason")
            if iqr_flag:
                score += iqr
                
        gw_flag = False
        gw_anomalies = []
        if "groupwise_details" in anom_details:
            gw_flag = bool(anom_details["groupwise_details"].get("flagged", False))
            gw_anomalies = anom_details["groupwise_details"].get("anomalies", [])
            if gw_flag:
                score += gw
                
                # Dynamic Group-wise Severity Boost Calculations (avoiding double-counting)
                boosts = [0]  # Initialize with 0 to safely handle empty cases
                for g_anom in gw_anomalies:
                    case = g_anom.get("case", "")
                    if any(c in case for c in critical_gwcases): 
                        boosts.append(35)
                    elif any(c in case for c in high_gwcases):
                        boosts.append(20)
                    elif any(c in case for c in medium_gwcases):
                        boosts.append(10)
                
                # Apply only the maximum single boost triggered
                score += max(boosts)
                
        # III) Unsupervised ML Scoring
        if_flag = False
        if_score = 0.0
        if_reason = None
        if "isolation_forest_details" in anom_details:
            if_flag = bool(anom_details["isolation_forest_details"].get("flagged", False))
            if_score = float(anom_details["isolation_forest_details"].get("score", 0.0))
            if_reason = anom_details["isolation_forest_details"].get("reason")
            if if_flag:
                score += isfo

        # IV) Cap Unified Score at 100
        score = int(min(100, max(0, score)))

                # D) Categorize Risk Severity & Recommended CFO Action
        if score <= 24:
            severity = "LOW"
            total_low += 1
        elif score <= 49:
            severity = "MEDIUM"
            total_medium += 1
        elif score <= 74:
            severity = "HIGH"
            total_high += 1
        else:
            severity = "CRITICAL"
            total_critical += 1

        # DYNAMIC ACTION ENRICHMENT CALL:
        recommended_action = generate_action_recommendation(severity, score, rules_violated, gw_anomalies)

        # E) Build Consolidated Risk Profile (Sanitizing to prevent float NaN leaks in JSON)
        profile = {
            "transaction_id": txn_id,
            "source_row": source_row,
            "customer_id": sanitize_str(row['customer_id']),
            "project_id": sanitize_str(row['project_id']),
            "unit_id": sanitize_str(row['unit_id']),
            "demand_amount": sanitize_float(row['demand_amount']),
            "collected_amount": sanitize_float(row['collected_amount']),
            "outstanding_amount": sanitize_float(row['outstanding_amount']),
            "discount_amount": sanitize_float(row['discount_amount']),
            "refund_amount": sanitize_float(row['refund_amount']),
            "payment_delay_days": sanitize_int(row['payment_delay_days']),
            "demand_date": sanitize_str(row['demand_date']),
            "payment_date": sanitize_str(row['payment_date']),
            
            "risk_score": score,
            "risk_severity": severity,
            "recommended_action": recommended_action,
            
            "violations_count": len(rules_violated),
            # Defensive Dictionary Retrieval
            "violations": [
                {
                    "rule_id": v.get("rule_id"),
                    "description": v.get("rule_description", "No description available")
                }
                for v in rules_violated
            ],
            "anomaly_details": {
                "zscore_flagged": z_flag,
                "zscore_reason": z_reason,
                "iqr_flagged": iqr_flag,
                "iqr_reason": iqr_reason,
                "groupwise_flagged": gw_flag,
                "groupwise_anomalies_count": len(gw_anomalies),
                "groupwise_anomalies": gw_anomalies,
                "isolation_forest_flagged": if_flag,
                "isolation_forest_score": round(if_score, 4),
                "isolation_forest_reason": if_reason
            }
        }
        risk_records.append(profile)

    # 6. Save consolidated risk ledger report to reports/risk_report.json
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    output_path = os.path.abspath(os.path.join(reports_dir, "risk_report.json"))
    
    with open(output_path, "w") as f:
        json.dump(risk_records, f, indent=4)

    # 7. Print Console Summary Metrics
    print(f"\n=======================================================")
    print(f"         UNIFIED RISK SCORING ENGINE COMPLETE")
    print(f"=======================================================")
    print(f"Total Transactions Scored   : {len(df)}")
    print(f"Consolidated Ledger Saved To: {output_path}")
    print(f"\nThreat Risk Breakdown:")
    print(f"  - LOW (0-24 score)        : {total_low} records")
    print(f"  - MEDIUM (25-49 score)    : {total_medium} records")
    print(f"  - HIGH (50-74 score)       : {total_high} records")
    print(f"  - CRITICAL (75-100 score)  : {total_critical} records")
    print(f"=======================================================\n")

    return risk_records


if __name__ == "__main__":
    run_risk_scoring()
