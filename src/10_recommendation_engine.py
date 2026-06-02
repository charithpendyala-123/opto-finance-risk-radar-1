import sys
import json
import os

# Dynamically add the parent directory (project root) to search path for safe imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mapping of hard compliance rules to exact corporate auditor SOP instructions
RULE_SOP_MAPPING = {
    # Identity
    "RV-01": "Verify original booking form. Patched transaction ID assigned. Reconcile with ERP sequence logs.",
    "RV-02": "Duplicate ID threat. Reconcile unit ledger balances and check for duplicate billing vouchers.",
    "RV-03": "Halt transaction. Missing Customer ID. Reconcile with CRM registration logs before payment clearance.",
    "RV-04": "Halt transaction. Missing Unit ID. Reconcile with property registry.",
    "RV-05": "Halt transaction. Missing Project ID. Reconcile with master project list.",
    "RV-06": "Identity Format Alert. Non-standard transaction ID format. Verify entry source.",
    "RV-07": "Identity Format Alert. Non-standard customer ID format. Perform KYC verification.",
    "RV-08": "Sequential gap in transaction IDs detected. Audit database for missing or deleted invoice records.",
    
    # Financial
    "RV-09": "Flag negative demand. Reconcile if transaction is a credit note; require director sign-off.",
    "RV-10": "Flag negative collection. Check for debit reversals, bounced checks, or allocation errors.",
    "RV-11": "Reconcile refund entry. Reconcile refund vouchers against associated bank cash receipts.",
    "RV-12": "Review negative discount. Reconcile unapplied credit allocations.",
    "RV-13": "Zero Demand Warning. Check for unapproved complimentary bookings or dummy entries.",
    "RV-14": "Over-collection Flag. Reconcile collected amount; check if refund is required.",
    "RV-15": "Refund exceeds demand. Halt payment. Verify physical sales contract pricing terms.",
    "RV-16": "Discount exceeds demand. Halt transaction. Review unapproved manager promotion clearances.",
    "RV-17": "Refund exceeds collection. High risk fraud alert. Audit refund authorization records immediately.",
    "RV-18": "Evasion Warning. Aggregate discounts and refunds exceed demand value. Freeze client ledger.",
    "RV-19": "Ledger mismatch: Outstanding != Demand - Collected - Discount + Refund. Reconcile manual postings.",
    "RV-20": "Credit Balance Flag. Negative outstanding balance. Reconcile over-collections.",
    "RV-21": "Unsecured Refund. Refund issued with zero collections. Reconcile reference slips for phantom receipts.",
    "RV-22": "Halt transaction. All financial fields are blank. Retrieve original paper voucher.",
    
    # Date & Time
    "RV-23": "Re-enter date parameters. System date parsing error. Locate physical timestamp records.",
    "RV-24": "Temporal Anomaly. Payment logged prior to demand date. Verify CRM logging latency.",
    "RV-25": "Date mismatch. payment_delay_days does not match calendar date difference. Audit trigger latency.",
    "RV-26": "Future-dated demand date warning. Verify system date limits and check billing queue leaks.",
    "RV-27": "Future-dated payment warning. Verify post-dated check authorization schedules.",
    "RV-28": "Date anomaly. Delay logged as 0 days but payment and demand dates differ. Check batch processing logs.",
    "RV-29": "Date anomaly. Negative delay logged but calendar dates show payment arrived late. Check chronological sync.",
    "RV-30": "Date anomaly. Same calendar dates but positive delay logged. Reconcile system logging triggers.",
    "RV-31": "Verify holiday/weekend billing authorization. Ensure invoice generation complies with bank operating hours.",
    "RV-32": "Verify holiday/weekend collection voucher. Confirm bank clearings and check deposit timestamps.",
    
    # Cross-Record
    "RV-33": "Missing receipt voucher. Payment date logged but zero cash collected. Audit cash desk reports.",
    "RV-34": "Multiple customers linked to identical transaction. Suspend ledger to prevent double-crediting.",
    "RV-35": "Multiple demand amounts logged on identical transaction ID. Reconcile invoices immediately.",
    "RV-36": "Duplicate full row submission. Purge redundant row from sub-ledger.",
    "RV-37": "Suspicious identical demands logged on same property unit. Check for double billing glitch.",
    "RV-38": "Likely duplicate submission. Identical customer, unit, and date with different transaction IDs.",
    "RV-39": "Behavioral Watchlist. Customer exceeds standard property booking thresholds. Run AML risk check.",
    "RV-40": "System Glitch. Bulk submissions logged on identical demand dates. Reconcile automated invoicing queues."
}

# Mapping of Z-Score outlier columns to specific auditor SOP actions
Z_ANOMALY_MAPPING = {
    "outstanding_amount": "Audit massive outstanding balance mismatch. Reconcile ledger manual entries.",
    "demand_amount": "Verify abnormal invoice demand value. Reconcile pricing with Master Rate Card.",
    "collected_amount": "Verify large cash collection voucher against corresponding bank deposit receipts.",
    "discount_amount": "Audit extreme discount allocation. Verify manager authorization signatures.",
    "refund_amount": "Halt refund payout. Reconcile refund voucher references against active collections.",
    "payment_delay_days": "Inspect temporal delay logs. Verify invoice timestamp synchronization."
}


def run_recommendation_engine(risk_report_path="reports/risk_report.json"):
    # 1. Retrieve the base directory of your project safely
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if not os.path.isabs(risk_report_path):
        risk_report_path = os.path.abspath(os.path.join(base_dir, risk_report_path))

    if not os.path.exists(risk_report_path):
        print(f"Error: Master risk report not found at: {risk_report_path}")
        return None
        # Dynamic import to generate fresh high-level actions and prevent duplicate prepends
    import importlib
    try:
        risk_score_module = importlib.import_module("src.09_risk_score")
        generate_action_recommendation = risk_score_module.generate_action_recommendation
    except Exception:
        generate_action_recommendation = None
    # Load master profiles generated by 09_risk_score.py
    with open(risk_report_path, "r") as f:
        profiles = json.load(f)

    print(f"[Recommendation Engine] Generating expert audit SOP checklists for {len(profiles)} records...")

    enriched_profiles = []
    enriched_count = 0

    for p in profiles:
        violations = p.get("violations", [])
        anom_details = p.get("anomaly_details", {})
        gw_anomalies = anom_details.get("groupwise_anomalies", [])
        
        actions = []

        # 1. Parse Hard Rule Violations
        for v in violations:
            rule_id = v.get("rule_id")
            sop_text = RULE_SOP_MAPPING.get(rule_id, "Conduct standard transaction audit.")
            actions.append(f"[{rule_id}] - {sop_text}")

        # 2. Check Z-Score Outlier details dynamically
        if anom_details.get("zscore_flagged", False):
            z_reason = anom_details.get("zscore_reason", "")
            
            matched = False
            for col, sop in Z_ANOMALY_MAPPING.items():
                if col in z_reason:
                    actions.append(f"[SOP-Z-{col.upper()}] - {sop}")
                    matched = True
            
            if not matched:
                actions.append(f"[SOP-Z-OUTLIER] - Statistical Z-Score Outlier: {z_reason}")

        # 3. Check IQR Outlier details dynamically
        if anom_details.get("iqr_flagged", False):
            iqr_reason = anom_details.get("iqr_reason", "")
            actions.append(f"[SOP-IQR-OUTLIER] - Statistical IQR Bounds Outlier: {iqr_reason}")

        # 4. Check Group-Wise Contextual anomalies
        for gw in gw_anomalies:
            case = gw.get("case", "")
            reason = gw.get("reason", "")
            
            if "Phantom Settlement" in case:
                actions.append("[SOP-GW-PHANTOM] - CFO Action: Freeze customer account. Verify physical cash ledger, check if collections were diverted without credit reduction, and reconcile bank statements.")
            elif "Ghost Collection" in case:
                actions.append("[SOP-GW-GHOST] - Audit Alert: Reconcile collected amount; check if funds are unallocated or misapplied to incorrect invoices.")
            elif "Laundering Cash" in case:
                actions.append("[SOP-GW-AML] - Reconcile source of funds. Reconcile check deposit receipts, verify customer KYC profiles, and file anti-money laundering (AML) regulatory review.")
            elif "Evasion" in case or "Evasion" in reason:
                actions.append("[SOP-GW-EVASION] - Reconcile outstanding balance evasion. Verify rate discounts and check executive manager discount approvals.")
            elif "Billing Escalation" in case:
                actions.append("[SOP-GW-ESCALATION] - Inspect billing escalation patterns. Check Rate Card deviations and demand schedules.")
            elif "Glitch" in case or "Burst" in case:
                actions.append("[SOP-GW-BURST] - Reconcile automated invoicing triggers. Check sequential timestamps for billing loop glitches.")

        # 5. Check ML Isolation Forest flags
        if anom_details.get("isolation_forest_flagged", False):
            if_reason = anom_details.get("isolation_forest_reason", "Abnormal multi-variable parameters.")
            actions.append(f"[SOP-ML-OUTLIER] - ML Outlier Reconciliations: {if_reason}")

        
        # 6. Enrich recommended_action field by prepending a fresh high-level action
        if generate_action_recommendation:
            # Re-generate the pristine high-level CFO recommendation dynamically
            original_action = generate_action_recommendation(
                p.get("risk_severity", "LOW"),
                p.get("risk_score", 0),
                p.get("violations", []),
                gw_anomalies
            )
        else:
            # Fallback cleanup to prevent duplication if running standalone
            original_action = p.get("recommended_action", "")
            if "; [" in original_action:
                original_action = original_action.split("; [")[0]
                if original_action.startswith("["):
                    original_action = ""

        if len(actions) > 0:
            if original_action:
                p["recommended_action"] = f"{original_action}; {'; '.join(actions)}"
            else:
                p["recommended_action"] = "; ".join(actions)
            enriched_count += 1
        else:
            p["recommended_action"] = original_action if original_action else "Approved: Ledger entry within safe financial parameters. Archive transaction record."

        enriched_profiles.append(p)

    # Save the enriched risk report back to risk_report.json
    with open(risk_report_path, "w") as f:
        json.dump(enriched_profiles, f, indent=4)

    print(f"=======================================================")
    print(f"       MITIGATION RECOMMENDATION ENGINE COMPLETE")
    print(f"=======================================================")
    print(f"Enriched Master Risk Profiles: {enriched_count} records")
    print(f"Updated Master Risk Report   : {risk_report_path}")
    print(f"=======================================================\n")

    return enriched_profiles


if __name__ == "__main__":
    run_recommendation_engine()