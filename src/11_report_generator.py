import os
import json
import csv
import numpy as np

def format_indian_currency(val):
    """
    Formats large currency amounts into Indian Rupees (Crores/Lakhs) for executive dashboards.
    Safe against non-numeric and malformed string/None types.
    """
    try:
        val = float(val)
    except (ValueError, TypeError):
        return "₹0.00"
        
    if val == 0 or np.isnan(val):
        return "₹0.00"
    
    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    
    if abs_val >= 10000000:  # 1 Crore = 10 Million
        cr_val = abs_val / 10000000
        return f"{sign}₹{cr_val:.2f} Cr"
    elif abs_val >= 100000:  # 1 Lakh = 100,000
        lakh_val = abs_val / 100000
        return f"{sign}₹{lakh_val:.2f} L"
    else:
        return f"{sign}₹{abs_val:,.2f}"


def load_risk_ledger(json_path):
    """Loads 100% scored transaction profiles from risk_report.json"""
    if not os.path.exists(json_path):
        print(f"Error: Scored ledger file not found at: {json_path}")
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading ledger JSON: {e}")
        return None


def generate_executive_summary(records):
    """Layer 1: Dataset Overview & Risk Classification Breakdown"""
    # Defensive cleanup of stray empty profiles
    clean_records = [r for r in records if r]
    total_scored = len(clean_records)
    breakdown = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    
    for r in clean_records:
        severity = r.get("risk_severity", "LOW")
        breakdown[severity] = breakdown.get(severity, 0) + 1
        
    total_flagged = breakdown["CRITICAL"] + breakdown["HIGH"] + breakdown["MEDIUM"]
    exposure_pct = (total_flagged / total_scored * 100) if total_scored > 0 else 0.0
    
    return {
        "total_audited": total_scored,
        "total_flagged": total_flagged,
        "exposure_pct": round(exposure_pct, 2),
        "breakdown": breakdown
    }


def generate_engine_statistics(records):
    """Layer 2: Engine Performance Summary (counts flags triggered per sub-engine)"""
    stats = {
        "rule_violations": 0,
        "zscore_flags": 0,
        "iqr_flags": 0,
        "groupwise_flags": 0,
        "isolation_forest_flags": 0
    }
    
    for r in records:
        if not r:
            continue
        if r.get("violations_count", 0) > 0:
            stats["rule_violations"] += 1
            
        anom = r.get("anomaly_details")
        if not isinstance(anom, dict):
            anom = {}
            
        if anom.get("zscore_flagged", False):
            stats["zscore_flags"] += 1
        if anom.get("iqr_flagged", False):
            stats["iqr_flags"] += 1
        if anom.get("groupwise_flagged", False):
            stats["groupwise_flags"] += 1
        if anom.get("isolation_forest_flagged", False):
            stats["isolation_forest_flags"] += 1
            
    return stats


def generate_risk_breakdown(records):
    """Layer 3 & 4: Priority Queues & Severity Splits"""
    clean_records = [r for r in records if r]
    
    # Sort descending by risk score, then transaction id
    sorted_records = sorted(
        clean_records,
        key=lambda x: (x.get("risk_score", 0), abs(x.get("collected_amount", 0) or 0)),
        reverse=True
    )
    
    top_10 = []
    for r in sorted_records[:10]:
        top_10.append({
            "transaction_id": r.get("transaction_id"),
            "risk_score": r.get("risk_score"),
            "severity": r.get("risk_severity"),
            "recommended_action": r.get("recommended_action")
        })
        
    return {
        "top_10_queue": top_10,
        "sorted_ledger": sorted_records
    }


def generate_financial_exposure(records):
    """Layer 5: Summarizes absolute and formatted exposure in Rupees (Cr/L)"""
    demand_exposure = 0.0
    outstanding_exposure = 0.0
    refund_exposure = 0.0
    
    for r in records:
        if not r:
            continue
        score = r.get("risk_score", 0)
        
        # 1. Demand Value at Risk (Sum of demand_amount for Critical and High, score >= 50)
        if score >= 50:
            demand_exposure += float(r.get("demand_amount") or 0.0)
            
        # 2. Outstanding & Refund Exposure (Sum for all flagged anomalies, score >= 25)
        if score >= 25:
            outstanding_exposure += float(r.get("outstanding_amount") or 0.0)
            refund_exposure += float(r.get("refund_amount") or 0.0)
            
    return {
        "raw": {
            "demand_exposure": demand_exposure,
            "outstanding_exposure": outstanding_exposure,
            "refund_exposure": refund_exposure
        },
        "formatted": {
            "demand_exposure": format_indian_currency(demand_exposure),
            "outstanding_exposure": format_indian_currency(outstanding_exposure),
            "refund_exposure": format_indian_currency(refund_exposure)
        }
    }


def generate_pattern_summary(records):
    """Layer 6: Summarizes fraud pattern tallies from Groupwise Outliers"""
    patterns = {}
    
    for r in records:
        if not r:
            continue
        score = r.get("risk_score", 0)
        if score >= 25:  # Scan only flagged transactions
            anom = r.get("anomaly_details")
            if not isinstance(anom, dict):
                anom = {}
            if anom.get("groupwise_flagged", False):
                groupwise_list = anom.get("groupwise_anomalies") or []
                for g_anom in groupwise_list:
                    case = g_anom.get("case", "Groupwise Outlier")
                    # Clean the names for visual reports
                    case_clean = case.replace(" Flag", "").replace(" Warning", "").strip()
                    patterns[case_clean] = patterns.get(case_clean, 0) + 1
                    
    return dict(sorted(patterns.items(), key=lambda x: x[1], reverse=True))


def generate_recommendations_summary(records):
    """Layer 7: Groups dynamic recommendations into CFO action types"""
    actions = {
        "Immediate Ledger Freeze Required": 0,
        "High Priority Forensic Audit": 0,
        "Manual Reconciliation Required": 0,
        "Senior Forensic Compliance Review": 0,
        "AML Review Required": 0,
        "Routine Policy Validation Check": 0
    }
    
    for r in records:
        if not r:
            continue
        score = r.get("risk_score", 0)
        if score >= 25:  # Group recommendations for flagged transactions
            action_text = r.get("recommended_action") or ""
            
            if "IMMEDIATE LEDGER SUSPENSION" in action_text:
                actions["Immediate Ledger Freeze Required"] += 1
            elif "HIGH PRIORITY FORENSIC AUDIT" in action_text:
                actions["High Priority Forensic Audit"] += 1
            elif "CRITICAL COMPLIANCE THREAT" in action_text:
                actions["Senior Forensic Compliance Review"] += 1
            elif "HIGH RISK COMPLIANCE ALERT" in action_text:
                actions["Manual Reconciliation Required"] += 1
            elif "HIGH RISK WATCHLIST" in action_text:
                actions["AML Review Required"] += 1
            else:
                actions["Routine Policy Validation Check"] += 1
                
    # Filter out empty categories for cleaner summary
    return {k: v for k, v in actions.items() if v > 0}


def export_csv(records, csv_path, flagged_only=False):
    """Layer 8: Exports ledger transactions to clean spreadsheet format"""
    headers = [
        "Transaction ID", "Source Row", "Customer ID", "Project ID", "Unit ID",
        "Demand Amount", "Collected Amount", "Outstanding Amount", "Discount Amount", "Refund Amount",
        "Payment Delay Days", "Demand Date", "Payment Date",
        "Risk Score", "Risk Severity", "Violations Count",
        "Z-Score Flagged", "IQR Flagged", "Groupwise Flagged", "Isolation Forest Flagged"
    ]
    
    try:
        # Guarantee output directory exists
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for r in records:
                if not r:
                    continue
                score = r.get("risk_score", 0)
                if flagged_only and score < 25:
                    continue
                    
                anom = r.get("anomaly_details")
                if not isinstance(anom, dict):
                    anom = {}
                
                writer.writerow([
                    r.get("transaction_id", ""),
                    r.get("source_row", ""),
                    r.get("customer_id", ""),
                    r.get("project_id", ""),
                    r.get("unit_id", ""),
                    r.get("demand_amount", ""),
                    r.get("collected_amount", ""),
                    r.get("outstanding_amount", ""),
                    r.get("discount_amount", ""),
                    r.get("refund_amount", ""),
                    r.get("payment_delay_days", ""),
                    r.get("demand_date", ""),
                    r.get("payment_date", ""),
                    score,
                    r.get("risk_severity", ""),
                    r.get("violations_count", 0),
                    anom.get("zscore_flagged", False),
                    anom.get("iqr_flagged", False),
                    anom.get("groupwise_flagged", False),
                    anom.get("isolation_forest_flagged", False)
                ])
        print(f"[Exporter] Successfully exported spreadsheet to: {csv_path}")
    except Exception as e:
        print(f"Error exporting CSV to {csv_path}: {e}")


def export_summary_json(summary_data, json_path):
    """Layer 8: Writes structured executive JSON report"""
    try:
        # Guarantee output directory exists
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=4)
        print(f"[Exporter] Successfully exported JSON summary to: {json_path}")
    except Exception as e:
        print(f"Error exporting JSON summary to {json_path}: {e}")


def generate_ascii_dashboard(summary_data, txt_path):
    """Layer 8: Formats a stunning, corporate-level ASCII console summary"""
    overview = summary_data["executive_summary"]
    breakdown = overview["breakdown"]
    engines = summary_data["engine_performance"]
    exposure = summary_data["financial_exposure"]["formatted"]
    patterns = summary_data["fraud_patterns"]
    actions = summary_data["grouped_recommendations"]
    top_10 = summary_data["top_10_high_priority"]
    
    lines = []
    lines.append("┌" + "─" * 78 + "┐")
    lines.append("│" + " OPTOxCRM FINANCE RISK RADAR - EXECUTIVE AUDIT REPORT ".center(78, " ") + "│")
    lines.append("├" + "─" * 78 + "┤")
    
    # 1. Dataset Overview Panel
    lines.append("│  1. DATASET OVERVIEW & CLASS BREAKDOWN                                       │")
    lines.append("├" + "─" * 78 + "┤")
    
    line1 = f"  Total Transactions Scored : {overview['total_audited']:<10}  Risk Exposure : {overview['exposure_pct']}%"
    lines.append(f"│{line1:<78}│")
    
    line2 = f"  Total Flagged Anomalies   : {overview['total_flagged']:<10}  Critical Risk : {breakdown['CRITICAL']}"
    lines.append(f"│{line2:<78}│")
    
    line3 = f"  High Severity Risks       : {breakdown['HIGH']:<10}  Medium Risks  : {breakdown['MEDIUM']}"
    lines.append(f"│{line3:<78}│")
    
    line4 = f"  Low Severity/Safe Entries : {breakdown['LOW']}"
    lines.append(f"│{line4:<78}│")
    
    lines.append("├" + "─" * 78 + "┤")
    
    # 2. Financial Exposure Summary
    lines.append("│  2. FINANCIAL VALUE AT RISK (EXPOSURE)                                       │")
    lines.append("├" + "─" * 78 + "┤")
    
    line_exp1 = f"  Total Demand Value at Risk (Critical + High)  : {exposure['demand_exposure']}"
    lines.append(f"│{line_exp1:<78}│")
    
    line_exp2 = f"  Flagged Outstanding Account Exposure          : {exposure['outstanding_exposure']}"
    lines.append(f"│{line_exp2:<78}│")
    
    line_exp3 = f"  Flagged Outflow/Refund Exposure               : {exposure['refund_exposure']}"
    lines.append(f"│{line_exp3:<78}│")
    
    lines.append("├" + "─" * 78 + "┤")
    
    # 3. Engine Statistics Panel
    lines.append("│  3. SUB-ENGINE DETECTION PERFORMANCE STATS                                   │")
    lines.append("├" + "─" * 78 + "┤")
    
    left1 = f"  Rule Violations (Hard Rules)  : {engines['rule_violations']}"
    right1 = f"Z-Score Flags                  : {engines['zscore_flags']}"
    lines.append(f"│{left1:<37} │ {right1:<38}│")
    
    left2 = f"  IQR Flags (Statistical Out)   : {engines['iqr_flags']}"
    right2 = f"Isolation Forest (ML Outlier)  : {engines['isolation_forest_flags']}"
    lines.append(f"│{left2:<37} │ {right2:<38}│")
    
    left3 = f"  Groupwise Systemic Outliers   : {engines['groupwise_flags']}"
    right3 = ""
    lines.append(f"│{left3:<37} │ {right3:<38}│")
    
    lines.append("├" + "─" * 78 + "┤")
    
    # 4. Fraud Patterns & Recommendations
    lines.append("│  4. SUSPECTED FRAUD PATTERNS & DYNAMIC CFO RECOMMENDATIONS                   │")
    lines.append("├" + "─" * 78 + "┤")
    
    left_h = "  [Fraud Patterns Detected]"
    right_h = "[CFO Action Mitigation Groups]"
    lines.append(f"│{left_h:<37} │ {right_h:<38}│")
    
    max_patterns = max(len(patterns), len(actions))
    patterns_list = list(patterns.items())
    actions_list = list(actions.items())
    
    for i in range(max_patterns):
        p_str = ""
        if i < len(patterns_list):
            k, v = patterns_list[i]
            p_str = f"  - {k[:23]}: {v}"
            
        a_str = ""
        if i < len(actions_list):
            k, v = actions_list[i]
            # Abbreviate for text column formatting
            k_abbrev = k.replace(" Required", "").replace(" Policy", "")
            a_str = f" * {k_abbrev[:28]}: {v}"
            
        lines.append(f"│{p_str:<37} │ {a_str:<38}│")
        
    lines.append("├" + "─" * 78 + "┤")
    
    # 5. Top 10 High Priority Audit Queue
    lines.append("│  5. HIGH PRIORITY AUDIT Remediation QUEUE (TOP 10)                           │")
    lines.append("├" + "─" * 78 + "┤")
    
    # Perfectly aligned table header using exact mathematical column sizing:
    lines.append(
        f"│  {'Rank':<4} │ {'Transaction ID':<19} │ {'Risk Score':<10} │ {'Severity':<8} │ {'CFO Forensic Action':<25} │"
    )
    lines.append("├" + "─" * 78 + "┤")
    
    for rank, item in enumerate(top_10, 1):
        action = item.get('recommended_action') or "Approved: Standard ledger archiving."
        # Shorten action text for horizontal alignment (fits in 25 chars)
        action_short = action.split(";")[0][:25]
        # Force-truncate transaction_id to 19 characters to guarantee it never overflows
        txn_id_short = item.get('transaction_id', '')[:19]
        
        lines.append(
            f"│  #{rank:<3} │ {txn_id_short:<19} │ {item.get('risk_score', 0):<10} │ {item.get('severity', ''):<8} │ {action_short:<25} │"
        )
        
    lines.append("└" + "─" * 78 + "┘")
    
    try:
        # Guarantee output directory exists
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[Exporter] Successfully exported ASCII summary to: {txt_path}")
    except Exception as e:
        print(f"Error exporting ASCII summary to {txt_path}: {e}")


def run_report_generator():
    """
    Consolidated Exporter Entry point. Runs the full 8-Layer Report Generation Engine.
    """
    print("[Report Engine] Starting Business Report Compilation...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_ledger_path = os.path.join(base_dir, "reports", "risk_report.json")
    
    records = load_risk_ledger(json_ledger_path)
    if records is None:
        print("[Report Engine] Compilation aborted. Scored ledger is missing.")
        return
        
    # Layer 1: Executive KPI overview
    exec_summary = generate_executive_summary(records)
    
    # Layer 2: Sub-engine statistics
    engine_stats = generate_engine_statistics(records)
    
    # Layer 3 & 4: Top 10 Priority Queue & categorizations
    breakdown_data = generate_risk_breakdown(records)
    
    # Layer 5: Exposure summation
    exposure_data = generate_financial_exposure(records)
    
    # Layer 6: Tally groupwise patterns
    patterns = generate_pattern_summary(records)
    
    # Layer 7: CFO recommendations consolidation
    recommendations = generate_recommendations_summary(records)
    
    # Package everything for Layer 8 exporters
    summary_package = {
        "executive_summary": exec_summary,
        "engine_performance": engine_stats,
        "financial_exposure": exposure_data,
        "fraud_patterns": patterns,
        "grouped_recommendations": recommendations,
        "top_10_high_priority": breakdown_data["top_10_queue"]
    }
    
    # Export File 1:Structured JSON Aggregates
    summary_json_path = os.path.join(base_dir, "reports", "executive_summary.json")
    export_summary_json(summary_package, summary_json_path)
    
    # Export File 2: Clean spreadsheet for Excel (100% records)
    report_csv_path = os.path.join(base_dir, "reports", "risk_report.csv")
    export_csv(records, report_csv_path, flagged_only=False)
    
    # Export File 3: Flagged spreadsheet for active audit queues (score >= 25)
    flagged_csv_path = os.path.join(base_dir, "reports", "high_risk_transactions.csv")
    export_csv(records, flagged_csv_path, flagged_only=True)
    
    # Export File 4: Elegant Corporate ASCII Console Summary
    summary_txt_path = os.path.join(base_dir, "reports", "summary_report.txt")
    generate_ascii_dashboard(summary_package, summary_txt_path)
    
    print("[Report Engine] Report Compilation Successful! All 8 layers completed.")


if __name__ == "__main__":
    run_report_generator()