import sys
import os
import importlib

# Dynamically add the parent directory (project root) to search path for safe imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("=========================================================================")
    print("                 OPTOxCRM FINANCE RISK RADAR SYSTEM                      ")
    print("=========================================================================")
    
    # 1. Dynamically import modules to handle numerical prefixes safely
    print("\n[Orchestrator] Importing pipeline sub-engines...")
    
    try:
        rule_validator = importlib.import_module("src.03_rule_validator")
        anomaly_detector = importlib.import_module("src.08_anomaly_detector")
        risk_score = importlib.import_module("src.09_risk_score")
        rec_engine = importlib.import_module("src.10_recommendation_engine")
        report_gen = importlib.import_module("src.11_report_generator")
    except ModuleNotFoundError as e:
        print(f"Error importing modules: {e}")
        print("Please ensure your working directory is set to the project root.")
        return

    csv_path = "data/sample_finance_data.csv"

    # Step 1: Rule Validation
    print("\n[Step 1/5] Running Hard Rules Validation Engine...")
    df, violations = rule_validator.validate_finance_csv(csv_path)
    if df is None:
        print("Pipeline aborted. Rule validation failed structurally.")
        return

    # Step 2: Anomaly Outlier Detection
    print("\n[Step 2/5] Running Statistical & ML Anomaly Detection Engines...")
    anomalies = anomaly_detector.run_anomaly_pipeline(csv_path)
    if anomalies is None:
        print("Pipeline aborted. Anomaly detection failed.")
        return

    # Step 3: Risk Scoring & Classification
    print("\n[Step 3/5] Running Unified Risk Scoring Engine...")
    scores = risk_score.run_risk_scoring(csv_path)
    if scores is None:
        print("Pipeline aborted. Risk scoring failed.")
        return

    # Step 4: CFO Audit Recommendations
    print("\n[Step 4/5] Running Mitigation Recommendation Engine...")
    rec_engine.run_recommendation_engine()

    # Step 5: Consolidated Exporter
    print("\n[Step 5/5] Packaging Consolidated Corporate Reports...")
    report_gen.run_report_generator()

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