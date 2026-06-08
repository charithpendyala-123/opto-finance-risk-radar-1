# ==============================================================================
# OPTOxCRM FINANCE RISK RADAR - STAKEHOLDER PORTAL & FORENSIC DASHBOARD
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import io
import sys

# Ensure parent directory is in python path for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─────────────────────────────────────────────────────────────────────────────
# PAGE STYLING & PREMIUM MATTE-CARBON DIRECTIVE
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OPTOxCRM Finance Risk Radar",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject modern CSS style parameters for slate/carbon theme & glassmorphic metrics
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght=300;400;500;600;700&family=Outfit:wght=400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #E2E8F0;
}

/* Completely hide the sidebar expand/collapse control arrow */
[data-testid="collapsedControl"] {
    display: none !important;
}
.main-title {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    background: linear-gradient(135deg, #F8FAFC 30%, #94A3B8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2px;
}

/* Glassmorphic Metric Cards */
            
.metric-card {
    background: rgba(30, 41, 59, 0.45);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(255, 255, 255, 0.1);
}
.metric-title {
    color: #94A3B8;
    font-size: 13px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
}
.metric-value {
    color: #F8FAFC;
    font-size: 21px;          /* Adjusted from 28px to fit perfectly inside 6-column layouts */
    font-weight: 700;
    font-family: 'Outfit', sans-serif;
    white-space: nowrap;      /* Guarantees currency symbols and Cr never wrap vertically */
}
.metric-sub {
    font-size: 11px;
    margin-top: 6px;
}

/* Severity Indicator Dots */
.dot {
    height: 8px;
    width: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
}
.dot-critical { background-color: #EF4444; box-shadow: 0 0 8px #EF4444; }
.dot-high { background-color: #F97316; box-shadow: 0 0 8px #F97316; }
.dot-medium { background-color: #EAB308; box-shadow: 0 0 8px #EAB308; }
.dot-low { background-color: #10B981; box-shadow: 0 0 8px #10B981; }

/* CFO Forensic Sheet Container */
.forensic-sheet {
    background: #0B0F19;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 24px;
    margin-top: 15px;
}

/* Styled Download Buttons */
.stDownloadButton > button {
    background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    font-weight: 600 !important;
    width: 100%;
}
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #60A5FA 0%, #2563EB 100%) !important;
    opacity: 0.95;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATABASE LOADERS & WORKSPACE PORTABILITY
# ─────────────────────────────────────────────────────────────────────────────
REPORTS_DIR = r"C:\Projects\OPTOxCRM Finance Risk Radar\reports"
if not os.path.exists(REPORTS_DIR):
    REPORTS_DIR = os.path.join(os.getcwd(), "reports")

@st.cache_resource
def load_ml_model():
    import joblib
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "best_fraud_model.pkl")
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception:
            pass
    return None

@st.cache_data(show_spinner="📊 Loading transaction reports from database...")
def load_database_reports(user_id="system_default", batch_id="All Batches"):
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        import src.db as db
    except ImportError:
        return None, None

    conn = db.get_connection()
    if conn is None:
        return None, None

    # Load ML Model using cached resource load
    ml_model = load_ml_model()

    try:
        with conn.cursor() as cur:
            query = """
                SELECT 
                    t.id AS transaction_row_id, t.transaction_id, t.customer_id, t.project_id, t.unit_id,
                    t.demand_date, t.payment_date, t.demand_amount, t.collected_amount,
                    t.outstanding_amount, t.discount_amount, t.refund_amount,
                    t.payment_delay_days, t.payment_gap_days, t.uploaded_at, t.upload_batch_id,
                    r.risk_score, r.severity, r.recommendation,
                    f.fraud_label, f.auditor_comments, f.reviewed_at,
                    trf.customer_txn_count_30d, trf.customer_flags_30d, trf.customer_avg_delay_30d,
                    trf.customer_avg_discount_30d, trf.customer_avg_refund_30d, trf.customer_avg_outstanding_30d,
                    trf.customer_max_outstanding_30d, trf.customer_avg_outstanding_ratio_30d,
                    trf.customer_refund_count_90d, trf.customer_refund_count_180d, trf.customer_total_refund_180d,
                    trf.customer_total_refund_lifetime, trf.customer_discount_count_lifetime, trf.customer_fraud_count_lifetime,
                    trf.project_txn_count_30d, trf.project_flags_30d, trf.project_avg_demand_30d,
                    trf.project_avg_outstanding_30d, trf.project_avg_outstanding_ratio_30d, trf.project_avg_refund_30d,
                    trf.unit_txn_count_30d, trf.unit_flags_30d, trf.unit_owner_changes_30d,
                    trf.unit_avg_outstanding_30d, trf.unit_avg_demand_30d, trf.unit_unique_customer_count_30d
                FROM transactions t
                LEFT JOIN risk_results r ON t.id = r.transaction_row_id
                LEFT JOIN auditor_feedback f ON t.id = f.transaction_row_id
                LEFT JOIN transaction_rolling_features trf ON t.id = trf.transaction_row_id
                WHERE t.user_id = %s
            """
            
            params = [user_id]
            if batch_id != "All Batches":
                query += " AND t.upload_batch_id = %s"
                params.append(batch_id)
                
            cur.execute(query, params)
            rows = cur.fetchall()
            colnames = [desc[0] for desc in cur.description]
            db_txs = [dict(zip(colnames, row)) for row in rows]
            
            if batch_id != "All Batches":
                cur.execute("""
                    SELECT ar.transaction_row_id, ar.engine_name, ar.anomaly_flag, ar.anomaly_score, ar.reason 
                    FROM anomaly_results ar
                    JOIN transactions t ON ar.transaction_row_id = t.id
                    WHERE t.user_id = %s AND t.upload_batch_id = %s;
                """, (user_id, batch_id))
            else:
                cur.execute("""
                    SELECT ar.transaction_row_id, ar.engine_name, ar.anomaly_flag, ar.anomaly_score, ar.reason 
                    FROM anomaly_results ar
                    JOIN transactions t ON ar.transaction_row_id = t.id
                    WHERE t.user_id = %s;
                """, (user_id,))
            anom_rows = cur.fetchall()
            anom_map = {}
            for row_id, engine, flag, score, reason in anom_rows:
                if row_id not in anom_map:
                    anom_map[row_id] = []
                anom_map[row_id].append({
                    "engine_name": engine,
                    "anomaly_flag": flag,
                    "anomaly_score": score,
                    "reason": reason
                })

        # Single consolidated pass over transaction rows
        total_audited = len(db_txs)
        breakdown = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        outstanding_exposure = 0.0
        demand_exposure = 0.0
        refund_exposure = 0.0
        fraud_patterns = {}
        engine_stats = {
            "rule_violations": 0,
            "zscore_flags": 0,
            "iqr_flags": 0,
            "groupwise_flags": 0,
            "isolation_forest_flags": 0
        }
        rec_actions = {
            "Immediate Ledger Freeze Required": 0,
            "High Priority Forensic Audit": 0,
            "Manual Reconciliation Required": 0,
            "Senior Forensic Compliance Review": 0,
            "AML Review Required": 0,
            "Routine Policy Validation Check": 0
        }
        
        feature_rows = []
        risk_list = []
        
        for tx in db_txs:
            row_id = tx["transaction_row_id"]
            tx_anoms = anom_map.get(row_id, [])
            
            tx_anom_by_engine = {x["engine_name"]: x for x in tx_anoms}
            
            rule_row = tx_anom_by_engine.get("RuleEngine")
            z_row = tx_anom_by_engine.get("ZScore")
            iqr_row = tx_anom_by_engine.get("IQR")
            gw_row = tx_anom_by_engine.get("Groupwise")
            if_row = tx_anom_by_engine.get("IsolationForest")
            
            violations = []
            if rule_row and rule_row["reason"]:
                parts = rule_row["reason"].split("; ")
                for part in parts:
                    if "]" in part:
                        r_id, r_desc = part.split("]", 1)
                        violations.append({
                            "rule_id": r_id.replace("[", "").strip(),
                            "description": r_desc.strip()
                        })
            
            gw_anoms_list = []
            if gw_row and gw_row["reason"]:
                gw_parts = gw_row["reason"].split("; ")
                for g_part in gw_parts:
                    if "]" in g_part:
                        case_name, case_desc = g_part.split("]", 1)
                        gw_anoms_list.append({
                            "case": case_name.replace("[", "").strip(),
                            "reason": case_desc.strip()
                        })
            
            anomaly_details = {
                "zscore_flagged": z_row is not None,
                "zscore_reason": z_row["reason"] if z_row else None,
                "iqr_flagged": iqr_row is not None,
                "iqr_reason": iqr_row["reason"] if iqr_row else None,
                "groupwise_flagged": gw_row is not None,
                "groupwise_anomalies_count": len(gw_anoms_list),
                "groupwise_anomalies": gw_anoms_list,
                "isolation_forest_flagged": if_row is not None,
                "isolation_forest_score": float(if_row["anomaly_score"]) if (if_row and if_row["anomaly_score"]) else 0.0,
                "isolation_forest_reason": if_row["reason"] if if_row else None
            }
            
            risk_severity = tx["severity"] or "LOW"
            breakdown[risk_severity] += 1
            
            if risk_severity != "LOW":
                if tx["outstanding_amount"] is not None:
                    outstanding_exposure += float(tx["outstanding_amount"])
                if tx["demand_amount"] is not None:
                    demand_exposure += float(tx["demand_amount"])
                if tx["refund_amount"] is not None:
                    refund_exposure += float(tx["refund_amount"])
            
            if len(violations) > 0:
                engine_stats["rule_violations"] += 1
            if anomaly_details["zscore_flagged"]:
                engine_stats["zscore_flags"] += 1
            if anomaly_details["iqr_flagged"]:
                engine_stats["iqr_flags"] += 1
            if anomaly_details["groupwise_flagged"]:
                engine_stats["groupwise_flags"] += 1
            if anomaly_details["isolation_forest_flagged"]:
                engine_stats["isolation_forest_flags"] += 1
                
            score = tx["risk_score"] or 0
            if score >= 25:
                action_text = tx["recommendation"] or ""
                if "IMMEDIATE LEDGER SUSPENSION" in action_text:
                    rec_actions["Immediate Ledger Freeze Required"] += 1
                elif "HIGH PRIORITY FORENSIC AUDIT" in action_text:
                    rec_actions["High Priority Forensic Audit"] += 1
                elif "CRITICAL COMPLIANCE THREAT" in action_text:
                    rec_actions["Senior Forensic Compliance Review"] += 1
                elif "HIGH RISK COMPLIANCE ALERT" in action_text:
                    rec_actions["Manual Reconciliation Required"] += 1
                elif "HIGH RISK WATCHLIST" in action_text:
                    rec_actions["AML Review Required"] += 1
                else:
                    rec_actions["Routine Policy Validation Check"] += 1
            
            patterns = []
            for g in gw_anoms_list:
                case = g.get("case", "")
                if case:
                    clean = case.replace(" Flag", "").replace(" Outlier", "").strip()
                    patterns.append(clean)
            for v in violations:
                desc = v.get("description", "")
                r_id = v.get("rule_id", "")
                if "refund" in desc.lower() or "over-collection" in desc.lower() or r_id in ["RV-14", "RV-15"]:
                    patterns.append("Outflow / Refund Exposure")
                elif "duplicate" in desc.lower() or r_id in ["RV-02", "RV-34", "RV-35"]:
                    patterns.append("Duplicate Ledger Posting")
                elif "gap" in desc.lower() or r_id == "RV-08":
                    patterns.append("Sequential Gap Outlier")
                elif "missing" in desc.lower() or "blank" in desc.lower() or r_id in ["RV-01", "RV-03", "RV-04", "RV-05"]:
                    patterns.append("Missing Identity Anchor")
            if anomaly_details["isolation_forest_flagged"]:
                patterns.append("Isolation Forest (ML Outlier)")
            if anomaly_details["zscore_flagged"]:
                patterns.append("Statistical Z-Score Outlier")
            if anomaly_details["iqr_flagged"]:
                patterns.append("IQR Deviation Outlier")
            
            unique_patterns = list(set(patterns))
            if not unique_patterns:
                unique_patterns = ["Standard Policy Outlier"]
            for p in unique_patterns:
                fraud_patterns[p] = fraud_patterns.get(p, 0) + 1

            risk_list.append({
                "transaction_row_id": tx["transaction_row_id"],
                "transaction_id": tx["transaction_id"],
                "customer_id": tx["customer_id"],
                "project_id": tx["project_id"],
                "unit_id": tx["unit_id"],
                "demand_amount": float(tx["demand_amount"]) if tx["demand_amount"] is not None else None,
                "collected_amount": float(tx["collected_amount"]) if tx["collected_amount"] is not None else None,
                "outstanding_amount": float(tx["outstanding_amount"]) if tx["outstanding_amount"] is not None else None,
                "discount_amount": float(tx["discount_amount"]) if tx["discount_amount"] is not None else None,
                "refund_amount": float(tx["refund_amount"]) if tx["refund_amount"] is not None else None,
                "payment_delay_days": tx["payment_delay_days"],
                "payment_gap_days": tx["payment_gap_days"],
                "demand_date": str(tx["demand_date"]) if tx["demand_date"] is not None else None,
                "payment_date": str(tx["payment_date"]) if tx["payment_date"] is not None else None,
                "risk_score": tx["risk_score"] or 0,
                "risk_severity": risk_severity,
                "recommended_action": tx["recommendation"] or "",
                "violations_count": len(violations),
                "violations": violations,
                "anomaly_details": anomaly_details,
                "fraud_label": tx.get("fraud_label"),
                "auditor_comments": tx.get("auditor_comments"),
                "reviewed_at": str(tx.get("reviewed_at")) if tx.get("reviewed_at") is not None else None,
                "uploaded_at": tx.get("uploaded_at"),
                "ml_probability": None
            })
            
            if ml_model:
                feature_rows.append({
                    'demand_amount': float(tx['demand_amount']) if tx['demand_amount'] is not None else 0.0,
                    'collected_amount': float(tx['collected_amount']) if tx['collected_amount'] is not None else 0.0,
                    'outstanding_amount': float(tx['outstanding_amount']) if tx['outstanding_amount'] is not None else 0.0,
                    'discount_amount': float(tx['discount_amount']) if tx['discount_amount'] is not None else 0.0,
                    'refund_amount': float(tx['refund_amount']) if tx['refund_amount'] is not None else 0.0,
                    'payment_delay_days': int(tx['payment_delay_days']) if tx['payment_delay_days'] is not None else 0,
                    'payment_gap_days': int(tx['payment_gap_days']) if tx['payment_gap_days'] is not None else 0,
                    'customer_txn_count_30d': int(tx['customer_txn_count_30d']) if tx['customer_txn_count_30d'] is not None else 0,
                    'customer_flags_30d': int(tx['customer_flags_30d']) if tx['customer_flags_30d'] is not None else 0,
                    'customer_avg_delay_30d': float(tx['customer_avg_delay_30d']) if tx['customer_avg_delay_30d'] is not None else 0.0,
                    'customer_avg_discount_30d': float(tx['customer_avg_discount_30d']) if tx['customer_avg_discount_30d'] is not None else 0.0,
                    'customer_avg_refund_30d': float(tx['customer_avg_refund_30d']) if tx['customer_avg_refund_30d'] is not None else 0.0,
                    'customer_avg_outstanding_30d': float(tx['customer_avg_outstanding_30d']) if tx['customer_avg_outstanding_30d'] is not None else 0.0,
                    'customer_max_outstanding_30d': float(tx['customer_max_outstanding_30d']) if tx['customer_max_outstanding_30d'] is not None else 0.0,
                    'customer_avg_outstanding_ratio_30d': float(tx['customer_avg_outstanding_ratio_30d']) if tx['customer_avg_outstanding_ratio_30d'] is not None else 0.0,
                    'customer_refund_count_90d': int(tx['customer_refund_count_90d']) if tx['customer_refund_count_90d'] is not None else 0,
                    'customer_refund_count_180d': int(tx['customer_refund_count_180d']) if tx['customer_refund_count_180d'] is not None else 0,
                    'customer_total_refund_180d': float(tx['customer_total_refund_180d']) if tx['customer_total_refund_180d'] is not None else 0.0,
                    'customer_total_refund_lifetime': float(tx['customer_total_refund_lifetime']) if tx['customer_total_refund_lifetime'] is not None else 0.0,
                    'customer_discount_count_lifetime': int(tx['customer_discount_count_lifetime']) if tx['customer_discount_count_lifetime'] is not None else 0,
                    'customer_fraud_count_lifetime': int(tx['customer_fraud_count_lifetime']) if tx['customer_fraud_count_lifetime'] is not None else 0,
                    'project_txn_count_30d': int(tx['project_txn_count_30d']) if tx['project_txn_count_30d'] is not None else 0,
                    'project_flags_30d': int(tx['project_flags_30d']) if tx['project_flags_30d'] is not None else 0,
                    'project_avg_demand_30d': float(tx['project_avg_demand_30d']) if tx['project_avg_demand_30d'] is not None else 0.0,
                    'project_avg_outstanding_30d': float(tx['project_avg_outstanding_30d']) if tx['project_avg_outstanding_30d'] is not None else 0.0,
                    'project_avg_outstanding_ratio_30d': float(tx['project_avg_outstanding_ratio_30d']) if tx['project_avg_outstanding_ratio_30d'] is not None else 0.0,
                    'project_avg_refund_30d': float(tx['project_avg_refund_30d']) if tx['project_avg_refund_30d'] is not None else 0.0,
                    'unit_txn_count_30d': int(tx['unit_txn_count_30d']) if tx['unit_txn_count_30d'] is not None else 0,
                    'unit_flags_30d': int(tx['unit_flags_30d']) if tx['unit_flags_30d'] is not None else 0,
                    'unit_owner_changes_30d': int(tx['unit_owner_changes_30d']) if tx['unit_owner_changes_30d'] is not None else 0,
                    'unit_avg_outstanding_30d': float(tx['unit_avg_outstanding_30d']) if tx['unit_avg_outstanding_30d'] is not None else 0.0,
                    'unit_avg_demand_30d': float(tx['unit_avg_demand_30d']) if tx['unit_avg_demand_30d'] is not None else 0.0,
                    'unit_unique_customer_count_30d': int(tx['unit_unique_customer_count_30d']) if tx['unit_unique_customer_count_30d'] is not None else 0,
                    'rule_violation_count': float(len(violations)),
                    'zscore_count': 1.0 if z_row else 0.0,
                    'iqr_count': 1.0 if iqr_row else 0.0,
                    'groupwise_count': 1.0 if gw_row else 0.0,
                    'isolation_forest_score': float(if_row['anomaly_score']) if (if_row and if_row['anomaly_score']) else 0.0,
                    'total_anomaly_count': float(len(tx_anoms))
                })

        if ml_model and feature_rows:
            try:
                feature_cols_ordered = [
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
                    'unit_avg_outstanding_30d', 'unit_avg_demand_30d', 'unit_unique_customer_count_30d',
                    'rule_violation_count', 'zscore_count', 'iqr_count', 'groupwise_count', 
                    'isolation_forest_score', 'total_anomaly_count'
                ]
                import pandas as pd
                feat_df = pd.DataFrame(feature_rows)[feature_cols_ordered]
                probs = ml_model.predict_proba(feat_df)[:, 1]
                for idx, p in enumerate(probs):
                    risk_list[idx]["ml_probability"] = float(p)
            except Exception as e:
                print(f"Batch prediction error: {e}")

        rec_actions = {k: v for k, v in rec_actions.items() if v > 0}
        total_flagged = breakdown["CRITICAL"] + breakdown["HIGH"] + breakdown["MEDIUM"]
        exposure_pct = (total_flagged / total_audited * 100) if total_audited > 0 else 0.0
        
        def format_cr(val):
            cr = val / 10000000
            return f"₹{cr:.2f} Cr"

        exec_summary = {
            "executive_summary": {
                "total_audited": total_audited,
                "total_flagged": total_flagged,
                "exposure_pct": exposure_pct,
                "breakdown": breakdown
            },
            "engine_performance": engine_stats,
            "grouped_recommendations": rec_actions,
            "fraud_patterns": fraud_patterns,
            "financial_exposure": {
                "raw": {
                    "outstanding_exposure": outstanding_exposure,
                    "demand_exposure": demand_exposure,
                    "refund_exposure": refund_exposure
                },
                "formatted": {
                    "outstanding_exposure": format_cr(outstanding_exposure),
                    "demand_exposure": format_cr(demand_exposure),
                    "refund_exposure": format_cr(refund_exposure)
                }
            },
            "top_10_high_priority": sorted(
                [{"transaction_id": r["transaction_id"], "risk_score": r["risk_score"], "severity": r["risk_severity"], "recommended_action": r["recommended_action"]} for r in risk_list if r["fraud_label"] is None],
                key=lambda x: x["risk_score"],
                reverse=True
            )[:10]
        }
        
        return exec_summary, risk_list
    except Exception as e:
        print(f"[Database Load Error] Failed to load from DB: {e}")
        return None, None
    finally:
        conn.close()

def load_local_reports():
    """Reads execution engine output JSONs from reports folder if available."""
    exec_summary = None
    risk_list = []
    
    summary_path = os.path.join(REPORTS_DIR, "executive_summary.json")
    report_path = os.path.join(REPORTS_DIR, "risk_report.json")
    
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                exec_summary = json.load(f)
        except Exception:
            pass
            
    if os.path.exists(report_path):
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                risk_list = json.load(f)
        except Exception:
            pass
            
    return exec_summary, risk_list

def generate_ascii_report_text(summary_data):
    """Formats a stunning, corporate-level ASCII console summary directly in memory"""
    if not summary_data:
        return "No report data available."
        
    overview = summary_data.get("executive_summary") or {}
    breakdown = overview.get("breakdown") or {}
    breakdown_upper = {k.upper(): v for k, v in breakdown.items()}
    
    crit_count = breakdown_upper.get("CRITICAL", 0)
    high_count = breakdown_upper.get("HIGH", 0)
    med_count = breakdown_upper.get("MEDIUM", 0)
    low_count = breakdown_upper.get("LOW", 0)
    
    engines = summary_data.get("engine_performance") or {}
    exposure = (summary_data.get("financial_exposure") or {}).get("formatted") or {}
    
    demand_exp = exposure.get("demand_exposure") or "₹0.00 Cr"
    outstanding_exp = exposure.get("outstanding_exposure") or "₹0.00 Cr"
    refund_exp = exposure.get("refund_exposure") or "₹0.00 Cr"
    
    patterns = summary_data.get("fraud_patterns") or {}
    actions = summary_data.get("grouped_recommendations") or {}
    top_10 = summary_data.get("top_10_high_priority") or []
    
    lines = []
    lines.append("┌" + "─" * 78 + "┐")
    lines.append("│" + " OPTOxCRM FINANCE RISK RADAR - EXECUTIVE AUDIT REPORT ".center(78, " ") + "│")
    lines.append("├" + "─" * 78 + "┤")
    
    # 1. Dataset Overview Panel
    lines.append("│  1. DATASET OVERVIEW & CLASS BREAKDOWN                                       │")
    lines.append("├" + "─" * 78 + "┤")
    
    line1 = f"  Total Transactions Scored : {overview.get('total_audited', 0):<10}  Risk Exposure : {overview.get('exposure_pct', 0.0)}%"
    lines.append(f"│{line1:<78}│")
    
    line2 = f"  Total Flagged Anomalies   : {overview.get('total_flagged', 0):<10}  Critical Risk : {crit_count}"
    lines.append(f"│{line2:<78}│")
    
    line3 = f"  High Severity Risks       : {high_count:<10}  Medium Risks  : {med_count}"
    lines.append(f"│{line3:<78}│")
    
    line4 = f"  Low Severity/Safe Entries : {low_count}"
    lines.append(f"│{line4:<78}│")
    
    lines.append("├" + "─" * 78 + "┤")
    
    # 2. Financial Exposure Summary
    lines.append("│  2. FINANCIAL VALUE AT RISK (EXPOSURE)                                       │")
    lines.append("├" + "─" * 78 + "┤")
    
    line_exp1 = f"  Total Demand Value at Risk (Critical + High)  : {demand_exp}"
    lines.append(f"│{line_exp1:<78}│")
    
    line_exp2 = f"  Flagged Outstanding Account Exposure          : {outstanding_exp}"
    lines.append(f"│{line_exp2:<78}│")
    
    line_exp3 = f"  Flagged Outflow/Refund Exposure               : {refund_exp}"
    lines.append(f"│{line_exp3:<78}│")
    
    lines.append("├" + "─" * 78 + "┤")
    
    # 3. Engine Statistics Panel
    lines.append("│  3. SUB-ENGINE DETECTION PERFORMANCE STATS                                   │")
    lines.append("├" + "─" * 78 + "┤")
    
    left1 = f"  Rule Violations (Hard Rules)  : {engines.get('rule_violations', 0)}"
    right1 = f"Z-Score Flags                  : {engines.get('zscore_flags', 0)}"
    lines.append(f"│{left1:<37} │ {right1:<38}│")
    
    left2 = f"  IQR Flags (Statistical Out)   : {engines.get('iqr_flags', 0)}"
    right2 = f"Isolation Forest (ML Outlier)  : {engines.get('isolation_forest_flags', 0)}"
    lines.append(f"│{left2:<37} │ {right2:<38}│")
    
    left3 = f"  Groupwise Systemic Outliers   : {engines.get('groupwise_flags', 0)}"
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
            k_abbrev = k.replace(" Required", "").replace(" Policy", "")
            a_str = f" * {k_abbrev[:28]}: {v}"
            
        lines.append(f"│{p_str:<37} │ {a_str:<38}│")
        
    lines.append("├" + "─" * 78 + "┤")
    
    # 5. Top 10 High Priority Audit Queue
    lines.append("│  5. HIGH PRIORITY AUDIT Remediation QUEUE (TOP 10)                           │")
    lines.append("├" + "─" * 78 + "┤")
    
    lines.append(
        f"│  {'Rank':<4} │ {'Transaction ID':<19} │ {'Risk Score':<10} │ {'Severity':<8} │ {'CFO Forensic Action':<25} │"
    )
    lines.append("├" + "─" * 78 + "┤")
    
    for rank, item in enumerate(top_10, 1):
        action = item.get('recommended_action') or "Approved: Standard ledger archiving."
        action_short = action.split(";")[0][:25]
        txn_id_short = item.get('transaction_id', '')[:19]
        
        lines.append(
            f"│  #{rank:<3} │ {txn_id_short:<19} │ {item.get('risk_score', 0):<10} │ {item.get('severity', ''):<8} │ {action_short:<25} │"
        )
        
    lines.append("└" + "─" * 78 + "┘")
    return "\n".join(lines) + "\n"

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL USER CONFIG & WORKSPACE ROUTING
# ─────────────────────────────────────────────────────────────────────────────
active_user = "system_default"

# Initialize Session States
if "active_page" not in st.session_state:
    st.session_state["active_page"] = "analytics"

if "pipeline_run" not in st.session_state:
    st.session_state["pipeline_run"] = False

# Establish database active connection status
db_active = True
import src.db as db
conn = db.get_connection()
if conn:
    try:
        conn.close()
    except Exception:
        pass
else:
    db_active = False

pipeline_run = st.session_state["pipeline_run"]
if not pipeline_run:
    if "pending_verdicts" in st.session_state:
        del st.session_state["pending_verdicts"]
active_batch = st.session_state.get("active_batch", "All Batches")

# Load live database dataset or local files when page requires it
real_summary, real_report = None, None
if pipeline_run:
    if db_active:
        real_summary, real_report = load_database_reports(active_user, active_batch)
    else:
        real_summary, real_report = load_local_reports()

if real_summary is None or real_report is None:
    real_summary = {}
    real_report = []

# Fetch active notifications strictly for Auditor console
active_notifications = []
if st.session_state["active_page"] == "auditor" and pipeline_run and db_active:
    conn = db.get_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                # Fetch all unread value updates from the DB
                cur.execute("""
                    SELECT n.id, n.transaction_id, n.message, n.created_at 
                    FROM system_notifications n
                    WHERE n.user_id = %s AND n.read_status = FALSE 
                    ORDER BY n.created_at DESC;
                """, (active_user,))
                rows = cur.fetchall()
                for r in rows:
                    active_notifications.append({
                        "id": r[0],
                        "txn_id": r[1],
                        "message": r[2],
                        "type": "VALUE_UPDATED"
                    })
        except Exception:
            pass
        finally:
            conn.close()

# Dynamically compute runtime SLA alerts for Auditor console
if st.session_state["active_page"] == "auditor":
    import datetime
    SLA_THRESHOLDS = {
        "CRITICAL": 3,
        "HIGH": 7,
        "MEDIUM": 10
    }
    for tx in real_report:
        uploaded_at = tx.get("uploaded_at")
        verdict = tx.get("fraud_label")
        severity = (tx.get("risk_severity") or "").upper()
        
        sla_limit = SLA_THRESHOLDS.get(severity)
        if uploaded_at and verdict is None and sla_limit is not None:
            if isinstance(uploaded_at, str):
                try:
                    uploaded_dt = datetime.datetime.fromisoformat(uploaded_at.replace("Z", "+00:00"))
                except Exception:
                    uploaded_dt = datetime.datetime.now()
            else:
                uploaded_dt = uploaded_at
                
            now_dt = datetime.datetime.now(uploaded_dt.tzinfo) if uploaded_dt.tzinfo else datetime.datetime.now()
            pending_days = (now_dt - uploaded_dt).days
            if pending_days > sla_limit:
                active_notifications.append({
                    "txn_id": tx["transaction_id"],
                    "message": f"⚠️ SLA Breach: Transaction {tx['transaction_id']} ({severity.title()}) pending review for {pending_days} days.",
                    "type": "SLA_BREACH"
                })

# Initialize session state for verdicts
if "pending_verdicts" not in st.session_state or not st.session_state["pending_verdicts"]:
    st.session_state["pending_verdicts"] = {}
    for idx, tx in enumerate(real_report):
        row_key = tx.get("transaction_row_id")
        if row_key is None:
            row_key = f"{tx['transaction_id']}_{idx}"
        st.session_state["pending_verdicts"][row_key] = tx.get("fraud_label")

# ─────────────────────────────────────────────────────────────────────────────
# MODAL DIALOGS
# ─────────────────────────────────────────────────────────────────────────────
@st.dialog("Audit Review Saved", width="small")
def show_success_modal(message):
    st.markdown("""
    <div style="text-align: center; padding: 10px 0px 10px 0px;">
        <div style="display: inline-flex; align-items: center; justify-content: center; width: 70px; height: 70px; border-radius: 50%; background-color: #22C55E; color: white; font-size: 35px; margin-bottom: 15px; box-shadow: 0 0 15px rgba(34, 197, 94, 0.4);">
            ✓
        </div>
        <h3 style="font-family: 'Outfit', sans-serif; font-weight: 700; color: #F8FAFC; margin-bottom: 10px;">Success</h3>
    </div>
    """, unsafe_allow_html=True)
    st.write(message)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("OK", type="primary", use_container_width=True):
        st.rerun()

# Trigger Success Modal if flag is set in session state
if "show_success_dialog" in st.session_state:
    dialog_msg = st.session_state["show_success_dialog"]
    del st.session_state["show_success_dialog"]
    show_success_modal(dialog_msg)

# ─────────────────────────────────────────────────────────────────────────────
# PREMIUM SIDEBAR PANEL
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="text-align: center; padding: 10px 0px 20px 0px;">
    <h2 style="font-family: 'Outfit', sans-serif; font-weight: 700; margin: 0; background: linear-gradient(135deg, #F8FAFC 30%, #94A3B8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        🛡️ OPTOxCRM
    </h2>
    <span style="color: #94A3B8; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em;">
        Finance Risk Radar
    </span>
</div>
<hr style="margin: 10px 0px 20px 0px; border-color: rgba(255, 255, 255, 0.05);">
""", unsafe_allow_html=True)

st.sidebar.markdown("### 🧭 Navigation")

btn_analytics_type = "primary" if st.session_state["active_page"] == "analytics" else "secondary"
btn_auditor_type = "primary" if st.session_state["active_page"] == "auditor" else "secondary"

if st.sidebar.button("📊 Executive Analytics", use_container_width=True, type=btn_analytics_type, key="sidebar_nav_analytics"):
    st.session_state["active_page"] = "analytics"
    st.rerun()

if st.sidebar.button("⚖️ Auditor Feedback Console", use_container_width=True, type=btn_auditor_type, key="sidebar_nav_auditor"):
    st.session_state["active_page"] = "auditor"
    st.rerun()

# Import New Ledger button when data is loaded
if pipeline_run:
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Import New Ledger", use_container_width=True, key="sidebar_reset_pipeline"):
        st.session_state["pipeline_run"] = False
        st.session_state["active_batch"] = "All Batches"
        if "pending_verdicts" in st.session_state:
            del st.session_state["pending_verdicts"]
        st.rerun()

# Display active batch badge if pipeline has run
if pipeline_run and active_batch:
    st.sidebar.markdown(f"""
    <div style="background: rgba(30, 41, 59, 0.45); border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 8px; padding: 12px; margin-top: 15px;">
        <span style="color: #94A3B8; font-size: 10px; font-weight: 600; text-transform: uppercase;">Active Ingest Batch</span>
        <div style="color: #60A5FA; font-weight: 700; font-size: 13px; margin-top: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
            📂 {active_batch}
        </div>
    </div>
    """, unsafe_allow_html=True)

# Page routing checks
if st.session_state["active_page"] == "auditor":
    # ─── AUDITOR FEEDBACK PAGE VIEW ───
    h_col1, h_col2 = st.columns([3, 1])
    with h_col1:
        st.markdown("""
        <h1 class='main-title' style='user-select: text; margin: 0;'>
            ⚖️ Auditor Feedback Console
        </h1>
        """, unsafe_allow_html=True)
        db_status_badge = "🟢 PostgreSQL Connected" if db_active else "🟡 Local Files Fallback"
        st.caption(f"Live Compliance Audit Log Queue · User: **{active_user}** · Connection: {db_status_badge}")
        
    with h_col2:
        notif_count = len(active_notifications)
        btn_label = f"🔔 Notifications ({notif_count})" if notif_count > 0 else "🔔 Notifications"
        with st.popover(btn_label, use_container_width=True):
            st.markdown("### 🔔 System Alert Center")
            if not active_notifications:
                st.info("No active notifications or SLA breaches logged.")
            else:
                for idx, n in enumerate(active_notifications):
                    icon = "🔄" if n["type"] == "VALUE_UPDATED" else "⚠️"
                    st.write(f"{icon} **{n['message']}**")
                    st.divider()
                
                # Button to clear persistent database notifications
                db_notifs = [n["id"] for n in active_notifications if n["type"] == "VALUE_UPDATED"]
                if db_notifs and st.button("✅ Mark All as Read", key="clear_notifs"):
                    conn = db.get_connection()
                    if conn:
                        try:
                            with conn.cursor() as cur:
                                cur.execute("UPDATE system_notifications SET read_status = TRUE WHERE id = ANY(%s);", (db_notifs,))
                            conn.commit()
                            st.success("Notifications marked as read!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
                        finally:
                            conn.close()
    st.markdown("<br>", unsafe_allow_html=True)

    # ─── STEP 1: CALCULATE COMPLIANCE METRICS ───
    total_cnt = 0
    fraud_cnt = 0
    clear_cnt = 0
    pending_cnt = 0
    if "pending_verdicts" in st.session_state and pipeline_run:
        total_cnt = len(st.session_state["pending_verdicts"])
        fraud_cnt = sum(1 for v in st.session_state["pending_verdicts"].values() if v is True)
        clear_cnt = sum(1 for v in st.session_state["pending_verdicts"].values() if v is False)
        pending_cnt = sum(1 for v in st.session_state["pending_verdicts"].values() if v is None)

    val_total = f"{total_cnt:,}" if pipeline_run else "N/A"
    val_fraud = f"{fraud_cnt:,}" if pipeline_run else "N/A"
    val_clear = f"{clear_cnt:,}" if pipeline_run else "N/A"
    val_pending = f"{pending_cnt:,}" if pipeline_run else "N/A"

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📊 Total Ledgers</div>
            <div class="metric-value">{val_total}</div>
            <div class="metric-sub" style="color: #64748B;">Total active audit files</div>
        </div>
        """, unsafe_allow_html=True)
    with mc2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🔴 Flagged Fraud</div>
            <div class="metric-value" style="color: #F87171;">{val_fraud}</div>
            <div class="metric-sub" style="color: #EF4444;">Transactions marked as fraud</div>
        </div>
        """, unsafe_allow_html=True)
    with mc3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🟢 Clear Status</div>
            <div class="metric-value" style="color: #34D399;">{val_clear}</div>
            <div class="metric-sub" style="color: #10B981;">Clean transaction records</div>
        </div>
        """, unsafe_allow_html=True)
    with mc4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">⏳ Pending Review</div>
            <div class="metric-value" style="color: #94A3B8;">{val_pending}</div>
            <div class="metric-sub" style="color: #64748B;">Transactions needing audit</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ─── STEP 2: RENDER FILTERS ───
    st.markdown("### 🔍 Filter Work Queue")
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        filter_severity = st.selectbox("Risk Severity:", ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"])
    with f_col2:
        filter_status = st.selectbox("Auditor Status:", ["All", "Flagged Fraud", "Clear"])
    with f_col3:
        search_txn = st.text_input("Search Transaction ID:", value="", placeholder="e.g. TXN313").strip()

    # Apply filters
    filtered_report = []
    for idx, tx in enumerate(real_report):
        txn_id = tx["transaction_id"]
        row_id = tx.get("transaction_row_id")
        if row_id is None:
            row_id = f"{txn_id}_{idx}"
        verdict = st.session_state["pending_verdicts"].get(row_id)
        
        # Search ID
        if search_txn and search_txn.upper() not in txn_id.upper():
            continue
            
        # Severity
        if filter_severity != "All" and tx.get("risk_severity") != filter_severity:
            continue
            
        # Verdict status
        if filter_status == "Flagged Fraud" and verdict is not True:
            continue
        elif filter_status == "Clear" and verdict is not False:
            continue
            
        filtered_report.append(tx)

    # ─── STEP 3: PAGINATION ───
    total_records = len(filtered_report)
    total_pages = max(1, (total_records + 24) // 25)

    if "current_page_idx" not in st.session_state:
        st.session_state["current_page_idx"] = 1
        
    if st.session_state["current_page_idx"] > total_pages:
        st.session_state["current_page_idx"] = total_pages

    st.markdown("---")
    st.markdown("### 📋 Transaction Auditing Queue")

    p_col1, p_col2, p_col3 = st.columns([1, 2, 1])
    with p_col1:
        if st.button("⬅️ Previous Page", disabled=(st.session_state["current_page_idx"] == 1)):
            st.session_state["current_page_idx"] -= 1
            st.rerun()
    with p_col3:
        if st.button("Next Page ➡️", disabled=(st.session_state["current_page_idx"] == total_pages)):
            st.session_state["current_page_idx"] += 1
            st.rerun()
    with p_col2:
        current_page = st.selectbox(
            "Select Page:",
            options=range(1, total_pages + 1),
            index=st.session_state["current_page_idx"] - 1,
            format_func=lambda x: f"Page {x} of {total_pages} (Rows {(x-1)*25+1} - {min(x*25, total_records)})",
            key="jump_page_select"
        )
    if current_page != st.session_state["current_page_idx"]:
        st.session_state["current_page_idx"] = current_page
        st.rerun()

    # Get page slice
    start_idx = (st.session_state["current_page_idx"] - 1) * 25
    end_idx = min(start_idx + 25, total_records)
    page_records = filtered_report[start_idx:end_idx]

    if not page_records:
        if not pipeline_run:
            st.info("💡 Please upload a financial ledger CSV under Executive Analytics to start auditing.")
        else:
            st.info("No transactions match the selected filters.")
    else:
        # Table layout
        st.markdown("<br>", unsafe_allow_html=True)
        h_col1, h_col2, h_col3, h_col4, h_col5, h_col6 = st.columns([1.5, 1.5, 1.2, 1.2, 2.0, 2.0])
        with h_col1: st.markdown("**Transaction ID**")
        with h_col2: st.markdown("**Customer ID**")
        with h_col3: st.markdown("**Risk Score**")
        with h_col4: st.markdown("**Severity**")
        with h_col5: st.markdown("**Review Status**")
        with h_col6: st.markdown("**Action Toggle**")
        st.markdown("<hr style='margin: 5px 0px; border-color: rgba(255,255,255,0.15);'>", unsafe_allow_html=True)

        for idx, tx in enumerate(page_records):
            txn_id = tx["transaction_id"]
            global_idx = start_idx + idx
            row_id = tx.get("transaction_row_id")
            if row_id is None:
                row_id = f"{txn_id}_{global_idx}"
            cust_id = tx["customer_id"] or "N/A"
            risk_score = tx["risk_score"]
            severity = tx["risk_severity"]
            verdict = st.session_state["pending_verdicts"].get(row_id)

            # --- Calculate SLA Pending Age in Days ---
            import datetime
            uploaded_at = tx.get("uploaded_at")
            pending_days = 0
            is_sla_breach = False
            
            SLA_THRESHOLDS = {
                "CRITICAL": 3,
                "HIGH": 7,
                "MEDIUM": 10
            }
            curr_severity = (severity or "").upper()
            sla_limit = SLA_THRESHOLDS.get(curr_severity)
            if uploaded_at and sla_limit is not None:
                if isinstance(uploaded_at, str):
                    try:
                        clean_ts = uploaded_at.replace("Z", "+00:00")
                        uploaded_dt = datetime.datetime.fromisoformat(clean_ts)
                    except Exception:
                        uploaded_dt = datetime.datetime.now()
                else:
                    uploaded_dt = uploaded_at
                
                now_dt = datetime.datetime.now(uploaded_dt.tzinfo) if uploaded_dt.tzinfo else datetime.datetime.now()
                pending_days = (now_dt - uploaded_dt).days
                
                # SLA Breach: Unreviewed for more than the severity threshold
                if verdict is None and pending_days > sla_limit:
                    is_sla_breach = True

            r_col1, r_col2, r_col3, r_col4, r_col5, r_col6 = st.columns([1.5, 1.5, 1.2, 1.2, 2.0, 2.0])

            # Determine if the Auditor manually cleared a machine-flagged risk
            is_override = False
            ml_prob = tx.get("ml_probability")
            if verdict is False and (
                (ml_prob is not None and ml_prob >= 0.5) or 
                (severity in ["CRITICAL", "HIGH", "MEDIUM"])
            ):
                is_override = True

            with r_col1:
                st.markdown(f"**{txn_id}**")
            with r_col2:
                st.markdown(f"`{cust_id}`")
            with r_col3:
                if is_override:
                    st.markdown(f"~~{risk_score} / 100~~")
                else:
                    st.markdown(f"{risk_score} / 100")
            with r_col4:
                if is_override:
                    st.markdown("<span style='color: #10B981; font-weight: 600;'>🟢 CLEARED</span>", unsafe_allow_html=True)
                else:
                    color = "#EF4444" if severity == "CRITICAL" else "#F97316" if severity == "HIGH" else "#EAB308" if severity == "MEDIUM" else "#10B981"
                    st.markdown(f"<span style='color: {color}; font-weight: 600;'>{severity}</span>", unsafe_allow_html=True)
            
            with r_col5:
                # Check if this row ID has an active "value updated" alert
                has_value_update = any(n["txn_id"] == txn_id and n["type"] == "VALUE_UPDATED" for n in active_notifications)
                update_badge = " <span style='background-color:#1E293B; border: 1px solid #3B82F6; color:#60A5FA; font-size:10px; padding:2px 6px; border-radius:4px; margin-left:6px; font-weight:600;'>🔄 Updated</span>" if has_value_update else ""

                # Construct ML Verdict Badge dynamically
                ml_badge = ""
                if ml_prob is not None:
                    if ml_prob >= 0.5:
                        ml_badge = " <span style='background-color:rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.25); color:#F87171; font-size:10px; padding:2px 6px; border-radius:4px; margin-left:6px; font-weight:600;'>🤖 ML: Fraud</span>"
                    else:
                        ml_badge = " <span style='background-color:rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.25); color:#34D399; font-size:10px; padding:2px 6px; border-radius:4px; margin-left:6px; font-weight:600;'>🤖 ML: Clean</span>"

                # Construct Overridden Badge
                override_badge = " <span style='background-color:rgba(16,185,129,0.1); border: 1px solid #10B981; color:#34D399; font-size:10px; padding:2px 6px; border-radius:4px; margin-left:6px; font-weight:600;'>🔄 Overridden</span>" if is_override else ""

                if verdict is True:
                    st.markdown(f"🔴 <span style='color: #F87171; font-weight: 600;'>Flagged Fraud</span>{update_badge}{ml_badge}", unsafe_allow_html=True)
                elif verdict is False:
                    st.markdown(f"🟢 <span style='color: #34D399; font-weight: 600;'>Clear</span>{override_badge}{update_badge}{ml_badge}", unsafe_allow_html=True)
                else:
                    if is_sla_breach:
                        st.markdown(f"⚠️ <span style='color: #EF4444; font-weight: 700; text-shadow: 0 0 8px rgba(239,68,68,0.3);'>SLA BREACH ({pending_days}d)</span>{update_badge}{ml_badge}", unsafe_allow_html=True)
                    else:
                        st.markdown(f"⏳ <span style='color: #808495;'>Pending ({pending_days}d)</span>{update_badge}{ml_badge}", unsafe_allow_html=True)
            with r_col6:
                if verdict is True:
                    if st.button("✅ Mark Clear", key=f"clear_{row_id}", type="primary", use_container_width=True):
                        st.session_state["pending_verdicts"][row_id] = False
                        st.rerun()
                elif verdict is False:
                    if st.button("🚨 Flag Fraud", key=f"flag_{row_id}", type="secondary", use_container_width=True):
                        st.session_state["pending_verdicts"][row_id] = True
                        st.rerun()
                else:  # verdict is None (Pending)
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        if st.button("🚨 Flag", key=f"flag_{row_id}", type="secondary", use_container_width=True):
                            st.session_state["pending_verdicts"][row_id] = True
                            st.rerun()
                    with col_btn2:
                        if st.button("✅ Clear", key=f"clear_{row_id}", type="secondary", use_container_width=True):
                            st.session_state["pending_verdicts"][row_id] = False
                            st.rerun()

            st.markdown("<hr style='margin: 3px 0px; border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)

        # ─── STEP 4: SUBMIT PANEL ───
        page_row_ids = []
        for idx, tx in enumerate(page_records):
            rid = tx.get("transaction_row_id")
            if rid is None:
                global_idx = start_idx + idx
                rid = f"{tx['transaction_id']}_{global_idx}"
            page_row_ids.append(rid)
        page_verdicts = {rid: st.session_state["pending_verdicts"].get(rid) for rid in page_row_ids}
        flagged_count = sum(1 for v in page_verdicts.values() if v is True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 💾 Batch Submit Panel")
        st.write(f"**Current Page Status:** {flagged_count} transactions flagged as Fraud on this page.")

        if not db_active:
            st.warning("🟡 Live Database Connection Offline. Running in Local Files Fallback mode. Submitting reviews to PostgreSQL is disabled.")
            st.button("💾 Submit Page Reviews to PostgreSQL", disabled=True, use_container_width=True)
        else:
            st.info("💡 You can submit reviews to the database at any time. All status values on this page will be saved.")
            if st.button("💾 Submit Page Reviews to PostgreSQL", type="primary", use_container_width=True):
                import src.db as db
                import src.feedback_repository as feedback_repo
                conn = db.get_connection()
                if conn:
                    success_count = 0
                    for rid, verdict in page_verdicts.items():
                        if verdict is not None:
                            if feedback_repo.save_feedback(conn, rid, verdict, f"Reviewed via portal (Page {st.session_state['current_page_idx']})"):
                                success_count += 1
                    conn.close()
                    
                    # --- AUTOMATIC ML TRAINING TRIGGER ---
                    train_status = ""
                    if success_count > 0:
                        try:
                            import src.train_model as trainer
                            trained = trainer.run_model_training(user_id=active_user, batch_id=active_batch)
                            if trained:
                                train_status = " and successfully trained the ML model!"
                            else:
                                train_status = " but ML training was skipped (requires both classes and 50% coverage)."
                        except Exception as e:
                            train_status = f" but ML training failed: {e}"
                    
                    # Store success message in session state to trigger the modal dialog on reload
                    st.session_state["show_success_dialog"] = f"Successfully saved {success_count} reviews to PostgreSQL{train_status}"
                    
                    # Force database reload by deleting session state
                    if "pending_verdicts" in st.session_state:
                        del st.session_state["pending_verdicts"]
                    
                    # Clear the Streamlit cache to load updated records from PostgreSQL
                    load_database_reports.clear()
                    
                    st.rerun()
                else:
                    st.error("Error: Failed to connect to PostgreSQL database.")

    # Halt execution here so that the original Analytics page doesn't execute
    st.stop()

# ─── EXECUTIVE ANALYTICS BLANK STATE INTERCEPT ───
if st.session_state["active_page"] == "analytics" and not pipeline_run:
    st.markdown("""
    <div style="text-align: center; padding: 50px 0px;">
        <h1 style="font-family: 'Outfit'; font-weight: 700;">🚀 Finance Risk Radar Ingestion Portal</h1>
        <p style="color: #94A3B8;">Please upload your financial voucher ledger sheet to initiate real-time forensic scanning.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        custom_batch_name = st.text_input("Custom Batch Name (Mandatory):", value="", placeholder="e.g. Q2_Audit_v1", key="landing_batch_name").strip()
        uploaded_file = st.file_uploader("Upload new financial ledger (CSV or XLSX):", type=["csv", "xlsx"], key="landing_file_uploader")
        
        if uploaded_file is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            if not custom_batch_name:
                st.warning("⚠️ Please provide a custom batch name to trigger the ingestion pipeline.")
                st.button("🔥 Run Risk Analysis Pipeline", type="primary", use_container_width=True, disabled=True, key="run_pipeline_btn_disabled")
            else:
                if st.button("🔥 Run Risk Analysis Pipeline", type="primary", use_container_width=True, key="run_pipeline_btn"):
                    # Save uploaded file to a temporary location inside workspace
                    temp_dir = "data/uploads"
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_path = os.path.join(temp_dir, f"uploaded_{active_user}.csv")
                    
                    # 1. Load the uploaded file into a pandas DataFrame first
                    try:
                        if uploaded_file.name.endswith(".xlsx"):
                            df_uploaded = pd.read_excel(uploaded_file)
                        else:
                            df_uploaded = pd.read_csv(uploaded_file)
                    except Exception as e:
                        st.error(f"Error reading file: {e}")
                        st.stop()

                    # 2. Heal / map truncated column headers back to expected names
                    heal_map = {
                        "transaction": "transaction_id",
                        "customer_": "customer_id",
                        "demand_a": "demand_amount",
                        "collected_": "collected_amount",
                        "outstandin": "outstanding_amount",
                        "discount_a": "discount_amount",
                        "refund_am": "refund_amount",
                        "payment_d": "payment_delay_days",
                        "demand_d": "demand_date",
                        "payment_d.1": "payment_date"
                    }
                    
                    new_cols = []
                    payment_d_seen = False
                    for col in df_uploaded.columns:
                        col_clean = str(col).strip()
                        if col_clean == "payment_d":
                            if not payment_d_seen:
                                new_cols.append("payment_delay_days")
                                payment_d_seen = True
                            else:
                                new_cols.append("payment_date")
                        elif col_clean == "payment_d.1":
                            new_cols.append("payment_date")
                        elif col_clean in heal_map:
                            new_cols.append(heal_map[col_clean])
                        else:
                            new_cols.append(col)
                    
                    df_uploaded.columns = new_cols

                    # 3. Save the healed DataFrame to standard CSV
                    df_uploaded.to_csv(temp_path, index=False)
                    # Show loading spinner
                    with st.spinner("Processing ledger pipeline through Hard Rules, Z-Score, IQR, Groupwise & Isolation Forest engines..."):
                        try:
                            import importlib
                            import src.main as pipeline
                            importlib.reload(pipeline)
                            # Run the pipeline
                            pipeline.main(user_id=active_user, csv_path=temp_path, batch_id=custom_batch_name)
                            
                            st.session_state["pipeline_run"] = True
                            st.session_state["active_batch"] = custom_batch_name
                            if "pending_verdicts" in st.session_state:
                                del st.session_state["pending_verdicts"]
                            
                            st.success(f"Pipeline executed successfully for batch: {custom_batch_name}!")
                            
                            # Clear cache so the dashboard retrieves the newly uploaded batch
                            load_database_reports.clear()
                            
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error running pipeline: {e}")
    # Halt execution here so that the dashboard doesn't execute
    st.stop()

# Parse values
exec_data = (real_summary or {}).get("executive_summary") or {}
total_audited = exec_data.get("total_audited", 0)
total_flagged = exec_data.get("total_flagged", 0)
exposure_pct = exec_data.get("exposure_pct", 0.0)

breakdown = exec_data.get("breakdown") or {}
crit_count = breakdown.get("CRITICAL", 0)
high_count = breakdown.get("HIGH", 0)
med_count = breakdown.get("MEDIUM", 0)
low_count = breakdown.get("LOW", 0)

fraud_distribution = (real_summary or {}).get("fraud_patterns") or {}

# ─────────────────────────────────────────────────────────────────────────────
# HEADER PANEL FOR EXECUTIVE ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
h_col1, h_col2 = st.columns([3, 1])
with h_col1:
    st.markdown("""
    <h1 class='main-title' style='user-select: text; margin: 0;'>
        🛡️ OPTOxCRM Finance Risk Radar
    </h1>
    """, unsafe_allow_html=True)
    db_status_badge = "🟢 PostgreSQL Connected" if db_active else "🟡 Local Files Fallback"
    st.caption(f"Consolidated Ledger Forensics Dashboard · Batch: **{active_batch}** · Connection: {db_status_badge}")

with h_col2:
    st.markdown(f"""
    <div style="text-align: right; padding-top: 10px;">
        <span style="background-color: rgba(30, 41, 59, 0.6); border: 1px solid rgba(255, 255, 255, 0.05); color: #94A3B8; font-size: 12px; padding: 6px 12px; border-radius: 20px; font-weight: 500;">
            👤 System Default Session
        </span>
    </div>
    """, unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CARD ROW: EXECUTIVE SUMMARY METRICS (DYNAMIC 6-COLUMN GRID)
# ─────────────────────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5, m6 = st.columns(6)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">📊 Total Audited</div>
        <div class="metric-value">{total_audited:,}</div>
        <div class="metric-sub" style="color: #64748B;">Total voucher receipts parsed</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">🚨 Flagged Transactions</div>
        <div class="metric-value" style="color: #F87171;">{total_flagged:,}</div>
        <div class="metric-sub" style="color: #EF4444;">⚠️ Requiring forensic evaluation</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">🔥 Exposure %</div>
        <div class="metric-value" style="color: #FB923C;">{exposure_pct:.2f}%</div>
        <div class="metric-sub" style="color: #F97316;">Voucher-to-Ledger deviation index</div>
    </div>
    """, unsafe_allow_html=True)

# Fetch financial exposure values dynamically from output summary
financial_exp = (real_summary or {}).get("financial_exposure") or {}
formatted_exp = financial_exp.get("formatted") or {}

val_outstanding = formatted_exp.get("outstanding_exposure") or "₹0.00 Cr"
val_demand = formatted_exp.get("demand_exposure") or "₹0.00 Cr"
val_refund = formatted_exp.get("refund_exposure") or "₹0.00 Cr"

with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">💼 Outstanding Risk</div>
        <div class="metric-value" style="color: #FBBF24;">{val_outstanding}</div>
        <div class="metric-sub" style="color: #EAB308;">Total outstanding value at risk</div>
    </div>
    """, unsafe_allow_html=True)

with m5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">📈 Demand Risk</div>
        <div class="metric-value" style="color: #60A5FA;">{val_demand}</div>
        <div class="metric-sub" style="color: #3B82F6;">Total demand value at risk</div>
    </div>
    """, unsafe_allow_html=True)

with m6:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">💸 Refund Risk</div>
        <div class="metric-value" style="color: #F472B6;">{val_refund}</div>
        <div class="metric-sub" style="color: #EC4899;">Total outflow refund exposure</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# VISUAL INSIGHTS PANEL (RISK BREAKDOWN + DISTRIBUTION)
# ─────────────────────────────────────────────────────────────────────────────
c1, c2 = st.columns([2, 3], gap="large")

with c1:
    st.markdown("### 🍩 Severity Classification")
    
    # 1. Case-insensitive key lookup to fix the casing bug
    breakdown_lower = {k.lower(): v for k, v in breakdown.items()}
    crit_count = breakdown_lower.get("critical", 0)
    high_count = breakdown_lower.get("high", 0)
    med_count = breakdown_lower.get("medium", 0)
    low_count = breakdown_lower.get("low", 0)
    
    raw_values = [crit_count, high_count, med_count, low_count]
    total_count = sum(raw_values) or 1
    
    # 2. Build the mathematically exact percentage strings for display
    percent_texts = [
        f"{crit_count / total_count * 100:.1f}%",
        f"{high_count / total_count * 100:.2f}%",
        f"{med_count / total_count * 100:.3f}%",
        f"{low_count / total_count * 100:.3f}%"
    ]
    
    # 3. Only pad slices if they have a non-zero count. If count is 0, keep it 0!
    render_values = [
        crit_count,
        high_count,
        max(med_count, 12) if med_count > 0 else 0,
        max(low_count, 8) if low_count > 0 else 0
    ]
    
    labels = ["Critical", "High", "Medium", "Low"]
    colors = ["#EF4444", "#F97316", "#EAB308", "#10B981"]
    
    # 4. Build custom hover text using actual, unpadded transaction counts
    hover_texts = [
        f"<b>{label}</b><br>Incident Cases: {val}<br>Percentage: {pct}"
        for label, val, pct in zip(labels, raw_values, percent_texts)
    ]
    
    fig_donut = go.Figure(data=[go.Pie(
        labels=labels, 
        values=render_values,     # Padded strictly for visual rendering
        hole=.5,
        marker=dict(colors=colors, line=dict(color='#0F172A', width=2)),
        textinfo='text',        
        text=percent_texts,       # Exact mathematical percentages printed outside
        textposition='outside', 
        rotation=110,           
        hovertext=hover_texts,    # Strictly shows actual, unpadded database counts on hover!
        hoverinfo='text'          # Overrides default hover with our custom texts
    )])
    
    fig_donut.update_traces(
        automargin=True,
        pull=[0, 0, 0, 0.12]
    )

    fig_donut.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(
            font=dict(color="#94A3B8"),
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=40, b=80, l=80, r=80), 
        height=340,
        uniformtext=dict(mode="show", minsize=9)
    )
    
    st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})

with c2:
    st.markdown("### 📈 Fraud Pattern Frequency")
    
    patterns = list(fraud_distribution.keys())
    counts = list(fraud_distribution.values())
    
    df_bar = pd.DataFrame({"Pattern": patterns, "Cases Flagged": counts})
    df_bar = df_bar.sort_values(by="Cases Flagged", ascending=True)
    
    fig_bar = px.bar(
        df_bar,
        x="Cases Flagged",
        y="Pattern",
        orientation="h",
        color="Cases Flagged",
        color_continuous_scale=["#EAB308", "#F97316", "#EF4444"],
        text_auto=True
    )
    
    fig_bar.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color="#94A3B8",
        xaxis=dict(showgrid=False, title="Voucher Incidents Count"),
        yaxis=dict(title=None),
        coloraxis_showscale=False,
        margin=dict(t=10, b=10, l=10, r=10),
        height=320
    )
    
    st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# CFO PRIORITY AUDIT REMEDIATION QUEUE (TOP 10 LIVE QUEUE)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 📋 CFO Audit Remediation Queue (Top 10 High Priority)")

real_queue = (real_summary or {}).get("top_10_high_priority") or []
records = []
for item in real_queue:
    records.append({
        "TXN ID": item.get("transaction_id", ""),
        "Risk Score": item.get("risk_score", 100),
        "Severity": item.get("severity", "CRITICAL"),
        "Recommendation": (item.get("recommended_action") or "").split("; ")[0]
    })
if not records:
    for idx, item in enumerate((real_report or [])[:10]):
        records.append({
            "TXN ID": item.get("transaction_id", f"TXN{idx}"),
            "Risk Score": item.get("risk_score", 90),
            "Severity": item.get("risk_severity", "CRITICAL"),
            "Recommendation": (item.get("recommended_action") or "").split("; ")[0]
        })
top_10_queue = pd.DataFrame(records)

# Render standard clean tabular spreadsheet datagrid layout
st.dataframe(
    top_10_queue,
    use_container_width=True,
    hide_index=True,
    column_config={
        "TXN ID": st.column_config.TextColumn("Transaction ID", width="small"),
        "Risk Score": st.column_config.ProgressColumn("Risk Score", min_value=0, max_value=100, format="%d"),
        "Severity": st.column_config.TextColumn("Severity Status", width="small"),
        "Recommendation": st.column_config.TextColumn("Auditor Action Directives", width="large")
    }
)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FORENSIC TRANSACTION SEARCH PANEL (100% PRODUCTION-ONLY)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 🔍 Forensic Ledger Explorer")
search_id = st.text_input("Search Transaction ID:", value="TXN313", placeholder="e.g. TXN313, TXN773, TXN607...").strip()

if search_id:
    # Search strictly in the active live database list
    matched_tx = [t for t in (real_report or []) if (t.get("transaction_id") or "").upper() == search_id.upper()]
    
    if matched_tx:
        tx = matched_tx[0]
        severity = tx.get("risk_severity", "HIGH")
        dot_class = "dot-critical" if severity == "CRITICAL" else "dot-high" if severity == "HIGH" else "dot-medium"
        color = "#EF4444" if severity == "CRITICAL" else "#F97316" if severity == "HIGH" else "#EAB308"
        icon = "🚨" if severity in ["CRITICAL", "HIGH"] else "🔍"
        
        v_list = [f"[{v.get('rule_id')}] - {v.get('description')}" for v in (tx.get("violations") or []) if v]
        v_str = " · ".join(v_list) if v_list else "None (No hard-rule violations)"
        
        anom_details = tx.get("anomaly_details") or {}
        anom_list = []
        if anom_details.get("zscore_flagged"): anom_list.append(f"Z-Score ({anom_details.get('zscore_reason')})")
        if anom_details.get("iqr_flagged"): anom_list.append(f"IQR Outlier ({anom_details.get('iqr_reason')})")
        if anom_details.get("groupwise_flagged"):
            reasons = [g.get('case') for g in (anom_details.get('groupwise_anomalies') or []) if g]
            anom_list.append(f"Groupwise ({', '.join(reasons)})")
        if anom_details.get("isolation_forest_flagged"): anom_list.append("Isolation Forest (ML)")
        
        anom_str = " · ".join(anom_list) if anom_list else "None (No statistical outliers)"
        
        # Dynamic suspected fraud patterns extraction based on flagged indicators
        patterns_list = []
        groupwise_list = anom_details.get("groupwise_anomalies") or []
        for g in groupwise_list:
            case_name = g.get("case")
            if case_name:
                clean_name = case_name.replace(" Flag", "").replace(" Outlier", "").strip()
                if clean_name not in patterns_list:
                    patterns_list.append(clean_name)
        
        for v in (tx.get("violations") or []):
            desc = v.get("description") or ""
            rule_id = v.get("rule_id") or ""
            if "refund" in desc.lower() or "over-collection" in desc.lower() or rule_id in ["RV-14", "RV-15"]:
                if "Outflow / Refund Exposure" not in patterns_list:
                    patterns_list.append("Outflow / Refund Exposure")
            elif "duplicate" in desc.lower() or rule_id in ["RV-02", "RV-34", "RV-35"]:
                if "Duplicate Ledger Posting" not in patterns_list:
                    patterns_list.append("Duplicate Ledger Posting")
            elif "gap" in desc.lower() or rule_id == "RV-08":
                if "Sequential Gap Outlier" not in patterns_list:
                    patterns_list.append("Sequential Gap Outlier")
            elif "missing" in desc.lower() or rule_id in ["RV-01", "RV-03", "RV-04", "RV-05"]:
                if "Missing Identity Anchor" not in patterns_list:
                    patterns_list.append("Missing Identity Anchor")
        
        if anom_details.get("isolation_forest_flagged"):
            if "Isolation Forest (ML Outlier)" not in patterns_list:
                patterns_list.append("Isolation Forest (ML Outlier)")
        if anom_details.get("zscore_flagged"):
            if "Statistical Z-Score Outlier" not in patterns_list:
                patterns_list.append("Statistical Z-Score Outlier")
        if anom_details.get("iqr_flagged"):
            if "IQR Deviation Outlier" not in patterns_list:
                patterns_list.append("IQR Deviation Outlier")
        
        if not patterns_list:
            patterns_list.append("Standard Policy Outlier")
            
        patterns_html = "".join([f"<li><strong>{p}:</strong> Flagged transaction anomaly</li>" for p in patterns_list])
        
        # Display detailed 5-row table with the separate analysis block printed below it
        st.markdown(f"""
        <div class="forensic-sheet">
            <h4 style="color:#F8FAFC; margin-bottom:12px; font-family:'Outfit';">{icon} Forensic Profile Sheet: {tx.get('transaction_id') or 'N/A'} (Production Ledger)</h4>
            <table style="width:100%; border-collapse:collapse; color:#E2E8F0; font-size:13px; font-family:sans-serif;">
                <tr style="border-bottom: 1px solid #1E293B; height:32px;">
                    <td style="color:#94A3B8; font-weight:600; width:25%;">Customer ID Reference</td>
                    <td><strong>{tx.get('customer_id') or 'N/A (Missing)'}</strong></td>
                </tr>
                <tr style="border-bottom: 1px solid #1E293B; height:32px;">
                    <td style="color:#94A3B8; font-weight:600;">Risk Severity Score</td>
                    <td><span class="dot {dot_class}"></span><strong style="color:{color};">{tx.get('risk_score') or 0} / 100 ({severity})</strong></td>
                </tr>
                <tr style="border-bottom: 1px solid #1E293B; height:32px;">
                    <td style="color:#94A3B8; font-weight:600;">Sub-engine Hard Violations</td>
                    <td style="color:#F87171;">{v_str}</td>
                </tr>
                <tr style="border-bottom: 1px solid #1E293B; height:32px;">
                    <td style="color:#94A3B8; font-weight:600;">Ledger Outliers</td>
                    <td style="color:#FCD34D;">{anom_str}</td>
                </tr>
                <tr style="border-bottom: 1px solid #1E293B; height:32px;">
                    <td style="color:#94A3B8; font-weight:600;">Action Directive</td>
                    <td><em>{tx.get('recommended_action') or 'No recommendation registered'}</em></td>
                </tr>
            </table>
            <br>
            <h5 style="color:#F8FAFC; margin-bottom:8px; font-family:'Outfit';">📊 Suspected Fraud Pattern Analysis:</h5>
            <ul>
                {patterns_html}
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ Transaction ID '{search_id}' not found in active Production Audit logs. Try searching for an active transaction like 'TXN313' or 'TXN773'.")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# EXPORT CENTER (PRODUCTION DOWNLOADS ONLY)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 💾 Stakeholder Export Center")
st.caption("Instantly output audited reports in corporate-ready CSV, JSON, or text summary sheets.")
st.markdown("<br>", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def get_json_bytes(summary_data):
    import json
    return json.dumps(summary_data, indent=4).encode('utf-8')

@st.cache_data(show_spinner=False)
def get_csv_bytes(report_data):
    import pandas as pd
    import io
    df_temp = pd.DataFrame(report_data or [])
    buffer = io.StringIO()
    df_temp.to_csv(buffer, index=False)
    return buffer.getvalue().encode('utf-8')

@st.cache_data(show_spinner=False)
def get_txt_bytes(summary_data):
    txt = generate_ascii_report_text(summary_data)
    return txt.encode('utf-8')

e1, e2, e3 = st.columns(3)

with e1:
    json_bytes = get_json_bytes(real_summary)
    st.download_button(
        label="📥 Download JSON Report (.json)",
        data=json_bytes,
        file_name="risk_report.json",
        mime="application/json"
    )

with e2:
    csv_bytes = get_csv_bytes(real_report)
    st.download_button(
        label="📥 Download CSV Ledger (.csv)",
        data=csv_bytes,
        file_name="risk_report.csv",
        mime="text/csv"
    )

with e3:
    summary_txt_bytes = get_txt_bytes(real_summary)
    st.download_button(
        label="📥 Download Executive Summary (.txt)",
        data=summary_txt_bytes,
        file_name="summary_report.txt",
        mime="text/plain"
    )

st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("OPTOxCRM Compliance Operations. Generated automatically on secure transaction schedules.")