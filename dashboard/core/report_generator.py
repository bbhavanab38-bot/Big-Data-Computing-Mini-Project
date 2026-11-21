"""Pure logic for generating self-contained Executive Summary HTML reports.
No Streamlit dependencies allowed.
"""
from datetime import datetime
from typing import Dict, Optional
import pandas as pd


def generate_executive_report_html(
    kpis: Dict[str, float],
    python_models: pd.DataFrame,
    spark_models: Optional[pd.DataFrame],
    segment_profiles: Optional[pd.DataFrame],
    top_drivers: Optional[pd.DataFrame],
    high_risk_count: int,
    assumptions: Optional[Dict[str, float]] = None,
) -> str:
    """Generate a clean, standalone, printable one-page HTML executive briefing."""
    generated_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    total = int(kpis.get("total", 0))
    churned = int(kpis.get("churned", 0))
    retained = int(kpis.get("retained", 0))
    churn_rate = float(kpis.get("churn_rate", 0.0))
    avg_score = float(kpis.get("avg_probability", 0.0))

    # Format Python models table
    py_rows = ""
    if python_models is not None and not python_models.empty:
        for _, r in python_models.iterrows():
            model_name = r.get("Model", "Unknown")
            acc = float(r.get("Accuracy", 0)) * (100 if float(r.get("Accuracy", 0)) <= 1 else 1)
            prec = float(r.get("Precision", 0)) * (100 if float(r.get("Precision", 0)) <= 1 else 1)
            rec = float(r.get("Recall", 0)) * (100 if float(r.get("Recall", 0)) <= 1 else 1)
            f1 = float(r.get("F1 Score", 0)) * (100 if float(r.get("F1 Score", 0)) <= 1 else 1)
            auc = float(r.get("ROC-AUC", 0))
            py_rows += f"""
            <tr>
                <td style="font-weight:600;">{model_name}</td>
                <td>{acc:.2f}%</td>
                <td>{prec:.2f}%</td>
                <td>{rec:.2f}%</td>
                <td>{f1:.2f}%</td>
                <td style="font-weight:600; color:#2f6655;">{auc:.4f}</td>
            </tr>"""

    # Format Spark models table
    spark_rows = ""
    if spark_models is not None and not spark_models.empty:
        for _, r in spark_models.iterrows():
            model_name = r.get("Model", "Unknown")
            acc = float(r.get("Accuracy", 0)) * 100
            prec = float(r.get("ChurnPrecision", 0)) * 100
            rec = float(r.get("ChurnRecall", 0)) * 100
            f1 = float(r.get("ChurnF1", 0)) * 100
            auc = float(r.get("ROC_AUC", 0))
            spark_rows += f"""
            <tr>
                <td style="font-weight:600;">{model_name}</td>
                <td>{acc:.2f}%</td>
                <td>{prec:.2f}%</td>
                <td>{rec:.2f}%</td>
                <td>{f1:.2f}%</td>
                <td style="font-weight:600; color:#2f6655;">{auc:.4f}</td>
            </tr>"""

    # Format Segment profiles table
    seg_rows = ""
    if segment_profiles is not None and not segment_profiles.empty:
        for _, r in segment_profiles.iterrows():
            seg_id = int(r.get("Segment", 0))
            seg_name = r.get("SegmentName", f"Segment {seg_id}")
            cust = int(r.get("Customers", 0))
            share = float(r.get("CustomerSharePct", 0.0))
            age = float(r.get("AvgAge", 0.0))
            bal = float(r.get("AvgBalance", 0.0))
            act = float(r.get("AvgActivity", 0.0))
            crate = float(r.get("ChurnRatePct", 0.0))
            risk = r.get("RiskLevel", "Normal")
            seg_rows += f"""
            <tr>
                <td><strong>#{seg_id} {seg_name}</strong></td>
                <td>{cust:,} ({share:.1f}%)</td>
                <td>{age:.1f} yrs</td>
                <td>${bal:,.0f}</td>
                <td>{act:.3f}</td>
                <td style="font-weight:600;">{crate:.2f}%</td>
                <td><span class="badge badge-{risk.lower()}">{risk}</span></td>
            </tr>"""

    # Format Top Drivers
    drivers_rows = ""
    if top_drivers is not None and not top_drivers.empty:
        for _, r in top_drivers.head(6).iterrows():
            feat = r.get("Feature", "N/A")
            imp = float(r.get("ImportanceMean", 0.0))
            drivers_rows += f"""
            <tr>
                <td>{feat}</td>
                <td><strong>{imp:.4f}</strong> (drop in ROC-AUC if shuffled)</td>
            </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Executive Briefing — Banking Churn Analytics</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #20342f;
        background: #fbfcf9;
        margin: 0;
        padding: 32px;
        line-height: 1.5;
    }}
    .container {{
        max-width: 960px;
        margin: 0 auto;
        background: #ffffff;
        padding: 40px;
        border-radius: 12px;
        border: 1px solid #e1e7e1;
        box-shadow: 0 4px 16px rgba(0,0,0,0.04);
    }}
    .header {{
        border-bottom: 2px solid #2f6655;
        padding-bottom: 16px;
        margin-bottom: 24px;
    }}
    .eyebrow {{
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #2f6655;
    }}
    h1 {{
        font-size: 26px;
        margin: 6px 0 4px 0;
        color: #162a25;
    }}
    .meta {{
        font-size: 13px;
        color: #60736c;
    }}
    .disclaimer-banner {{
        background: #f4f7f4;
        border-left: 4px solid #2f6655;
        padding: 12px 16px;
        font-size: 12px;
        color: #29443c;
        margin-bottom: 28px;
        border-radius: 0 6px 6px 0;
    }}
    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 32px;
    }}
    .kpi-card {{
        background: #f8faf8;
        border: 1px solid #dfe5df;
        border-top: 3px solid #2f6655;
        border-radius: 8px;
        padding: 14px 16px;
    }}
    .kpi-label {{
        font-size: 12px;
        color: #556962;
        font-weight: 600;
        text-transform: uppercase;
    }}
    .kpi-value {{
        font-size: 24px;
        font-weight: 700;
        color: #2f6655;
        margin-top: 4px;
    }}
    .section-title {{
        font-size: 17px;
        font-weight: 700;
        color: #1b332c;
        margin-top: 28px;
        margin-bottom: 12px;
        border-bottom: 1px solid #edf2ed;
        padding-bottom: 6px;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        margin-bottom: 20px;
    }}
    th {{
        text-align: left;
        background: #f1f5f1;
        color: #2b453d;
        padding: 8px 12px;
        border: 1px solid #e0e6e0;
        font-weight: 600;
    }}
    td {{
        padding: 8px 12px;
        border: 1px solid #e8ede8;
    }}
    .badge {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
    }}
    .badge-critical {{ background: #fde8e8; color: #9b1c1c; }}
    .badge-elevated {{ background: #fef08a; color: #854d0e; }}
    .badge-moderate {{ background: #fed7aa; color: #9a3412; }}
    .badge-low {{ background: #def7ec; color: #03543f; }}
    .notes-box {{
        background: #fafbfa;
        border: 1px solid #e5ebe5;
        border-radius: 8px;
        padding: 16px;
        font-size: 12px;
        color: #445650;
        margin-top: 32px;
    }}
    @media print {{
        body {{ padding: 0; background: #fff; }}
        .container {{ border: none; box-shadow: none; padding: 0; }}
    }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <div class="eyebrow">Enterprise Banking Intelligence • YARN Analytics</div>
        <h1>Finance & Banking Churn Risk Executive Briefing</h1>
        <div class="meta">Generated: {generated_time} | Stack: Hadoop 3.5.0 • PySpark 3.5.9 on YARN • scikit-learn / XGBoost</div>
    </div>

    <div class="disclaimer-banner">
        <strong>Academic & Architecture Notice:</strong> Customer dataset comprises <strong>10,000 synthetic customer records</strong>. All reported performance indicators and ROI simulations represent controlled test benchmarks on single-node YARN infrastructure.
    </div>

    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Total Portfolio</div>
            <div class="kpi-value">{total:,}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Churn Rate</div>
            <div class="kpi-value">{churn_rate:.2f}%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">High-Risk Count</div>
            <div class="kpi-value">{high_risk_count:,}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Mean Churn Score</div>
            <div class="kpi-value">{avg_score:.4f}</div>
        </div>
    </div>

    <div class="section-title">1. Machine Learning Performance Benchmark</div>
    <div style="font-size:12px; color:#50645d; margin-bottom:6px;"><strong>Track A: Python / scikit-learn (Held-Out Test Set)</strong></div>
    <table>
        <thead>
            <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1 Score</th>
                <th>ROC-AUC</th>
            </tr>
        </thead>
        <tbody>
            {py_rows}
        </tbody>
    </table>

    <div style="font-size:12px; color:#50645d; margin-bottom:6px;"><strong>Track B: Spark MLlib on YARN (Distributed Test Evaluation)</strong></div>
    <table>
        <thead>
            <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>Churn Precision</th>
                <th>Churn Recall</th>
                <th>Churn F1</th>
                <th>ROC-AUC</th>
            </tr>
        </thead>
        <tbody>
            {spark_rows}
        </tbody>
    </table>

    <div class="section-title">2. Primary Churn Drivers (Permutation Importance)</div>
    <table>
        <thead>
            <tr>
                <th>Feature / Driver</th>
                <th>Permutation Impact</th>
            </tr>
        </thead>
        <tbody>
            {drivers_rows}
        </tbody>
    </table>

    <div class="section-title">3. Behavioral Segmentation (Spark K-Means Clusters)</div>
    <table>
        <thead>
            <tr>
                <th>Cluster Archetype</th>
                <th>Customer Volume</th>
                <th>Avg Age</th>
                <th>Avg Balance</th>
                <th>Activity Score</th>
                <th>Churn Rate</th>
                <th>Risk Level</th>
            </tr>
        </thead>
        <tbody>
            {seg_rows}
        </tbody>
    </table>

    <div class="notes-box">
        <strong>Strategic Recommendations & Project Boundaries:</strong>
        <ul style="margin: 6px 0 0 0; padding-left: 20px;">
            <li><strong>Priority Intervention:</strong> Focus outreach on Cluster #0 (Inactive High-Balance Disengaged), which exhibits the highest portfolio churn risk (22.64%) despite holding over $71,000 average deposit balance.</li>
            <li><strong>Class Weighting vs Raw Probabilities:</strong> Risk scores reflect class-weighted loss optimization for sensitivity to churners, not uncalibrated natural probabilities.</li>
            <li><strong>Cluster Configuration:</strong> Single-node YARN ResourceManager and NodeManager operating over HDFS block storage with client-mode driver execution.</li>
        </ul>
    </div>
</div>
</body>
</html>"""
    return html

