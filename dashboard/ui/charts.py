"""Plotly Chart Templates and Builders for the Banking Retention Intelligence Platform.

Consistent with the dark finance command center theme.
"""
from typing import Dict, List, Optional, Any, Tuple
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from dashboard.ui.theme import (
    CARD_BG,
    PRIMARY_ACCENT,
    SECONDARY_ACCENT,
    SUCCESS,
    WARNING,
    DANGER,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    PALETTE_CHART,
)


def apply_theme_to_fig(fig: go.Figure, title: Optional[str] = None) -> go.Figure:
    """Apply the standardized enterprise dark finance theme to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            color=TEXT_PRIMARY,
            size=12,
        ),
        title=dict(
            text=title or "",
            font=dict(size=13, color=TEXT_PRIMARY),
            x=0.01,
            xanchor="left",
            y=0.98,
        ) if title else None,
        margin=dict(l=20, r=20, t=50 if title else 20, b=55),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(148, 163, 184, 0.12)",
            borderwidth=1,
            font=dict(size=10, color=TEXT_SECONDARY),
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
        ),
        hoverlabel=dict(
            bgcolor=CARD_BG,
            bordercolor=PRIMARY_ACCENT,
            font=dict(size=12, color=TEXT_PRIMARY),
        ),
    )
    fig.update_xaxes(
        gridcolor="rgba(148, 163, 184, 0.08)",
        zerolinecolor="rgba(148, 163, 184, 0.15)",
        tickfont=dict(size=11, color=TEXT_SECONDARY),
        title_font=dict(size=12, color=TEXT_SECONDARY),
    )
    fig.update_yaxes(
        gridcolor="rgba(148, 163, 184, 0.08)",
        zerolinecolor="rgba(148, 163, 184, 0.15)",
        tickfont=dict(size=11, color=TEXT_SECONDARY),
        title_font=dict(size=12, color=TEXT_SECONDARY),
    )
    return fig


def create_donut_chart(
    labels: List[str],
    values: List[float],
    title: Optional[str] = None,
    colors: Optional[List[str]] = None,
    hole: float = 0.65,
) -> go.Figure:
    """Create a sleek donut chart."""
    theme_colors = colors or PALETTE_CHART
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=hole,
                marker=dict(colors=theme_colors, line=dict(color="#0B1120", width=2)),
                textinfo="percent",
                hoverinfo="label+value+percent",
                textfont=dict(size=12, color=TEXT_PRIMARY),
            )
        ]
    )
    return apply_theme_to_fig(fig, title=title)


def create_bar_chart(
    x: List[Any],
    y: List[float],
    title: Optional[str] = None,
    color: str = PRIMARY_ACCENT,
    x_title: Optional[str] = None,
    y_title: Optional[str] = None,
    text_template: str = "%{y}",
) -> go.Figure:
    """Create a vertical bar chart with rounded corners."""
    fig = go.Figure(
        data=[
            go.Bar(
                x=x,
                y=y,
                marker=dict(
                    color=color,
                    line=dict(color="rgba(255,255,255,0.05)", width=1),
                ),
                text=y,
                texttemplate=text_template,
                textposition="outside",
                textfont=dict(size=11, color=TEXT_SECONDARY),
            )
        ]
    )
    fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)
    return apply_theme_to_fig(fig, title=title)


def create_horizontal_bar_chart(
    y: List[Any],
    x: List[float],
    title: Optional[str] = None,
    color: str = PRIMARY_ACCENT,
    x_title: Optional[str] = None,
    y_title: Optional[str] = None,
    text_template: str = "%{x}",
) -> go.Figure:
    """Create a clean horizontal bar chart."""
    fig = go.Figure(
        data=[
            go.Bar(
                y=y,
                x=x,
                orientation="h",
                marker=dict(
                    color=color,
                    line=dict(color="rgba(255,255,255,0.05)", width=1),
                ),
                text=x,
                texttemplate=text_template,
                textposition="outside",
                textfont=dict(size=11, color=TEXT_SECONDARY),
            )
        ]
    )
    fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)
    return apply_theme_to_fig(fig, title=title)


def create_funnel_chart(
    stages: List[str],
    values: List[float],
    title: Optional[str] = None,
) -> go.Figure:
    """Create a funnel chart representing portfolio risk attrition."""
    fig = go.Figure(
        go.Funnel(
            y=stages,
            x=values,
            textinfo="value+percent initial",
            marker=dict(
                color=[
                    PRIMARY_ACCENT,
                    "#60A5FA",
                    WARNING,
                    "#F97316",
                    DANGER,
                ],
                line=dict(color="#0B1120", width=1.5),
            ),
            textfont=dict(size=12, color=TEXT_PRIMARY),
        )
    )
    return apply_theme_to_fig(fig, title=title)


def create_histogram(
    values: List[float],
    title: Optional[str] = None,
    nbins: int = 25,
    color: str = PRIMARY_ACCENT,
    x_title: Optional[str] = None,
) -> go.Figure:
    """Create an activity or balance distribution histogram."""
    fig = go.Figure(
        data=[
            go.Histogram(
                x=values,
                nbinsx=nbins,
                marker=dict(color=color, line=dict(color="#0B1120", width=1)),
            )
        ]
    )
    fig.update_layout(xaxis_title=x_title, yaxis_title="Customer Count")
    return apply_theme_to_fig(fig, title=title)


def create_radar_chart(
    categories: List[str],
    series_dict: Dict[str, List[float]],
    title: Optional[str] = None,
) -> go.Figure:
    """Create a radar chart comparing normalized cluster metrics."""
    fig = go.Figure()
    colors = PALETTE_CHART

    for idx, (name, values) in enumerate(series_dict.items()):
        color = colors[idx % len(colors)]
        # Close the loop
        closed_cats = categories + [categories[0]]
        closed_vals = values + [values[0]]
        fig.add_trace(
            go.Scatterpolar(
                r=closed_vals,
                theta=closed_cats,
                fill="toself",
                name=name,
                line=dict(color=color, width=2),
                fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.15)",
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor="rgba(148, 163, 184, 0.15)",
                tickfont=dict(size=9, color=TEXT_SECONDARY),
            ),
            angularaxis=dict(
                gridcolor="rgba(148, 163, 184, 0.15)",
                tickfont=dict(size=11, color=TEXT_PRIMARY),
            ),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
    )
    return apply_theme_to_fig(fig, title=title)


def create_calibration_plot(
    cal_table_df: pd.DataFrame,
    title: str = "Empirical Reliability / Calibration Curve",
) -> go.Figure:
    """Create a reliability diagram comparing predicted probability to observed churn."""
    fig = go.Figure()

    # 45-degree ideal calibration diagonal
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect Calibration (y = x)",
            line=dict(color=TEXT_SECONDARY, dash="dash", width=1.5),
        )
    )

    if not cal_table_df.empty and "MeanPredictedProb" in cal_table_df.columns:
        fig.add_trace(
            go.Scatter(
                x=cal_table_df["MeanPredictedProb"],
                y=cal_table_df["ObservedChurnRate"],
                mode="lines+markers",
                name="Calibrated Production Model",
                line=dict(color=PRIMARY_ACCENT, width=2.5),
                marker=dict(size=8, color=PRIMARY_ACCENT, symbol="circle"),
            )
        )

    fig.update_layout(
        xaxis=dict(range=[0, 1], title="Mean Predicted Probability"),
        yaxis=dict(range=[0, 1], title="Observed Churn Rate"),
    )
    return apply_theme_to_fig(fig, title=title)


def create_roc_curve_plot(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    title: str = "Receiver Operating Characteristic (ROC)",
) -> go.Figure:
    """Create an interactive ROC curve plot."""
    fig = go.Figure()
    # Diagonal random guess
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance Baseline (AUC=0.50)",
            line=dict(color=TEXT_SECONDARY, dash="dash", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"Production Model (AUC={roc_auc:.4f})",
            line=dict(color=PRIMARY_ACCENT, width=2.5),
        )
    )
    fig.update_layout(
        xaxis=dict(title="False Positive Rate (1 - Specificity)", range=[0, 1]),
        yaxis=dict(title="True Positive Rate (Recall)", range=[0, 1]),
    )
    return apply_theme_to_fig(fig, title=title)


def create_pr_curve_plot(
    recall: np.ndarray,
    precision: np.ndarray,
    pr_auc: float,
    baseline: float = 0.164,
    title: str = "Precision-Recall (PR) Curve",
) -> go.Figure:
    """Create an interactive Precision-Recall curve plot."""
    fig = go.Figure()
    # Baseline horizontal line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[baseline, baseline],
            mode="lines",
            name=f"Base Rate ({baseline:.1%})",
            line=dict(color=TEXT_SECONDARY, dash="dash", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name=f"Production Model (PR-AUC={pr_auc:.4f})",
            line=dict(color=SECONDARY_ACCENT, width=2.5),
        )
    )
    fig.update_layout(
        xaxis=dict(title="Recall (Sensitivity)", range=[0, 1]),
        yaxis=dict(title="Precision (PPV)", range=[0, 1]),
    )
    return apply_theme_to_fig(fig, title=title)


def create_confusion_matrix_heatmap(
    tn: int,
    fp: int,
    fn: int,
    tp: int,
    title: str = "Confusion Matrix",
) -> go.Figure:
    """Create a confusion matrix heatmap."""
    z = [[tn, fp], [fn, tp]]
    x = ["Pred Retained (0)", "Pred Churned (1)"]
    y = ["Actual Retained (0)", "Actual Churned (1)"]

    annotations = []
    labels = [
        [f"TN: {tn:,}<br>{tn/(tn+fp):.1%}", f"FP: {fp:,}<br>{fp/(tn+fp):.1%}"],
        [f"FN: {fn:,}<br>{fn/(fn+tp):.1%}", f"TP: {tp:,}<br>{tp/(fn+tp):.1%}"],
    ]

    for i in range(2):
        for j in range(2):
            annotations.append(
                dict(
                    x=x[j],
                    y=y[i],
                    text=labels[i][j],
                    showarrow=False,
                    font=dict(color=TEXT_PRIMARY, size=9.5, weight=600),
                )
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale=[[0, "#172235"], [1, PRIMARY_ACCENT]],
            showscale=False,
        )
    )
    fig.update_layout(
        annotations=annotations,
        margin=dict(l=85, r=15, t=45 if title else 20, b=35),
    )
    return apply_theme_to_fig(fig, title=title)


def build_policy_comparison_bar(
    optimizer_data: Dict[str, Any],
    baseline_data: Dict[str, Any],
    title: str = "Policy Comparison: Uplift Optimizer vs. Standard Churn-Risk Targeting",
) -> go.Figure:
    """Grouped bar chart comparing Optimizer vs Churn-Risk Baseline under identical budget."""
    categories = ["Contacted Customers", "Incremental Retained", "Net Benefit (€)"]
    optim_vals = [
        optimizer_data.get("customers_contacted", 0),
        round(optimizer_data.get("expected_incremental_retained", 0), 1),
        round(optimizer_data.get("expected_net_benefit", 0), 0),
    ]
    base_vals = [
        baseline_data.get("customers_contacted", 0),
        round(baseline_data.get("expected_incremental_retained", 0), 1),
        round(baseline_data.get("expected_net_benefit", 0), 0),
    ]

    fig = go.Figure(data=[
        go.Bar(
            name="Uplift & Net Benefit Optimizer",
            x=categories,
            y=optim_vals,
            marker_color=PRIMARY_ACCENT,
            text=[f"{v:,.0f}" if i != 1 else f"{v:,.1f}" for i, v in enumerate(optim_vals)],
            textposition="auto",
        ),
        go.Bar(
            name="Top-N Churn Risk Baseline",
            x=categories,
            y=base_vals,
            marker_color=WARNING,
            text=[f"{v:,.0f}" if i != 1 else f"{v:,.1f}" for i, v in enumerate(base_vals)],
            textposition="auto",
        ),
    ])
    fig.update_layout(barmode="group")
    return apply_theme_to_fig(fig, title=title)


def build_qini_curve_chart(
    qini_res: Dict[str, Any],
    title: str = "Causal Uplift Gain Curve (Qini / AUUC Benchmark)",
) -> go.Figure:
    """Plot Qini cumulative incremental gain curve vs random baseline."""
    fractions = [f * 100 for f in qini_res.get("fractions", [])]
    qini_gains = qini_res.get("qini_gains", [])
    rand_gains = qini_res.get("random_gains", [])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0] + fractions,
        y=[0] + qini_gains,
        mode="lines+markers",
        name="Uplift Model (Causal Targeting)",
        line=dict(color=PRIMARY_ACCENT, width=3),
    ))
    fig.add_trace(go.Scatter(
        x=[0] + fractions,
        y=[0] + rand_gains,
        mode="lines",
        name="Random Targeting Baseline",
        line=dict(color=TEXT_SECONDARY, dash="dash", width=2),
    ))
    fig.update_layout(
        xaxis=dict(title="% of Customer Population Contacted", range=[0, 105]),
        yaxis=dict(title="Cumulative Incremental Customers Retained"),
    )
    return apply_theme_to_fig(fig, title=title)


def build_net_benefit_uncertainty_curve(
    curve_df: pd.DataFrame,
    title: str = "Campaign Net Benefit Curve with 90% Uncertainty Band",
) -> go.Figure:
    """Plot net benefit across contact volumes with 5th to 95th percentile confidence ribbon."""
    x = curve_df["ContactsCount"].tolist()
    y_mean = curve_df["ExpectedNetBenefit"].tolist()
    y_upper = curve_df["UpperBound_P95"].tolist()
    y_lower = curve_df["LowerBound_P5"].tolist()

    fig = go.Figure()

    # Upper bound (invisible line)
    fig.add_trace(go.Scatter(
        x=x,
        y=y_upper,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        name="95th Percentile",
    ))

    # Lower bound with shaded fill to upper
    fig.add_trace(go.Scatter(
        x=x,
        y=y_lower,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(56, 189, 248, 0.15)",
        name="90% Confidence Interval (P5 - P95)",
    ))

    # Expected Net Benefit mean line
    fig.add_trace(go.Scatter(
        x=x,
        y=y_mean,
        mode="lines+markers",
        line=dict(color=PRIMARY_ACCENT, width=3),
        name="Expected Net Benefit (€)",
    ))

    # Zero break-even line
    fig.add_hline(y=0, line_dash="dash", line_color=DANGER, annotation_text="Break-Even (€0)", annotation_position="bottom right")

    fig.update_layout(
        xaxis=dict(title="Contacted Customer Volume (N)"),
        yaxis=dict(title="Net Benefit (€)"),
    )
    return apply_theme_to_fig(fig, title=title)


def build_monte_carlo_distribution_chart(
    mc_res: Any,
    title: str = "5,000-Draw Monte Carlo Simulation: Net Benefit Distribution",
) -> go.Figure:
    """Histogram of simulated net benefit draws with P5, Median, and P95 markers."""
    draws = mc_res.net_benefit_draws

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=draws,
        nbinsx=50,
        marker_color="rgba(56, 189, 248, 0.65)",
        name="Simulated Net Benefit",
    ))

    # Add vertical markers for percentiles
    fig.add_vline(x=mc_res.percentile_5, line_dash="dash", line_color=DANGER, annotation_text=f"5th %: €{mc_res.percentile_5:,.0f}")
    fig.add_vline(x=mc_res.percentile_50, line_dash="solid", line_color=SUCCESS, annotation_text=f"Median: €{mc_res.percentile_50:,.0f}")
    fig.add_vline(x=mc_res.percentile_95, line_dash="dash", line_color=PRIMARY_ACCENT, annotation_text=f"95th %: €{mc_res.percentile_95:,.0f}")
    fig.add_vline(x=0, line_dash="dot", line_color=TEXT_SECONDARY, annotation_text="€0 Threshold")

    fig.update_layout(
        xaxis=dict(title="Simulated Campaign Net Benefit (€)"),
        yaxis=dict(title="Draw Frequency (out of 5,000)"),
    )
    return apply_theme_to_fig(fig, title=title)


def build_tornado_chart(
    sensitivity_dict: Dict[str, float],
    title: str = "Tornado Sensitivity Analysis: Key Risk Drivers",
) -> go.Figure:
    """Horizontal bar chart showing Spearman rank correlation of inputs with Net Benefit."""
    sorted_items = sorted(sensitivity_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    features = [item[0] for item in reversed(sorted_items)]
    corrs = [item[1] for item in reversed(sorted_items)]
    colors = [SUCCESS if c > 0 else DANGER for c in corrs]

    fig = go.Figure(go.Bar(
        x=corrs,
        y=features,
        orientation="h",
        marker_color=colors,
        text=[f"{c:+.2f} corr" for c in corrs],
        textposition="auto",
    ))
    fig.update_layout(
        xaxis=dict(title="Spearman Rank Correlation with Net Benefit", range=[-1.0, 1.0]),
    )
    return apply_theme_to_fig(fig, title=title)


def build_shap_waterfall_bar(
    drivers: List[Dict[str, Any]],
    title: str = "Individual Customer Risk Drivers (SHAP Log-Odds Contributions)",
) -> go.Figure:
    """Horizontal bar chart of SHAP drivers (red = risk increasing, green = risk decreasing)."""
    # Top 8 drivers by magnitude
    top_drivers = sorted(drivers, key=lambda d: abs(d.get("shap_value", 0.0)), reverse=True)[:8]
    top_drivers = list(reversed(top_drivers))

    y_labels = [f"{d.get('title', d.get('feature'))} ({d.get('customer_value')})" for d in top_drivers]
    x_vals = [d.get("shap_value", 0.0) for d in top_drivers]
    colors = [DANGER if v > 0 else SUCCESS for v in x_vals]

    fig = go.Figure(go.Bar(
        x=x_vals,
        y=y_labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in x_vals],
        textposition="auto",
    ))
    fig.add_vline(x=0, line_color=TEXT_SECONDARY, line_width=1)
    fig.update_layout(
        xaxis=dict(title="SHAP Value (Contribution to Log-Odds of Churn)"),
    )
    return apply_theme_to_fig(fig, title=title)


def build_fairness_metric_bar(
    group_names: List[str],
    selection_rates: List[float],
    tprs: List[float],
    fprs: List[float],
    selection_rate_cis: Optional[List[Tuple[float, float]]] = None,
    tpr_cis: Optional[List[Tuple[float, float]]] = None,
    fpr_cis: Optional[List[Tuple[float, float]]] = None,
    title: str = "Group Fairness Comparison: Selection Rate, TPR & FPR",
) -> go.Figure:
    """Group comparison bar chart with optional bootstrap 95% confidence interval error bars."""
    fig = go.Figure()

    # Selection rate trace
    sr_err_y = None
    if selection_rate_cis:
        sr_err_y = dict(
            type="data",
            symmetric=False,
            array=[hi - val for (lo, hi), val in zip(selection_rate_cis, selection_rates)],
            arrayminus=[val - lo for (lo, hi), val in zip(selection_rate_cis, selection_rates)],
            color="rgba(56, 189, 248, 0.6)",
        )

    fig.add_trace(go.Bar(
        name="Selection Rate (Intervention Rate)",
        x=group_names,
        y=selection_rates,
        error_y=sr_err_y,
        marker_color=PRIMARY_ACCENT,
        text=[f"{v*100:.1f}%" for v in selection_rates],
        textposition="auto",
    ))

    # TPR trace
    tpr_err_y = None
    if tpr_cis:
        tpr_err_y = dict(
            type="data",
            symmetric=False,
            array=[hi - val for (lo, hi), val in zip(tpr_cis, tprs)],
            arrayminus=[val - lo for (lo, hi), val in zip(tpr_cis, tprs)],
            color="rgba(16, 185, 129, 0.6)",
        )

    fig.add_trace(go.Bar(
        name="True Positive Rate (Recall)",
        x=group_names,
        y=tprs,
        error_y=tpr_err_y,
        marker_color=SUCCESS,
        text=[f"{v*100:.1f}%" for v in tprs],
        textposition="auto",
    ))

    # FPR trace
    fpr_err_y = None
    if fpr_cis:
        fpr_err_y = dict(
            type="data",
            symmetric=False,
            array=[hi - val for (lo, hi), val in zip(fpr_cis, fprs)],
            arrayminus=[val - lo for (lo, hi), val in zip(fpr_cis, fprs)],
            color="rgba(239, 68, 68, 0.6)",
        )

    fig.add_trace(go.Bar(
        name="False Positive Rate",
        x=group_names,
        y=fprs,
        error_y=fpr_err_y,
        marker_color=DANGER,
        text=[f"{v*100:.1f}%" for v in fprs],
        textposition="auto",
    ))

    fig.update_layout(
        barmode="group",
        yaxis=dict(title="Rate", range=[0, 1.05], tickformat=".0%"),
        xaxis=dict(title="Demographic Group"),
    )
    return apply_theme_to_fig(fig, title=title)


def build_fairness_comparison_grouped_bar(
    categories: List[str],
    original_gaps: List[float],
    excluded_gaps: List[float],
    threshold: float = 0.10,
    title: str = "Fairness Disparities: Baseline Model vs Attribute-Excluded Model",
) -> go.Figure:
    """Side-by-side grouped bar chart comparing demographic disparities before and after feature exclusion."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Baseline Model (All Features)",
        x=categories,
        y=original_gaps,
        marker_color=WARNING,
        text=[f"{v:.3f}" for v in original_gaps],
        textposition="auto",
    ))

    fig.add_trace(go.Bar(
        name="Retrained Model (Excluding Protected Features)",
        x=categories,
        y=excluded_gaps,
        marker_color=PRIMARY_ACCENT,
        text=[f"{v:.3f}" for v in excluded_gaps],
        textposition="auto",
    ))

    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color=DANGER,
        annotation_text=f"Compliance Warning Threshold ({threshold:.2f})",
        annotation_position="top left",
    )

    max_val = max(original_gaps + excluded_gaps) if (original_gaps or excluded_gaps) else 0.20
    fig.update_layout(
        barmode="group",
        yaxis=dict(title="Disparity Gap (Abs Difference)", range=[0, max(0.25, max_val * 1.15)]),
        xaxis=dict(title="Evaluated Disparity"),
    )
    return apply_theme_to_fig(fig, title=title)


def build_campaign_outcomes_bar(
    action_names: List[str],
    targets: List[int],
    contacted: List[int],
    responded: List[int],
    retained: List[int],
    title: str = "Campaign Execution Funnel by Action",
) -> go.Figure:
    """Grouped bar chart showing customer progression through campaign lifecycle by action."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Assigned Targets",
        x=action_names,
        y=targets,
        marker_color=TEXT_SECONDARY,
    ))

    fig.add_trace(go.Bar(
        name="Contacted",
        x=action_names,
        y=contacted,
        marker_color=PRIMARY_ACCENT,
    ))

    fig.add_trace(go.Bar(
        name="Responded",
        x=action_names,
        y=responded,
        marker_color=WARNING,
    ))

    fig.add_trace(go.Bar(
        name="Retained (Saved)",
        x=action_names,
        y=retained,
        marker_color=SUCCESS,
    ))

    fig.update_layout(
        barmode="group",
        yaxis=dict(title="Number of Customers"),
        xaxis=dict(title="Retention Intervention"),
    )
    return apply_theme_to_fig(fig, title=title)


def build_event_volume_timeseries_chart(
    events_df: pd.DataFrame,
    title: str = "Streaming Event Volumes Over Time by Type",
) -> go.Figure:
    """Stacked bar / line chart showing volume of streaming events bucketed by time."""
    fig = go.Figure()

    if events_df.empty or "timestamp" not in events_df.columns:
        return apply_theme_to_fig(fig, title="No streaming events available")

    df = events_df.copy()
    df["ts"] = pd.to_datetime(df["timestamp"])
    # Bucket by 6 hours or 1 day depending on range
    df["time_bucket"] = df["ts"].dt.floor("6h")

    type_colors = {
        "transaction": PRIMARY_ACCENT,
        "login": SUCCESS,
        "balance_change": WARNING,
        "complaint": DANGER,
    }

    for evt_type in ["transaction", "login", "balance_change", "complaint"]:
        sub = df[df["event_type"] == evt_type]
        if not sub.empty:
            counts = sub.groupby("time_bucket").size().reset_index(name="count")
            fig.add_trace(go.Bar(
                name=evt_type.replace("_", " ").title(),
                x=counts["time_bucket"],
                y=counts["count"],
                marker_color=type_colors.get(evt_type, TEXT_SECONDARY),
            ))

    fig.update_layout(
        barmode="stack",
        xaxis=dict(title="Event Timestamp (UTC)"),
        yaxis=dict(title="Event Count"),
    )
    return apply_theme_to_fig(fig, title=title)


def build_customer_risk_trajectory_chart(
    baseline_risk: float,
    current_risk: float,
    customer_id: int | str,
    title: Optional[str] = None,
) -> go.Figure:
    """Gauge or trajectory bar showing baseline risk vs real-time streaming risk escalation."""
    fig = go.Figure()

    labels = ["Initial Baseline Risk", "Current Streaming Risk"]
    risks = [baseline_risk, current_risk]
    colors = [
        PRIMARY_ACCENT,
        DANGER if current_risk >= 0.25 else (WARNING if current_risk >= 0.10 else SUCCESS),
    ]

    fig.add_trace(go.Bar(
        x=labels,
        y=risks,
        marker_color=colors,
        text=[f"{r*100:.1f}%" for r in risks],
        textposition="auto",
        width=[0.4, 0.4],
    ))

    # Add reference line for high risk threshold (25%)
    fig.add_hline(
        y=0.25,
        line_dash="dash",
        line_color=DANGER,
        annotation_text="High Risk Threshold (25%)",
        annotation_position="top left",
    )
    fig.add_hline(
        y=0.10,
        line_dash="dot",
        line_color=WARNING,
        annotation_text="Elevated Risk (10%)",
        annotation_position="bottom left",
    )

    chart_title = title or f"Customer #{customer_id}: Risk Escalation Trajectory"
    fig.update_layout(
        yaxis=dict(title="Calibrated Churn Probability", range=[0, 1.0], tickformat=".0%"),
        xaxis=dict(title="Scoring Horizon"),
    )
    return apply_theme_to_fig(fig, title=chart_title)


def build_drift_psi_horizontal_bar(
    drift_metrics: Dict[str, Any],
    title: str = "Population Stability Index (PSI) by Feature",
) -> go.Figure:
    """Horizontal bar chart of feature PSI values color-coded by regulatory drift thresholds."""
    features = []
    psi_values = []
    colors = []

    sorted_items = sorted(drift_metrics.items(), key=lambda x: getattr(x[1], "psi", 0.0), reverse=False)

    for feat_name, metric in sorted_items:
        psi = getattr(metric, "psi", 0.0)
        features.append(feat_name)
        psi_values.append(psi)

        if psi < 0.10:
            colors.append(SUCCESS)
        elif psi < 0.25:
            colors.append(WARNING)
        else:
            colors.append(DANGER)

    fig = go.Figure(go.Bar(
        x=psi_values,
        y=features,
        orientation="h",
        marker_color=colors,
        text=[f"{p:.3f}" for p in psi_values],
        textposition="auto",
    ))

    # Add vertical threshold lines
    fig.add_vline(x=0.10, line_dash="dash", line_color=WARNING, annotation_text="Moderate (0.10)")
    fig.add_vline(x=0.25, line_dash="dash", line_color=DANGER, annotation_text="Significant (0.25)")

    max_x = max(0.35, max(psi_values) * 1.15) if psi_values else 0.35
    fig.update_layout(
        xaxis=dict(title="Population Stability Index (PSI)", range=[0, max_x]),
        yaxis=dict(title="Feature"),
        height=max(450, len(features) * 22),
    )
    return apply_theme_to_fig(fig, title=title)


def build_activity_over_time_chart(
    events_df: pd.DataFrame,
    title: str = "Simulated Customer Event Activity Over Time",
) -> go.Figure:
    """Activity timeseries chart for the executive Overview page."""
    fig = go.Figure()

    if events_df.empty or "timestamp" not in events_df.columns:
        return apply_theme_to_fig(fig, title="Simulated Activity: No stream events generated yet")

    df = events_df.copy()
    df["ts"] = pd.to_datetime(df["timestamp"])
    daily = df.groupby(df["ts"].dt.floor("1D")).size().reset_index(name="event_count")

    fig.add_trace(go.Scatter(
        x=daily["ts"],
        y=daily["event_count"],
        mode="lines+markers",
        line=dict(color=PRIMARY_ACCENT, width=3),
        marker=dict(size=6, color=PRIMARY_ACCENT),
        fill="tozeroy",
        fillcolor="rgba(56, 189, 248, 0.1)",
        name="Daily Events",
    ))

    fig.update_layout(
        xaxis=dict(title="Date (UTC)"),
        yaxis=dict(title="Daily Events Logged"),
    )
    return apply_theme_to_fig(fig, title=title)


