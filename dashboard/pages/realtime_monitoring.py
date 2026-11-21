"""Real-Time Monitoring Page.

Displays streaming telemetry, engine heartbeats (Spark Structured Streaming vs Micro-Batch Fallback),
recent synthetic events, risk movement watchlist, customer trajectory charts, and event volume trends.
Auto-refreshes periodically using st.fragment with pause control.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Guarded bootstrap: ensure project root is on sys.path when script is executed directly
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import pandas as pd

from dashboard.core.streaming_engine import (
    get_stream_heartbeat,
    load_recent_stream_events,
    load_risk_snapshots,
    trigger_microbatch_cycle,
)
from dashboard.core.event_simulator import run_event_generator
from dashboard.ui.charts import (
    build_customer_risk_trajectory_chart,
    build_event_volume_timeseries_chart,
)
from dashboard.ui.components import (
    apply_page_styles,
    empty_state,
    kpi_card,
    page_header,
    section_header,
)


def render():
    apply_page_styles()
    page_header(
        title="Real-Time Streaming & Risk Monitoring",
        description="Event-time stream processing, micro-batch risk scoring, and customer health movement.",
        badges=["STRUCTURED STREAMING", "MICRO-BATCH", "REACTIVE RISK", "SIMULATED TELEMETRY"],
    )

    # Mandatory Simulation Disclaimer
    st.info(
        "⚠️ **Simulated Event Stream**: All streaming events, timestamps, and customer telemetry are synthetically "
        "generated for demonstration. No real banking events or external messaging integrations exist in this project."
    )

    # Top Control Bar: Stream Generation & Micro-batch trigger
    with st.expander("⚙️ Stream Simulation & Engine Controls", expanded=False):
        c1, c2, c3 = st.columns([1.5, 1.5, 2])
        with c1:
            if st.button("⚡ Generate 500 Simulated Events", type="primary", width="stretch"):
                with st.spinner("Synthesizing events with 20% deteriorating cohort..."):
                    n = run_event_generator(events=500, at_risk_ratio=0.20)
                    st.success(f"Generated {n:,} simulated events!")
                    st.rerun()

        with c2:
            if st.button("▶️ Execute Micro-Batch Cycle", width="stretch"):
                with st.spinner("Running micro-batch feature aggregation & model scoring..."):
                    hb = trigger_microbatch_cycle()
                    st.success(f"Processed {hb.get('events_processed', 0):,} events via `{hb.get('engine')}`!")
                    st.rerun()

        with c3:
            st.caption(
                "CLI Commands:\n"
                "- Event Generator: `python -m src.streaming.event_simulator --events 1000`\n"
                "- Micro-Batch Engine: `python -m src.streaming.microbatch_runner`\n"
                "- PySpark Streaming: `spark-submit spark/spark_streaming_risk.py`"
            )

    # Auto-refreshing Monitoring Dashboard via st.fragment
    render_monitoring_fragment()


@st.fragment(run_every=5)
def render_monitoring_fragment():
    """Streaming dashboard fragment auto-refreshing every 5 seconds."""
    heartbeat = get_stream_heartbeat()
    snapshots_df = load_risk_snapshots()
    events_df = load_recent_stream_events(limit=200)

    # Empty State check
    if heartbeat is None and snapshots_df.empty and events_df.empty:
        empty_state(
            title="Streaming Engine Idle / No Telemetry Detected",
            message="No stream events or risk snapshots are currently available. "
                    "Use the controls above or run 'python -m src.streaming.event_simulator' to start the stream.",
            icon="📡",
        )
        return

    # Engine Status & Telemetry KPIs
    engine_name = heartbeat.get("engine", "Pure-Python Micro-Batch Fallback") if heartbeat else "Pure-Python Micro-Batch Fallback"
    status = heartbeat.get("status", "IDLE") if heartbeat else "IDLE"
    last_time = heartbeat.get("last_batch_time", "N/A")[:19].replace("T", " ") if heartbeat else "N/A"
    events_proc = heartbeat.get("events_processed", len(events_df)) if heartbeat else len(events_df)
    throughput = heartbeat.get("throughput_eps", 0.0) if heartbeat else 0.0
    cust_tracked = len(snapshots_df) if not snapshots_df.empty else 0

    st.markdown("<br>", unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        kpi_card(
            label="Streaming Engine",
            value=engine_name.split()[0],  # "PySpark" or "Pure-Python"
            delta=f"Status: {status}",
            delta_basis=engine_name,
            delta_direction="normal" if status == "RUNNING" else "inverse",
        )

    with k2:
        kpi_card(
            label="Last Processed Batch",
            value=last_time.split()[1] if " " in last_time else last_time,
            delta=last_time.split()[0] if " " in last_time else "UTC",
            delta_basis="Processing timestamp",
        )

    with k3:
        kpi_card(
            label="Events Ingested",
            value=f"{events_proc:,}",
            delta=f"{throughput:,.1f} eps",
            delta_basis="Throughput (events / sec)",
        )

    with k4:
        kpi_card(
            label="Customers Monitored",
            value=f"{cust_tracked:,}",
            delta=f"{len(events_df):,} Recent Events",
            delta_basis="Dynamic customer profiles",
        )

    with k5:
        escalated_count = (
            int((snapshots_df["DeltaRisk"] > 0.10).sum()) if not snapshots_df.empty and "DeltaRisk" in snapshots_df.columns else 0
        )
        kpi_card(
            label="Risk Escalations (Δ > 10%)",
            value=f"{escalated_count:,}",
            delta=f"{float(escalated_count/cust_tracked*100):.1f}% of cohort" if cust_tracked > 0 else "0%",
            delta_basis="Accounts with rising churn risk",
            delta_direction="inverse" if escalated_count > 0 else "normal",
        )

    # Section 1: Risk Escalation Watchlist & Trajectory Chart
    st.markdown("<br>", unsafe_allow_html=True)
    section_header(
        title="Real-Time Risk Escalation Watchlist",
        subtitle="Identifies accounts whose churn hazard increased most vs baseline due to recent events",
    )

    if not snapshots_df.empty and "DeltaRisk" in snapshots_df.columns:
        w_col1, w_col2 = st.columns([1.3, 1.1])

        sorted_snaps = snapshots_df.sort_values("DeltaRisk", ascending=False).reset_index(drop=True)

        with w_col1:
            st.markdown("##### Highest Risk Increases")
            table_rows = []
            for _, r in sorted_snaps.head(15).iterrows():
                table_rows.append({
                    "CustomerId": int(r["CustomerId"]),
                    "Baseline": f"{r['BaselineProbability']*100:.1f}%",
                    "Current Risk": f"{r['StreamProbability']*100:.1f}%",
                    "Risk Change": f"{r['DeltaRisk']*100:+.1f}%",
                    "Category": r["RiskCategory"],
                    "Recent Tx": int(r.get("window_tx_count", 0)),
                    "Complaints": int(r.get("window_complaints", 0)),
                    "Balance": f"€{r.get('Balance', 0):,.0f}",
                })
            st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)

        with w_col2:
            st.markdown("##### Customer Risk Movement Trajectory")
            top_candidates = sorted_snaps["CustomerId"].tolist()[:20]
            selected_cid = st.selectbox(
                "Select Customer to Inspect Trajectory",
                options=top_candidates,
                format_func=lambda x: f"Customer #{x} (Δ: {sorted_snaps[sorted_snaps['CustomerId']==x]['DeltaRisk'].iloc[0]*100:+.1f}%)",
                index=0,
            )

            cust_row = sorted_snaps[sorted_snaps["CustomerId"] == selected_cid].iloc[0]
            fig_traj = build_customer_risk_trajectory_chart(
                baseline_risk=float(cust_row["BaselineProbability"]),
                current_risk=float(cust_row["StreamProbability"]),
                customer_id=selected_cid,
            )
            st.plotly_chart(fig_traj, width="stretch")

    # Section 2: Event Volume Over Time & Recent Stream Log
    st.markdown("---")
    section_header(
        title="Streaming Event Telemetry",
        subtitle="Recent simulated event stream arrivals categorized by interaction type",
    )

    e_col1, e_col2 = st.columns([1.2, 1.2])

    with e_col1:
        st.markdown("##### Event Arrivals Over Time")
        if not events_df.empty:
            fig_vol = build_event_volume_timeseries_chart(events_df)
            st.plotly_chart(fig_vol, width="stretch")
        else:
            st.info("No events loaded.")

    with e_col2:
        st.markdown("##### Live Ingestion Stream (Recent 20 Events)")
        if not events_df.empty:
            display_events = []
            for _, ev in events_df.head(20).iterrows():
                payload = ev.get("payload", {})
                detail_str = ""
                if isinstance(payload, dict):
                    if "amount" in payload:
                        detail_str = f"Amount: €{payload['amount']:,.2f} ({payload.get('merchant', '')})"
                    elif "category" in payload:
                        detail_str = f"Complaint: {payload.get('category')} ({payload.get('severity', '')})"
                    elif "channel" in payload:
                        detail_str = f"Login via {payload.get('channel')}"
                    elif "change_amount" in payload:
                        detail_str = f"Balance Change: €{payload.get('change_amount'):,.2f} ({payload.get('reason', '')})"

                display_events.append({
                    "Time (UTC)": str(ev.get("timestamp", ""))[:19].replace("T", " "),
                    "CustomerId": ev.get("customer_id"),
                    "Type": str(ev.get("event_type", "")).replace("_", " ").title(),
                    "Details": detail_str,
                    "Source": ev.get("source", "simulated"),
                })
            st.dataframe(pd.DataFrame(display_events), width="stretch", hide_index=True)


if __name__ == "__main__":
    render()
