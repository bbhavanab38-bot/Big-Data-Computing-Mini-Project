"""Reusable enterprise UI components for the Banking Retention Intelligence Platform.

No raw Streamlit styling hacks in views; all presentation logic is encapsulated here.
All HTML-emitting components route through `clean_html` / `render_html` to prevent
Markdown code block interpretation caused by leading indentation (4+ spaces) or blank lines.
"""
from __future__ import annotations

import html
import math
import re
from typing import List, Optional
import streamlit as st

from dashboard.ui.theme import (
    BORDER_COLOR,
    PRIMARY_ACCENT,
    SECONDARY_ACCENT,
    SUCCESS,
    WARNING,
    DANGER,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    get_custom_css,
)


def clean_html(html_str: str) -> str:
    """Clean HTML string to prevent Markdown code block or inline escaping artifacts.

    Collapses multi-line HTML into a single compact line without newlines,
    so that CommonMark / React-Markdown treats the entire markup as an atomic
    HTML block rather than breaking into inline markdown text on line breaks.
    """
    if not html_str:
        return ""
    if "<style" in html_str:
        lines = [line.strip() for line in str(html_str).splitlines()]
        return "\n".join([line for line in lines if line])

    collapsed = " ".join(line.strip() for line in str(html_str).splitlines() if line.strip())
    return re.sub(r">\s+<", "><", collapsed)


def render_html(html_str: str, container=None):
    """Render cleaned HTML safely without Markdown code block artifacts.

    Parameters
    ----------
    html_str : str
        HTML markup string (may be multi-line or indented).
    container : streamlit.delta_generator.DeltaGenerator, optional
        Target Streamlit container (e.g. st, st.sidebar, col). Defaults to st.
    """
    target = container if container is not None else st
    target.markdown(clean_html(html_str), unsafe_allow_html=True)


def apply_page_styles():
    """Inject global CSS rules for dark enterprise finance styling."""
    render_html(get_custom_css())


def status_pill(status: str = "LIVE", variant: str = "live") -> str:
    """Generate inline HTML for a live system status pill."""
    var = str(variant).strip().lower()
    if var in {"live", "success", "ok"}:
        status_class = "pill-live"
        dot_color = SUCCESS
    elif var in {"warn", "warning"}:
        status_class = "pill-warn"
        dot_color = WARNING
    else:
        status_class = "pill-neutral"
        dot_color = TEXT_SECONDARY

    return (
        f'<span class="pill-status {status_class}">'
        f'<span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: {dot_color};"></span>'
        f'{html.escape(status)}'
        f'</span>'
    )


def data_badge(text: str = "SYNTHETIC DATASET") -> str:
    """Generate inline HTML for an institutional governance dataset badge."""
    return (
        f'<span class="badge-synthetic" title="Project uses 10,000 synthetic banking customer records. Never present simulated outcomes as real data.">'
        f'<span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: {WARNING};"></span>'
        f'{html.escape(text)}'
        f'</span>'
    )


def sidebar_brand():
    """Render the official brand header in the sidebar."""
    badge = data_badge("SYNTHETIC DATASET")
    html_content = (
        f'<div style="padding: 0.5rem 0.25rem 1.25rem 0.25rem; border-bottom: 1px solid {BORDER_COLOR}; margin-bottom: 1rem;">'
        f'<div style="display: flex; align-items: center; gap: 0.65rem;">'
        f'<div style="background: linear-gradient(135deg, {PRIMARY_ACCENT} 0%, {SECONDARY_ACCENT} 100%); width: 28px; height: 28px; border-radius: 7px; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 0.95rem; color: #0B1120;">R</div>'
        f'<div>'
        f'<div style="font-size: 0.92rem; font-weight: 700; color: {TEXT_PRIMARY}; letter-spacing: -0.01em; line-height: 1.2;">RETENTION IQ</div>'
        f'<div style="font-size: 0.68rem; font-weight: 500; color: {TEXT_SECONDARY}; letter-spacing: 0.04em; text-transform: uppercase;">Banking Intelligence Platform</div>'
        f'</div>'
        f'</div>'
        f'<div style="margin-top: 0.75rem;">{badge}</div>'
        f'</div>'
    )
    render_html(html_content, container=st.sidebar)


def page_header(title: str, description: str, badges: Optional[List[str]] = None):
    """Render a clean top-level page header with optional metadata badges."""
    badge_html = ""
    if badges:
        badge_spans = "".join(f'<span class="badge-synthetic" style="margin-left: 0.5rem;">{html.escape(b)}</span>' for b in badges)
        badge_html = f'<div style="margin-top: 0.4rem;">{badge_spans}</div>'

    html_content = (
        f'<div style="margin-bottom: 1.25rem;">'
        f'<div style="display: flex; align-items: baseline; justify-content: space-between;">'
        f'<h1 style="font-size: 1.65rem; font-weight: 700; color: {TEXT_PRIMARY}; margin: 0; letter-spacing: -0.02em;">{html.escape(title)}</h1>'
        f'</div>'
        f'<div style="font-size: 0.85rem; color: {TEXT_SECONDARY}; margin-top: 0.25rem; line-height: 1.4;">{html.escape(description)}</div>'
        f'{badge_html}'
        f'</div>'
    )
    render_html(html_content)


def top_bar_system_status(model_loaded: bool, data_loaded: bool, total_rows: int = 10000) -> str:
    """Render top right system status indicators computed from actual runtime checks."""
    if model_loaded and data_loaded:
        pill = status_pill("LIVE", "live")
    else:
        pill = status_pill("WARN", "warn")

    badge = data_badge("SYNTHETIC DATA")

    return (
        f'<div style="display: flex; align-items: center; justify-content: flex-end; gap: 0.65rem;">'
        f'{pill}'
        f'{badge}'
        f'</div>'
    )


def kpi_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_basis: Optional[str] = None,
    tooltip: Optional[str] = None,
    delta_direction: str = "positive",
):
    """Render a compact, styled metric card without Markdown code block artifacts.

    delta_direction: 'positive' / 'normal' (green if positive, red if negative),
                     'negative_is_good' / 'inverse' (green if negative, red if positive),
                     'neutral' (muted grey).
    """
    delta_html = ""
    if delta is not None:
        direction = str(delta_direction).strip().lower()
        if direction in {"neutral", "none"}:
            d_class = "delta-neutral"
        elif direction in {"negative_is_good", "inverse"}:
            d_class = "delta-positive" if str(delta).startswith("-") else "delta-negative"
        else:
            d_class = "delta-positive" if (str(delta).startswith("+") or not str(delta).startswith("-")) else "delta-negative"

        basis_html = f'<span class="delta-basis">({html.escape(str(delta_basis))})</span>' if delta_basis else ""
        delta_html = f'<div class="kpi-delta {d_class}">{html.escape(str(delta))}{basis_html}</div>'

    tip_attr = f' title="{html.escape(tooltip)}"' if tooltip else ""
    info_icon = f'<span style="cursor: help; color: {TEXT_SECONDARY}; font-size: 0.75rem; margin-left: 0.35rem;">ⓘ</span>' if tooltip else ""

    html_content = (
        f'<div class="retention-card"{tip_attr}>'
        f'<div class="kpi-title"><span>{html.escape(label)}</span>{info_icon}</div>'
        f'<div class="kpi-value">{html.escape(value)}</div>'
        f'{delta_html}'
        f'</div>'
    )
    render_html(html_content)


def insight_banner(title: str, text: str, icon: str = "💡", badge: str = "SYSTEM INSIGHT"):
    """Render a highlighted dynamic insight banner."""
    html_content = (
        f'<div class="insight-banner">'
        f'<div class="insight-banner-icon">{icon}</div>'
        f'<div class="insight-banner-text">'
        f'<span class="insight-banner-badge">{html.escape(badge)}</span>'
        f'<strong>{html.escape(title)}:</strong> {html.escape(text)}'
        f'</div>'
        f'</div>'
    )
    render_html(html_content)


def section_header(title: str, subtitle: Optional[str] = None):
    """Render a section header with secondary subtext."""
    sub_html = f'<div class="section-subtitle">{html.escape(subtitle)}</div>' if subtitle else ""
    html_content = (
        f'<div style="margin-top: 1rem; margin-bottom: 0.5rem;">'
        f'<div class="section-title">{html.escape(title)}</div>'
        f'{sub_html}'
        f'</div>'
    )
    render_html(html_content)


def risk_chip(risk_category: str) -> str:
    """Return inline HTML string for a risk band chip (Low, Medium, High)."""
    cat = str(risk_category).strip().capitalize()
    if cat == "High":
        cls = "risk-high"
    elif cat == "Medium":
        cls = "risk-medium"
    else:
        cls = "risk-low"
    return f'<span class="risk-chip {cls}">{cat} Risk</span>'


def risk_gauge_svg(probability: float, risk_category: str) -> str:
    """Generate an SVG semicircle gauge indicating calibrated churn hazard."""
    prob_pct = max(0.0, min(100.0, float(probability) * 100.0))
    cat = str(risk_category).strip().capitalize()

    if cat == "High":
        color = DANGER
    elif cat == "Medium":
        color = WARNING
    else:
        color = SUCCESS

    # Angle calculation: 0% -> -90 deg, 100% -> +90 deg
    angle = -90 + (prob_pct / 100.0) * 180.0
    rad = math.radians(angle)
    needle_x = 100 + 55 * math.cos(rad)
    needle_y = 100 + 55 * math.sin(rad)

    return (
        f'<div style="text-align: center; padding: 0.5rem 0;">'
        f'<svg viewBox="0 0 200 125" style="width: 100%; max-width: 220px; margin: 0 auto; display: block;">'
        f'<path d="M 30 100 A 70 70 0 0 1 170 100" fill="none" stroke="rgba(148, 163, 184, 0.15)" stroke-width="14" stroke-linecap="round"/>'
        f'<path d="M 30 100 A 70 70 0 0 1 45 60" fill="none" stroke="{SUCCESS}" stroke-width="14" opacity="0.4"/>'
        f'<path d="M 45 60 A 70 70 0 0 1 85 32" fill="none" stroke="{WARNING}" stroke-width="14" opacity="0.4"/>'
        f'<path d="M 85 32 A 70 70 0 0 1 170 100" fill="none" stroke="{DANGER}" stroke-width="14" opacity="0.4"/>'
        f'<line x1="100" y1="100" x2="{needle_x:.1f}" y2="{needle_y:.1f}" stroke="{color}" stroke-width="3.5" stroke-linecap="round"/>'
        f'<circle cx="100" cy="100" r="6" fill="{TEXT_PRIMARY}"/>'
        f'</svg>'
        f'<div style="margin-top: -12px;">'
        f'<div style="font-size: 1.75rem; font-weight: 700; color: {color}; line-height: 1.1;">{prob_pct:.1f}%</div>'
        f'<div style="margin-top: 4px;">{risk_chip(risk_category)}</div>'
        f'<div style="font-size: 0.70rem; color: {TEXT_SECONDARY}; margin-top: 4px;">Calibrated Churn Probability</div>'
        f'</div>'
        f'</div>'
    )


def empty_state(message: str, icon: str = "ℹ️"):
    """Render an empty/no-results placeholder."""
    html_content = (
        f'<div class="retention-card" style="text-align: center; padding: 2.5rem 1rem;">'
        f'<div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>'
        f'<div style="font-size: 0.95rem; font-weight: 600; color: {TEXT_PRIMARY};">{html.escape(message)}</div>'
        f'<div style="font-size: 0.80rem; color: {TEXT_SECONDARY}; margin-top: 0.25rem;">Adjust search filters to display records.</div>'
        f'</div>'
    )
    render_html(html_content)


def render_in_cell_bar(value: float, min_val: float, max_val: float, color: str = PRIMARY_ACCENT) -> str:
    """Generate HTML for an in-cell mini progress bar."""
    pct = 0.0
    if max_val > min_val:
        pct = max(0.0, min(100.0, (value - min_val) / (max_val - min_val) * 100.0))
    return (
        f'<div style="display: flex; align-items: center; justify-content: space-between; gap: 0.5rem;">'
        f'<span style="font-size: 0.80rem; color: {TEXT_PRIMARY};">{value:.2f}</span>'
        f'<div class="bar-container" style="max-width: 70px;">'
        f'<div class="bar-fill" style="width: {pct:.1f}%; background-color: {color};"></div>'
        f'</div>'
        f'</div>'
    )


def pill_toggle(
    label: str,
    options: List[str],
    default: Optional[str] = None,
    key: Optional[str] = None,
) -> str:
    """Wrapper around st.pills or st.segmented_control with fallback."""
    if hasattr(st, "pills"):
        sel = st.pills(label, options, default=default, key=key)
        return sel if sel is not None else (default or options[0])
    elif hasattr(st, "segmented_control"):
        sel = st.segmented_control(label, options, default=default, key=key)
        return sel if sel is not None else (default or options[0])
    else:
        return st.radio(label, options, index=options.index(default) if default in options else 0, horizontal=True, key=key)
