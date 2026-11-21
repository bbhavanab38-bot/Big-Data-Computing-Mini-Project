# Phase 3 Design System & Visual Reference Comparison Report

## 1. Executive Summary & Design System Architecture
In Phase 3, the Banking Retention Intelligence Platform was upgraded with an enterprise-grade dark finance design system rooted in the layout, hierarchy, and density patterns of modern financial command centers.

The new product was built cleanly as an independent entrypoint at `dashboard/platform_app.py`, leaving the legacy `dashboard/app.py` and its test suite completely intact and passing.

### Design Tokens & Palette
- **Background**: `#0B1120` (Command Center Midnight)
- **Sidebar**: `#101827` (Deep Charcoal)
- **Card Containers**: `#172235` (Slate Navy, 1px border at `rgba(148, 163, 184, 0.12)`, radius 12px)
- **Secondary Background**: `#1D2B40`
- **Primary Accent**: `#38BDF8` (Sky Blue)
- **Secondary Accent**: `#8B5CF6` (Violet)
- **Success / Low Risk**: `#22C55E` (Emerald Green)
- **Warning / Medium Risk**: `#F59E0B` (Amber)
- **Danger / High Risk**: `#EF4444` (Rose Red)
- **Primary Text**: `#F8FAFC` (Snow White)
- **Secondary Text**: `#94A3B8` (Muted Slate)
- **Typography**: Inter (with clean system sans-serif fallback), small uppercase tracked labels (`letter-spacing: 0.05em; font-size: 0.72rem; font-weight: 600`), large semi-bold metric figures (`font-size: 1.85rem; font-weight: 600`).

---

## 2. Visual Reference Comparison: Matched vs Streamlit Constraints

### Reference A: Dark Finance Command Center & Overview
**Model for Shell & Overview Page (`dashboard/pages/overview.py`)**

#### What We Matched:
1. **Header & Status Banner**:
   - Implemented real-time system status indicator (`LIVE` / `WARN`) computed directly from runtime state (calibrated model bundle loaded, scored dataset loaded, record count verified).
   - Prominent text-mark branding block ("RETENTION IQ: Banking Intelligence Platform") in the deep charcoal sidebar.
   - Persistent "SYNTHETIC DATASET" compliance badge prominently displayed on every screen.
2. **Automated Insight Banner**:
   - Dynamic banner rendered via rule templates analyzing empirical data in memory (identifies top churn region and highest concentration risk segment with exact deltas).
3. **5x2 KPI Grid**:
   - Structured 10-card command grid: Total Customers, Observed Churn Rate, Predicted Churners, High-Risk Customers, Avg Calibrated Probability, Estimated Portfolio CLV, Test ROC-AUC, Test Brier Score, High-Risk Deposit Balance, and Active Segments.
   - Honest, mathematically sound deltas with explicit reference bases (e.g. `vs 16.4% baseline`, `vs chance 0.50`, `vs uncalibrated 0.203`).
4. **Symmetrical Central Layout**:
   - Placed the empirical risk band churn distribution chart side-by-side with a 5-tier Risk Attrition Funnel (`Total Portfolio` $\rightarrow$ `Scored Customers` $\rightarrow$ `Elevated Risk` $\rightarrow$ `Critical Risk` $\rightarrow$ `Predicted Churners`).
5. **Cohort Breakdown Grid**:
   - Demographic breakdowns by geography, gender, age cohorts, segment shares, and customer activity distributions.
6. **Reactive Global Filtering**:
   - Multi-dimensional sidebar filters (Geography, Gender, Risk Tier, Segment, Age Slider) that dynamically filter every metric, funnel stage, and chart on the page.

#### Streamlit Constraints & Discrepancies:
- **Free-floating Canvas Widgets**: Reference A features overlapping floating cards and animated gradient meshes. In Streamlit's container model, layout is strictly column- and row-based.
- **Top Bar Integration**: Streamlit reserves the top browser edge for its status indicators and hamburger menu. Consequently, top-bar search and system status pills were embedded in the upper grid of the page and top of the sidebar.

---

### Reference B: Three-Column Customer Intelligence Workspace
**Model for Customer Intelligence (`dashboard/pages/customer_intelligence.py`)**

#### What We Matched:
1. **Three-Column Command Layout**:
   - **Left Column (1.1 ratio)**: Searchable customer roster showing Masked ID (`ID-***2427`), calibrated probability, risk chip, and segment. Includes search-by-ID, pagination, filtering by geography/risk/segment, and sorting (Risk, CLV, Balance). Raw `Surname` is strictly suppressed for compliance.
   - **Center Column (1.7 ratio)**: Selected customer profile featuring an SVG semicircle calibrated risk gauge, 8-box financial/demographic facts grid, assigned persona card, and "Detected Behavioral Signals" evaluated strictly from empirical data rules (e.g. digital inactivity, capital flight hazard, complaint history, single-product concentration, sub-median transaction velocity).
   - **Right Rail (1.2 ratio)**: Runtime model health metrics (version hash, sigmoid calibration, honest OOF data source), directional factor drivers extracted from ensemble linear coefficients, and an explicit placeholder noting prescriptive decision engine actions arrive in Phase 4.
2. **Integrated Inference Tabs**:
   - Interactive prediction workspace with domain bound validation, calling the shared `predict_single` service.
   - Batch CSV scoring with 5MB byte limit and 5,000 row limit enforcement, complete with preview and downloadable scored results.

#### Streamlit Constraints & Discrepancies:
- **Client-Side Scrolling Virtual List**: Reference B uses an infinite-scroll virtualized DOM list in the left rail. In Streamlit, this is implemented using clean pagination (`page_size=12`) with radio selection to prevent client browser memory exhaustion.
- **Micro-Interactions**: Client-side animated gauge needles on hover were substituted with lightweight, responsive SVG vector gauges rendered directly from Python.

---

### Reference C: Dense Multi-Model Evaluation & Segmentation
**Model for Segmentation & Model Performance (`dashboard/pages/segmentation.py` & `model_performance.py`)**

#### What We Matched:
1. **Interactive Pill Toggles**:
   - Utilized Streamlit's native `st.pills` / `st.segmented_control` for rapid decision threshold exploration (`0.30`, `0.40`, `0.50`, `0.60`).
2. **Test Split Focus**:
   - All performance metrics (ROC-AUC 0.7295, PR-AUC 0.3866, Precision, Recall, F1, Brier Score 0.1216) are evaluated strictly on the untouched holdout test partition ($N=2,000$).
3. **Confusion Matrix Heatmap**:
   - Dynamically recomputes True Positives, False Positives, True Negatives, and False Negatives at the chosen threshold.
4. **Interactive Discrimination & Calibration Curves**:
   - High-resolution Plotly ROC Curve, Precision-Recall Curve, and Empirical Reliability Diagram (plotting 5 risk quantiles against the ideal 45-degree diagonal).
5. **Tables with In-Cell Bar Visualizers**:
   - Rendered styled HTML tables featuring inline horizontal progress bars for deposit balances and churn rates in the segmentation view.
6. **Overfitting Audit & Spark MLlib Benchmark**:
   - Explicitly contrasted in-sample training metrics with holdout test metrics.
   - Partitioned Spark MLlib clustering and training results into separate, labeled cards to prevent conflating distributed Spark and local Python experiments.

#### Streamlit Constraints & Discrepancies:
- **Native Table Progress Bars**: Native `st.dataframe` progress bar column configurations have limited CSS styling capabilities in dark mode; custom responsive HTML tables were implemented to achieve the visual density of Reference C.

---

## 3. Verification & Test Evidence
- **AppTest Test Suite**: All 9 tests in `tests/test_platform_app.py` passed cleanly.
- **Full Suite**: All 67 tests in the project pass with 0 failures and 0 errors.
- **Clean Shell**: Both `dashboard/platform_app.py` (new platform) and `dashboard/app.py` (legacy app) run cleanly side-by-side.

