#!/usr/bin/env python3
"""Automated End-to-End QA Walk-Through across all 11 platform pages.

Simulates user interactions:
- Navigating every sidebar route
- Interacting with every filter, slider, selectbox, and text input
- Clicking action buttons (optimizations, simulations, state transitions)
- Testing download button payloads (PDF, CSVs)
- Inspecting exceptions, error banners, and UI warnings

Outputs structured findings for QA_REPORT.md.
"""
import sys
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

# Auto-delegate to venv
venv_python = ROOT_DIR / ".venv" / "bin" / "python"
if venv_python.exists() and sys.executable != str(venv_python):
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)

from streamlit.testing.v1 import AppTest

PAGES = [
    ("Overview", "dashboard/pages/overview.py"),
    ("Customer Intelligence", "dashboard/pages/customer_intelligence.py"),
    ("Real-Time Monitoring", "dashboard/pages/realtime_monitoring.py"),
    ("Customer Segmentation", "dashboard/pages/segmentation.py"),
    ("Model Performance", "dashboard/pages/model_performance.py"),
    ("Decision Engine", "dashboard/pages/decision_engine.py"),
    ("Explainable AI (SHAP)", "dashboard/pages/explainability.py"),
    ("ROI Simulator", "dashboard/pages/roi_simulator.py"),
    ("Fairness & Risk Audit", "dashboard/pages/fairness_audit.py"),
    ("Campaign Management", "dashboard/pages/campaigns.py"),
    ("Executive Reports", "dashboard/pages/reports.py"),
    ("System Settings", "dashboard/pages/settings.py"),
]


def test_page_walkthrough(name: str, path_str: str):
    print("\n========================================================")
    print(f"  TESTING PAGE: {name} ({path_str})")
    print("========================================================")

    page_path = ROOT_DIR / path_str
    if not page_path.exists():
        print(f"  ❌ File not found: {page_path}")
        return False, ["File not found"]

    issues = []
    try:
        at = AppTest.from_file(str(page_path))
        at.run(timeout=40)

        # 1. Initial Render Check
        if len(at.exception) > 0:
            for exc in at.exception:
                issues.append(f"Initial render Exception: {exc.message}")
                print(f"  ❌ Exception: {exc.message}")
            return False, issues
        else:
            print("  ✔ Initial render clean (0 exceptions)")

        # 2. Inspect Widgets
        selectboxes = at.selectbox
        sliders = at.slider
        buttons = at.button
        download_buttons = at.download_button
        tabs = at.tabs if hasattr(at, "tabs") else []
        
        print(f"  Widgets: {len(selectboxes)} selectboxes, {len(sliders)} sliders, {len(buttons)} buttons, {len(download_buttons)} downloads")

        # 3. Test Selectboxes
        for idx, sb in enumerate(selectboxes):
            try:
                opts = sb.options
                if opts and len(opts) > 1:
                    target_val = opts[1]
                    sb.select(target_val).run(timeout=30)
                    if len(at.exception) > 0:
                        issues.append(f"Selectbox '{sb.label}' error when selecting '{target_val}': {at.exception[0].message}")
                    else:
                        print(f"  ✔ Selectbox '{sb.label}' selected '{target_val}' successfully")
            except Exception as e:
                issues.append(f"Selectbox interaction failed: {e}")

        # 4. Test Sliders
        for idx, sl in enumerate(sliders):
            try:
                min_v, max_v = sl.min, sl.max
                new_v = (min_v + max_v) / 2 if (min_v is not None and max_v is not None) else sl.value
                sl.set_value(new_v).run(timeout=30)
                if len(at.exception) > 0:
                    issues.append(f"Slider '{sl.label}' error at value {new_v}: {at.exception[0].message}")
                else:
                    print(f"  ✔ Slider '{sl.label}' updated to {new_v}")
            except Exception as e:
                issues.append(f"Slider interaction failed: {e}")

        # 5. Test Buttons (Non-destructive / Active Run Buttons)
        for idx, btn in enumerate(buttons):
            label = btn.label or f"Button-{idx}"
            # Skip disabled buttons (e.g. state-machine transitions not valid in current state)
            is_disabled = getattr(btn, "disabled", False) or getattr(btn.proto, "disabled", False)
            if is_disabled:
                print(f"  [info] Button '{label}' correctly disabled in current state")
                continue

            # Skip dangerous actions like delete campaign or cancel in initial sweep
            if any(term in label.lower() for term in ["delete", "drop", "purge"]):
                print(f"  [skip] Destructive button '{label}' skipped in auto-walk")
                continue
            try:
                btn.click().run(timeout=35)
                if len(at.exception) > 0:
                    issues.append(f"Button '{label}' raised exception: {at.exception[0].message}")
                else:
                    print(f"  ✔ Button '{label}' clicked successfully")
            except Exception as e:
                issues.append(f"Button click '{label}' error: {e}")

        # 6. Test Downloads
        for idx, db in enumerate(download_buttons):
            label = db.label or f"Download-{idx}"
            try:
                url = getattr(db.proto, "url", None)
                if not url:
                    issues.append(f"Download button '{label}' has empty media payload!")
                else:
                    print(f"  ✔ Download button '{label}' verified (media URL='{url}')")
            except Exception as e:
                issues.append(f"Download check '{label}' error: {e}")

        # Check for deprecation warnings or red banners
        for err in at.error:
            issues.append(f"UI Error banner displayed: {err.value}")
            print(f"  ⚠️ Error banner: {err.value}")

        if not issues:
            print(f"  🌟 ALL INTERACTIONS PASSED CLEANLY FOR {name}")
            return True, []
        else:
            print(f"  ⚠️ Encountered {len(issues)} issues on {name}")
            return False, issues

    except Exception as exc:
        issues.append(f"Test runner failure: {str(exc)}")
        print(f"  ❌ Fatal test failure: {exc}")
        return False, issues


def main():
    print("==================================================================")
    print("  BANKING RETENTION INTELLIGENCE PLATFORM - COMPLETE QA AUDIT")
    print("==================================================================")

    results = {}
    for name, path_str in PAGES:
        passed, issues = test_page_walkthrough(name, path_str)
        results[name] = {"passed": passed, "issues": issues}

    print("\n\n==================================================================")
    print("  QA SUMMARY MATRIX")
    print("==================================================================")
    total_passed = 0
    for name, r in results.items():
        status_sym = "✔ PASS" if r["passed"] else "✖ FAIL/WARN"
        print(f"  {status_sym:12} | {name}")
        if r["issues"]:
            for iss in r["issues"]:
                print(f"               ↳ {iss}")
        if r["passed"]:
            total_passed += 1

    print("==================================================================")
    print(f"  SCORE: {total_passed}/{len(PAGES)} Pages Passed Cleanly")
    print("==================================================================\n")
    return 0 if total_passed == len(PAGES) else 1


if __name__ == "__main__":
    sys.exit(main())
