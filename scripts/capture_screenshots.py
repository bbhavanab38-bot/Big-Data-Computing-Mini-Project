"""Automated screenshot capture for Phase 3 Platform App using Chrome DevTools Protocol."""

import asyncio
import base64
import json
import subprocess
import time
from pathlib import Path
import urllib.request
import websockets

SCREENSHOT_DIR = Path("docs/screenshots")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PORT = 9222

PAGES = [
    ("01_overview.png", "http://localhost:8501/overview", "PORTFOLIO CHURN & RISK COMMAND CENTER"),
    ("02_customer_intelligence.png", "http://localhost:8501/customer_intelligence", "CUSTOMER 360 & INTERACTIVE PREDICTION"),
    ("03_segmentation.png", "http://localhost:8501/segmentation", "Behavioral Customer Segmentation"),
    ("04_model_performance.png", "http://localhost:8501/model_performance", "Model Evaluation & Calibration Reliability Lab"),
    ("05_decision_engine.png", "http://localhost:8501/decision_engine", "Prescriptive Retention Decision Engine"),
    ("06_explainability.png", "http://localhost:8501/explainability", "Explainable AI (XAI) & Counterfactual Guidance"),
    ("07_roi_simulator.png", "http://localhost:8501/roi_simulator", "Prescriptive ROI & Financial Optimization Simulator"),
    ("08_fairness_audit.png", "http://localhost:8501/fairness_audit", "Fairness & Risk Committee Audit"),
    ("09_campaigns.png", "http://localhost:8501/campaigns", "Retention Campaign Management & Execution Hub"),
    ("10_realtime_monitoring.png", "http://localhost:8501/realtime_monitoring", "Real-Time Risk Monitoring & Event Stream"),
    ("11_reports.png", "http://localhost:8501/reports", "Executive Reporting & Governance Exports"),
    ("12_settings.png", "http://localhost:8501/settings", "System Settings & Platform Diagnostics"),
]


async def capture_page(ws_url: str, url: str, output_path: Path, expected_text: str = ""):
    async with websockets.connect(ws_url, max_size=20 * 1024 * 1024) as ws:
        msg_id = 1

        async def send(method, params=None):
            nonlocal msg_id
            m = {"id": msg_id, "method": method}
            if params:
                m["params"] = params
            msg_id += 1
            await ws.send(json.dumps(m))
            while True:
                resp = json.loads(await ws.recv())
                if resp.get("id") == m["id"]:
                    return resp

        # Enable domains
        await send("Page.enable")
        await send("Runtime.enable")

        # Navigate
        print(f"Navigating to {url}...")
        await send("Page.navigate", {"url": url})

        # Wait for Streamlit to render elements
        print(f"Waiting for Streamlit render (looking for '{expected_text}')...")
        for _ in range(30):
            await asyncio.sleep(0.5)
            # Check if expected text is in body
            eval_res = await send(
                "Runtime.evaluate",
                {
                    "expression": f"document.body.innerText.includes('{expected_text}')"
                    if expected_text
                    else "document.querySelectorAll('[data-testid=\"stMarkdownContainer\"]').length > 5"
                },
            )
            val = eval_res.get("result", {}).get("result", {}).get("value", False)
            if val:
                print(f"Content matched for {expected_text or url}!")
                break

        # Give 2.5 seconds for Plotly charts and SVG gauges to complete animations
        await asyncio.sleep(2.5)

        # Capture screenshot
        print(f"Capturing screenshot to {output_path}...")
        ss_res = await send(
            "Page.captureScreenshot",
            {
                "format": "png",
                "captureBeyondViewport": False,
            },
        )
        data = ss_res.get("result", {}).get("data", "")
        if data:
            output_path.write_bytes(base64.b64decode(data))
            print(f"Successfully saved {output_path} ({len(data)} bytes b64)")
        else:
            print(f"Failed to capture screenshot for {url}: {ss_res}")


def main():
    chrome_proc = subprocess.Popen(
        [
            CHROME_BIN,
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--user-data-dir=/tmp/chrome_ss_profile",
            f"--remote-debugging-port={PORT}",
            "--window-size=1600,1050",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("Started Chrome with PID", chrome_proc.pid)

    try:
        # Wait for Chrome CDP endpoint
        ws_url = None
        for _ in range(15):
            time.sleep(0.5)
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/version", timeout=1) as r:
                    ver_data = json.loads(r.read())
                    ws_url = ver_data.get("webSocketDebuggerUrl")
                    if ws_url:
                        break
            except Exception:
                pass

        if not ws_url:
            print("Failed to connect to Chrome CDP endpoint")
            return

        # Open a new target tab
        with urllib.request.urlopen(
            urllib.request.Request(f"http://127.0.0.1:{PORT}/json/new?about:blank", method="PUT")
        ) as r:
            tab_data = json.loads(r.read())
            tab_ws_url = tab_data["webSocketDebuggerUrl"]

        for filename, page_url, expected_text in PAGES:
            out_file = SCREENSHOT_DIR / filename
            asyncio.run(capture_page(tab_ws_url, page_url, out_file, expected_text))

    finally:
        chrome_proc.terminate()
        chrome_proc.wait(timeout=5)
        print("Chrome stopped.")


if __name__ == "__main__":
    main()
