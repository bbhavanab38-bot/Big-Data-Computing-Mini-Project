"""Automated full-page screenshot capture across 1440px and 1024px viewports on http://localhost:8504."""

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
PORT = 9223  # use distinct port from any running instance
BASE_URL = "http://localhost:8504"

PAGES = [
    ("overview", f"{BASE_URL}/", "PORTFOLIO CHURN & RISK COMMAND CENTER"),
    ("customer_intelligence", f"{BASE_URL}/customer_intelligence", "CUSTOMER 360 & INTERACTIVE PREDICTION"),
    ("realtime_monitoring", f"{BASE_URL}/realtime_monitoring", "Real-Time Streaming & Risk Monitoring"),
    ("segmentation", f"{BASE_URL}/segmentation", "Behavioral Customer Segmentation"),
    ("model_performance", f"{BASE_URL}/model_performance", "Model Evaluation & Calibration Reliability Lab"),
    ("decision_engine", f"{BASE_URL}/decision_engine", "Policy Comparison"),
    ("explainability", f"{BASE_URL}/explainability", "Explainable AI (XAI) & Counterfactual Guidance"),
    ("roi_simulator", f"{BASE_URL}/roi_simulator", "Retention ROI & Uncertainty Lab"),
    ("fairness_audit", f"{BASE_URL}/fairness_audit", "Fairness & Algorithmic Risk Audit"),
    ("campaigns", f"{BASE_URL}/campaigns", "Campaign Management & Lifecycle Execution"),
    ("reports", f"{BASE_URL}/reports", "Executive Reporting & Governance Exports"),
    ("settings", f"{BASE_URL}/settings", "System Settings & Platform Diagnostics"),
]

VIEWPORTS = [
    (1440, 1000, "1440"),
    (1024, 900, "1024"),
]


async def capture_all():
    # Wait for Chrome CDP endpoint
    ws_url = None
    for _ in range(20):
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

    # Create new tab
    with urllib.request.urlopen(
        urllib.request.Request(f"http://127.0.0.1:{PORT}/json/new?about:blank", method="PUT")
    ) as r:
        tab_data = json.loads(r.read())
        tab_ws_url = tab_data["webSocketDebuggerUrl"]

    async with websockets.connect(tab_ws_url, max_size=40 * 1024 * 1024) as ws:
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

        await send("Page.enable")
        await send("Runtime.enable")
        await send("DOM.enable")

        for width, height, tag in VIEWPORTS:
            print(f"\n================ Setting viewport: {width}x{height} ({tag}) ================")
            await send(
                "Emulation.setDeviceMetricsOverride",
                {
                    "width": width,
                    "height": height,
                    "deviceScaleFactor": 1,
                    "mobile": False,
                },
            )

            for page_name, url, expected_text in PAGES:
                print(f"[{tag}px] Navigating to {url}...")
                await send("Page.navigate", {"url": url})

                # Wait for content
                for _ in range(60):
                    await asyncio.sleep(0.5)
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
                        break

                # Additional stabilization time for charts, tables and animations
                await asyncio.sleep(3.0)

                # Capture full page screenshot
                ss_res = await send(
                    "Page.captureScreenshot",
                    {
                        "format": "png",
                        "captureBeyondViewport": True,
                    },
                )
                data = ss_res.get("result", {}).get("data", "")
                if data:
                    raw_bytes = base64.b64decode(data)
                    # Save tag version (e.g. overview_1440.png or overview_1024.png)
                    tagged_path = SCREENSHOT_DIR / f"{page_name}_{tag}.png"
                    tagged_path.write_bytes(raw_bytes)
                    print(f"  -> Saved {tagged_path} ({len(raw_bytes):,} bytes)")

                    # If 1440, also save as standard <page>.png
                    if tag == "1440":
                        std_path = SCREENSHOT_DIR / f"{page_name}.png"
                        std_path.write_bytes(raw_bytes)
                        print(f"  -> Saved {std_path}")
                else:
                    print(f"  [ERROR] Failed to capture {page_name} at {tag}px")


def main():
    chrome_proc = subprocess.Popen(
        [
            CHROME_BIN,
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--user-data-dir=/tmp/chrome_qa_profile",
            f"--remote-debugging-port={PORT}",
            "--window-size=1600,1200",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("Started Chrome with PID", chrome_proc.pid)

    try:
        asyncio.run(capture_all())
    finally:
        chrome_proc.terminate()
        chrome_proc.wait(timeout=5)
        print("Chrome terminated.")


if __name__ == "__main__":
    main()
