"""Legacy Flask Prediction REST API (Deprecated).

Run: python src/api.py
Note: This API is maintained for legacy backward compatibility.
Modern applications should use the FastAPI v2 service in src/api_v2/app.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flask import Flask, jsonify, request
from src.predict import predict_churn

app = Flask(__name__)

DEPRECATION_MESSAGE = "This endpoint is deprecated. Please migrate to the modern FastAPI service at POST /v1/predict."


def validate_legacy_payload(payload: dict) -> tuple[bool, str]:
    """Validate legacy input fields to prevent unhandled 500 crashes and absurd inputs."""
    if not payload or not isinstance(payload, dict):
        return False, "Payload must be a non-empty JSON object."

    # Validate Age if provided
    if "Age" in payload:
        age = payload["Age"]
        if age is None or not isinstance(age, (int, float)) or isinstance(age, bool):
            return False, "Age must be a valid numeric value."
        if age < 18 or age > 100:
            return False, f"Age {age} is outside valid bounds (18 to 100)."

    # Validate Balance if provided
    if "Balance" in payload:
        bal = payload["Balance"]
        if bal is None or not isinstance(bal, (int, float)) or isinstance(bal, bool):
            return False, "Balance must be a valid numeric value."
        if bal < 0 or bal > 1e10:
            return False, f"Balance {bal} is invalid (must be >= 0)."

    # Validate CreditScore if provided
    if "CreditScore" in payload:
        cs = payload["CreditScore"]
        if cs is None or not isinstance(cs, (int, float)) or isinstance(cs, bool):
            return False, "CreditScore must be a valid numeric value."
        if cs < 300 or cs > 850:
            return False, f"CreditScore {cs} is outside valid bounds (300 to 850)."

    # Validate Geography if provided
    if "Geography" in payload:
        geo = payload["Geography"]
        if geo not in ("France", "Germany", "Spain"):
            return False, f"Unknown Geography '{geo}'. Valid values: France, Germany, Spain."

    # Validate Gender if provided
    if "Gender" in payload:
        gen = payload["Gender"]
        if gen not in ("Male", "Female"):
            return False, f"Unknown Gender '{gen}'. Valid values: Male, Female."

    return True, ""


@app.get("/health")
def health():
    return jsonify(status="ok", service="flask-legacy-api", status_notice="deprecated")


@app.post("/predict")
def predict():
    payload = request.get_json(force=True, silent=True)
    is_valid, err_msg = validate_legacy_payload(payload)
    if not is_valid:
        resp = jsonify(error=err_msg, deprecation=DEPRECATION_MESSAGE)
        resp.headers["X-API-Deprecation"] = DEPRECATION_MESSAGE
        return resp, 400

    try:
        result = predict_churn(payload)
        resp = jsonify(result)
        resp.headers["X-API-Deprecation"] = DEPRECATION_MESSAGE
        return resp
    except Exception as e:
        resp = jsonify(error=f"Prediction error: {str(e)}", deprecation=DEPRECATION_MESSAGE)
        resp.headers["X-API-Deprecation"] = DEPRECATION_MESSAGE
        return resp, 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
