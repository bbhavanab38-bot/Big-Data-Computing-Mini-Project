"""Unit tests for FastAPI v2 service: validation, probe cases, batch limits, API key."""
import os
import io
import unittest
import pandas as pd
from fastapi.testclient import TestClient

from src.api_v2.app import app


class TestFastAPIV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.valid_customer = {
            "Age": 45,
            "CreditScore": 650,
            "Balance": 50000.0,
            "Geography": "France",
            "Gender": "Female",
            "Tenure": 5,
            "NumOfProducts": 2,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 75000.0,
            "TransactionCount": 20,
            "OnlineBankingUsage": 50.0,
            "CustomerComplaints": 0,
            "LoanAmount": 10000.0,
            "LoanStatus": "Approved",
        }

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["service"], "banking-retention-api-v2")

    def test_model_metadata_endpoint(self):
        resp = self.client.get("/v1/model")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("name", data)
        self.assertIn("model_version", data)
        self.assertIn("metrics", data)

    def test_predict_valid_customer(self):
        resp = self.client.post("/v1/predict", json=self.valid_customer)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("prediction", data)
        self.assertIn("churn_probability", data)
        self.assertIn("risk_category", data)
        self.assertIn("model_version", data)
        self.assertGreaterEqual(data["churn_probability"], 0.0)
        self.assertLessEqual(data["churn_probability"], 1.0)
        self.assertIn(data["risk_category"], ["Low", "Medium", "High"])

    def test_probe_empty_payload(self):
        resp = self.client.post("/v1/predict", json={})
        self.assertEqual(resp.status_code, 422)
        data = resp.json()
        self.assertEqual(data["error_type"], "validation_error")

    def test_probe_string_age(self):
        payload = dict(self.valid_customer)
        payload["Age"] = "abc"
        resp = self.client.post("/v1/predict", json=payload)
        self.assertEqual(resp.status_code, 422)

    def test_probe_null_age(self):
        payload = dict(self.valid_customer)
        payload["Age"] = None
        resp = self.client.post("/v1/predict", json=payload)
        self.assertEqual(resp.status_code, 422)

    def test_probe_negative_age(self):
        payload = dict(self.valid_customer)
        payload["Age"] = -5
        resp = self.client.post("/v1/predict", json=payload)
        self.assertEqual(resp.status_code, 422)

    def test_probe_absurd_balance(self):
        payload = dict(self.valid_customer)
        payload["Balance"] = -100.0
        resp = self.client.post("/v1/predict", json=payload)
        self.assertEqual(resp.status_code, 422)

    def test_probe_unknown_geography(self):
        payload = dict(self.valid_customer)
        payload["Geography"] = "Mars"
        resp = self.client.post("/v1/predict", json=payload)
        self.assertEqual(resp.status_code, 422)

    def test_predict_batch_csv_success(self):
        csv_data = (
            "CustomerId,Age,CreditScore,Balance,Geography,Gender\n"
            "1001,45,650,50000.0,France,Female\n"
            "1002,32,720,12000.0,Spain,Male\n"
            "1003,58,580,95000.0,Germany,Female\n"
        )
        files = {"file": ("batch.csv", io.BytesIO(csv_data.encode("utf-8")), "text/csv")}
        resp = self.client.post("/v1/predict/batch", files=files)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total_records"], 3)
        self.assertEqual(len(data["predictions"]), 3)
        self.assertIn("churn_likely_count", data)

    def test_predict_batch_non_csv_rejected(self):
        files = {"file": ("batch.txt", io.BytesIO(b"hello world"), "text/plain")}
        resp = self.client.post("/v1/predict/batch", files=files)
        self.assertEqual(resp.status_code, 400)

    def test_predict_batch_byte_limit(self):
        # Override byte limit
        old_val = os.environ.get("MAX_BATCH_BYTES")
        import src.api_v2.app as app_module
        app_module.DEFAULT_MAX_BATCH_BYTES = 50  # 50 bytes limit

        try:
            large_content = b"a" * 100
            files = {"file": ("batch.csv", io.BytesIO(large_content), "text/csv")}
            resp = self.client.post("/v1/predict/batch", files=files)
            self.assertEqual(resp.status_code, 413)
        finally:
            app_module.DEFAULT_MAX_BATCH_BYTES = 5 * 1024 * 1024

    def test_probe_missing_required_fields(self):
        """Omitting required fields such as CreditScore returns 422."""
        payload = dict(self.valid_customer)
        del payload["CreditScore"]
        resp = self.client.post("/v1/predict", json=payload)
        self.assertEqual(resp.status_code, 422)
        data = resp.json()
        self.assertEqual(data["error_type"], "validation_error")

    def test_probe_credit_score_out_of_range(self):
        """Credit score outside valid banking bounds (300-850) returns 422."""
        payload = dict(self.valid_customer)
        payload["CreditScore"] = 9999
        resp = self.client.post("/v1/predict", json=payload)
        self.assertEqual(resp.status_code, 422)

        payload["CreditScore"] = 150
        resp = self.client.post("/v1/predict", json=payload)
        self.assertEqual(resp.status_code, 422)

    def test_predict_batch_empty_csv(self):
        """A CSV with only headers and 0 data rows returns 400."""
        csv_data = "CustomerId,Age,CreditScore,Balance,Geography,Gender\n"
        files = {"file": ("empty.csv", io.BytesIO(csv_data.encode("utf-8")), "text/csv")}
        resp = self.client.post("/v1/predict/batch", files=files)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("0 data rows", resp.json()["detail"])

    def test_predict_batch_row_limit(self):
        """Exceeding DEFAULT_MAX_BATCH_ROWS triggers 400 Bad Request."""
        import src.api_v2.app as app_module
        orig_limit = app_module.DEFAULT_MAX_BATCH_ROWS
        app_module.DEFAULT_MAX_BATCH_ROWS = 2
        try:
            csv_data = (
                "CustomerId,Age,CreditScore,Balance,Geography,Gender\n"
                "1001,45,650,50000.0,France,Female\n"
                "1002,32,720,12000.0,Spain,Male\n"
                "1003,58,580,95000.0,Germany,Female\n"
            )
            files = {"file": ("batch.csv", io.BytesIO(csv_data.encode("utf-8")), "text/csv")}
            resp = self.client.post("/v1/predict/batch", files=files)
            self.assertEqual(resp.status_code, 400)
            self.assertIn("exceeds maximum allowed limit", resp.json()["detail"])
        finally:
            app_module.DEFAULT_MAX_BATCH_ROWS = orig_limit

    def test_predict_endpoint_api_key_enforcement(self):
        """POST /v1/predict enforces X-API-Key when API_KEY is set."""
        os.environ["API_KEY"] = "prod-test-key-xyz"
        try:
            # Without key -> 401
            resp = self.client.post("/v1/predict", json=self.valid_customer)
            self.assertEqual(resp.status_code, 401)

            # With valid key -> 200
            resp = self.client.post(
                "/v1/predict",
                json=self.valid_customer,
                headers={"X-API-Key": "prod-test-key-xyz"},
            )
            self.assertEqual(resp.status_code, 200)
        finally:
            del os.environ["API_KEY"]

    def test_api_key_authentication(self):
        # Set API_KEY in environment
        os.environ["API_KEY"] = "secret-test-token"
        try:
            # Request without header -> 401
            resp = self.client.get("/v1/model")
            self.assertEqual(resp.status_code, 401)

            # Request with wrong header -> 401
            resp = self.client.get("/v1/model", headers={"X-API-Key": "wrong-token"})
            self.assertEqual(resp.status_code, 401)

            # Request with valid header -> 200
            resp = self.client.get("/v1/model", headers={"X-API-Key": "secret-test-token"})
            self.assertEqual(resp.status_code, 200)
        finally:
            del os.environ["API_KEY"]

    def test_hdfs_status_endpoint(self):
        resp = self.client.get("/v1/hdfs/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("available", data)
        self.assertIn("mode", data)
        self.assertIn("status", data)

    def test_ingest_endpoint_valid_and_quarantine(self):
        csv_content = (
            "CustomerId,CreditScore,Age,Balance,EstimatedSalary,Geography,Gender,Exited\n"
            "501,720,40,15000,55000,France,Male,0\n"
            "502,999,12,-100,50000,Atlantis,Alien,5\n"
        )
        files = {"file": ("ingest_test.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
        resp = self.client.post("/v1/ingest", files=files)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["total_received"], 2)
        self.assertEqual(data["valid_count"], 1)
        self.assertEqual(data["quarantined_count"], 1)


if __name__ == "__main__":
    unittest.main()

