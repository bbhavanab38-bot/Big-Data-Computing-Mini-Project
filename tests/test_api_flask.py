"""Unit tests for legacy Flask API: input validation, probe cases, and deprecation notices."""
import unittest
from src.api import app


class TestFlaskLegacyAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()
        cls.valid_customer = {
            "Age": 45,
            "Balance": 20000,
            "CreditScore": 700,
            "TransactionCount": 5,
            "Geography": "France",
            "Gender": "Female",
        }

    def test_health(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "ok")

    def test_valid_predict(self):
        resp = self.client.post("/predict", json=self.valid_customer)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("X-API-Deprecation", resp.headers)
        data = resp.get_json()
        self.assertIn("prediction", data)
        self.assertIn("probability", data)

    def test_probe_empty_payload(self):
        resp = self.client.post("/predict", json={})
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn("error", data)
        self.assertIn("deprecation", data)

    def test_probe_string_age(self):
        payload = dict(self.valid_customer)
        payload["Age"] = "abc"
        resp = self.client.post("/predict", json=payload)
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn("Age", data["error"])

    def test_probe_null_age(self):
        payload = dict(self.valid_customer)
        payload["Age"] = None
        resp = self.client.post("/predict", json=payload)
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn("Age", data["error"])

    def test_probe_negative_age(self):
        payload = dict(self.valid_customer)
        payload["Age"] = -5
        resp = self.client.post("/predict", json=payload)
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn("Age", data["error"])

    def test_probe_absurd_balance(self):
        payload = dict(self.valid_customer)
        payload["Balance"] = -100.0
        resp = self.client.post("/predict", json=payload)
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn("Balance", data["error"])

    def test_probe_unknown_geography(self):
        payload = dict(self.valid_customer)
        payload["Geography"] = "Mars"
        resp = self.client.post("/predict", json=payload)
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn("Geography", data["error"])


if __name__ == "__main__":
    unittest.main()

