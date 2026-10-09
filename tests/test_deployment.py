"""Check the deployed website and API together with the committed model."""
import json
from pathlib import Path
import unittest

from fastapi.testclient import TestClient
from backend.main import app

ROOT = Path(__file__).resolve().parent.parent


class DeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_website_and_assets_are_available(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn('id="mushroom-form"', response.text)
        for path in ("/index.html", "/config.js", "/app.js", "/styles.css",
                     "/metadata.json", "/citibike_metrics.json"):
            with self.subTest(path=path):
                asset = self.client.get(path)
                self.assertEqual(asset.status_code, 200)
                self.assertEqual(asset.headers["cache-control"], "no-cache")

    def test_docs_and_health_keep_their_api_routes(self):
        self.assertEqual(self.client.get("/docs").status_code, 200)
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json()["mushroom_model_loaded"])
        schema = self.client.get("/openapi.json").json()
        self.assertIn("/predict/mushroom", schema["paths"])

    def test_website_example_produces_a_prediction(self):
        metadata = json.loads((ROOT / "frontend/metadata.json").read_text(encoding="utf-8"))
        response = self.client.post("/predict/mushroom", json=metadata["example"])
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertIn(result["label"], ["e", "p"])
        self.assertGreaterEqual(result["probability_poisonous"], 0)
        self.assertLessEqual(result["probability_poisonous"], 1)

    def test_invalid_prediction_input_is_still_rejected(self):
        response = self.client.post("/predict/mushroom", json={})
        self.assertEqual(response.status_code, 422)

    def test_backend_and_model_files_are_not_served(self):
        for path in ("/backend/main.py", "/render.yaml",
                     "/secondary_mushroom/models/baseline_decision_tree.joblib"):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 404)


if __name__ == "__main__":
    unittest.main()
