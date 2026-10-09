"""Check the deployed website and API together with the committed model."""
import json
from pathlib import Path
import unittest

from fastapi.testclient import TestClient
from backend.main import app
from backend.mushroom_model import MODEL_ID, FEATURES

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
        for path in ("/index.html", "/config.js", "/api.js", "/app.js", "/styles.css",
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
        self.assertTrue(health.json()["citibike_model_loaded"])
        self.assertEqual(health.json()["mushroom_model_id"], MODEL_ID)
        schema = self.client.get("/openapi.json").json()
        self.assertIn("/predict/mushroom", schema["paths"])
        self.assertIn("/predict/citibike", schema["paths"])

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

    def test_frontend_fields_match_andrews_model(self):
        metadata = self.client.get("/metadata.json").json()
        self.assertEqual(metadata["model_id"], MODEL_ID)
        self.assertEqual({field["name"] for field in metadata["fields"]}, set(FEATURES))
        self.assertEqual(set(metadata["example"]), set(FEATURES))
        self.assertEqual(app.state.mushroom_model.named_steps["classifier"].n_estimators, 200)

    def test_zero_measurements_match_unknown_values(self):
        example = self.client.get("/metadata.json").json()["example"]
        zeros = {**example, "stem-height": 0, "stem-width": 0}
        unknown = {**example, "stem-height": None, "stem-width": None}
        zero_result = self.client.post("/predict/mushroom", json=zeros)
        unknown_result = self.client.post("/predict/mushroom", json=unknown)
        self.assertEqual(zero_result.status_code, 200, zero_result.text)
        self.assertEqual(zero_result.json(), unknown_result.json())

    def test_optional_noise_fields_can_be_omitted(self):
        example = self.client.get("/metadata.json").json()["example"]
        example.pop("jumbled_noise_0")
        example.pop("jumbled_noise_1")
        response = self.client.post("/predict/mushroom", json=example)
        self.assertEqual(response.status_code, 200, response.text)

    def test_city_prediction_matches_the_frontend_chart_contract(self):
        response = self.client.post("/predict/citibike", json={"date": "2025-01-25", "hour": 8})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["date"], "2025-01-25")
        self.assertEqual(result["hour"], 8)
        self.assertEqual(result["timezone"], "America/New_York")
        rows = result["daily_predictions"]
        self.assertEqual([row["hour"] for row in rows], list(range(24)))
        self.assertTrue(all(row["predicted_ride_starts"] >= 0 for row in rows))
        self.assertEqual(result["predicted_ride_starts"], rows[8]["predicted_ride_starts"])
        metrics = self.client.get("/citibike_metrics.json").json()
        self.assertEqual(result["test_mae"], metrics["results"]["decision_tree"]["mae"])

    def test_city_prediction_warns_outside_the_training_month(self):
        response = self.client.post("/predict/citibike", json={"date": "2026-10-09", "hour": 23})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("buiten de beschikbare dataperiode", response.json()["warning"])

    def test_invalid_city_date_hour_and_extra_fields_are_rejected(self):
        valid = {"date": "2025-01-25", "hour": 8}
        invalid = [{}, {**valid, "date": "not-a-date"}, {**valid, "hour": -1},
                   {**valid, "hour": 24}, {**valid, "hour": "8"}, {**valid, "extra": True}]
        for payload in invalid:
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post("/predict/citibike", json=payload).status_code, 422)

    def test_missing_city_model_returns_json_and_keeps_mushroom_inference(self):
        artifact = app.state.city_model
        try:
            app.state.city_model = None
            health = self.client.get("/health")
            self.assertEqual(health.status_code, 503)
            self.assertFalse(health.json()["citibike_model_loaded"])
            response = self.client.post("/predict/citibike", json={"date": "2025-01-25", "hour": 8})
            self.assertEqual(response.status_code, 503)
            self.assertIn("niet beschikbaar", response.json()["detail"])
            example = self.client.get("/metadata.json").json()["example"]
            self.assertEqual(self.client.post("/predict/mushroom", json=example).status_code, 200)
        finally:
            app.state.city_model = artifact

    def test_backend_and_model_files_are_not_served(self):
        for path in ("/backend/main.py", "/render.yaml",
                     "/secondary_mushroom/models/baseline_decision_tree.joblib",
                     "/nyc_citi_bike/models/baseline_hourly_tree.joblib",
                     "/SolutionAndrew/MushroomDataset/models/random_forest.joblib"):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 404)


if __name__ == "__main__":
    unittest.main()
