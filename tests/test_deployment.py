"""Check the deployed website and API together with the committed model."""
import json
import hashlib
from pathlib import Path
import unittest
import tempfile
import numpy as np
import pandas as pd

from fastapi.testclient import TestClient
from backend.main import app
from backend.mushroom_model import MODEL_ID, FEATURES, DEFAULT_MODEL, PortableMushroom
from backend.raoul_citibike import MODEL_ID as CITY_MODEL_ID

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
                     "/metadata.json", "/citibike_metrics.json", "/citibike.js", "/mushroom.js"):
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
        self.assertEqual(health.json()["citibike_model_id"], CITY_MODEL_ID)
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
        if result["probability_poisonous"] < .5:
            self.assertEqual(result["decision"], "uncertain")
            self.assertEqual(result["prediction"], "Onzeker / mogelijk giftig")
        else:
            self.assertEqual(result["decision"], "suspected_poisonous")

    def test_invalid_prediction_input_is_still_rejected(self):
        response = self.client.post("/predict/mushroom", json={})
        self.assertEqual(response.status_code, 422)

    def test_frontend_fields_match_selected_rf300(self):
        metadata = self.client.get("/metadata.json").json()
        self.assertEqual(metadata["model_id"], MODEL_ID)
        self.assertEqual({field["name"] for field in metadata["fields"]}, set(FEATURES))
        self.assertEqual(set(metadata["example"]), set(FEATURES))
        self.assertEqual(len(metadata["fields"]), 11)
        self.assertNotIn("jumbled_noise_0", metadata["example"])
        self.assertEqual(app.state.mushroom_model.metadata["n_estimators"], 300)
        manifest = app.state.mushroom_model.metadata
        source = (ROOT/manifest["source_pipeline"]).resolve()
        self.assertTrue(source.is_relative_to(ROOT))
        self.assertEqual(metadata["source_pipeline_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())

    def test_serving_runtime_matches_all_saved_rf300_test_predictions(self):
        saved = pd.read_csv(ROOT/"backend/artifacts/mushroom_reference_predictions.csv", float_precision="round_trip")
        data = pd.read_csv(ROOT/"MushroomFinal/mushroom_project_dataset.csv")
        data.columns = data.columns.str.replace("-", "_", regex=False)
        frame = data[FEATURES].iloc[saved.source_row]
        actual = app.state.mushroom_model.predict_proba(frame)
        self.assertEqual(len(saved), 900)
        np.testing.assert_array_equal(np.argmax(actual, axis=1), saved.prediction)
        np.testing.assert_allclose(actual[:, 1], saved.probability_p, rtol=0, atol=1e-12)
        self.assertEqual(int(((saved.true_class == 1) & (saved.prediction == 0)).sum()), app.state.mushroom_model.metadata["metrics"]["fn"])
        for index in [0, 1, 2, 10, 40, 100, 400, 899]:
            payload = json.loads(frame.iloc[[index]].to_json(orient="records"))[0]
            response = self.client.post("/predict/mushroom", json=payload)
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertEqual(result["model_id"], MODEL_ID)
            self.assertEqual(result["label"], "p" if saved.prediction.iloc[index] else "e")
            self.assertAlmostEqual(result["probability_poisonous"], saved.probability_p.iloc[index], places=12)

    def test_zero_and_missing_measurements_remain_valid(self):
        example = self.client.get("/metadata.json").json()["example"]
        cases = [{**example, "stem_height": 0, "stem_width": 0},
                 {field: None for field in FEATURES}]
        for payload in cases:
            response = self.client.post("/predict/mushroom", json=payload)
            self.assertEqual(response.status_code, 200, response.text)

    def test_optional_noise_feature_can_be_omitted(self):
        example = self.client.get("/metadata.json").json()["example"]
        example.pop("jumbled_noise_1")
        response = self.client.post("/predict/mushroom", json=example)
        self.assertEqual(response.status_code, 200, response.text)

    def test_retired_features_and_invalid_values_are_rejected(self):
        example = self.client.get("/metadata.json").json()["example"]
        for payload in [{**example, "jumbled_noise_0": "a"}, {**example, "stem-height": 2},
                        {**example, "stem_height": -1}, {**example, "cap_shape": "invalid"},
                        {**example, "stem_width": "2.4"}]:
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post("/predict/mushroom", json=payload).status_code, 422)

    def test_changed_forest_is_rejected_before_loading(self):
        metadata = json.loads(DEFAULT_MODEL.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder/metadata["forest_file"]).write_bytes(b"changed artifact")
            manifest = folder/"model.json"
            manifest.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                PortableMushroom(manifest)

    def test_city_prediction_matches_the_frontend_chart_contract(self):
        response = self.client.post("/predict/citibike", json={"date": "2026-04-30", "hour": 8, "mode": "demo"})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["date"], "2026-04-30")
        self.assertEqual(result["hour"], 8)
        self.assertEqual(result["timezone"], "America/New_York")
        rows = result["daily_predictions"]
        self.assertEqual([row["hour"] for row in rows], list(range(24)))
        self.assertTrue(all(row["predicted_ride_starts"] >= 0 for row in rows))
        self.assertEqual(result["predicted_ride_starts"], rows[8]["predicted_ride_starts"])
        metrics = self.client.get("/citibike_metrics.json").json()
        self.assertEqual(result["test_mae"], metrics["metrics"]["MAE"])
        self.assertEqual(result["model_id"], CITY_MODEL_ID)
        self.assertAlmostEqual(result["daily_total"], sum(row["predicted_ride_starts"] for row in rows))

    def test_city_requires_recent_history_for_other_dates(self):
        response = self.client.post("/predict/citibike", json={"date": "2026-10-09", "hour": 23})
        self.assertEqual(response.status_code, 422, response.text)
        self.assertIn("eigen historie", response.json()["detail"])

    def test_city_matches_original_saved_model_and_dst_contract(self):
        fixtures = json.loads((ROOT/"backend/fixtures/citibike_parity.json").read_text(encoding="utf-8"))
        for fixture in fixtures:
            response = self.client.post("/predict/citibike", json={"date": fixture["date"], "hour": 8,
                                        "mode": "history", "history": fixture["history"]})
            self.assertEqual(response.status_code, 200, response.text)
            rows = response.json()["daily_predictions"]
            np.testing.assert_allclose([row["predicted_ride_starts"] for row in rows], fixture["expected"], rtol=1e-12, atol=1e-9)
            self.assertEqual([row["clock_hours"] for row in rows], fixture["clock_hours"])
        self.assertEqual(self.client.get("/citibike/model").json()["model_id"], CITY_MODEL_ID)

    def test_city_rejects_jersey_missing_duplicates_future_and_timezone(self):
        fixture = json.loads((ROOT/"backend/fixtures/citibike_parity.json").read_text(encoding="utf-8"))[0]
        valid = {"date": fixture["date"], "hour": 8, "mode": "history", "history": fixture["history"]}
        cases = [None, fixture["history"][:-1]]
        for field, value in [("area", "JC"), ("rides", -1), ("rides", "123"),
                             ("timestamp", "2026-04-30T00:00:00"), ("timestamp", "2026-04-16T00:00:00Z"),
                             ("timestamp", fixture["history"][1]["timestamp"])]:
            history = [dict(row) for row in fixture["history"]]
            history[0][field] = value
            cases.append(history)
        for history in cases:
            with self.subTest(history=history[0] if history else None):
                response = self.client.post("/predict/citibike", json={**valid, "history": history})
                self.assertEqual(response.status_code, 422, response.text)

    def test_invalid_city_date_hour_and_extra_fields_are_rejected(self):
        valid = {"date": "2026-04-30", "hour": 8}
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
                     "/backend/artifacts/mushroom_rf300.npz", "/backend/artifacts/mushroom_rf300.json",
                     "/secondary_mushroom/models/baseline_decision_tree.joblib",
                     "/nyc_citi_bike/models/baseline_hourly_tree.joblib",
                     "/SolutionAndrew/MushroomDataset/models/random_forest.joblib"):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 404)


if __name__ == "__main__":
    unittest.main()
