"""The published Netlify site can call Citybike; unrelated sites stay excluded."""
import unittest
from fastapi.testclient import TestClient
from backend.main import app


class CityCorsTests(unittest.TestCase):
    def test_city_preflight_from_netlify(self):
        response = TestClient(app).options("/predict/citibike", headers={
            "Origin": "https://10messi10.netlify.app",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"],
                         "https://10messi10.netlify.app")

    def test_unrelated_origin_is_not_allowed(self):
        response = TestClient(app).options("/predict/citibike", headers={
            "Origin": "https://other.netlify.app",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        })
        self.assertEqual(response.status_code, 400)
        self.assertNotIn("access-control-allow-origin", response.headers)
