import unittest
from backend.mushroom_policy import decision_for_score, apply_policy, POLICY_ID


class MushroomPolicyTests(unittest.TestCase):
    def test_no_score_confirms_edibility(self):
        for score in [0, .005, .01, .05, .202, .499999]:
            self.assertEqual(decision_for_score(score)["decision"], "uncertain")

    def test_tie_and_poisonous_scores_warn(self):
        for score in [.5, .75, 1]:
            self.assertEqual(decision_for_score(score)["decision"], "suspected_poisonous")

    def test_scientific_label_and_probability_are_preserved(self):
        result = apply_policy({"label": "e", "prediction": "eetbaar", "probability_poisonous": .03465518938833522})
        self.assertEqual(result["label"], "e")
        self.assertEqual(result["probability_poisonous"], .03465518938833522)
        self.assertEqual(result["prediction"], "Onzeker / mogelijk giftig")
        self.assertEqual(result["decision_policy_id"], POLICY_ID)

    def test_invalid_probabilities_are_rejected(self):
        for score in [float("nan"), float("inf"), -.01, 1.01]:
            with self.assertRaises(ValueError): decision_for_score(score)
