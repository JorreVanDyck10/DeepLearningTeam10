"""Separate the statistical class from an edibility decision."""
import math

POLICY_ID = "mushroom_no_edibility_confirmation_v1"


def decision_for_score(score):
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid Mushroom probability")
    if score >= 0.5:
        return {"decision": "suspected_poisonous", "prediction": "Model voorspelt: giftig"}
    # Even >=95% edible scores contained poisonous out-of-fold examples.
    return {"decision": "uncertain", "prediction": "Onzeker / mogelijk giftig"}


def apply_policy(prediction):
    return {**prediction, **decision_for_score(prediction["probability_poisonous"]),
            "decision_policy_id": POLICY_ID}
