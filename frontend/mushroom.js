"use strict";
const MushroomDecision = {
  POLICY_ID: "mushroom_no_edibility_confirmation_v1",
  present(result) {
    const score = result.probability_poisonous;
    if (!["e", "p"].includes(result.label) || !Number.isFinite(score) || score < 0 || score > 1) {
      throw new Error("De API gaf een onverwacht resultaat terug.");
    }
    const expected = score >= 0.5 ? "suspected_poisonous" : "uncertain";
    if (result.decision_policy_id !== this.POLICY_ID || result.decision !== expected) {
      throw new Error("De backend gebruikt nog de oude beslisregel. Wacht op de deployment en vernieuw de pagina.");
    }
    return expected === "suspected_poisonous"
      ? { title: "Model voorspelt: giftig", explanation: "Het model wijst op de giftige klasse. Dit is een educatieve voorspelling." }
      : { title: "Onzeker / mogelijk giftig", explanation: "Ook een hoge score voor de eetbare klasse sluit giftigheid niet uit. Dit model bevestigt geen eetbaarheid." };
  }
};
if (typeof module === "object" && module.exports) module.exports = MushroomDecision;
else window.MushroomDecision = MushroomDecision;
