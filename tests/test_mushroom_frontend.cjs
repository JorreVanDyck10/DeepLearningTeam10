const test = require('node:test');
const assert = require('node:assert/strict');
const policy = require('../frontend/mushroom.js');
test('low poison scores, including 0% and 95% edible, never confirm edibility', () => {
  for (const probability_poisonous of [0, 0.01, 0.05, 0.202, 0.4999]) {
    const view = policy.present({label:'e', probability_poisonous, decision:'uncertain', decision_policy_id:policy.POLICY_ID});
    assert.equal(view.title,'Onzeker / mogelijk giftig');
  }
});
test('tie and higher poison scores warn about the poisonous class', () => {
  for (const probability_poisonous of [0.5, 0.8, 1]) {
    const view = policy.present({label:'p', probability_poisonous, decision:'suspected_poisonous', decision_policy_id:policy.POLICY_ID});
    assert.equal(view.title,'Model voorspelt: giftig');
  }
});
test('old, contradictory and malformed API responses are rejected', () => {
  for (const result of [
    {label:'e', prediction:'eetbaar', probability_poisonous:0.202},
    {label:'e',probability_poisonous:0.01,decision:'edible',decision_policy_id:policy.POLICY_ID},
    {label:'p',probability_poisonous:0.8,decision:'uncertain',decision_policy_id:policy.POLICY_ID},
    {label:'e',probability_poisonous:NaN}, {label:'e',probability_poisonous:-0.1},
    {label:'e',probability_poisonous:1.1}, {label:'e',probability_poisonous:'0.02'}
  ]) assert.throws(()=>policy.present(result));
});
