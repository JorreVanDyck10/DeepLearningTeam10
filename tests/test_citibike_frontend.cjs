const { test } = require("node:test");
const assert = require("node:assert/strict");
const { parseHistoryCsv, recentHistory, validateCityResult, MODEL_ID } = require("../frontend/citibike.js");

test("CSV preserves counts and accepts BOM, quotes and CRLF", () => {
  assert.deepEqual(parseHistoryCsv('\uFEFFtimestamp,rides,area\r\n"2026-04-29 08:00:00",1234,NYC\r\n'),
    [{ timestamp: "2026-04-29T08:00:00", rides: 1234, area: "NYC" }]);
});
test("invalid counts, timezone, region and malformed CSV are rejected", () => {
  for (const row of ["2026-04-29 08:00:00,-1,NYC", "2026-04-29 08:00:00,,NYC",
      "2026-04-29 08:00:00,NaN,NYC", "2026-04-29 08:00:00,12,JC", "2026-04-29T08:00:00Z,12,NYC",
      "2026-04-29 08:00:00,12", '"2026-04-29 08:00:00,12,NYC']) {
    assert.throws(() => parseHistoryCsv(`timestamp,rides,area\n${row}`));
  }
  assert.throws(() => parseHistoryCsv("timestamp,rides,rides,area\na,1,1,NYC"));
});
test("only fourteen previous calendar days go to the API", () => {
  const origin = Date.parse("2026-11-01T00:00:00Z");
  const rows = Array.from({ length: 15 * 24 }, (_, i) => ({
    timestamp: new Date(origin - 14 * 86400000 + i * 3600000).toISOString().replace("Z", ""), rides: 100, area: "NYC",
  }));
  const recent = recentHistory(rows, "2026-11-01");
  assert.equal(recent.length, 336);
  assert.ok(recent.every(row => row.timestamp < "2026-11-01"));
  assert.throws(() => recentHistory(rows.slice(1), "2026-11-01"), /336/);
});
test("a baseline response can never be presented as Raouls model", () => {
  assert.throws(() => validateCityResult({ model_id: "baseline_hourly_tree" }), /nog niet Raouls/);
  const daily_predictions = Array.from({ length: 24 }, (_, hour) => ({ hour, predicted_ride_starts: 100, clock_hours: 1 }));
  assert.doesNotThrow(() => validateCityResult({ model_id: MODEL_ID, daily_total: 2400, predicted_ride_starts: 100, daily_predictions }));
  assert.throws(() => validateCityResult({ model_id: MODEL_ID, daily_total: 2400, predicted_ride_starts: 100, daily_predictions: daily_predictions.slice(1) }));
});
