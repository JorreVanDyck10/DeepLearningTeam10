const { test, afterEach } = require("node:test");
const assert = require("node:assert/strict");
const { requestJson } = require("../frontend/api.js");

const originalFetch = global.fetch;
afterEach(() => { global.fetch = originalFetch; });
function respond(body, status = 200) {
  global.fetch = async () => new Response(body, { status });
}

test("valid prediction JSON is returned and request body is preserved", async () => {
  const result = { label: "e", probability_poisonous: 0.415 };
  const payload = JSON.stringify({ "cap-diameter": 13.76 });
  global.fetch = async (url, options) => {
    assert.equal(url, "/predict/mushroom");
    assert.equal(options.method, "POST");
    assert.equal(options.body, payload);
    assert.equal(options.cache, "no-store");
    assert.ok(options.signal instanceof AbortSignal);
    return new Response(JSON.stringify(result));
  };
  assert.deepEqual(await requestJson("/predict/mushroom", { method: "POST", body: payload }), result);
});

test("empty successful response gets a useful message", async () => {
  respond("");
  await assert.rejects(requestJson("/predict/mushroom"), /leeg antwoord/);
});

test("empty gateway error reports its HTTP status", async () => {
  respond("", 502);
  await assert.rejects(requestJson("/predict/mushroom"), /tijdelijk.*HTTP 502/);
});

test("HTML gateway response is not shown as JSON or raw HTML", async () => {
  respond("<html>Bad gateway</html>", 504);
  await assert.rejects(requestJson("/predict/mushroom"), /tijdelijk.*HTTP 504/);
});

test("empty missing route still reports that the route is unavailable", async () => {
  for (const status of [404, 405]) {
    respond("", status);
    await assert.rejects(requestJson("/predict/citibike"), /route is nog niet beschikbaar/);
  }
});

test("model unavailable detail is retained", async () => {
  respond(JSON.stringify({ detail: "Model niet beschikbaar." }), 503);
  await assert.rejects(requestJson("/predict/mushroom"), /^Error: Model niet beschikbaar\.$/);
});

test("field validation messages are retained", async () => {
  respond(JSON.stringify({ detail: [{ loc: ["body", "stem-width"], msg: "Must be positive" }] }), 422);
  await assert.rejects(requestJson("/predict/mushroom"), /stem-width: Must be positive/);
});

test("invalid or non-object successful response is rejected", async () => {
  for (const body of ["<html>Login</html>", '{"label":', "null", "[]"]) {
    respond(body);
    await assert.rejects(requestJson("/predict/mushroom"), /geen geldig JSON-antwoord/);
  }
});

test("network failures remain distinguishable from response failures", async () => {
  const failure = new TypeError("Failed to fetch");
  global.fetch = async () => { throw failure; };
  await assert.rejects(requestJson("/health"), (error) => error === failure);
});
