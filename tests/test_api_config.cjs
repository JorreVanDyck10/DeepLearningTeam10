const { test } = require("node:test");
const assert = require("node:assert/strict");
const { resolveApiUrl } = require("../frontend/config.js");

function location(url) {
  return new URL(url);
}

test("Netlify uses Render on a new visit and migrates its old stored default", () => {
  const site = location("https://10messi10.netlify.app");
  for (const saved of [undefined, "", site.origin, `${site.origin}/`]) {
    assert.equal(resolveApiUrl(site, saved), "https://deeplearningteam10-api.onrender.com");
  }
});

test("Render remains same-origin and the separate local frontend uses port 8000", () => {
  assert.equal(resolveApiUrl(location("https://deeplearningteam10-api.onrender.com")),
    "https://deeplearningteam10-api.onrender.com");
  for (const host of ["localhost", "127.0.0.1", "[::1]"]) {
    assert.equal(resolveApiUrl(location(`http://${host}:5173`)), `http://${host}:8000`);
    assert.equal(resolveApiUrl(location(`http://${host}:8000`)), `http://${host}:8000`);
  }
});

test("obsolete browser settings cannot override the automatic backend", () => {
  assert.equal(resolveApiUrl(location("https://10messi10.netlify.app"), "https://custom.example/"),
    "https://deeplearningteam10-api.onrender.com");
});
