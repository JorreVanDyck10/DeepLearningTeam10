"use strict";

// Render or a proxy may return an empty/HTML error instead of API JSON.
async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    cache: "no-store", ...options,
    signal: options.signal ?? AbortSignal.timeout(120000),
  });
  const body = await response.text();
  let data;
  try { data = JSON.parse(body); } catch { /* Report the HTTP error below. */ }

  if (!response.ok) {
    if ([404, 405].includes(response.status)) {
      throw new Error("Deze route is nog niet beschikbaar op de backend. Controleer de backend-URL bij Verbindingsinstellingen.");
    }
    if (response.status >= 500) {
      const detail = typeof data?.detail === "string" ? data.detail : null;
      throw new Error(detail ?? `De server is tijdelijk niet beschikbaar (HTTP ${response.status}). Wacht even en probeer opnieuw.`);
    }
    let message = typeof data?.detail === "string" ? data.detail : "Controleer de invoer en probeer opnieuw.";
    if (Array.isArray(data?.detail)) {
      const details = data.detail.filter((item) => typeof item?.msg === "string").map((item) => {
        const field = Array.isArray(item.loc) ? item.loc.slice(1).join(".") : "";
        return field ? `${field}: ${item.msg}` : item.msg;
      });
      if (details.length) message = details.join(" · ");
    }
    throw new Error(message);
  }
  if (!body.trim()) throw new Error("De server gaf een leeg antwoord. Wacht even en probeer opnieuw.");
  if (!data || typeof data !== "object" || Array.isArray(data)) {
    throw new Error("De server gaf geen geldig JSON-antwoord. Controleer de backend-URL en probeer opnieuw.");
  }
  return data;
}

if (typeof module === "object" && module.exports) {
  module.exports = { requestJson };
} else {
  window.Team10API = { requestJson };
}
