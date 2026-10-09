"use strict";

const CITY_MODEL_ID = "raoul_tuned_random_forest_320";

function parseHistoryCsv(text) {
  // Small aggregate CSV only; quoted fields, CRLF and BOM are supported.
  if (text.length > 5 * 1024 * 1024) throw new Error("Gebruik een klein CSV-bestand met uurtellingen, maximaal 5 MB.");
  const rows = []; let row = [], field = "", quoted = false;
  text = text.replace(/^\uFEFF/, "");
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '"') {
      if (quoted && text[i + 1] === '"') { field += '"'; i++; }
      else quoted = !quoted;
    } else if (!quoted && (c === "," || c === "\n" || c === "\r")) {
      row.push(field.trim()); field = "";
      if (c !== ",") {
        if (row.some(Boolean)) rows.push(row);
        row = [];
        if (c === "\r" && text[i + 1] === "\n") i++;
      }
    } else field += c;
  }
  if (quoted) throw new Error("Het CSV-bestand bevat een onafgesloten aanhalingsteken.");
  if (field || row.length) { row.push(field.trim()); if (row.some(Boolean)) rows.push(row); }
  const header = rows.shift();
  const columns = ["timestamp", "rides", "area"];
  if (!header || columns.some(name => header.filter(value => value === name).length !== 1)) {
    throw new Error("Het CSV-bestand moet timestamp, rides en area bevatten.");
  }
  const indices = columns.map(name => header.indexOf(name));
  return rows.map((values, i) => {
    if (values.length !== header.length) throw new Error(`CSV-regel ${i + 2} heeft een onjuist aantal kolommen.`);
    const [timestamp, count, area] = indices.map(index => values[index]);
    if (area !== "NYC") throw new Error("Gebruik uitsluitend NYC-data; Jersey City hoort hier niet bij.");
    if (!/^\d{4}-\d{2}-\d{2}[T ]\d{2}:00(?::00(?:\.0+)?)?$/.test(timestamp)) {
      throw new Error(`CSV-regel ${i + 2}: gebruik lokale uurvakken zonder tijdzone, bijvoorbeeld 2026-04-29 08:00:00.`);
    }
    const rides = Number(count);
    if (!count || !Number.isFinite(rides) || rides < 0) throw new Error(`CSV-regel ${i + 2}: ongeldig aantal ritten.`);
    return { timestamp: timestamp.replace(" ", "T"), rides, area };
  });
}

function recentHistory(rows, date) {
  const origin = new Date(`${date}T00:00:00Z`);
  if (!Number.isFinite(origin.getTime())) throw new Error("Kies een geldige voorspeldatum.");
  const first = new Date(origin.getTime() - 14 * 86400000).toISOString().slice(0, 10);
  const recent = rows.filter(row => row.timestamp.slice(0, 10) >= first && row.timestamp.slice(0, 10) < date)
    .sort((a, b) => a.timestamp.localeCompare(b.timestamp));
  if (recent.length !== 336) throw new Error("Voor deze datum ontbreken 14 volledige voorafgaande dagen (336 lokale uurvakken).");
  return recent;
}

function validateCityResult(data) {
  if (data.model_id !== CITY_MODEL_ID) throw new Error("Deze backend gebruikt nog niet Raouls gekozen model. Wacht op de deployment of controleer de backend-URL.");
  if (!Number.isFinite(data.predicted_ride_starts) || !Number.isFinite(data.daily_total) ||
      !Array.isArray(data.daily_predictions) || data.daily_predictions.length !== 24 ||
      data.daily_predictions.some((row, index) => row.hour !== index ||
        !Number.isFinite(row.predicted_ride_starts) || row.predicted_ride_starts < 0 || ![0, 1, 2].includes(row.clock_hours))) {
    throw new Error("Onverwacht API-resultaat.");
  }
}

const CityBike = { MODEL_ID: CITY_MODEL_ID, parseHistoryCsv, recentHistory, validateCityResult };
if (typeof module === "object" && module.exports) module.exports = CityBike;
else window.CityBike = CityBike;
