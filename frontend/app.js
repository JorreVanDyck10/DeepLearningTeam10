"use strict";
const $ = (selector) => document.querySelector(selector);
let metadata;
let busy = false;
let apiUrl = window.GREENLAB_API_URL;
try { apiUrl = localStorage.getItem("greenlab-api") || apiUrl; } catch { /* Storage may be disabled. */ }
apiUrl = apiUrl.replace(/\/+$/, "");
$("#api-url").value = apiUrl;
const percent = (n) => new Intl.NumberFormat("nl-BE", { style: "percent", maximumFractionDigits: 1 }).format(n);

function route() {
  const id = ["mushroom", "citibike", "project"].includes(location.hash.slice(1)) ? location.hash.slice(1) : "mushroom";
  document.querySelectorAll(".page").forEach((page) => { page.hidden = page.id !== id; });
  document.querySelectorAll(".nav-link").forEach((link) => {
    const active = link.hash === `#${id}`;
    link.classList.toggle("active", active);
    if (active) link.setAttribute("aria-current", "page"); else link.removeAttribute("aria-current");
  });
}
window.addEventListener("hashchange", route);
route();

async function api(path, options = {}) {
  return window.Team10API.requestJson(`${apiUrl}${path}`, options);
}
async function checkConnection() {
  const status = $("#connection");
  status.textContent = "Verbinding controleren…";
  status.classList.remove("ready");
  try {
    const health = await api("/health");
    if (!health.mushroom_model_loaded) throw new Error("Model niet geladen");
    status.textContent = health.citibike_model_loaded ? "Backend verbonden" : "Paddenstoelenmodel verbonden";
    status.classList.add("ready");
  } catch { status.textContent = "Backend niet bereikbaar"; }
}
$("#settings-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const url = new URL($("#api-url").value);
  if (!["http:", "https:"].includes(url.protocol) || url.username || url.password || url.search || url.hash) {
    $("#settings-message").textContent = "Gebruik een HTTP(S)-URL zonder inloggegevens, query of fragment.";
    return;
  }
  apiUrl = url.href.replace(/\/+$/, "");
  try { localStorage.setItem("greenlab-api", apiUrl); } catch { /* Still usable for this visit. */ }
  $("#settings-message").textContent = "URL ingesteld voor deze browser.";
  checkConnection();
});

function fieldElement(field) {
  const wrapper = document.createElement("div");
  wrapper.className = "field";
  const label = document.createElement("label");
  label.htmlFor = field.name;
  label.textContent = field.label + (field.unit ? ` (${field.unit})` : "");
  const input = document.createElement(field.options ? "select" : "input");
  input.id = field.name;
  input.name = field.name;
  if (field.options) {
    input.append(new Option("Onbekend", ""));
    field.options.forEach(([code, text]) => input.append(new Option(text, code)));
  } else { input.type = "number"; input.min = "0"; input.step = "any"; input.placeholder = "Onbekend"; }
  wrapper.append(label, input);
  return wrapper;
}
function clearResult() {
  $("#result").innerHTML = '<p class="muted">Nog geen voorspelling.</p>';
}
$("#example").addEventListener("click", () => {
  if (!metadata || busy) return;
  for (const [name, value] of Object.entries(metadata.example)) $("#mushroom-form").elements.namedItem(name).value = value ?? "";
  clearResult();
});
$("#reset-button").addEventListener("click", () => { if (!busy) { $("#mushroom-form").reset(); clearResult(); } });
$("#mushroom-form").addEventListener("input", () => { if (!busy) clearResult(); });
$("#mushroom-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!metadata || busy) return;
  const payload = {};
  for (const field of metadata.fields) {
    const value = event.target.elements.namedItem(field.name).value;
    payload[field.name] = value === "" ? null : field.options ? value : Number(value);
  }
  busy = true;
  const controls = [...document.querySelectorAll("#mushroom-form input, #mushroom-form select, #mushroom-form button, #example, #settings-form button")];
  controls.forEach((el) => { el.disabled = true; });
  $("#predict-button").textContent = "Voorspelling ophalen…";
  $("#result").innerHTML = '<p>Voorspelling ophalen…</p><p class="muted">De server kan ongeveer een minuut nodig hebben om op te starten.</p>';
  try {
    const result = await api("/predict/mushroom", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    if (!["p", "e"].includes(result.label) || !Number.isFinite(result.probability_poisonous) || result.probability_poisonous < 0 || result.probability_poisonous > 1) throw new Error("De API gaf een onverwacht resultaat terug.");
    $("#result").innerHTML = '<div class="result-label"></div><p class="score"></p><p class="muted">De modelscore geeft geen zekerheid over eetbaarheid.</p>';
    $(".result-label").textContent = result.label === "p" ? "Giftig / onbekend" : "Eetbaar";
    $(".score").textContent = `Modelscore giftige klasse: ${percent(result.probability_poisonous)}`;
    $("#connection").textContent = "Backend verbonden";
    $("#connection").classList.add("ready");
  } catch (error) {
    $("#result").innerHTML = '<p>Voorspelling niet gelukt.</p><p class="error"></p><p class="muted">Controleer de URL bij Verbindingsinstellingen en of de backend draait.</p>';
    $(".error").textContent = error instanceof TypeError ? "Geen verbinding met de API. Controleer de server en CORS-instellingen." : error.name === "TimeoutError" ? "De server reageerde niet binnen twee minuten. Probeer opnieuw." : error.message;
  } finally {
    busy = false;
    controls.forEach((el) => { el.disabled = false; });
    $("#predict-button").textContent = 'Voorspellen';
  }
});

async function init() {
  try {
    metadata = await window.Team10API.requestJson("metadata.json");
    metadata.fields.forEach((field, i) => $(i < 6 ? "#basic-fields" : "#advanced-fields").append(fieldElement(field)));
    for (const [label, value] of [["Accuracy", percent(metadata.metrics.accuracy)], ["Recall giftige klasse", percent(metadata.metrics.recall_poisonous)], ["Giftig als eetbaar voorspeld", String(metadata.metrics.poisonous_predicted_edible)]]) {
      const tile = document.createElement("div"); tile.className = "metric";
      const strong = document.createElement("strong"); strong.textContent = value;
      const caption = document.createElement("span"); caption.textContent = label;
      tile.append(strong, caption); $("#metrics").append(tile);
    }
    checkConnection();
  } catch (error) {
    const message = error instanceof TypeError ? "Geen verbinding met de server." : error.name === "TimeoutError" ? "De server reageerde niet op tijd." : error.message;
    $("#result").textContent = `Het formulier kon niet worden geladen. ${message} Vernieuw de pagina en probeer opnieuw.`;
    $("#predict-button").disabled = true;
  }
}
init();

const number = (value) => new Intl.NumberFormat("nl-BE", { maximumFractionDigits: 0 }).format(value);
for (let hour = 0; hour < 24; hour++) $("#city-hour").append(new Option(`${String(hour).padStart(2, "0")}:00`, hour));
$("#city-hour").value = "8";
let cityBusy = false;
$("#city-form").addEventListener("input", () => {
  if (!cityBusy) { $("#city-result").innerHTML = '<p class="muted">Nog geen voorspelling.</p>'; $("#city-day").hidden = true; }
});
$("#city-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (cityBusy) return;
  cityBusy = true;
  const controls = [...document.querySelectorAll("#city-form input, #city-form select, #city-form button")];
  controls.forEach((el) => { el.disabled = true; });
  $("#city-day").hidden = true;
  $("#city-result").innerHTML = '<p>Voorspelling ophalen…</p><p class="muted">De server kan ongeveer een minuut nodig hebben om op te starten.</p>';
  try {
    const data = await api("/predict/citibike", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({date:$("#city-date").value, hour:Number($("#city-hour").value)})});
    if (!Number.isFinite(data.predicted_ride_starts) || !Array.isArray(data.daily_predictions) || data.daily_predictions.length !== 24 || data.daily_predictions.some((row, index) => row.hour !== index || !Number.isFinite(row.predicted_ride_starts) || row.predicted_ride_starts < 0)) throw new Error("Onverwacht API-resultaat.");
    $("#city-result").innerHTML = '<div class="result-label"></div><p class="city-time"></p><p class="muted city-warning"></p>';
    $("#city-result .result-label").textContent = `${number(data.predicted_ride_starts)} ritstarts`;
    $(".city-time").textContent = `${data.date}, ${String(data.hour).padStart(2,"0")}:00–${String(data.hour+1).padStart(2,"0")}:00 (New York)`;
    $(".city-warning").textContent = data.warning;
    drawCityChart(data.daily_predictions, data.hour);
    $("#city-day").hidden = false;
  } catch (error) {
    $("#city-result").innerHTML = '<p>Voorspelling niet gelukt.</p><p class="error"></p>';
    $("#city-result .error").textContent = error instanceof TypeError ? "Geen verbinding met de backend. Controleer de verbindingsinstellingen." : error.name === "TimeoutError" ? "De server reageerde niet binnen twee minuten. Probeer opnieuw." : error.message;
  } finally {
    cityBusy = false;
    controls.forEach((el) => { el.disabled = false; });
  }
});

function drawCityChart(rows, selectedHour) {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns,"svg");
  svg.setAttribute("viewBox","0 0 760 250"); svg.setAttribute("role","img");
  svg.setAttribute("aria-label","Voorspelde ritstarts per uur. Exacte waarden staan in de tabel eronder.");
  const maximum = Math.max(1, ...rows.map((row) => row.predicted_ride_starts));
  const top = Math.ceil(maximum / 1000) * 1000;
  function element(tag, attributes, text) {
    const node = document.createElementNS(ns, tag);
    Object.entries(attributes).forEach(([key,value])=>node.setAttribute(key,String(value)));
    if(text!==undefined) node.textContent=text;
    svg.append(node); return node;
  }
  for(let step=0;step<=4;step++) {
    const y=20+step*45;
    element("line",{x1:56,y1:y,x2:752,y2:y,stroke:"#e5e5e5"});
    element("text",{x:48,y:y+4,"text-anchor":"end","font-size":11,fill:"#666"},number(top*(1-step/4)));
  }
  $("#city-table").replaceChildren();
  rows.forEach((row)=>{
    const height=row.predicted_ride_starts/top*180;
    const rect=element("rect",{x:60+row.hour*28.5,y:200-height,width:20,height,rx:3,fill:row.hour===selectedHour?"#245e8e":"#83adce"});
    const title=document.createElementNS(ns,"title"); title.textContent=`${row.hour}:00: ${number(row.predicted_ride_starts)} ritstarts`; rect.append(title);
    if(row.hour%3===0) element("text",{x:70+row.hour*28.5,y:221,"text-anchor":"middle","font-size":11,fill:"#666"},`${row.hour}:00`);
    const tr=document.createElement("tr");
    for(const text of [`${String(row.hour).padStart(2,"0")}:00`,number(row.predicted_ride_starts)]) {const td=document.createElement("td");td.textContent=text;tr.append(td);}
    $("#city-table").append(tr);
  });
  element("text",{x:405,y:245,"text-anchor":"middle","font-size":11,fill:"#666"},"Uur in New York");
  $("#city-chart").replaceChildren(svg);
}

window.Team10API.requestJson("citibike_metrics.json").then(metrics=>{
  const entries=[["MAE beslisboom",number(metrics.results.decision_tree.mae)], ["MAE vast gemiddelde",number(metrics.results.global_mean.mae)], ["MAE per weekdag / uur",number(metrics.results.weekday_hour_mean.mae)]];
  for(const [label,value] of entries) {
    const tile=document.createElement("div");tile.className="metric";
    const strong=document.createElement("strong");strong.textContent=value;
    const span=document.createElement("span");span.textContent=label;tile.append(strong,span);$("#city-metrics").append(tile);
  }
}).catch(()=>{$("#city-metrics").textContent="Modelresultaten niet beschikbaar.";});
