# Frontend Team 10

Een statische HTML/CSS/JavaScript-website in dezelfde repository als de API.
Geen Node-installatie of buildtool nodig. De paddenstoelenpagina gebruikt echte API-inference;
Citi Bike voorspelt een gekozen uur en toont daarnaast 24 uurvoorspellingen als grafiek en tabel. De projectpagina toont gemeten baseline-metrics voor beide datasets.

Alle websitebestanden staan in deze map: `index.html`, `styles.css`, `app.js`,
`config.js`, `metadata.json` en `citibike_metrics.json`. `export_metadata.py`
is een hulpscript om de paddenstoelenmetadata opnieuw te exporteren.
De standaard backend-URL is `https://deeplearningteam10-api.onrender.com`.
De Citi Bike-pagina vereist dat ook `/predict/citibike` op die backend beschikbaar is.

## Lokaal

Laat de backend draaien op poort 8000. Open een tweede PowerShell-venster:

```powershell
cd C:\DeepLearning\Opdracht\DeepLearningTeam10
python -m http.server 5173 --bind 127.0.0.1 --directory frontend
```

Open http://127.0.0.1:5173, klik **Voorbeeld invullen** en **Voorspellen**.
Open de pagina via HTTP, niet door dubbelklikken op index.html.
Vul voor de lokale backend `http://127.0.0.1:8000` in bij
**Verbindingsinstellingen**; standaard gebruikt de site de publieke Render-API.

## Render Static Site

1. Stel `window.GREENLAB_API_URL` in `frontend/config.js` in op de echte HTTPS-URL van de backend.
2. Commit en push de frontendmap naar jullie GitHub-repository.
3. Kies op Render **New > Static Site**, dezelfde repo en branch.
4. Laat Root Directory leeg, gebruik `echo "Static site: no build required"` als Build Command
   en `frontend` als Publish Directory.
5. Voeg de werkelijke frontend-origin (zonder trailing slash) toe aan `CORS_ORIGINS` van de backend
   in `render.yaml`, naast de lokale origins. Push en laat de backend opnieuw deployen.
6. Test de publieke site, ook na minstens 15 minuten zonder API-verkeer.

De Verbindingsinstellingen op de website kunnen de URL voor de huidige browser overschrijven.
Dit is handig voor ontwikkelen; `config.js` moet voor bezoekers al de juiste URL bevatten.
De frontend wacht maximaal twee minuten op de API en toont uitleg over de koude start.

## Modelmetadata bijwerken

```powershell
.\.venv\Scripts\python.exe frontend/export_metadata.py
```

Dit leest alleen de vertrouwde lokale baseline en exporteert de categorieÃ«n, het voorbeeld
en de metrics naar `metadata.json`. Controleer vertalingen bij model/schemawijzigingen.
Categoriebetekenissen en meeteenheden komen uit de [UCI-metadata](https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset).
De UI toont alleen categorieÃ«n die de opgeslagen encoder kent. Onbekende waarden worden als null verstuurd.
Voor een toekomstig automatisch bijgewerkt model moet metadata samen met de modelversie worden bijgewerkt.

Codex maakte deze frontend, labels en koppeling. Controleer deze bijdrage volgens de GenAI-policy.

De Citi Bike-pagina verwacht `/predict/citibike`. Het trainingsscript vernieuwt `citibike_metrics.json`; commit de bijgewerkte metrics samen met het kleine model. De eerste versie gebruikt januari 2025 en vermeldt die beperking op de pagina.
