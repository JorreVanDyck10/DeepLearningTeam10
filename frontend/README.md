# Frontend Team 10

Een statische HTML/CSS/JavaScript-website in dezelfde repository als de API.
Geen Node-installatie of buildtool nodig. De paddenstoelenpagina gebruikt echte API-inference;
Citi Bike voorspelt een gekozen uur en toont daarnaast 24 uurvoorspellingen als grafiek en tabel. De projectpagina toont gemeten baseline-metrics voor beide datasets.

Alle websitebestanden staan in deze map: `index.html`, `styles.css`, `app.js`,
`config.js`, `api.js`, `metadata.json` en `citibike_metrics.json`. `export_metadata.py`
is een hulpscript om de paddenstoelenmetadata opnieuw te exporteren.
De standaard backend-URL is automatisch hetzelfde domein als de website.
Website en API worden samen gehost op `https://deeplearningteam10-api.onrender.com/`.
Beide voorspelroutes zijn beschikbaar op dezelfde backend.

Het voorlopige paddenstoelenmodel is Andrew's Random Forest uit
`SolutionAndrew/MushroomDataset/`. Het formulier toont de twaalf bijbehorende
velden en de testresultaten van deze specifieke dataset/modelversie.
De twee ruisvelden zijn optionele extra datasetkenmerken.

## Lokaal

Laat de backend draaien op poort 8000. Open een tweede PowerShell-venster:

```powershell
cd C:\DeepLearning\Opdracht\DeepLearningTeam10
python -m http.server 5173 --bind 127.0.0.1 --directory frontend
```

Open http://127.0.0.1:5173, klik **Voorbeeld invullen** en **Voorspellen**.
Open de pagina via HTTP, niet door dubbelklikken op index.html.
Op deze lokale ontwikkelserver kiest de website automatisch de API op poort 8000.
Je kunt ook alleen de backend starten en http://127.0.0.1:8000/ openen:
die serveert nu zowel de website als de API.

## Automatisch publiceren op Render

Push de website- en backendwijzigingen naar `main`. De bestaande Render Web Service
publiceert beide samen. `config.js` gebruikt automatisch het domein waarop de site
draait. Voor deze opstelling is geen afzonderlijke Static Site of CORS-origin nodig.
Websitebestanden worden bij het opnieuw openen/vernieuwen van de pagina opnieuw
gevalideerd; voorspellingen worden rechtstreeks bij de actuele API opgevraagd.

Zie [de deployment-instructies](../docs/deployment.md) voor de eenmalige
Render-instellingen, GitHub-controles en de lokale testopdracht.

## Optioneel: een afzonderlijke Render Static Site

1. Stel `window.GREENLAB_API_URL` in `frontend/config.js` in op de echte HTTPS-URL van de backend.
2. Commit en push de frontendmap naar jullie GitHub-repository.
3. Kies op Render **New > Static Site**, dezelfde repo en branch.
4. Laat Root Directory leeg, gebruik `echo "Static site: no build required"` als Build Command
   en `frontend` als Publish Directory.
5. Voeg de werkelijke frontend-origin (zonder trailing slash) toe aan `CORS_ORIGINS` van de backend
   in `render.yaml`, naast de lokale origins. Push en laat de backend opnieuw deployen.
6. Test de publieke site, ook na minstens 15 minuten zonder API-verkeer.

De Verbindingsinstellingen op de website kunnen de URL voor de huidige browser overschrijven.
Dit is handig voor ontwikkelen. Bij een afzonderlijke Static Site moet je `config.js`
aanpassen naar de publieke API-URL; de standaard gebruikt hetzelfde domein als de site.
De frontend wacht maximaal twee minuten op de API en toont uitleg over de koude start.

## Modelmetadata bijwerken

```powershell
.\.venv\Scripts\python.exe frontend/export_metadata.py
```

Dit leest alleen de vertrouwde pipeline van Andrew en exporteert de categorieÃ«n, het voorbeeld
en de metrics naar `metadata.json`. Controleer vertalingen bij model/schemawijzigingen.
Categoriebetekenissen en meeteenheden komen uit de [UCI-metadata](https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset).
De UI toont alleen categorieÃ«n die de opgeslagen encoder kent. Onbekende waarden worden als null verstuurd.
Voor een toekomstig automatisch bijgewerkt model moet metadata samen met de modelversie worden bijgewerkt.

Codex maakte deze frontend, labels en koppeling. Controleer deze bijdrage volgens de GenAI-policy.

De Citi Bike-pagina verwacht `/predict/citibike`. Het trainingsscript vernieuwt `citibike_metrics.json`; commit de bijgewerkte metrics samen met het kleine model. De eerste versie gebruikt januari 2025 en vermeldt die beperking op de pagina.
