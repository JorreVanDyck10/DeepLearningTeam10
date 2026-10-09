# Backend starten

Deze FastAPI-backend gebruikt de bestaande Mushroom-baseline en preprocessing.
Het model wordt eenmaal bij het opstarten geladen. De API traint geen modellen.
Citi Bike is nog niet aangesloten: daarvoor moet eerst een model en invoercontract bestaan.

## Lokaal op Windows

Voer uit in PowerShell, vanaf de repositorymap:

```powershell
cd C:\DeepLearning\Opdracht\DeepLearningTeam10
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

Gebruik Python 3.13, overeenkomstig de baselineomgeving. Je hoeft de venv niet te activeren.
Stop de server met Ctrl+C. Open http://127.0.0.1:8000/docs om de API uit te proberen.

| Route | Functie |
| --- | --- |
| `GET /health` | 200 als het model geladen is, anders 503 |
| `POST /predict/mushroom` | Eén Mushroom-record classificeren |
| `GET /docs` | Interactieve documentatie en invoervelden |

## Eerste voorspelling

Open een tweede PowerShell-venster in de repositorymap:

```powershell
$examples = Get-Content secondary_mushroom/example_mushroom.json -Raw | ConvertFrom-Json
$body = $examples[0] | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict/mushroom -Method Post -ContentType 'application/json' -Body $body
```

Het bestaande voorbeeldbestand bevat een lijst; de API verwacht één object uit die lijst.
Ook via `/docs`: kies `POST /predict/mushroom`, `Try it out`, plak dat object en klik `Execute`.
Alle 20 velden zijn verplicht. Gebruik de oorspronkelijke datasetcodes en `null` voor ontbrekende
metingen; numerieke metingen moeten getallen zijn, geen strings. Extra velden, negatieve metingen
en onbekende categorische codes worden geweigerd met HTTP 422.

De respons bevat `label`, `prediction`, `probability_poisonous` en een educatieve disclaimer.
De kanswaarde is de output van de baseline en is niet als gekalibreerde zekerheid gevalideerd.

## Model en frontend aansluiten

Standaard wordt `secondary_mushroom/models/baseline_decision_tree.joblib` geladen.
Als het ontbreekt, voer eerst `04_predict_baseline.ipynb` uit. Alleen vertrouwde, zelf gemaakte
joblib-bestanden laden. Een ander pad kan via de omgevingsvariabele `MUSHROOM_MODEL_PATH`.
Een vervangend model moet dezelfde 20 features, de `p`-klasse en de bestaande preprocessingstructuur
hebben. Pas de API aan als die structuur verandert en herstart na een modelvervanging.

Voor de frontend staan localhost en 127.0.0.1 op poort 5173 al in de CORS-lijst.
Stel later `CORS_ORIGINS` in op jullie echte frontend-URL. Meerdere origins zijn kommagescheiden.

## Eerste deployment op Render

In de repositoryroot staat `render.yaml` met de onderstaande instellingen op het Free-plan.
Het bestaande baselinemodel van ongeveer 12 kB heeft een specifieke uitzondering in `.gitignore`.
Commit dat bestand samen met de backend en push naar GitHub. Grote modellen blijven genegeerd.
De eerste deployment gebruikt deze opgeslagen baseline en traint geen nieuw model.

Open het Render-dashboard, kies **New > Blueprint**, koppel jullie GitHub-repository en
selecteer de branch waarop de backend en `render.yaml` staan. Controleer dat de service
het **Free**-plan gebruikt voordat je hem aanmaakt. Na deployment bezoek je de getoonde
publieke service-URL met `/health` en `/docs` erachter. De naam/URL kan Render aanpassen
als deze al bezet is. Je lokale PowerShell-server hoeft dan niet te draaien.

Als alternatief kun je de service handmatig instellen:

Maak een Python Web Service, gekoppeld aan de repository. Laat Root Directory leeg, zodat
de server zowel `backend` als `secondary_mushroom` kan importeren.

- Python-versie: `PYTHON_VERSION=3.13.7`.
- Build command: `pip install -r backend/requirements.txt`.
- Start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`.
- Health check path: `/health`.
- `CORS_ORIGINS`: de publieke URL van jullie frontend.

Voor toekomstige grotere modellen laat je de build een specifieke, vertrouwde modelversie
uit artifactopslag downloaden naar `MUSHROOM_MODEL_PATH`, vóór de server start. Bewaar daarbij
dezelfde sklearn-versie als bij training. Automatische training/modelpublicatie is nog niet
geïmplementeerd. Een nieuwe commit deployen is op zichzelf geen model opnieuw trainen.

De gratis service slaapt na 15 minuten zonder verkeer; de eerste aanvraag kan ongeveer een
minuut duren. Voeg later de echte frontend-URL toe aan `CORS_ORIGINS` in `render.yaml`.

## Bijdrage en controle

De backend serveert ook `frontend/` op `/`. Website en API worden samen bijgewerkt
bij een push naar de gekoppelde `main`-branch. De frontend gebruikt automatisch
hetzelfde domein als de API, zodat deze opstelling geen extra CORS-origin nodig heeft.
Zie [docs/deployment.md](../docs/deployment.md) voor automatische deployment en tests.

Codex maakte deze backendstarter en koppelde hem aan de bestaande baseline.
Controleer de code en vul de vereiste GenAI-disclosure aan volgens het opleidingsbeleid.
