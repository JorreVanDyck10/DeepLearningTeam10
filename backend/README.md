# Backend en website

FastAPI gebruikt voorlopig Andrew's Random Forest met 200 bomen en twaalf invoervelden.
Bij het starten laadt de API de volledige pipeline, inclusief preprocessing.
De backend serveert ook de website op hetzelfde domein.

## Lokaal starten op Windows

Voer uit vanaf de repositorymap, met Python 3.13:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

Open http://127.0.0.1:8000/ voor de website of `/docs` voor de interactieve API.

| Route | Functie |
| --- | --- |
| `GET /` | Website met het Mushroom-formulier |
| `GET /health` | Modelstatus en model-ID; 503 als laden mislukt |
| `POST /predict/mushroom` | Een Mushroom-record classificeren |
| `GET /docs` | Interactieve API-documentatie |

## Eerste voorspelling

```powershell
$body = Get-Content SolutionAndrew/MushroomDataset/example_mushroom.json -Raw
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict/mushroom -Method Post -ContentType 'application/json' -Body $body
```

De tien gewone kenmerken zijn verplicht; ontbrekende waarden mogen `null` zijn.
De twee extra datasetvelden `jumbled_noise_0` en `jumbled_noise_1` zijn optioneel
en krijgen standaard `null`. Gebruik de oorspronkelijke datasetcodes. Extra velden,
negatieve metingen en onbekende categorische codes geven HTTP 422.
Nul voor `stem-height` of `stem-width` wordt, zoals in Andrew's notebook, een
ontbrekende waarde voordat de pipeline deze invult.

De respons bevat `label`, `prediction`, `probability_poisonous` en een educatieve
disclaimer. De modeluitkomst is geen advies om een paddenstoel te eten.

## Opgeslagen model

Standaard laadt de API:
`SolutionAndrew/MushroomDataset/models/random_forest.joblib`.
Het modelpad kan worden overschreven met `ANDREW_MUSHROOM_MODEL_PATH`.
De oude instelling `MUSHROOM_MODEL_PATH` selecteert deze pipeline niet.

Andrew's notebook blijft ongewijzigd. Voor het opnieuw exporteren van zijn model
in dezelfde omgeving:

```powershell
.\.venv\Scripts\python.exe -m backend.mushroom_model
.\.venv\Scripts\python.exe frontend/export_metadata.py
```

Zie [modeldocumentatie](../SolutionAndrew/MushroomDataset/DEPLOYMENT.md) voor de
train/test-verdeling, opnieuw berekende scores en preprocessing.

## Deployment op Render

De bestaande Render-service volgt `main` met **Auto-Deploy: On Commit**.
Een push publiceert backend, opgeslagen model en frontend samen. De website
gebruikt automatisch de API op hetzelfde domein.
Deployment traint het model niet opnieuw.

- Build: `pip install -r backend/requirements.txt`.
- Start: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`.
- Health check: `/health`.
- Python-versie: `3.13.7`.
- Root Directory: leeg, zodat de hele repository beschikbaar blijft.

Zie [deploymentdocumentatie](../docs/deployment.md) voor GitHub-controles en
Render-instellingen. Citi Bike is nog niet aangesloten in deze gepubliceerde versie.

## Bijdrage

Codex exporteerde Andrew's bestaande trainingspipeline en paste API, formulier,
metadata en controles aan. Vul de GenAI-disclosure aan volgens het opleidingsbeleid.
