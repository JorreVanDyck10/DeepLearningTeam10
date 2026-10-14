# Backend en website

FastAPI gebruikt de gekozen RF300 uit de vergelijking op de juiste cursusdata.
Bij het starten laadt de API een verliesvrije export met alle preprocessing.
Het formulier gebruikt elf underscore-velden; de response vermeldt de model-ID.
De backend serveert ook de website op hetzelfde domein.
Het bestaande Citi Bike-model wordt ook geladen. `/health` is gereed wanneer
beide modellen beschikbaar zijn.

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
| `POST /predict/citibike` | Ritstarts per uur en 24 uurvoorspellingen |
| `GET /docs` | Interactieve API-documentatie |

## Eerste voorspelling

```powershell
$body = (Get-Content frontend/metadata.json -Raw | ConvertFrom-Json).example | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict/mushroom -Method Post -ContentType 'application/json' -Body $body
```

De tien gewone kenmerken zijn verplicht en nullable. Alleen `jumbled_noise_1` is
optioneel, met standaard null. `jumbled_noise_0` is verwijderd. Extra velden,
negatieve metingen en onbekende categorische codes geven HTTP 422. Nul blijft
hier een echte meting. De response bevat `model_id`, `label`, `prediction`,
`probability_poisonous` en een educatieve disclaimer.
`label` is de ruwe e/p-modelklasse; `prediction` is nu de voorzichtige boodschap.
`decision` en `decision_policy_id` scheiden beide expliciet. De API bevestigt
geen eetbaarheid, ook niet bij 95% eetbare score. Zie
[de analyse en beslisregel](../docs/mushroom_decisions.md).

## Opgeslagen model

Standaard laadt de API `backend/artifacts/mushroom_rf300.json` en het bijbehorende
NPZ-bos. `MUSHROOM_RF300_MANIFEST` kan het manifestpad aanpassen; de oude
`ANDREW_MUSHROOM_MODEL_PATH` selecteert geen RF200 meer. De originele gekozen
pipeline blijft onder `MushroomFinal/comparison_correct/selected_pipeline.joblib`.
Zie [RF300 deployment en export](../docs/mushroom_rf300.md) voor bronhash,
compatibiliteit, reproduceerbare export en controle op alle 900 testrecords.

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
Render-instellingen.

## Citi Bike gebruiken

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict/citibike -Method Post -ContentType 'application/json' -Body '{"date":"2026-04-30","hour":8,"mode":"demo"}'
```

De datum is lokale tijd in New York. Het uur is een geheel getal van 0 tot 23.
De respons bevat de gekozen voorspelling en 24 voorspellingen voor de daggrafiek.
Raouls gekozen Random Forest gebruikt kalenderkenmerken en eerdere tellingen.
Gebruik `{"date":"2026-04-30","hour":8,"mode":"demo"}` voor de historische demo.
Voor andere datums stuur je `mode="history"` en precies 336 records in `history`
met `timestamp`, `rides` en `area="NYC"`: de 14 volledige voorafgaande lokale dagen.
De toekomst en Jersey City zijn verboden als historie. Er zijn altijd 24 kloklabels:
het ontbrekende voorjaaruur is nul; het herhaalde najaarsuur wordt samengevoegd.

De API laadt een lossless export van exact hetzelfde getunede 320-bomenmodel.
Alle splits en bladwaarden blijven behouden; er wordt niet opnieuw getraind.
De arrays worden memory-mapped en bij ontbrekende bestanden uit een gepinde
GitHub Release gedownload, met SHA-256-controle vóór gebruik. De release staat
buiten Git. `RAOUL_CITIBIKE_MODEL_DIR` kan de lokale opslagmap aanpassen.
De oude `CITIBIKE_MODEL_PATH` selecteert geen baseline meer. Bij een ontbrekend
of fout model geeft de route 503 in plaats van op de baseline terug te vallen.
`GET /citibike/model` en `/health` rapporteren `raoul_tuned_random_forest_320`.
Zie [de nieuwe deploymentdocumentatie](../docs/citibike_raoul.md).

## Bijdrage

Codex koppelde de gekozen RF300 via een verliesvrije export aan API en formulier
en voegde metadata- en voorspelcontroles toe. Vul de GenAI-disclosure aan volgens het opleidingsbeleid.
