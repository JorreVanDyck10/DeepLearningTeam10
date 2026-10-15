# Citi Bike — bestaande teamfrontend gebruiken

De juiste frontend is `C:\Deeplearning\DeepLearningTeam10\frontend`.
Dit is de bestaande website op [Netlify](https://10messi10.netlify.app/#citibike).
De backend staat in `C:\Deeplearning\DeepLearningTeam10\backend` en draait op
[Render](https://deeplearningteam10-api.onrender.com).

De losse frontend/API die bij de eerste audit in `Citybike_full` waren toegevoegd,
zijn teruggetrokken en lokaal bewaard in `.local_archive/standalone_web_20261014`.
Die map is uitgesloten van Git. Er is geen vervangende website nodig.

## Bestaande keten

`frontend/app.js` en `citibike.js` versturen de invoer naar
`POST /predict/citibike`. `frontend/config.js` kiest op de Netlify-site automatisch
jullie bestaande Render-backend. `/citibike/model` beschrijft het gebruikte model;
`/health` rapporteert `citibike_model_id="raoul_tuned_random_forest_320"`.

De backend gebruikt de bestaande compacte export van hetzelfde gekozen Random
Forest met 320 bomen. De oorspronkelijke modelchecksum is
`aea5c81d306facc1fadd8dd1888a08219d3afabda16a11b63db7271a694f509f`.
De export verandert geen boomgrenzen of bladwaarden en gebruikt de gedeelde
`predict_day`. Er wordt geen model getraind wanneer de API start of voorspelt.
Zie `DeepLearningTeam10/docs/citibike_raoul.md` voor het bestaande exportbewijs.

## Lokaal starten — PowerShell

```powershell
Set-Location 'C:\Deeplearning\DeepLearningTeam10'
.\.venv-serving\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/#citibike`. Dit toont dezelfde frontendbestanden die
Netlify publiceert. Stoppen: Ctrl+C. De PyCaret/modelomgeving in `Citybike_full`
blijft afzonderlijk; Streamlit blijft als lokale analysedemo beschikbaar.

## Controleren

```powershell
Set-Location 'C:\Deeplearning\DeepLearningTeam10'
.\.venv-serving\Scripts\python.exe -m unittest discover -s tests -v
node --test tests/test_api_client.cjs tests/test_api_config.cjs tests/test_citibike_frontend.cjs
```

Bij de correctie op 14 oktober 2026 slagen 19 Python-tests en 16 JavaScript-tests
van de bestaande teamfrontend/backend. De modelanalyse heeft daarnaast haar eigen
tests in `Citybike_full/tests`. Er is niets naar GitHub gepusht of opnieuw gedeployed.

## Voorspelcontract

De demonstratie gebruikt ontwikkeling uit april 2026 en is geen onafhankelijke
test. Eigen historie bevat precies 336 volledige voorafgaande lokale uurvakken
met `timestamp`, `rides`, `area=NYC`. Voorspelling gebeurt om 00:00 New York-tijd.
Het geselecteerde uur is één van dezelfde 24 dagvoorspellingen. De API controleert
ook ontbrekende uren, duplicaten, gebied en ongeldige tellingen.

## Vaste hertraining: lokale uitvoering

De nieuwe pipeline gebruikt exact de 320-booms RF, dezelfde features en de
114.336 oorspronkelijke train+validatierijen. Hij start geen selectie, tuning
of testevaluatie. De vergelijking met referentievoorspellingen is een
reproductiecontrole; deze rijen zitten in de finale refit.

```powershell
Set-Location 'C:\Deeplearning\CloudAI\Citybike_full'
.\.venv\Scripts\python.exe -X utf8 tools/retrain_pipeline.py
```

Iedere run krijgt een aparte map in `data/pipeline_runs/`, met een gecomprimeerd
modelartefact en `run.json`. Het rapport bevat codeversie, bron- en modelchecksums,
fitduur, gebruikte rijen en voorspellingsovereenkomst na herladen. De oorspronkelijke
studie en het live model blijven behouden. De laatste lokale receipt staat in
`reports/retraining_pipeline_latest.json`.

## Automatisch bij push: handmatige inrichting nog nodig

De groepsrepo bevat `.github/workflows/citibike-retrain.yml`, naast de bestaande
website/API-testworkflow. Hij start bij iedere push op `main` of via
`workflow_dispatch`; hij wordt niet voor pull requests op de lokale runner gestart.
De workflow draait tests op de uitgecheckte code, fit en controleert de vaste RF
en uploadt een versiegebonden Actions-artefact met zeven dagen bewaartijd.

Nog uitvoeren door een beheerder van de groepsrepo:

1. Registreer op de trainingscomputer een GitHub Actions-runner voor
   `JorreVanDyck10/DeepLearningTeam10`, met label `citibike`, Windows en X64.
   Volg het platformspecifieke registratiecommando in GitHub Settings → Actions → Runners.
2. Configureer repositoryvariables:
   `CITIBIKE_TRAINING_ROOT=C:\Deeplearning\CloudAI\Citybike_full` en
   `CITIBIKE_TRAINING_PYTHON=C:\Deeplearning\CloudAI\Citybike_full\.venv\Scripts\python.exe`.
3. Zorg dat notebook 02 is uitgevoerd en dat de vaste referentie en voorbereide
   data lokaal aanwezig blijven. Start de runner tijdens de evaluatieperiode.
4. Commit/push de gecontroleerde groepswijzigingen en start de workflow eenmaal.
   Controleer de geslaagde run en download het model plus runreceipt.

Er zijn geen runnerregistratiecredentials ingesteld in dit project. De workflow
is lokaal voorbereid; automatische GitHub-uitvoering is pas aangetoond na die run.

## Live modelupdate blijft afzonderlijk

De API laadt momenteel een vastgezet compact model uit een GitHub-release.
Actions-artefacten worden niet automatisch door Render geladen. Een beheerder moet
een gecontroleerde compacte export publiceren als een **nieuwe** release, de
modelreceipt en parityfixture bijwerken, de API/frontendtests uitvoeren en de
Render-versie omschakelen. De oude release blijft beschikbaar voor terugschakelen.
Daarvoor is schrijftoegang tot GitHub en hosting nodig. Deze ronde publiceert of
vervangt geen live model en presenteert dat niet als voltooid.

## AWS-bewijs

Het lokale Mushroom-SageMaker-notebook en tuningexports staan onder
`SolutionJorre/MushroomDataset`. De generieke `08_predict_aws.ipynb`-bestanden zijn
placeholders. **OPENSTAAND: AWS / SageMaker model — cloudprovenance bevestigen**,
bijvoorbeeld met gedateerde SageMaker-runtimegegevens gekoppeld aan het notebook
en de bewaarde model-/tuningexports. Geen AWS-training uitgevoerd in deze ronde.
