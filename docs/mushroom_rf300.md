# Mushroom RF300 op de teamwebsite

De API serveert `MushroomFinal/comparison_correct/selected_pipeline.joblib` via een
verliesvrije NumPy-export. Dit is de gekozen Random Forest met 300 bomen en
maximale diepte 20: geen opnieuw getraind vervangmodel. De training gebruikt de
juiste cursus-CSV met 5.000 rijen, elf kenmerken en een gedeelde 4.100/900-split.
`jumbled_noise_0` is verwijderd; `jumbled_noise_1` blijft behouden.

De vergelijking koos RF300 op vijfvoudige CV macro F1 (0,761876). Op de 900
ontwikkeltestrecords is accuracy 81,33% en giftige recall 64,52% (121 gemiste
giftige records). Deze testset was eerder bekeken; dit is geen onafhankelijke
eindtest en geen model voor consumptiebeslissingen. Zie het
[vergelijkingsrapport](../secondary_mushroom/09_vergelijkingsrapport.md).

## Model en compatibiliteit

Het originele artifact is opgeslagen met scikit-learn 1.4.2. De bestaande
Python 3.13-backend gebruikt andere libraryversies. Daarom exporteert
`MushroomFinal/export_serving.py` alle oorspronkelijke splits, bladwaarden,
imputatiemedians, categorische modi en encoder-categorieën. De backend gebruikt
NumPy voor exact dezelfde boomberekening, inclusief de oorspronkelijke float32
modelinvoer. Er wordt geen oude sklearn-pickle in de servingomgeving geladen.

- Bronartifact SHA-256: `5b451a756dd5c03ace50e32d18d7a230198df40570e547c9085e87215bddf42c`.
- Manifest en bos: `backend/artifacts/mushroom_rf300.json` en `.npz`.
- `backend/artifacts/mushroom_export_audit.json` registreert de exportcontrole.
- Alle 900 opgeslagen testvoorspellingen en 5.006 aanvullende gevallen zijn
  gecontroleerd: nul labelverschillen, maximale kansafwijking 2,23 × 10⁻¹⁶.
- De oorspronkelijke pipeline blijft het herlaadbare onderzoeksartifact.

De export opnieuw maken vanuit de repository-root:

```powershell
# Alleen voor export: de Python 3.11-vergelijkingsomgeving met sklearn 1.4.2.
.\.venv-comparison\Scripts\python.exe MushroomFinal/export_serving.py
.\.venv-comparison\Scripts\python.exe frontend/export_metadata.py
```

Commit het bos, manifest, audit en `frontend/metadata.json` samen. Export doet
geen training. De SHA-256 van het bos wordt voor gebruik gecontroleerd.

## API-contract

`POST /predict/mushroom` verwacht underscore-namen:
`cap_diameter`, `stem_height`, `stem_width`, `spore_print_color`, `gill_color`,
`habitat`, `season`, `ring_type`, `cap_shape`, `stem_surface`, `jumbled_noise_1`.
De eerste tien zijn verplicht maar nullable. `jumbled_noise_1` is optioneel en
standaard null; de pipeline vult ontbrekende waarden in. Nul blijft een echte
meting, zoals in de gekozen RF300. Extra velden, oude hyphen-namen, negatieve
metingen en onbekende categorische codes geven HTTP 422.

De respons bevat `model_id="mushroom_final_rf300"`, `label="e"` of `"p"`,
`prediction`, `probability_poisonous` en een educatieve disclaimer.
De oorspronkelijke integerklassen 0 en 1 worden naar e en p vertaald.
`label` blijft de oorspronkelijke modelklasse voor vergelijking. De website
bevestigt geen eetbaarheid meer: `prediction` en `decision` volgen de
[voorzichtige beslisregel](mushroom_decisions.md), met een aparte policy-ID.
Ook een 95% eetbare modelscore bleek bij trainingsvalidatie nog giftige gevallen
te bevatten. Het formulier toont daarom giftig of onzeker / mogelijk giftig.
Het formulier en de metrics komen uit hetzelfde manifest. De website controleert
of de backend dezelfde model-ID gebruikt en meldt een onvoltooide deployment.

## Hosting en verificatie

Render blijft dezelfde Python 3.13-runtime en dependencies gebruiken. Standaard
laadt de API het meegeleverde RF300-manifest; alleen `MUSHROOM_RF300_MANIFEST` kan
dat pad aanpassen. `ANDREW_MUSHROOM_MODEL_PATH` en `MUSHROOM_MODEL_PATH` selecteren
geen oud model meer. Een bestaande handmatig ingestelde Render-service hoeft
voor deze migratie dus geen nieuwe environment variable te krijgen.

Een push naar main activeert de bestaande Render- en Netlify-deployments.
GitHub Actions controleert API-inference, alle 900 referentievoorspellingen,
frontendmetadata, JavaScript en de bestaande Citi Bike-routes. Metadata moet
zonder Git-diff reproduceerbaar zijn. Modeltraining gebeurt niet tijdens deploy.

```powershell
python -m pip install -r backend/requirements-dev.txt
python -m unittest discover -s tests -v
node --test tests/test_api_client.cjs tests/test_api_config.cjs tests/test_citibike_frontend.cjs
```

Controleer na beide deployments `/health` op `mushroom_final_rf300`, vul het
voorbeeld in op [de website](https://10messi10.netlify.app/) en klik Voorspellen.
Controleer ook Citi Bike op `raoul_tuned_random_forest_320`.

## Uitgevoerde controle

De deployment van commit `bcdc853` is op 14 oktober 2026 gecontroleerd:
22 Python-tests en 16 JavaScript-tests slagen, evenals de GitHub Actions-run
[Website and API checks](https://github.com/JorreVanDyck10/DeepLearningTeam10/actions/runs/38591551285).
Render-health en Netlify-metadata melden beide `mushroom_final_rf300`.
Acht publieke API-voorspellingen komen binnen 10⁻¹² overeen met de opgeslagen
testlabels en kansen. De Netlify-CORS-preflight slaagt.

Een browsercontrole heeft **Voorbeeld invullen → Voorspellen** uitgevoerd,
de elf velden en projectmetrics gecontroleerd, en daarna Citi Bike gebruikt.
Beide formulieren geven een geldige voorspelling zonder JavaScript-fouten.
Citi Bike geeft ook via de publieke API dezelfde 24 historievoorspellingen als
de bestaande referentiefixture. Het
[uitgevoerde deploymentnotebook](../secondary_mushroom/10_deploy.ipynb) bevat de
herhaalbare artifact-, inference- en publieke controles met hun uitvoer.

Codex maakte op verzoek van Jorre de export, API-koppeling, formuliermigratie,
controles en documentatie. Het team reviewt en verdedigt de modelkeuze.
