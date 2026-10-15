# Audit Citi Bike — Messi10

**Scopecorrectie frontend, 14 oktober:** de eerste controle miste de bestaande
groepsrepo `C:\Deeplearning\DeepLearningTeam10`. De juiste website staat daar in
`frontend/` en wordt op `https://10messi10.netlify.app/#citibike` gehost, met de
bestaande Render-backend. De onnodig toegevoegde losse frontend/API zijn
teruggetrokken en lokaal gearchiveerd onder `.local_archive/standalone_web_20261014`.
De model-/notebookverbeteringen blijven behouden. De relevante deploymentuitleg
staat nu in `DEPLOYMENT.md` en verwijst naar de bestaande teamketen.

Datum: 14 oktober 2026. Basis: de bestaande `Citybike_full`, vergeleken met
`../Discussion topics/project assignment.md`, hoofdstukken 1–7 en AWS,
en de uitgewerkte lesnotebooks in `../Exercises_solved/`.
De oudere `../Citybike` blijft behouden. Dit document beschrijft eerst de
beginsituatie; de uitgevoerde correcties en controles staan onderaan.

## ✅ CORRECT — behouden

| Vereiste | Gevonden bewijs / beoordeling |
|---|---|
| OBTAIN → SCRUB → EXPLORE → MODEL → INTERPRET → DEPLOY | Duidelijke modules, overzichtsnotebook en notebooks 01–07. |
| Automatisch downloaden en samenstellen | `prepare.download`, bronmanifest met SHA-256, ZIP/nested-ZIP streaming, geen handmatige CSV-merge. Nieuwe machine moet eerst downloaden. |
| Reproduceerbare cleaning | Schema-aliases, datatypes, timestamps, onbekende categorieën, kwaliteitsflags en cachecontroles in `prepare.py`. |
| NaN en foutieve waarden | Onbekende stations/fietstypen behouden; onbruikbare tijden gemarkeerd. Een geldige start blijft bruikbaar voor vertrektellingen als einde/duur ontbreekt. |
| Outliers | Extreme duur onderzocht en gemarkeerd; geen willekeurige cutoff voor het gekozen teldoel. Coördinaatgrenzen bewijzen geen correcte stationlocatie. |
| Duplicaten | Moderne ID-conflicten stoppen verwerking. Bronoverlap wordt met EXCEPT ALL bewezen; identieke historische ritten behouden hun multipliciteit. |
| Schemawijzigingen | Oude/nieuwe headers, historische ID-strings en onbekende categorieën expliciet behandeld; NYC en JC gescheiden. |
| EDA | Tijd, uur/weekdag, duur, user/bike type, stations, routes, ruimtelijke aggregaties en ACF; `plots.show` toont een conclusie na iedere figuur. |
| Statistisch bewijs vóór modelkeuze | Drie ontwikkelingshypothesen met H0/H1, alpha .05, teststatistiek, p, effect in oorspronkelijke eenheid, interval en conclusie. |
| Verdedigbaar target | Dagvooruit uurvolume volgt uit wekelijkse associatie; individuele ritduur is als alternatief onderzocht. Geregistreerd gebruik is geen onvervulde vraag. |
| Prediction-time/leakage | Kalender en historische daglags; rolling eerst verschoven; geen telling van voorspeldag D in de invoer. Beschikbaarheid van recente tellingen is een expliciete aanname. |
| Split en preprocessing | Training vóór mei 2026; validatie mei–juni; test juli–augustus. Vijf chronologische folds; Ridge-scaler alleen binnen trainfold. Geen shuffle. |
| Baseline / AutoML / eigen modellen | Weekbaseline, Ridge, PyCaret-vergelijking, Random Forest en LightGBM. RF en LightGBM zijn apart getuned. |
| Modelkwaliteit | RMSE hoofdmetric, MAE/R² aanvullend; train, CV en validatie per model; eindtest alleen gekozen model plus baseline. |
| Eindtestprocedure | Selectie, modelchecksum en codecontract vóór test vastgelegd. Historie komt dag voor dag beschikbaar; model blijft vast. Nieuwe selectie na de test wordt geblokkeerd. |
| Residuen | actual − prediction; verdeling, tijd, voorspelde waarde, uur/weekdag/seizoen en grote validatiefouten onderzocht. |
| Time series | Trend/rolling gemiddelden, seizoen via kalender, ACF, lagfeatures en juiste lokale zomertijdconventie. ARIMA/deep learning is hiervoor niet verplicht. |
| Lokale inference | Streamlit laadt een bestaand artefact. Gedeelde featurefunctie, vaste featurevolgorde, geen training bij opstart. |
| Git / dependencies | Data, modellen en venv genegeerd. Python 3.11, directe requirements en lockfile. Eerste controle: 27 tests slagen; pip check zonder conflicten. |

## ⚠️ VERBETEREN — gericht aanpassen

1. Notebook 02 toont vooral een manifest in plaats van de uitvoerbare stap van ruwe
   data naar voorbereide data. Voeg die stap toe, zonder EDA-figuren en zonder
   nieuwe modelselectie.
2. De losse modelnotebooks tonen rapporten en een korte voorbeeldfit. Toon de
   werkelijke trainings-/tuningfuncties en leg modellen in gewone taal uit.
3. Niet iedere belangrijke codecel heeft een voorafgaande markdowncel. Voeg
   gerichte uitleg toe; stel geen nieuwe technieken voor.
4. De statistische aanpak met tijdsblokken is uitgebreider dan de eenvoudige
   lesvoorbeelden. Behoud het bestaande onderzoek, maar leg H0/H1 concreet uit,
   waarom ritten geen onafhankelijke steekproeven zijn, en wat de aannamen zijn.
   De minimale Monte-Carlo-p is 1/20001, geen bewijs van p=0. De vragen zijn
   geïnformeerd door EDA; dit is exploratief bewijs, geen onafhankelijke bevestiging.
5. Vergelijking mist één leesbare tabel met features en testkolommen. Toon test
   uitsluitend voor winnaar/baseline; lege cellen bij andere modellen zijn juist.
   Bestaande `fit_cv_seconds` meet CV plus fit/opslag, geen zuivere fitduur.
6. RF train-RMSE 536,13 versus validatie-RMSE 1940,49: bespreek overfit én
   veranderde vraag/systeemgrootte. Tuningwinst van circa 0,27% tegenover basis-RF
   is klein; het model van circa 647 MiB is een reële hostingbeperking. Niet na
   inzage in de test een kleiner model kiezen en dezelfde testclaim behouden.
7. README mengt volledige herbouw en opnieuw openen van een afgeronde studie.
   Splits die routes, zet downloaden vóór prepare op een nieuwe machine en
   maak alle commando's bruikbaar in PowerShell.
8. Teamnaam toevoegen: **Messi10**. Teamleden/bijdragen: **moet nog ingevuld worden**.

## ❌ ONTBREEKT / FOUT — beginsituatie

| Onderdeel | Beoordeling |
|---|---|
| Eigen Citi Bike-frontend + API | Correctie: aanwezig in `DeepLearningTeam10/frontend` en `backend`; de eerste audit inspecteerde alleen CloudAI en miste die bestaande keten. |
| Automatische modelupdate bij push | Geen workflow gevonden. `local_training.py retrain` is wel een bruikbare vaste refit, zonder eindtest. |
| Online controle Render / Netlify | Bestaande Netlify-URL en Render-configuratie gevonden in groepsrepo; gebruik deze bestaande frontend. |
| AWS | **OPENSTAAND: AWS / SageMaker model — bewijs in dit Citi Bike-pakket ontbreekt.** Gebruiker meldt één AWS-model in het groepsproject. De opdracht vraagt minstens één getraind én getuned AWS-model en het lokale gebruikte notebook. Dat hoeft niet noodzakelijk een tweede Citi Bike-model te zijn. Geen AWS-werk binnen deze correctieronde. |
| Teamgegevens | Namen en echte bijdragen ontbreken; op verzoek voorlopig zichtbaar laten. |
| Definitieve GitHub-inlevering | De ingestelde remote is de cursusrepo `mjochen/CloudAI`, geen aangetoonde groepsrepo. De projectmappen zijn lokaal nog untracked. Niet pushen naar de cursusrepo. |
| Groepsbrede onderdelen | Mushroom, toegang docent, presentatie en GenAI-beleid vallen buiten deze Citi Bike-correctie; blijven teamverantwoordelijkheid. |

## Verbeterplan op prioriteit — vastgelegd vóór aanpassingen

1. Bewaar bestaande modelkeuze, inferencecode, brondata en eindtestresultaten;
   leg checksums vast en voer bestaande tests/dependencycontrole uit.
2. Verbeter notebook 02, uitleg, hypotheseprotocol en de gezamenlijke
   vergelijking. Behoud de modellen en onderzoekskeuzes.
3. Gecorrigeerd: gebruik de bestaande teamfrontend/API; behoud Streamlit.
   Test foutieve invoer en overeenstemming van voorspellingen.
4. Maak de keten voorbereid op jullie Render/Netlify-opzet, met duidelijke
   configuratie en zonder een live deployment te claimen. Een automatische
   refit mag de oorspronkelijke studie niet overschrijven.
5. Voer notebooks, tests, artefactcontrole en frontend opnieuw uit. Leg exacte
   resultaten en resterende teamacties vast in `reports/audit_validation.json`.

## Cursusverwijzingen en keuzes

- `2 first model/1 - linear regression.ipynb`: RMSE en residual = actual − prediction.
- `2 first model/2 - MPG model.ipynb`: X/y, verboden targetinformatie, opgeslagen model.
- `3 model quality` en discussion chapter 3: train/validatie/test, generalisatie.
- `4 models/2 - Model overview.ipynb`: Ridge/trees/bagging/boosting; ook LightGBM staat in het overzicht.
- `4 models/5.2 - Random Forest.ipynb` en `5.7 - PyCaret.ipynb`: ensembles/AutoML.
- `5 data augmentation`: tidy data, ontbrekende waarden, outliers en bias.
- `6 time series/1 - Stationarity.ipynb`, `2 - Autocorrelation.ipynb`,
  `3 - Stubborn lamp.ipynb`, `5/6 - Bike highways`: trend, autocorrelatie,
  aggregatie, rolling en toekomstige observaties apart houden.

De code voor bronkopieën is technisch ingewikkeld doordat de concrete ZIPs
overlappen. Dat is geen reden om correcte dataverwerking opnieuw te schrijven.
De mondelinge kern is eenvoudig: één geregistreerde rit één keer tellen,
zonder echte identieke historische ritten ten onrechte te verwijderen.

## Laatste controle — uitgevoerde correcties

- Groepsnaam Messi10 toegevoegd. Teamleden/bijdragen blijven op uitdrukkelijk
  verzoek gemarkeerd als **moet nog ingevuld worden**.
- Notebook 02 voert `prepare(SOURCE)` werkelijk uit en hergebruikt gecontroleerde
  caches. Geen ruwe-data-rewrite, geen EDA-grafieken in finale preparation.
- Alle notebooks kregen uitleg vóór codecellen; concrete hypothesen, aannamen,
  modeluitleg en echte trainingsfuncties zijn toegevoegd. De bestaande bloktoets
  is opnieuw berekend en komt overeen met de opgeslagen cijfers.
- De complete vergelijking toont features, train/CV/validatie, alleen toegestane
  testscores en de juiste betekenis van gemeten looptijd.
- De los toegevoegde frontend/API zijn na gebruikerscorrectie teruggetrokken.
  De bestaande Netlify-frontend en FastAPI/Render-backend zijn de juiste keten.
  Die gebruiken reeds dezelfde geselecteerde RF en gedeelde inference.
- De modelanalyse behoudt haar 27 tests. Voor de bestaande teamwebsite slagen
  19 Python-tests en 16 JavaScript-tests, inclusief Citi Bike-inference, CORS,
  CSV-invoer en controle op de juiste modelversie. De 7 tests van de ingetrokken
  losse API zijn met die code gearchiveerd. `pip check` meldde geen conflicten.
- Alle 291 partitions zijn opnieuw met checksums gecontroleerd. De vaste
  modelconfiguratie is opnieuw gefit zonder tuning/testlabels; maximaal verschil
  3,64×10⁻¹² ritten. Het echte Streamlit-formulier gaf dezelfde 24 waarden.
- Train- en validatiemetrics van alle acht opgeslagen estimators en bestaande
  eindtestmetrics zijn opnieuw nagerekend. Originele dataset-, selectie- en
  modelfingerprints komen nog overeen. Geen nieuwe prestatieclaim na testinspectie.
- De acht notebooks zijn opnieuw uitgevoerd. Notebook 02 hergebruikte bij de
  laatste uitlegcorrectie alleen zijn zojuist uitgevoerde, identieke code/uitvoer;
  de andere zeven zijn opnieuw uitgevoerd. Zie `delivery_check.json` voor checksums.

| Officiële vereiste na deze ronde | Status |
|---|---|
| Data automatisch verkrijgen/cleanen, EDA en statistiek | Lokaal aanwezig en gecontroleerd; volledige rauwe herbouw niet nogmaals uitgevoerd. |
| Definitieve preparation zonder grafieken | Uitvoerbare notebook 02, uitgevoerd met geldige caches. |
| Baseline, AutoML, twee eigen getunede modellen | Aanwezig; opgeslagen scores gecontroleerd, volledige zoekprocedure niet opnieuw gedraaid. |
| Modelvergelijking, generalisatie en foutenanalyse | Aanwezig en uitgelegd; bestaande eindtest blijft vastgelegd. |
| Backend/API + eigen Citi Bike-frontend | De bestaande teamversie wordt gebruikt en lokaal getest. |
| Render/Netlify online bereikbaar | Bestaande Netlify-site opent met Citi Bike-pagina en meldt verbinding met Render; zie de correctiecontrole voor de voorspelling. |
| Automatische modelupdate bij push | Nog niet gekoppeld aan groepshosting; geen voltooiing claimen. |
| AWS-model getraind/getuned + lokaal gebruikt notebook | Gebruiker meldt één groepsmodel; niet gecontroleerd in dit Citi Bike-pakket. |
| Namen, bijdragen, definitieve groepsrepo en GenAI-beleid | Moeten door team worden ingevuld/gecontroleerd. |

Niet toegevoegd: extra modelarchitecturen, clustering, weerfeatures of nieuwe
targetkeuzes. Die zijn niet nodig om de bestaande Citi Bike-uitwerking te verbeteren.


## Gerichte notebook- en pipelineverbetering — 14 oktober 2026

De oorspronkelijke acht notebooks zijn vervangen door negen genummerde notebooks.
Unieke bron-, kwaliteits-, statistiek- en foutresultaten zijn overgenomen; de grote
overview is na migratie lokaal gearchiveerd. `notebook_content_migration.json` toont
de bestemming per oorspronkelijke sectie.

Notebook 02 maakt zelfstandig dezelfde modelinvoer. Het echte notebook is ook
uitgevoerd op synthetische ZIPs in een project zonder onderzoeksrapporten of modellen.
Daarbij bleek de stagingmap op een nieuwe computer nog te ontbreken; de nieuwe
voorbereidingmodule maakt die aan. De oorspronkelijke cleaningcode is behouden.

03, 05 en 06 bouwen en fitten dezelfde acht estimators werkelijk. 04 voert PyCaret
opnieuw uit met dezelfde folds en expliciete validatie. Reproducties staan apart
in `reports/notebook_reproduction`. Alle negen notebooks hebben nieuwe outputs.
De bestaande tuning is bewaard, geen nieuwe modelselectie gestart.

De bestaande team-API is via echte lokale HTTP-aanvragen vergeleken met de offline
functie, inclusief beide DST-overgangen. Streamlit AppTest bedient het echte formulier.
19 Python- en 16 JavaScript-tests voor de bestaande teamketen slagen.

De lokale retrainingpipeline fit de vaste 320-booms RF op oorspronkelijke train+validatie,
controleert serialisatie en voorspellingen en schrijft een apart gecomprimeerd artefact.
De workflow voor een lokale Windows-runner staat in de groepsrepo. Registratie,
een geslaagde GitHub-run en live omschakeling zijn nog handmatig nodig.

Het lokale Mushroom-AWS-notebook en de tuningexports zijn gecontroleerd. Het
notebook bevat conda_python3 en een /home/ec2-user/-runtimepad. De generieke
AWS-slots zijn placeholders. Er is geen afzonderlijke AWS-cloudverificatie gedaan;
cloudprovenance blijft expliciet openstaand.

`OPDRACHTCONTROLE.md` en `implementation_checks.json` zijn de actuele eindcontrole.
De originele model-, selectie-, feature-, cleaning- en eindtestchecksums blijven
behouden. Er is niet gepusht of live gedeployed.
