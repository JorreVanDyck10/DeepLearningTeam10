# Raoul Buffels — Citi Bike

De [openbare teamwebsite](https://deeplearningteam10-api.onrender.com/#citibike)
gebruikt inmiddels het hier geselecteerde 320-bomenmodel. Zie
[frontend/API-deployment](../../docs/citibike_raoul.md) voor de lossless export,
historische demonstratie en eigen recente historie. De hieronder beschreven
export uit de oorspronkelijke lokale studie behoudt haar provenance.

Dit is de deelbare export van de volledig uitgevoerde Citi Bike-uitwerking. Begin bij [`notebooks/citybike_analysis.ipynb`](notebooks/citybike_analysis.ipynb). De code, acht uitgevoerde notebooks, figuren, tests, requirements, manifesten, modelkaart en lokale trainingsmetingen staan in deze map.

Oorspronkelijke uitvoering: `C:\Deeplearning\CloudAI\Citybike_full`. Rapporten behouden de oorspronkelijke paden en fingerprints als provenance; ze zijn geen bewijs van een nieuwe run in deze repository. `EXPORT_MANIFEST.json` legt de gekopieerde bestanden en eventuele documentatieaanpassingen vast. Kerncode, voorspelfunctie en modelregels blijven identiek.

Op deze computer zijn `data`, `models` en `.venv` lokaal via directory-junctions gekoppeld aan de originele uitwerking. Die koppelingen en hun inhoud worden niet gecommit. Op een andere computer maak je een eigen omgeving en download/verwerk je de bronnen volgens de commando's hieronder. Het modelbestand van circa 647 MiB blijft lokaal; training is reproduceerbaar met de vastgelegde code en configuratie.

De bestaande teamwebsite, Citi Bike-baseline, API en bijdragen van teamgenoten worden door deze export niet vervangen. AWS-training blijft een latere taak.

## Lokale koppelingen opnieuw maken op Raouls computer

Alleen uitvoeren als de oorspronkelijke mappen aanwezig zijn en de doelmappen nog niet bestaan:

```powershell
Set-Location 'C:\Deeplearning\DeepLearningTeam10\SolutionRaoul\Citybike'
New-Item -ItemType Junction -Path 'data' -Target 'C:\Deeplearning\CloudAI\Citybike_full\data'
New-Item -ItemType Junction -Path 'models' -Target 'C:\Deeplearning\CloudAI\Citybike_full\models'
New-Item -ItemType Junction -Path '.venv' -Target 'C:\Deeplearning\CloudAI\Citybike_full\.venv'
```

# Citi Bike: volledig onderzoek met beschermde eindtest

Deze afzonderlijke uitwerking volgt **OBTAIN → SCRUB → EXPLORE → MODEL → INTERPRET
→ DEPLOY**. De bestaande `Citybike`-uitwerking wordt niet aangepast. NYC is het
hoofdonderzoek; Jersey City wordt afzonderlijk verwerkt en geanalyseerd.

Team: **DeepLearningTeam10** — Jorre Van Dyck, Milan Wouters, Raoul Buffels en Andrew Noeyens.
Deze map bevat de Citi Bike-bijdrage van **Raoul Buffels**. De concrete menselijke
taakverdeling moet het team zelf aanvullen; er worden geen bijdragen verzonnen.
Codex hielp met implementatie, uitleg, tests en uitvoering. Controleer de GenAI-regels
van de opleiding en zorg dat ieder teamlid de keuzes zelf kan verdedigen.

De ontwikkeling bevat ook extreme geregistreerde ritduren, met start en einde
weken of maanden uiteen. Dat bewijst geen voortdurend fietsen. Registratiefouten
of laat afgesloten ritten zijn mogelijke, onbevestigde verklaringen. De diagnose
en concrete voorbeelden staan in `reports/*_duration_sample_audit.json` en
`reports/*_extreme_duration_examples.csv`; we verwijderen deze waarden niet met
een willekeurige statistische cutoff. Ritduur is niet het gekozen modeldoel.

## Bronnen en mappen

Bron: [Citi Bike System Data](https://citibikenyc.com/system-data), volgens de
[Data Use Policy](https://citibikenyc.com/data-sharing-policy). Er zijn 175 lokale
ZIP-bestanden met bestandsdekking juni 2013–augustus 2026. De exacte omvang,
schema's, rij-aantallen en fingerprints staan na verwerking in
`reports/data_manifest.json`.

De ruwe bestanden blijven in `C:\Deeplearning\citibike_data`. De code leest ze
zonder handmatig uitpakken. Jaararchieven kunnen maand-ZIPs bevatten: daarvan
staat hoogstens één tijdelijk maandarchief op schijf. Geen volledige dataset
wordt in Pandas of RAM geladen. Arrow leest batches van 8 MiB; DuckDB gebruikt
standaard vier threads en maximaal 4 GiB intern geheugen. De totale proces-RSS
kan hoger zijn door Arrow/Python; gesamplede RSS staat per partition in het manifest.

```text
SolutionRaoul/Citybike/
  citibike/       Python-modules voor pipeline, onderzoek, features en modellen
  notebooks/     volledig overzicht plus genummerde ondersteunende notebooks
  reports/       kleine, deelbare manifesten, statistiek en modelresultaten
  tests/         controles van bronoverlap, tijd, gebied en inference
  data/          gegenereerde Parquet, caches en tijdelijk werk (Git-ignore)
  models/        lokaal opgeslagen modellen (Git-ignore)
  app.py         Streamlit-inference, zonder training bij opstart
  run.py         offline CLI
  requirements.txt
```

Ruwe data, grote tussenbestanden, caches, de omgeving en alle modelartefacten
horen niet in Git. Code, notebooks, requirements en rapporten worden wel gedeeld.
Gebruik deze aparte projectomgeving zodat de bestaande omgeving intact blijft.

## Gemeten resultaten van deze uitvoering

175 ZIP-bronnen leverden 355.199.439 ruwe records en 330.714.664 behouden
records in 291 maandpartitions: 323.817.126 NYC en 6.897.538 Jersey City.
De verwijderde records zijn bewezen bronkopieën; kwaliteitsflags blijven
afzonderlijk beschikbaar. De bronnen en bestaande Citybike-uitwerking zijn behouden.

| Periode | Getunede Random Forest: RMSE | Weekbaseline: RMSE |
|---|---:|---:|
| Validatie mei–juni 2026 | 1.940,49 | 2.826,16 |
| Eindtest juli–augustus 2026 | 1.979,51 | 2.608,78 |

Eenheid: ritten per lokaal klokvak. Eindtestverbetering: **24,12%**;
MAE 1.076,23; R² 0,8280; 1.488 testvakken. Het weekbootstrapinterval
voor de MSE-winst is indicatief door het kleine aantal weekblokken.

De 1%-regel rangschikt modeltypen; binnen dezelfde familie kiest de laagste
validatie-RMSE. De RF-tuning levert slechts **0,27%** extra validatiewinst
tegenover de oorspronkelijke RF, met meer bomen en een groter artefact
(647,3 MiB). Een afzonderlijke capaciteitsrang binnen dezelfde modelklasse
maakt geen deel uit van deze vooraf vastgelegde regel. LightGBM-tuning leverde
geen validatiewinst; beide Ridge-ablaties presteerden slechter.

Gemeten tijd: 77,3 minuten als som van geslaagde partitionbouwtijden,
27,1 minuten voor schrijfstappen van de maandgrenscorrectie, 18,5 minuten
ontwikkelings-EDA en 72,8 minuten modelonderzoek/selectie. Deze tijden sluiten
eerdere mislukte pogingen en sommige globale controles/checksums uit.
De laatste hervatte voorbereiding duurde 5,1 minuten, inclusief cachecontrole.
De hoogste gesamplede CLI-proces-RSS was 4,01 GiB; dit is geen gecombineerde
worker-RSS en geen exacte continue piekmeting. DuckDB zelf had een 4 GiB-limiet.

Alle **27 tests** slagen. Hertraining van de vaste configuratie duurde 98,9 seconden
en verschilde hoogstens 3,64×10⁻¹² ritten in voorspellingen. Het echte Streamlit-
formulier en offline code leverden dezelfde 24 waarden. De acht notebooks zijn
uitgevoerd; `reports/delivery_check.json` legt hun checksums en uitvoercellen vast.

## Installatie — Windows PowerShell

Voer de volgende commando's uit vanuit `C:\Deeplearning\DeepLearningTeam10\SolutionRaoul\Citybike`:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
$env:CITIBIKE_SOURCE = 'C:\Deeplearning\citibike_data'
.\.venv\Scripts\python.exe -X utf8 tools/run_tests.py
```

Geef vanuit deze geneste projectmap `CITIBIKE_SOURCE` of `--source` expliciet op;
op deze computer is de bronmap `C:\Deeplearning\citibike_data`, zoals hierboven.
Reserveer ruime vrije schijfruimte
voor Parquet, staging en DuckDB-spill; de code dupliceert niet alle ruwe CSV's.
`requirements-lock.txt` legt ook transitieve dependencies vast;
`requirements.txt` documenteert de direct gebruikte packages. Gebruik Python 3.11.
`reports/resource_profile.json` meet gesamplede RSS van CLI-hoofdprocessen;
het is geen som van alle eventuele PyCaret/joblib-workerprocessen. Korte pieken
en de vroegste voorbereiding kunnen buiten de meting vallen.

## Uitvoeringsvolgorde

```powershell
.\.venv\Scripts\python.exe -X utf8 run.py prepare
.\.venv\Scripts\python.exe -X utf8 run.py research
.\.venv\Scripts\python.exe -X utf8 run.py train
.\.venv\Scripts\python.exe -X utf8 run.py evaluate
.\.venv\Scripts\python.exe -X utf8 run.py verify
.\.venv\Scripts\python.exe -X utf8 run.py notebooks
.\.venv\Scripts\python.exe -X utf8 tools/check_delivery.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

`prepare` inventariseert, valideert bronoverlap en schrijft partitions. Een
onderbroken run hervat via gecontroleerde bron- en partitionreceipts. Een
gewijzigde bron of pipelinecode maakt de bijbehorende cache ongeldig. Een
incomplete partition heeft geen geldig receipt en wordt opnieuw gebouwd.
`discover` maakt alleen de broninventaris. Met het gedeelde bronmanifest kan
`download` exact die archieven opnieuw ophalen en SHA-256 controleren:

```powershell
.\.venv\Scripts\python.exe -X utf8 run.py download --source 'D:\citibike_data'
```

Als de provider een archief gewijzigd heeft, stopt de checksumcontrole. Dat is
een nieuwe bronversie en geen stilzwijgend identieke reproductie.

Open de notebooks in Jupyter of VS Code met de interpreter `.venv\Scripts\python.exe`.
`citybike_analysis.ipynb` bevat het volledige verhaal. De notebooks 01–07 geven
de EDA, definitieve voorbereiding zonder grafieken, afzonderlijke modellen en
eindvergelijking. De notebookrunner bewaart echte uitvoer en start expliciet
dezelfde interpreter; hij voert geen nieuwe modelselectie uit.

## Datakwaliteit en gebiedsscheiding

Brongebied wordt expliciet bepaald uit het gecontroleerde bestandsformaat.
NYC en JC hebben afzonderlijke directories, queries en uurreeksen. NYC-features
en inference weigeren ontbrekende of gemengde gebiedslabels.

Historische station-ID's blijven strings; een integer-`.0`-suffix wordt alleen
bij het oude bikeschema verwijderd. Coördinaten worden op zeven decimalen
gestandaardiseerd (~centimeter), omdat parallelle CSV-versies verschillen in
floating-pointserialisatie. Identieke maandrepresentaties worden gecontroleerd
met **EXCEPT ALL in beide richtingen**. Dit behoudt de multipliciteit van
identieke historische ritten. Andere verschillen stoppen de verwerking.

Geboortejaar en gender worden in de broninventaris herkend, maar zijn geen
onderzoeks- of modelkolommen in deze uitwerking. Ze worden niet gereconstrueerd
voor de moderne schema's. Ontbrekend fietstype en gebruikerscategorie blijven
expliciet onbekend. De coördinatenflags controleren numerieke wereldgrenzen;
ze bewijzen geen juiste stationlocatie. Het ruimtelijke plotvenster en ontbrekende
stations worden daarom afzonderlijk besproken.

Moderne ride-ID-conflicten stoppen de pipeline. Exacte herhalingen tellen eenmaal.
De technische controle ontdekte bovendien kopieën over maandgrenzen en geldige
vertrekken buiten de bestandsmaand. Een tweede, reproduceerbare assemblylaag
deelt die ritten in op hun **werkelijke lokale vertrekmaand**. De oorspronkelijke
maandcaches blijven intact, zodat veranderde naburige bronnen geen eerder verwijderde
rijen kunnen verbergen. Conflicterende ID-inhoud stopt de verwerking.

Een volledige historische bronsegmentkopie zonder ride-ID wordt alleen verwijderd
als EXCEPT ALL bewijst dat de hele segmentmultiset al in de natuurlijke maand
aanwezig is, inclusief multipliciteit. Er is geen generieke deduplicatie van
identieke historische ritten. Onbewezen segmenten blijven behouden. De audit,
verplaatsingen en verwijderingsredenen staan in `reports/month_boundary_audit.json`
en het manifest. Ritten buiten bekende maanddekking blijven apart geflagd;
zij rechtvaardigen geen volledig met nullen gevulde nieuwe maand.
Bij acht herhaalde NYC-rit-ID's verschilt uitsluitend de decimale schrijfwijze
van een station-ID (`4632.1` / `4632.10`). Alleen voor het bron-equivalentiebewijs
worden fractionele eindnullen genegeerd, en uitsluitend bij bekende gelijke
stationnaam en coördinaten plus gelijke overige ritvelden. Originele ID-strings
en voorloopnullen blijven in de dataset behouden. Stations kunnen daarom
historische ID-aliassen hebben; dit is een beperking van ruimtelijke aggregaties.
Voor 24 Jersey City-rit-ID's bestaat daarnaast een versie op hele seconden en
een versie met milliseconden. Alleen bij een bestaande volledige secondenversie,
gelijke secondenfloors voor start én einde en gelijke overige canonieke velden
wordt dit als precisiealias bewezen. De versie met hogere precisie blijft behouden.
Deze regel geldt uitsluitend voor JC; andere tijdverschillen stoppen de verwerking.
Een eenmalig gecontroleerde NYC-cacheovergang is beschreven in
`reports/nyc_cache_transition.json`; bron- en outputchecksums blijven daarbij gelijk.
Een volledige herbouw vanaf lege caches heeft die overgang niet nodig.

Train-, validatie- en testperioden volgen de werkelijke vertrekdatum, onafhankelijk
van de ZIP-bestandsnaam. De technische maandgrenscorrectie gebruikt geen
statistische patronen of modeluitkomsten. Juli–augustus-vertrekken blijven afgeschermd.
Parsingfouten worden geflagd; ze verdwijnen niet stil.

De Parquet-laag bewaart kwaliteitsflags: een geldig vertrek kan bruikbaar zijn
voor tellingen terwijl eindstation, duur of coördinaten onbruikbaar zijn.
Negatieve/nulduur wordt uitgesloten van duurstatistiek; duur >24 uur wordt
gemarkeerd, maar blijft aanwezig. Missing stations zijn geen reden om een geldig
vertrek niet te tellen. Onbekende categorieën worden zichtbaar gerapporteerd.

De Parquet-duur is de oorspronkelijke lokale klokduur. De EDA-duurlaag gebruikt
de expliciet gerapporteerde historische duur waar beschikbaar, anders het
verschil na New York-tijdzoneconversie. Ambigue najaarsuren zonder gerapporteerde
duur worden alleen uit duurstatistiek uitgesloten. Ook moeten de ritten vóór
het ontwikkelings-eindmoment afgerond zijn. Vertrektellingen blijven behouden.

`requirements-lock.txt` legt ook de werkelijk geïnstalleerde transitieve
dependencies vast; gebruik die voor een exacte omgevingsreproductie.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

## Onderzoekscontract en eindtest

De kandidaatcontracten worden vóór feature engineering opgeslagen. De doelkeuze
volgt uit ontwikkelings-EDA en de geblokte hypothesen, niet uit de bestaande
uitwerking of eindtest. `reports/target_decision.json` legt de gekozen taak vast.

Training: vóór mei 2026. Validatie: mei–juni 2026. **Juli–augustus 2026** blijft
gesloten voor EDA, hypotheseselectie, featureselectie, tuning en modelselectie.
Technische schema- en kwaliteitscontroles mogen vooraf plaatsvinden; geen
eindtestplots of modeluitkomsten. De onderzoeksdatasettoegang weigert de eindtest
tot er een evaluatievergrendeling bestaat.

Binnen een aanwezig, technisch intact en compleet gelezen maandarchief zijn
uurvakken zonder geldige vertrekken nul. De volledigheid van de publicatie is
daarbij een expliciete aanname, geen extern bewezen garantie. Ontbrekende
maanden veroorzaken een fout. Kwaliteitsverlies wordt apart gerapporteerd.

De drie hypothesen gebruiken vierweekblokken, sign-flip, effectgroottes,
bootstrapintervallen en Holm-Bonferroni. Achtweekblokken controleren gevoeligheid
voor langere afhankelijkheid. De methode veronderstelt voldoende onafhankelijke
blokken en symmetrie onder H0; grote aantallen ritten maken deze aannames niet
automatisch waar. Mediane duurkwantielen van volledige data worden benaderd
door DuckDB en als zodanig aangegeven.

De [officiële bronbeschrijving](https://citibikenyc.com/system-data) beschrijft
zowel de oude als moderne kolommen en voorafgaande filters van de publicatie.
De eigen pipeline voert geen extra generieke 60-seconden- of percentielgrens in.
De [PyCaret-API](https://pycaret.readthedocs.io/en/latest/api/regression.html)
documenteert expliciete `test_data`, eigen `fold_strategy` en shuffle-instellingen;
de opgeslagen audit controleert bovendien de werkelijk gebruikte rijen en folds.

De modellen vergelijken dezelfde vijf chronologische folds van elk 28 dagen.
PyCaret krijgt expliciete validatiedata en exact dezelfde opgeslagen foldgrenzen.
Random CV, shuffle en globale vooraf geleerde preprocessing zijn uitgeschakeld;
Ridge schaalt binnen iedere estimatorfold. RMSE is primair; MAE, R² en runtime
geven aanvullende context. Binnen 1% van de beste validatie-RMSE wint het
eenvoudigere model. Maximaal twintig tuningconfiguraties per veelbelovend model.

`train` slaat selectie en artefact vóór de eindtest op. `evaluate` vergrendelt
die fingerprints, evalueert eenmaal en schrijft rapporten. Een tweede oproep
leest uitsluitend de bestaande uitkomst. Na opening blokkeert de code nieuwe
onderzoeks- of modelkeuzes onder dezelfde eindtestclaim.
Bij een technische onderbreking vóór het voltooiingsrapport herstelt `evaluate`
uitsluitend hetzelfde vergrendelde experiment: model-, selectie- en datasetfingerprints
moeten overeenkomen. Er vindt geen nieuwe selectie of tuning plaats.
De vergrendeling omvat ook feature-, inference-, datatoegangs- en configuratiecode,
de operationele estimatorregels en packageversies. Evaluatie en model laden
weigeren afwijkingen van deze vastgelegde voorspelfingerprint.

## Voorspellen en modelartefact

Het definitieve artefact staat lokaal in `models/citibike_model.joblib` en is
uitgesloten van Git. Preprocessing, featurevolgorde, contract, metadata en model
worden samen opgeslagen. De modelkaart en werkelijke metrics staan in
`reports/MODEL_CARD.md` en `reports/final_evaluation.json`.

Als uurvraag door de EDA wordt gekozen: voorspel om 00:00 New York-tijd alle
uurvakken van één kalenderdag. Geef minimaal de veertien onmiddellijk voorafgaande
volledige dagen mee, met `timestamp`, `rides`, `area='NYC'`. Timestamps zijn lokale
kloklabels zonder offset. Geen telling van de voorspeldag mag in de invoer staan.
Er zijn altijd 24 labels: het niet-bestaande voorjaaruur is nul, het herhaalde
najaarsuur bevat beide fysieke uren. Verwachtingen worden niet afgerond in het model.

```python
from citibike.inference import predict_day
# history: DataFrame met volledige lokale dagen, strikt vóór de voorspeldag.
forecast = predict_day('2026-09-01', history)
```

De app biedt een ontwikkelingsdemonstratie en een CSV-upload voor eigen historie.
Een demonstratiedatum kan in de training zitten en is geen onafhankelijke test.
Het model voorspelt geregistreerd gebruik, geen stationbeschikbaarheid of
onvervulde vraag. Actuele tellingen moeten apart worden aangeleverd; maandarchieven
leveren niet automatisch live historie voor morgen.
De historische backtest veronderstelt dat de voorafgaande uurtellingen op het
voorspeltijdstip beschikbaar zijn via een externe tellingenbron. De openbare
maandarchieven tonen eventtijden, geen realtime aanlevering. Publicatie-,
ingestie- en rekenlatentie worden in deze backtest niet gemodelleerd. De gemeten
prestaties gelden daarom onder dit invoercontract; voor operationeel gebruik
moet een tijdige tellingenfeed worden gerealiseerd en gecontroleerd.

## Hertraining en volledige herbouw

### Lokaal opnieuw trainen en rekentijd meten

De oorspronkelijke studie is afgerond. Gebruik in dit project voor hertraining
de onderstaande commando's; `run.py train` start volledige modelselectie en wordt
terecht geblokkeerd zodra de oorspronkelijke eindtest is geopend.

```powershell
Set-Location 'C:\Deeplearning\DeepLearningTeam10\SolutionRaoul\Citybike'
# Alleen dezelfde vaste Random Forest opnieuw fitten en lokaal opslaan:
.\.venv\Scripts\python.exe -X utf8 tools/local_training.py retrain
# Drie nieuwe LightGBM-instellingen onderzoeken, zonder nieuwe eindtest:
.\.venv\Scripts\python.exe -X utf8 tools/local_training.py experiment
# Beide achtereenvolgens:
.\.venv\Scripts\python.exe -X utf8 tools/local_training.py all
```

Dit gebruikt de reeds voorbereide NYC-uurfeatures; de honderden miljoenen ruwe
ritten hoeven niet opnieuw verwerkt te worden. Iedere oproep krijgt een aparte
map in `reports/local_runs/` en een genegeerde modelmap in `models/local_runs/`.
`reports/local_runs/latest.json` verwijst naar het laatste voltooide rapport.
Het rapport bevat fit-, laad-, opslag- en totaaltijd, gesamplede procesboom-RSS,
CPU-tijd, machinegegevens, packages, code-, bron- en datasetfingerprints.
De RSS-meting omvat geladen modellen en data, bemonstert iedere 0,2 seconde
en is geen exacte continue piekmeting; andere apps vallen buiten die meting.

`retrain` gebruikt exact de vaste configuratie en 114.336 trainingsrijen tot
en met juni 2026. De opgeslagen kopie wordt opnieuw geladen en moet dezelfde
voorspellingen leveren. Deze reproductiecontrole is geen nieuwe accuracy-test:
mei–juni zit in deze definitieve refit.

`experiment` legt vóór fitting drie configuraties vast: Poisson, Huber en
geregulariseerde squared-error LightGBM. Alle drie gebruiken dezelfde features,
112.872 ontwikkelingsrijen en vijf opgeslagen chronologische folds. De laagste
gemiddelde CV-RMSE bepaalt de kandidaat; alleen daarna wordt deze op mei–juni
vergeleken met de oorspronkelijke RF en weekbaseline. Het kandidaatmodel wordt
uitsluitend op ontwikkeling getraind. Validatie is al eerder gebruikt en deze
vervolgstudie gebeurt na afloop van de oorspronkelijke studie: resultaten zijn
verkennend en vormen geen onafhankelijke bevestiging. Juli–augustus wordt niet
geladen of opnieuw geëvalueerd. Het bestaande model, selectie en eindtestrapport
worden via checksums beschermd en de Streamlit-app blijft hetzelfde model laden.

Een verbeterde CV-score garandeert geen betere validatiescore. Gebruik voor
verdere verbetering hetzelfde chronologische protocol, een vooraf begrensde
zoekruimte en uitsluitend informatie die om 00:00 beschikbaar was. Voor een
nieuwe onafhankelijke prestatieclaim moet een nieuwe, vooraf gereserveerde
testperiode beschikbaar zijn. Leg een nieuwe studie en eventuele deployment
apart vast; vervang niet stilzwijgend de geëvalueerde modelversie.

De twee sterkste CV-kandidaten krijgen een vastgelegde zoekbegroting: zes
Random Forest-configuraties en maximaal twintig voor LightGBM of Ridge.
De kleinere RF-zoekbegroting is vóór tuning gekozen op basis van de gemeten
rekentijd, zonder eindtestinformatie. `reports/tuning_protocol.json` legt dit,
de seed en dezelfde folds vast; alle daadwerkelijk uitgevoerde configuraties
staan in de tuningrapporten.
Voor de 1%-regel gebruiken we de vooraf vastgelegde volgorde van modelstructuren:
weekbaseline → Ridge → Random Forest → LightGBM. Rekentijd, artefactgrootte en
train/validatieverschil worden daarnaast gerapporteerd, zodat de praktische
kosten van een gekozen model zichtbaar blijven.

Voor een eerlijke nieuwe modelselectie na deze eindtest is een **nieuwe onafhankelijke
testperiode** nodig. Pas grensdata expliciet aan, archiveer eerdere rapporten en
herhaal de volledige workflow. Inspecteer geen oude test om nieuwe keuzes te sturen.

Een exacte reproductie van de bestaande studie gebruikt dezelfde code, bronnen,
dependencies, seeds en configuratie in een nieuwe projectkopie met lege gegenereerde
directories. Voer bovenstaande CLI-volgorde opnieuw uit. Bewaar de oorspronkelijke
rapporten apart en vergelijk metrics, bron- en datasetfingerprints. Parquetbytes
en getimede runtimes kunnen tussen platformen verschillen; de datasetfingerprint
identificeert de concrete gegenereerde bestanden, geen platformonafhankelijke semantiek.

`reports/data_manifest.json` en de modelrapporten bevatten de benodigde
reproductiegegevens; `verification.json` controleert partitions, folds, inference
en Git-ignore. `verify` fit ook de vastgelegde configuratie opnieuw op precies
dezelfde trainingsrijen, zonder tuning of testlabels, en vergelijkt de voorspellingen.
Daarna bedient Streamlit AppTest het echte formulier en controleert de 24 uitvoerwaarden
tegen de gedeelde offline voorspelfunctie. Dezelfde geëvalueerde modelversie wordt in de app gebruikt.

AWS-training blijft een afzonderlijke latere taak. De team-API en online
frontend gebruiken inmiddels het gekozen model; de lokale Streamlit-app blijft
ook beschikbaar. Automatische hertraining op nieuwe tellingen is een aparte taak.
