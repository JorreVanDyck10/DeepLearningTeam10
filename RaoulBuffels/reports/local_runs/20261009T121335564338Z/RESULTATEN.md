# Lokale training gemeten op 9 oktober 2026

Uitgevoerd vanuit `C:\Deeplearning\CloudAI\Citybike_full`:

```powershell
.\.venv\Scripts\python.exe -X utf8 tools/local_training.py all
```

Deze run gebruikte de bestaande voorbereide NYC-features. Ruwe ZIPs en
maandpartitions zijn niet opnieuw gebouwd. De machine heeft 10 fysieke en
16 logische CPU's en 15,71 GiB RAM; de modellen gebruiken twee threads/jobs.

| Stap | Gemeten duur | Gesamplede maximale procesboom-RSS |
|---|---:|---:|
| Vast RF-model fitten, 114.336 rijen | 74,41 seconden | 1,52 GiB |
| Hertraining inclusief laden, opslaan en herlaadcontrole | 80,51 seconden | 1,82 GiB |
| Drie LightGBM-kandidaten op vijf chronologische folds, plus winnaar fitten/evalueren | 20,13 seconden | 0,30 GiB |
| Beide achtereenvolgens, inclusief tussenstappen | 101,34 seconden | Zie afzonderlijke metingen |

De RSS wordt iedere 0,2 seconde bemonsterd. De meting omvat de geladen data en
modelkopieën, geen andere apps. Een volgende run kan door belasting en caches
langer of korter duren. De gekopieerde RF is 647,33 MiB groot en wordt door Git
genegeerd. Hij is opnieuw geladen en geeft dezelfde voorspellingen: maximale
numerieke afwijking 3,64 × 10⁻¹² ritten. Dat is een reproductiecontrole, geen
nieuwe accuracy-test: de definitieve RF-refit gebruikt mei–juni als training.

## Begrensd verbeterexperiment

Alle kandidaten gebruiken dezelfde 112.872 ontwikkelingsrijen, features en vijf
opgeslagen chronologische folds. Configuraties zijn vóór de fitting vastgelegd.
Alleen de CV-winnaar is daarna op de 1.464 validatie-uurvakken van mei–juni
geëvalueerd, na fitten uitsluitend op ontwikkeling vóór mei.

| Nieuwe kandidaat | Gemiddelde CV-RMSE | Tijd vijf folds |
|---|---:|---:|
| LightGBM Poisson | 1.421,31 | 7,69 s |
| LightGBM Huber | 3.032,97 | 6,67 s |
| LightGBM geregulariseerde squared error | 1.416,37 | 4,68 s |

| Model | Validatie-RMSE | Validatie-MAE | Validatie-R² |
|---|---:|---:|---:|
| Weekbaseline | 2.826,16 | 1.510,13 | 0,6681 |
| Oorspronkelijke getunede RF, gefit vóór mei | 1.940,49 | 1.114,48 | 0,8435 |
| Nieuwe CV-winnaar, gefit vóór mei | 1.922,88 | 1.140,33 | 0,8463 |

De nieuwe kandidaat heeft **0,91% lagere RMSE**, maar een hogere MAE. Het resultaat
is gemengd en de RMSE-winst blijft onder de oorspronkelijke 1%-grens. Onder de
oorspronkelijke vooraf vastgelegde volgorde van modeltypen blijft RF de keuze.
Het nieuwe model is apart opgeslagen en niet in de app geplaatst.

Dit is een verkennende vervolgstudie. Mei–juni is al gebruikt bij de oorspronkelijke
modelselectie en vormt geen onafhankelijke bevestiging. Juli–augustus is niet
opnieuw geladen of geëvalueerd; de oorspronkelijke eindtest kan geen nieuwe
kandidaat onafhankelijk bevestigen. Het model, de selectie en de oorspronkelijke
evaluatierapporten hebben na afloop exact dezelfde checksums.

## Zelf uitvoeren

```powershell
Set-Location 'C:\Deeplearning\CloudAI\Citybike_full'
# Alleen vaste hertraining:
.\.venv\Scripts\python.exe -X utf8 tools/local_training.py retrain
# Alleen het beperkte vervolgexperiment:
.\.venv\Scripts\python.exe -X utf8 tools/local_training.py experiment
```

Elke uitvoering maakt nieuwe rapport- en modelmappen met een unieke tijdcode.
`reports/local_runs/latest.json` wijst naar het laatste voltooide `run.json`.
`run.py train` is de volledige selectieworkflow en blijft in deze afgeronde
studie geblokkeerd. Voor verdere onafhankelijke verbetering: reserveer vooraf
een nieuwe, ongeziene testperiode in een aparte studie, behoud chronologische
folds en het D00-voorspelcontract, en begrens de zoekruimte vóór training.

De originele volledige modelvergelijking en tuning duurde circa 72,8 minuten;
deze kleine vervolgmeting is niet hetzelfde werk. Exacte machine-, package-,
code-, bron-, dataset-, rij- en splitgegevens staan in `run.json`, `retraining.json`
en `experiment.json`. De bestaande 27 tests zijn opnieuw uitgevoerd en geslaagd.
