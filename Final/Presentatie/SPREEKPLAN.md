# Spreekplan — 25 minuten inclusief vragen

> Historisch spreekplan voor de eerdere twaalfdia-versie. Gebruik voor de actuele
> groepspresentatie [SPREEKPLAN_GROEP.md](SPREEKPLAN_GROEP.md). Jorres bijdragen en
> de GenAI-verantwoording zijn inmiddels centraal ingevuld; oude statuszinnen
> hieronder beschrijven de voorbereiding van de vorige versie.

Plan ongeveer 16,5 minuut uitleg, 1,5 minuut demonstratie en 7 minuten vragen.
Sprekers spreken jullie zelf af; de definitieve bijdragen blijven stap 5.
De cover is een AI-illustratie, geen foto uit de dataset. Tabellen en grafieken
in de PowerPoint zijn bewerkbaar; bronverwijzingen staan in de spreeknotities.

## Dia 1 — Paddenstoelen en fietsverkeer

0:30 â€” Stel het team voor zonder een niet-bevestigde taakverdeling te verzinnen. De daadwerkelijke menselijke bijdragen worden door Jorre nog gecontroleerd (stap 5). Onderwerp: een classificatieprobleem met Mushroom en een tijdreeksprobleem met Citi Bike. Coverillustratie is AI-gegenereerd, decoratief en geen datasetfoto.

Bronnen: `README.md`

## Dia 2 — Data en voorspeltaken

1:15 â€” Mushroom gebruikt de afgesproken 5.000 cursusrijen. Het oorspronkelijke UCI-bestand is niet de definitieve vergelijkingsbron. Citi Bike start van ritregistraties, maar het model voorspelt geaggregeerde NYC-uurvraag. Houd NYC en Jersey City gescheiden. Citeer bronmanifesten voor omvang, niet alleen een bestandsnaam.

Bronnen: `Final/tools/data.py`, `SolutionRaoul/Citybike/reports/data_manifest.json`

## Dia 3 — Ontbrekende Mushroom-kenmerken

1:30 â€” Benoem de dominante ontbrekende kenmerken. We gaan ze niet willekeurig invullen op basis van alle data. De uiteindelijke pipeline leert medianen en modi binnen elke trainingsfold. Nulmetingen blijven bij RF300 echte nulmetingen. Het ruisveld jumbled_noise_0 is verwijderd, jumbled_noise_1 blijft.

Bronnen: `Final/Mushroom/01_EDA.ipynb`, `Final/Mushroom/02_voorbereiding.ipynb`

## Dia 4 — Validatie en preprocessing

1:30 â€” Laat zien waarom preprocessing vÃ³Ã³r cross-validatie leakage kan geven. De oorspronkelijke proximity-imputatie was vÃ³Ã³r CV geleerd; de gecorrigeerde vergelijking fit de volledige pipeline binnen elke fold. De 900 testrecords zijn eerder bekeken en mogen dus niet als onaangeraakte eindtest worden verkocht.

Bronnen: `MushroomFinal/final_model_support.py`, `MushroomFinal/comparison_correct/protocol.json`

## Dia 5 — Mushroom-modelvergelijking

2:00 â€” De primaire rangschikking is gemiddelde CV macro F1. RF300 is de voorlopige keuze, met een kleine voorsprong op AWS en RF500. Dit is geen bewijs dat ieder verschil statistisch significant is. De AWS-rij komt uit de afzonderlijk gecontroleerde echte SageMaker-run.

Bronnen: `MushroomFinal/comparison_correct/model_comparison.csv`, `MushroomFinal/comparison_correct/aws_artifact_audit.json`

## Dia 6 — Mushroom-fouten en de website

1:30 â€” Accuracy verbergt dat er 121 giftige voorbeelden worden gemist. De positief-klasse-recall is 220/341 = 64,52%. Ook >=95% eetbare scores zijn niet veilig geÃ¯nterpreteerd: 7 van 55 trainingsvalidatiegevallen zijn giftig. Daarom bevestigt de UI geen eetbaarheid; onder 50% giftigheidsscore onthoudt zij zich met onzeker/mogelijk giftig. Dat maakt de classifier niet nauwkeuriger.

Bronnen: `MushroomFinal/comparison_correct/model_comparison.csv`, `docs/mushroom_decisions.md`

## Dia 7 — Training en tuning op SageMaker

1:00 â€” Jorre voerde training en vier kandidaatinstellingen uit op het juiste cursusbestand. Het gedownloade notebook en de pipeline staan in Git. De artifactaudit herberekent de voorspellingen en vergelijkt ze met de lokale baseline. Het AWS-model hoeft niet automatisch het beste model te zijn.

Bronnen: `SolutionJorre/MushroomDataset/correct_dataset/04_aws_correct_dataset.ipynb`, `MushroomFinal/comparison_correct/aws_artifact_audit.json`

## Dia 8 — Citi Bike: een toetsbaar patroon

1:30 â€” De hypothese is vÃ³Ã³r de berekening vastgelegd. Ochtendrit-aandeel ligt op werkdagen gemiddeld 9,55 procentpunt hoger dan in het weekend. Holm-gecorrigeerde p is ongeveer 0,00015; vierweekblokken houden rekening met tijdsafhankelijkheid. Dit is associatie, geen causaliteit. Lagkenmerken vereisen tijdig beschikbare voorafgaande tellingen.

Bronnen: `SolutionRaoul/Citybike/reports/hypotheses.json`, `SolutionRaoul/Citybike/reports/hypothesis_protocol.json`

## Dia 9 — Citi Bike-modelvergelijking

1:30 â€” Dit zijn validatiescores van meiâ€“juni, geen eindtestscore. Lagere RMSE is beter. RF320 is de vastgelegde keuze. De RF-tuning leverde slechts een kleine extra verbetering; modelcomplexiteit en looptijd worden mee afgewogen. De eindtest bleef beschermd tot na de selectie.

Bronnen: `SolutionRaoul/Citybike/reports/model_comparison.csv`, `SolutionRaoul/Citybike/reports/model_selection.json`

## Dia 10 — Citi Bike: de vastgelegde eindtest

1:30 â€” Eindtest juliâ€“augustus: RF RMSE 1.979,51 tegenover weekbaseline 2.608,78; relatieve reductie 24,12%. MAE 1.076,23 en RÂ² 0,8280 over 1.488 uurvakken. Geen verdere tuning op deze eindtest. Onzekerheidsinterval is indicatief door ongeveer negen weekblokken.

Bronnen: `SolutionRaoul/Citybike/reports/final_evaluation.json`, `SolutionRaoul/Citybike/reports/evaluation_lock.json`

## Dia 11 — API, website en automatische publicatie

1:45 â€” Leg de datastroom uit. Een codepush triggert echte Mushroom-training op Python 3.11. Een afzonderlijke Python 3.13-job verifieert de lossless export en beide API-modellen. Publicatie gebeurt pas na kwaliteitsgates en tests; een stale broncommit wordt geweigerd. Render en Netlify volgen de gevalideerde commit. Citi Bike wordt uit een gepinde release geladen; deze workflow claimt geen nieuwe Citi Bike-training.

Bronnen: `Final/AUTOMATISERING.md`, `.github/workflows/model-training.yml`, `backend/main.py`

## Dia 12 — Resultaat en beperkingen

1:00 + live demo indien tijd. Toon eerst Voorbeeld invullen/Voorspellen voor Mushroom, daarna de historische Citi Bike-demo. De gitnotebooks zijn de fallback. Rond af met: reproduceerbare vergelijking, bewezen koppeling naar echte inference, maar geen eetadvies en geen gegarandeerde toekomstige City Bike-kwaliteit. Laat circa 7 minuten voor vragen. De definitieve bijdragen en GenAI-policy worden door Jorre in stap 5 gecontroleerd; dit spreekplan verzint geen taakverdeling.

Bronnen: `Final/README.md`, `Final/INLEVERCHECK.md`

## Demo voorbereiden

1. Open de website voor de presentatie zodat de backend kan opstarten.
2. Mushroom: Voorbeeld invullen, Voorspellen; leg modelscore en onthouding uit.
3. Citi Bike: historische demo van 30 april 2026, uur 8; toon de 24 uurvoorspellingen.
4. Gebruik bij netwerkproblemen de bewaarde notebookuitvoer en het websitebeeld in de slides.

## Samen oefenen

Bespreek waarom preprocessing per fold wordt geleerd, waarom accuracy alleen
onvoldoende is voor Mushroom, waarom een modelscore geen gekalibreerde kans is,
waarom Citi Bike chronologisch wordt gesplitst, welke tellingen een voorspelling
nodig heeft en hoe een gevalideerd model van GitHub naar de backend gaat.
Ieder teamlid moet de hoofdlijn van beide datasets kunnen uitleggen.
