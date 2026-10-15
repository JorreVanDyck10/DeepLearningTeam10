# Kritische controle van Final tegenover de lesinhoud

Controle: 15 oktober 2026. Dit concept is op verzoek van de gebruiker door Codex
opgesteld. Het staat in `solutions/milan`; de mapnaam betekent geen bevestiging
dat Milan deze code zelfstandig heeft geschreven of al inhoudelijk heeft goedgekeurd.

## Oordeel

Final bevat een uitgebreide, grotendeels verdedigbare oplevering. De opgeslagen
resultaten zijn onderling controleerbaar. **Ik kan niet bevestigen dat alles volgens
de lesaanpak is afgerond:** de Mushroom-eindtest is niet onafhankelijk en de
gecorrigeerde modelvergelijking herhaalt de oorspronkelijke tuning niet.
Een nieuwe modelnaam of een andere willekeurige split herstelt dit niet.

Dit is een inhoudelijke controle van de lokale cursusbestanden, Final, de aangeroepen
broncode en opgeslagen onderzoeksresultaten op basiscommit `e7ece14`. Alle elf
cursus-PowerPoints zijn als bron geïnventariseerd, inclusief notitietekst. De
relevante passages en bijbehorende oefeningen zijn hieronder expliciet genoemd.
De inhoudelijke claims van de groepspresentatie zijn ook met de cijfers vergeleken;
de visuele opmaak is hier niet in PowerPoint getest.

Het oorspronkelijke volledige opdrachtbestand, de actuele Canvas-rubric en de
genoemde `GenAI_Policy.pdf` staan niet in deze checkout. Daarom is dit geen
garantie van rubric- of policyconformiteit. De Intro-PowerPoint noemt nog 60k
Mushrooms; Final documenteert een latere afspraak over het cursusbestand met
5.000 rijen. Ik behoud die bronkeuze en kan die latere afspraak niet uit de Intro
alleen onafhankelijk bevestigen.

## Vergelijking met de lessen

| Lesonderdeel | Bewijs in de oplevering | Kritisch oordeel |
|---|---|---|
| Data science: variatie, covariatie en interpretatie — 1.1, dia 2–8 | Mushroom-EDA, Citi Bike-EDA en hypotheserapporten | Aanwezig. Mushroom-featuredictionary is breder dan de echte cursus-CSV; inkorten tot de aanwezige velden. |
| OSEMN en ontbrekende waarden — 5, dia 7–13 en 33–37 | Aparte voorbereiding, imputatie en one-hot encoding | Goed opgebouwd; medianen/modi worden in de gecorrigeerde pipelines binnen de fold geleerd. |
| Train/validatie/test — 3, dia 4–5; 4, dia 13 en 17–24 | Gedeelde Mushroom-folds en expliciete waarschuwing over de 900 records | Trainings-CV is geldig als ontwikkelvergelijking. Een onaangeraakte Mushroom-eindtest ontbreekt. |
| Tuning en overfitting — 4, dia 3–13; oefening 4/4 Winetrees en 4/5.2 Random Forest | Historische grids, nieuwe fits met vaste instellingen | Nieuwe scores zijn reproduceerbaar, maar bewijzen niet dat dit nog de beste instellingen voor de gecorrigeerde pipeline zijn. |
| Confusion matrix en metrics — 3, dia 8–20 | Macro F1, recall p, precision, FN/FP, ROC-AUC en AP | Goed. p=1 is consequent de positieve klasse; alleen accuracy zou onvoldoende zijn. |
| Beslisdrempels — 3, dia 22–29; oefening 3/2 Threshold tuning | Bestaande drempelaudit en voorzichtige websitebeslissing | Al aanwezig. Een drempel gericht op hogere giftige recall kan nog als afzonderlijk validatie-experiment worden uitgewerkt. |
| Bagging/boosting/lineaire modellen — 4, dia 31–54 | RF, GB, LR, Ridge en LightGBM | Passende modellen. Extra complexe imputatie is niet automatisch een verbetering. |
| Tijdreeksvolgorde en lookahead — 6, dia 3, 6–7 en 21–22 | Chronologische folds, daglags, shift vóór rolling, weekbaseline | Sterk. Geen gewone geschudde CV gebruikt voor Citi Bike. |
| Stationariteit en tijdreeksmodellen — 6, dia 13–29 | ADF/ACF-diagnose en lagmodellen | Aanwezig in brononderzoek. ARIMA/ETS toevoegen is een mogelijke vergelijking, geen bewezen verplichte correctie. |
| Unsupervised learning — 7 | Historische proximity-imputatie; supervised einddoelen | Geen reden om alleen voor het afvinken KMeans toe te voegen. De slides maken dit niet verplicht voor ieder project. |
| Volledige modelketen — 2, dia 15 | Herlaadbare pipelines, API, export en trainingsworkflow | Sterke artifactcontrole. Mushroom-refit gebruikt dezelfde 4.100 rijen en dezelfde vaste configuratie. Dit is geen automatisch onderzoek op nieuwe data. |
| Omgeving en notebooks — 1.4, dia 20–32 | Aparte gepinde omgevingen en opgeslagen notebookuitvoer | Terecht verschillende omgevingen voor PyCaret/training en serving. Final is niet een losstaand project. |

## Verbeterpunten in volgorde van belang

### 1. Een onafhankelijke Mushroom-eindtest ontbreekt

De les vraagt dat de eindtest pas na modelkeuze wordt geopend. Final noemt zelf
dat de 900 records al eerder zijn bekeken. Die eerlijkheid is goed, maar maakt
81,33% accuracy geen onbevooroordeelde eindtestscore.

**Actie:** behoud de term ontwikkeltest, ook mondeling. Voor een echte nieuwe
eindclaim zijn nieuwe, nog niet voor ontwikkeling gebruikte data en een vooraf
vastgelegd protocol nodig. Ook nested CV in dit concept blijft ontwikkelonderzoek
op reeds gebruikte projectdata.

Bronnen: [Final-overzicht](../../Final/README.md),
[vergelijking](../../Final/Mushroom/09_vergelijking.ipynb), les 4 dia 20 en les 5 dia 43.

### 2. Tuning is niet opnieuw uitgevoerd op de gecorrigeerde pipeline

Dit is concreter dan alleen een ontbrekende grafiek. In
[het oorspronkelijke projectnotebook](../../MushroomFinal/mushroom_project.ipynb)
fit celindex 64 de RF-grid op `X_train_imputed`, na de vooraf geleerde
proximity-imputatie. Celindex 67 fit de geselecteerde RF vervolgens op `X_train`.
De selectie en uiteindelijke fit gebruiken dus verschillende voorbereiding.
GB/LR hadden daarnaast de al gedocumenteerde preprocessing buiten CV.

De gecorrigeerde Final-versies leren preprocessing wel per fold, maar
`reproduce()` test alleen de reeds gekozen instellingen. De oorspronkelijke
grid wordt als tekst getoond. Dat valideert de vaste kandidaten; het herhaalt
niet de selectieprocedure en bewijst geen optimale hyperparameters.

**Actie:** voer een kleine expliciete search opnieuw uit met de complete pipeline
binnen CV. Houd evaluatie van de tuningprocedure apart van de folds waarop de
parameters gekozen worden. Dat is precies het alternatief in deze map: dezelfde
vijf buitenste folds, met drie binnenste trainingsfolds voor de grid. De cursus
leert de bouwstenen; deze extra scheiding past ze strikter toe.

Bronnen: [RF-leesversie](../../Final/Mushroom/05_Random_Forest.ipynb),
[model_workflow.py](../../Final/tools/model_workflow.py),
[kandidaten](../../MushroomFinal/compare_correct_dataset.py).

### 3. Maak het foutentype belangrijker dan één samenvattende score

RF300 mist op trainings-OOF ongeveer 40% van de giftige voorbeelden. Op de reeds
bekende ontwikkeltest zijn het 121 van 341, dus 35,48%. Macro F1 kiest een
evenwicht, terwijl de beoogde toepassing gemiste giftige gevallen zwaarder weegt.

Er is al een goede [drempelaudit](../../MushroomFinal/comparison_correct/decision_threshold_audit.json).
Deze toont dat zelfs hoge eetbaarheidsscores giftige records bevatten. De website
bevestigt daarom terecht geen eetbaarheid. Dit concept voegt een onderwijsproef
toe: kies binnen training de hoogste specificiteit bij ten minste 90% giftige
recall, en meet vervolgens de gemiste gevallen én de extra valse alarmen op een
afzonderlijke buitenfold. De 90% is een gekozen demonstratiedoel, geen veiligheidsnorm
en geen gegarandeerde prestatie op ongeziene data. Een drempel verandert ook
geen ruwe modelscores in gekalibreerde kansen.

**Actie:** neem alleen een gemeten afweging over. Houd de huidige voorzichtige
websitebeslissing aan totdat een afzonderlijke keuze en validatie zijn gedaan.

### 4. Verbeter de vindbaarheid en de controles van de leesmap

Twee links verwijzen naar het niet-bestaande `09_compare_models.ipynb` vanuit
`Final/Mushroom/09_vergelijking.ipynb` en `10_deployment.ipynb`.
Gebruik daar `09_vergelijking.ipynb`.

`Final/tools/verify_final.py` controleert Markdownlinks alleen in `.md`-bestanden,
niet in notebook-Markdown. Ook controleert het de aanwezigheid van historische
uitvoeringsregistraties; dat is geen nieuwe uitvoering van alle 19 notebooks.
De voorbereiding en verschillende Citi Bike-notebooks lezen eerder opgebouwde
rapporten/data. Dat staat terecht vermeld, maar moet ook duidelijk zijn voor
iemand die alleen de leesmap opent.

**Actie:** neem notebooklinks mee in de checker, voeg een duidelijke route naar de
bron-CLI toe, en onderscheid opnieuw trainen van opgeslagen resultaten tonen.
Onze [audit](audit_final.py) controleert ook notebooklinks.

### 5. Vereenvoudig en onderbouw de Mushroom-voorbereiding

De EDA-featuredictionary bevat velden uit de volledige UCI-bron die niet in de
5.000-rijen-CSV staan. De klasseverdeling is 62,14% eetbaar, geen exacte twee derde.
Een dictionary op de werkelijk aanwezige 11 invoervelden is duidelijker.

`spore_print_color` is 94,8% ontbrekend. Een complexe proximity-imputatie moet
daarom aantoonbaar beter zijn dan mediaan/modus of een expliciete missingcategorie.
De huidige verschillende modelpipelines isoleren dat effect niet. Automatisch
alle missings, nullen of uitschieters verwijderen volgt evenmin uit deze data.

**Actie:** test eenvoudige imputatiekeuzes binnen dezelfde search. Dit concept
vergelijkt categorische modus met een eigen missingcategorie, zonder proximity.
Het behoudt dezelfde elf kenmerken en `jumbled_noise_1`, zodat het geen wijziging
van het afgesproken invoercontract introduceert. Verwijderen van die resterende
noise-kolom is een eventuele latere ablatietest, geen hier bewezen verbetering.

### 6. Geef Citi Bike de juiste praktische afbakening

De historische test is methodologisch veel sterker dan bij Mushroom. De voorspelling
vereist wel veertien volledige dagen tellingen die vóór 00:00 bekend zijn. De
publieke maandarchieven leveren die operationele feed niet automatisch. Twee
zomermaanden bewijzen ook geen prestaties in alle seizoenen.

De getunede RF verbetert de oorspronkelijke RF op validatie slechts circa 0,27%,
met een groter model. De vooraf vastgelegde 1%-regel vergelijkt modeltypen; binnen
RF wordt toch de laagste RMSE gekozen. Dat is consistent met de code. Een toekomstige
studie kan complexiteit ook binnen dezelfde familie meewegen. De bestaande
eindtest mag daarvoor niet opnieuw als selectiecriterium worden gebruikt.

**Actie:** benoem feed, horizon en beperking in de demo. Een nieuwe wintertest of
ARIMA/ETS-vergelijking vraagt een nieuwe studie. De centrale workflow traint alleen
Mushroom automatisch; voor Citi Bike staat een vaste release gekoppeld.

Bronnen: [Citi Bike-modelkaart](../../SolutionRaoul/Citybike/reports/MODEL_CARD.md),
[selectiecode](../../SolutionRaoul/Citybike/citibike/modeling.py),
[automatisering](../../Final/AUTOMATISERING.md).

## Wat hier opnieuw gecontroleerd is

De [machineleesbare audit](results/audit_final.json) legt de concrete controles vast:

- 19 Final-notebooks gelezen; geen opgeslagen foutuitvoer gevonden.
- Alle 17 bronhashes uit het Final-manifest komen overeen.
- 396 Mushroom-test- en foldmetrics herberekend uit de opgeslagen voorspellingen.
- De actieve serving-export op alle 900 referentierijen gecontroleerd; de grootste
  kansafwijking is circa 2,22 × 10⁻¹⁶. De actieve pipelinehash klopt met de metadata.
- Citi Bike-RMSE, MAE en R² herberekend uit de 1.488 bestaande testvoorspellingen.
  Dit is een controle van het bestaande rapport, geen nieuwe modelkeuze of testfit.
- Alle vijf chronologische foldgrenzen gecontroleerd. Met synthetische data getest
  dat gewijzigde tellingen op/na dag D de features voor dag D niet veranderen.

De volledige oorspronkelijke PyCaret/AWS/Citi Bike-training is hier niet opnieuw
uitgevoerd. De huidige machine heeft Python 3.13; de referentie gebruikt sklearn
1.4.2 in een andere omgeving. Het alternatief gebruikt een afzonderlijk vastgelegd
sklearn 1.6.1-protocol, met beide RF-varianten opnieuw in dezelfde omgeving.
Publieke hosting en cloudruns zijn in deze review niet opnieuw live gecontroleerd.
De bijgevoegde broninventaris legt hashes vast, maar is geen vervanging voor de
hierboven genoemde passages en de oorspronkelijke cursusbestanden.

De nieuwe vijf-foldvergelijking is volledig uitgevoerd. De [uitkomst](RESULTATEN.md)
laat een zeer kleine macro-F1-winst zien (0,761876 naar 0,762934), met 21 minder
gemiste giftige gevallen en 24 extra valse alarmen. Dat is onvoldoende om algemene
superioriteit of onmiddellijke vervanging van het bestaande model te claimen.

De bijdragenvermelding is deels achterhaald: `BIJDRAGEN.md` bevestigt inmiddels ook
Raoul, terwijl delen van `INLEVERCHECK.md`/`GENAI_DISCLOSURE.md` nog alleen Jorre als
bevestigd beschrijven. Werk die verwijzingen bij zonder de open bevestigingen van
Andrew/Milan of een Canvas-indiening automatisch af te vinken.

## Wat ik zou veranderen vóór inlevering

Eerst de twee links en achterhaalde statuszinnen corrigeren. Vervolgens zorgen
dat iedereen het verschil tussen ontwikkeltest, CV en onafhankelijke eindtest kan
uitleggen. Het nieuwe tuningexperiment pas toevoegen als het team de aanpak en
de werkelijke uitkomst kan verdedigen. Een complexere pipeline zonder gemeten
winst hoeft de bestaande oplossing niet te vervangen.
