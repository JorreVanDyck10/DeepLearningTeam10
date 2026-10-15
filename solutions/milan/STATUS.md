# Actuele status: derde studie afgerond

15 oktober 2026. Uitsluitend `solutions/milan` gewijzigd.

## De beste gemeten afweging

Gewogen voting met een binnen training gekozen F1-drempel verbetert tegenover
kandidaat v2 zowel het aantal gemiste giftige gevallen als het aantal valse alarmen.

| Procedure | Macro F1 | Recall giftig | FN | FP | Accuracy |
|---|---:|---:|---:|---:|---:|
| Oorspronkelijke RF300 | 0,7619 | 60,44% | 614 | 256 | 78,78% |
| Kandidaat v2 met beslispolicy | 0,7691 | 66,82% | 515 | 356 | 78,76% |
| Voting v3 met F1-policy | **0,7788** | **67,79%** | **500** | **334** | **79,66%** |

Dit zijn vijf gelijke buitenfolds op de oorspronkelijke 4.100 trainingsrijen.
FN/FP zijn opgetelde aantallen; de andere cijfers zijn foldgemiddelden. De winst
tegenover v2 is **15 minder gemiste giftige gevallen en 22 minder valse alarmen**.
Macro F1 verbetert in vier van vijf folds. De verbetering is bescheiden; er worden
nog veel giftige gevallen gemist. De bekende 900 ontwikkeltestrecords zijn niet
opnieuw gescoord. Dit is verdere ontwikkeling, geen onafhankelijke eindtest.

Voting met het vooraf gekozen FPR-budget haalt 494 FN en 351 FP bij macro F1
0,7767. Die optie mist iets minder giftige gevallen dan de gekozen F1-policy,
maar geeft meer valse alarmen. Alle procedures staan in [RESULTATEN_V3.md](RESULTATEN_V3.md).

## Praktisch eenvoudiger alternatief

RF300 met interne featureselectie en standaardgrens haalt macro F1 **0,7749**,
accuracy **79,83%**, 576 FN en 251 FP. Tegenover de oorspronkelijke RF300 zijn dat
38 minder gemiste giftige gevallen en vijf minder valse alarmen. Tegenover v2
zijn er juist 61 extra gemiste giftige gevallen en 105 minder valse alarmen.
Dit is dus een andere afweging en haalt het volledige v2-verbeterdoel niet.

Het RF-pakket behoudt de bestaande modellezer en alle elf invoervelden. De finale
trainingsselectie negeert intern sporenkleur en het extra ruisveld. Het gebruikt
300 bomen, diepte 20, balanced class weights en een vaste grens van 0,5. Nieuwe
servingbibliotheken zijn hiervoor niet nodig. De publicatie-/trainingsworkflow
moet later wel hetzelfde kolomselectie- en exportrecept gaan gebruiken.

## Wat is werkelijk uitgevoerd?

- 24 vaste configuraties, vijf buitenfolds en drie binnenfolds: **360 gridfits**,
  naast refits en binnen-OOF-berekeningen. Alle modelkeuzes en drempels gebruiken
  alleen training binnen elke buitenfold; ook votinggewichten worden daar gekozen.
- Featureselectie, LightGBM, CatBoost en zeven vaste votingcombinaties. Dit sluit
  aan op de lessen over voorbereiding, regularisatie, boosting, ensembles en
  modelkwaliteit. Het [protocol](STUDIE_V3.md) noemt de gebruikte dia's.
- Twee afzonderlijke fulltrain-kandidaatpakketten. De votingfit kiest gelijke
  gewichten voor RF, LightGBM en CatBoost en een drempel van ongeveer 0,50.
  Buitenfolds kiezen hun eigen drempel; de CV-cijfers beoordelen de hele
  selectieprocedure en zijn geen onafhankelijke test van dit fulltrain-artifact.
- RF-array-export én native voting-export geven op alle 4.100 gecontroleerde
  invoerrijen exact dezelfde kansen als de getrainde pipelines. Ook nullen en
  onbekende categorieën zijn gecontroleerd. Dit zijn exportchecks, geen accuracytests.
- **20 Python-tests geslaagd:** acht onderzoekscontroles en twaalf pakket-/API-
  en regressiecontroles, zonder skips. Dat omvat de bestaande v2-controles.
- **Negen frontendtests geslaagd** op de echte gegenereerde beslisfuncties van
  alle drie previews: grensgevallen, versie, verkeerde grens en elf invoervelden.
- Drie nieuwe rapportnotebooks uitgevoerd en met `nbformat` gevalideerd. De
  uitvoering gebeurde via IPython met verse naamruimtes; er wordt geen aparte
  Jupyter-kernelrun geclaimd. De bouwer ondersteunt daarnaast gewone kernels.

Receipts: [onderzoek](results_v3/test_results.json),
[pakketten/API](results_v3/package_test_results.json),
[frontend](results_v3/frontend_test_results.json) en
[notebooks](results_v3/notebook_execution.json).

## Wat moet later nog gekoppeld worden?

Voting gebruikt een eigen native modellezer en extra boostingbibliotheken. Het
is geen losse artifactwissel bij de bestaande RF-lezer. Het RF-alternatief houdt
die bestaande lezer wel. Zie [INTEGRATIE_V3.md](INTEGRATIE_V3.md) voor beide routes,
de bestanden en voorbereide lokale startopdrachten.

De echte Mushroom-controller en het invoerschema zijn via de lokale testclient
gecontroleerd. Een aparte webserver/browserdemo is hier niet uitgevoerd. De
gezamenlijke website, gedeelde backend, Final, Citi Bike en automatische
publicatie zijn niet omgeschakeld. De volledige groepssuite moet na een latere
gezamenlijke integratie opnieuw worden uitgevoerd.

De eerdere kandidaten en resultaten blijven bewaard:
[v2](RESULTATEN_V2.md), [v1](RESULTATEN.md) en [kritische Final-review](REVIEW_FINAL.md).
