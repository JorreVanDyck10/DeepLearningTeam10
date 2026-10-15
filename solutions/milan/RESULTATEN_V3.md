# Derde studie: wat verbetert echt?

Vijf gelijke buitenfolds op de oorspronkelijke 4.100 trainingsrijen; alle
voorbereiding, configuraties, drempels en votinggewichten worden binnen de
training gekozen. De bekende 900 ontwikkeltestrijen zijn niet gescoord.

De lessen over featureselectie, boosting en ensembles vormen de basis.
Deze herhaalde ontwikkelstudie is geen nieuwe onafhankelijke eindtest.

| Procedure | Macro F1 | Recall giftig | FN | FP | Accuracy |
|---|---:|---:|---:|---:|---:|
| Oude RF300 | 0.7619 | 60.44% | 614 | 256 | 78.78% |
| Kandidaat v2 | 0.7691 | 66.82% | 515 | 356 | 78.76% |
| RF + featureselectie + standaardgrens | 0.7749 | 62.89% | 576 | 251 | 79.83% |
| RF + featureselectie + F1-drempel | 0.7690 | 65.66% | 533 | 332 | 78.90% |
| RF + featureselectie + FPR-budget | 0.7745 | 66.11% | 526 | 319 | 79.39% |
| LightGBM + standaardgrens | 0.7611 | 68.75% | 485 | 430 | 77.68% |
| LightGBM + F1-drempel | 0.7629 | 67.14% | 510 | 390 | 78.05% |
| LightGBM + FPR-budget | 0.7658 | 64.89% | 545 | 332 | 78.61% |
| CatBoost + standaardgrens | 0.7556 | 66.75% | 516 | 414 | 77.32% |
| CatBoost + F1-drempel | 0.7544 | 64.88% | 545 | 380 | 77.44% |
| CatBoost + FPR-budget | 0.7548 | 62.63% | 580 | 332 | 77.76% |
| Gewogen voting + standaardgrens | 0.7789 | 66.76% | 516 | 313 | 79.78% |
| Gewogen voting + F1-drempel | 0.7788 | 67.79% | 500 | 334 | 79.66% |
| Gewogen voting + FPR-budget | 0.7767 | 68.17% | 494 | 351 | 79.39% |

Macro F1, recall en accuracy zijn gemiddelden over vijf folds.
FN en FP zijn de opgetelde aantallen over de 4.100 buitenfoldvoorspellingen.

De hoogste macro F1 komt van **Gewogen voting + standaardgrens**: **0.7789**.

Het vooraf beschreven doel wordt gehaald door **Gewogen voting + F1-drempel**:
macro F1 stijgt tegenover v2 met **0.0097**,
er zijn **15 minder gemiste giftige gevallen**
en **334 valse alarmen** tegenover 356 bij v2.
De omvang en consistentie staan in het vergelijkingsnotebook; een kleine winst blijft klein.

De gekozen procedure verbetert macro F1 in **4 van 5 folds**.
De vijf verschillen staan in het notebook; dit wordt niet als een
significantietoets of veiligheidsbewijs voorgesteld.

## Een eenvoudiger alternatief voor aansluiting

RF met featureselectie en standaardgrens haalt **0.7749 macro F1**
en **79.83% accuracy**. Het pakket houdt de bestaande
modellezer en elf invoervelden. De fulltrain-selectie negeert intern
`spore_print_color, jumbled_noise_1`. De folds kiezen dit niet allemaal hetzelfde.
Tegenover v2 mist deze RF **61 meer giftige gevallen**,
met **105 minder valse alarmen**. Hij haalt het volledige v2-doel dus niet.
Tegenover de oorspronkelijke RF300 verlaagt hij wel beide foutaantallen.
Kies hem voor de eenvoudiger aansluiting en deze specifieke foutafweging;
noem hem niet beter dan voting in giftige recall. Zie [integratie v3](INTEGRATIE_V3.md).

## Wat is geleerd?

- Featureselectie is onderdeel van de pipeline en wordt via de binnenfolds gekozen.
- LightGBM krijgt expliciete categorieën; onbekende categorieën worden ontbrekend.
- CatBoost krijgt categorische ontbrekende waarden als aparte tekstcategorie.
- De regularisatie en complexiteit van beide boosters worden vergeleken.
- Gewogen soft voting combineert kansen; binnen-OOF kiest uit zeven vaste gewichten.
- Het FPR-budget van 14% geldt voor binnen-OOF. De buitenfold toetst of dit overgaat.
- Alle twaalf nieuwe procedures worden gerapporteerd, inclusief tegenvallende uitkomsten.

De 360 gridfits betreffen 24 configuraties × 3 binnenfolds × 5 buitenfolds,
naast de benodigde refits en binnen-OOF-berekeningen. Meer onderzoek maakt
eerder bekeken data niet weer onaangeraakt. Kies de laatste kandidaat op
ontwikkelbewijs en gebruik later echt nieuwe gelabelde data voor bevestiging.

Zie [onderzoeksafspraken](STUDIE_V3.md), [protocol](results_v3/protocol.json)
en de uitgevoerde notebooks 08–10 voor code, controleerbare uitvoer en foutanalyse.
