# Mushroom — modelvergelijking volgens de opdracht

**DeepLearningTeam10 · 9 oktober 2026**

Bijdragen: Andrew Noeyens maakte de oorspronkelijke lokale configuraties en exports;
Jorre Van Dyck trainde en tuneerde het AWS-model. Codex voerde op verzoek van Jorre de
gezamenlijke lokale evaluatie, soortcontrole, foutanalyse en rapportage uit. Het team
moet de keuzes reviewen en zelf kunnen verdedigen. AI-gebruik is expliciet vermeld.

Dit rapport hoort bij [09_compare_models.ipynb](09_compare_models.ipynb), het ene
vergelijkingsnotebook voor Mushroom. Het volgt de opdracht: modelmetrics verzamelen,
grondig vergelijken, fouten onderzoeken en een conclusie trekken. Citi Bike heeft
een eigen vergelijking; Milan heeft op dit moment geen beschikbaar getraind
Mushroom-model in deze vergelijking. Zijn voorbereiding wordt niet als modelscore behandeld.

## 1. Conclusie en keuze

**Ontwikkelkeuze voor ongeziene gesimuleerde soorten: Andrew RF500.**
Van de negen getrainde kandidaatconfiguraties heeft dit model de hoogste gemiddelde
balanced accuracy over vijf identieke soorten-folds: **0.6565**.
De gepoolde voorspellingen buiten training geven **65.93% accuracy**,
**68.37% recall p** en **F1 p 0.6897**.
Er zijn **10,674 FN** en **10,082 FP** op 60.923 unieke records.

De eerstvolgende configuratie op het keuzecriterium is Andrew RF200 (API-configuratie) met
0.6550 gemiddelde balanced accuracy.
Het verschil bedraagt slechts
0.148
procentpunt. RF200 haalt gepoolde accuracy
66.00% en recall p
70.26%; Andrew RF500 haalt
respectievelijk 65.93% en 68.37%.
De keuze voor Andrew RF500 volgt dus het vastgelegde criterium;
zij bewijst geen algemene verbetering op ieder belangrijk fouttype.
Deze volgorde is een beschrijvende CV-uitkomst, geen bewezen statistische superioriteit.
De scores en rangschikking hebben betrekking op **lokaal opnieuw gefitte configuraties**
met twintig kenmerken, niet op de oorspronkelijke opgeslagen modellen met een ander schema.
Er is nog geen onafhankelijke, ongebruikte eindtest. De API wordt door deze vergelijking
niet automatisch naar een ander model omgeschakeld.

## 2. Data, doel en eerlijk validatieprotocol

De bron is de [UCI Secondary Mushroom Dataset](https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset):
61.069 gesimuleerde records uit 173 soorten, 353 per soort. Na 146 exacte duplicaten
blijven **60.923 unieke records** over. De positieve klasse `p` omvat giftig én onbekend
of niet aanbevolen; `e` is eetbaar. Er zijn drie numerieke en zeventien categorische
bronfeatures. De doelkolom, soortnaam, bronrij en groeps-ID zijn uitgesloten van invoer.
Bronhash (LF-normalisatie): `a0d68cfc46c6900d67d30a49c6e1c3b8c37042dbd6e62ce38a9cf84a40c022e0`.

De eerdere random 80/20-split bevat alle 173 soorten in zowel train als test. Daarom
maken we de generalisatievraag expliciet: **kan de configuratie voorspellen voor een
gesimuleerde soort die helemaal niet in de training zat?** De oorspronkelijke UCI-
bronvolgorde wordt gecontroleerd tegen 173 primaire soortrecords: constante klasse
per blok, dezelfde klassevolgorde en 2.941 categorische consistentiechecks zonder afwijking.
Pas daarna gebruiken we de oorspronkelijke bronrij gedeeld door 353 als soortgroep.
Dit is een geverifieerde reconstructie voor deze bestandsversie, geen inputfeature.

Alle configuraties gebruiken dezelfde vijf `StratifiedGroupKFold`-folds, shuffle en seed
42. Een soort komt in precies één validatiefold; iedere fold heeft **nul soortoverlap**
tussen training en validatie. Elke unieke rij krijgt één voorspelling buiten training.
Imputatie, encoding, scaling en training worden uitsluitend binnen de trainingsfold
gefit. Er wordt geen nieuwe hyperparametersearch of drempeltuning uitgevoerd.
[Uitleg van grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).

**Selectieregel:** gemiddelde balanced accuracy over de vijf folds; bij een exacte
gelijke score daarna gemiddelde F1 p, recall p en de vaste configuratievolgorde.
De regel is vastgelegd voordat de aanvullende configuraties worden geëvalueerd.
De eerder bekende AWS/RF500-diagnose maakt dit geen volledig vooraf geblindeerd onderzoek.
Balanced accuracy weegt recall van `e` en `p` gelijk en voorkomt dat alleen de grootste
klasse voorspellen goed lijkt. F1, precision, recall, FN en FP tonen de afweging afzonderlijk.
De meerderheidsreferentie is een controle en wordt niet als eindmodel geselecteerd.

## 3. Kandidaten, motivatie en tuning

| Configuratie | Waarom vergelijken? | Belangrijkste instellingen |
| --- | --- | --- |
| Beslisboom baseline | Eenvoudige eerste voorspellingspipeline; referentie voor extra complexiteit | Depth 5, seed 42 |
| AWS RF100 baseline | Controle of AWS-tuning iets toevoegt | 100 bomen, geen depth-limiet, leaf 1 |
| AWS RF100 getuned | Jorre's op SageMaker getrainde en getunede configuratie | 100 bomen, depth 24, leaf 2 |
| Andrew RF200 API-config | Evalueert de configuratie die nu wordt gehost | 200 bomen, balanced, geen depth-limiet |
| Andrew RF500 | Meer bomen en begrensde diepte; ensemble van gebootstrapte bomen | 500 bomen, depth 20, split 3, balanced |
| Gradient Boosting | Bouwt opeenvolgend bomen die eerdere fouten corrigeren | 350 bomen, learning rate 0,08, depth 8 |
| XGBoost | Alternatieve geregulariseerde boosting met rij- en featuresubsampling | 500 bomen, learning rate 0,05, depth 5, subsample/colsample 0,8 |
| Logistic + poly | Vergelijkt bomen met een eenvoudige lineaire classifier plus numerieke interacties | Polynomial graad 2 + scaling; C=1, balanced |
| Logistic v1-config | Het opgeslagen Andrew-LR-model heeft andere preprocessing/instellingen | Missing indicators + scaling; C=1, geen class weights |
| Meerderheidsreferentie | Laat zien wat zonder bruikbare feature-informatie kan worden bereikt | Meest voorkomende trainingsklasse |

AWS/beslisboom gebruiken numerieke mediaan en categorisch `missing` met one-hot encoding.
Andrew-configuraties zetten nulmetingen van steelhoogte/breedte op ontbrekend, gebruiken
mediaan en categorische modus met one-hot encoding; de LR-varianten voegen hun eigen
transformaties toe. Dit vergelijkt **volledige pipelines**: een verschil kan uit de
preprocessing én de classifier komen. De nulmetingenregel is voor reproductie behouden;
een verbetering door deze regel is niet afzonderlijk bewezen.

Voor gelijke invoer worden Andrew's twaalf oorspronkelijke velden uitgebreid naar de
twintig echte UCI-features. De twee noisevelden worden niet meegenomen of verzonnen.
De hyperparameters blijven gelijk. De oorspronkelijke 5.000-rijenresultaten worden
behouden in [de historische vergelijking](09_andrew_vergelijkingsrapport.md).
De RF200- en beslisboomconfiguraties zijn overgenomen uit de bestaande deployment/
baseline. Hun opnieuw gefitte groepsscores horen niet bij de reeds opgeslagen artifacts.

Jorre's vier AWS-tuningkandidaten hadden alle random CV F1=1,0: er is daarmee geen
gemeten tuningwinst onder dat oude protocol. De originele AWS-export, tuninglogs en
SageMaker-notebook blijven in [SolutionJorre/MushroomDataset](../SolutionJorre/MushroomDataset/).
Deze nieuwe soortgebonden vergelijking is lokaal uitgevoerd en vervangt die AWS-run niet.
AutoML en extra systematische tuning zijn afzonderlijke opdrachtvereisten; deze
vergelijking claimt niet dat een ontbrekend AutoML-experiment hiermee is uitgevoerd.

## 4. Resultaten op ongeziene soorten

### Kruisvalidatie voor de keuze

| Configuratie | Gem. balanced accuracy ± SD | Gem. F1 p ± SD | Gem. recall p |
| --- | --- | --- | --- |
| Andrew RF500 | 0.6565 ± 0.0434 | 0.6884 ± 0.0471 | 0.6838 |
| Andrew RF200 (API-configuratie) | 0.6550 ± 0.0468 | 0.6950 ± 0.0448 | 0.7027 |
| Jorre AWS RF100 (baseline) | 0.6328 ± 0.0417 | 0.6657 ± 0.0405 | 0.6584 |
| Jorre AWS RF100 (getuned) | 0.6234 ± 0.0446 | 0.6649 ± 0.0487 | 0.6695 |
| Andrew Gradient Boosting | 0.6192 ± 0.0513 | 0.6586 ± 0.0426 | 0.6572 |
| Andrew Logistic v1-config | 0.6181 ± 0.0750 | 0.6668 ± 0.0536 | 0.6756 |
| Andrew XGBoost | 0.6165 ± 0.0311 | 0.6580 ± 0.0402 | 0.6637 |
| Andrew Logistic + poly | 0.5911 ± 0.0643 | 0.6409 ± 0.0465 | 0.6480 |
| Beslisboom depth 5 (baseline) | 0.5730 ± 0.0382 | 0.5346 ± 0.1188 | 0.4832 |
| Meerderheidsreferentie | 0.5000 ± 0.0000 | 0.7129 ± 0.0061 | 1.0000 |

SD beschrijft spreiding tussen vijf folds; het is geen betrouwbaarheidsinterval van
de rangschikking. Verschillen in samenstelling van de soorten beïnvloeden de uitkomsten.
De keuze gebruikt het gemiddelde van foldmetrics, niet achteraf de hoogste losse fold.

### Gepoolde voorspellingen buiten training

| Configuratie | Accuracy | Balanced acc. | Precision p | Recall p | F1 p | ROC-AUC | AP | FN | FP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Andrew RF500 | 0.6593 | 0.6564 | 0.6959 | 0.6837 | 0.6897 | 0.6637 | 0.6693 | 10674 | 10082 |
| Andrew RF200 (API-configuratie) | 0.6600 | 0.6549 | 0.6894 | 0.7026 | 0.6960 | 0.6600 | 0.6672 | 10034 | 10679 |
| Jorre AWS RF100 (baseline) | 0.6353 | 0.6325 | 0.6751 | 0.6584 | 0.6666 | 0.6517 | 0.6486 | 11526 | 10694 |
| Jorre AWS RF100 (getuned) | 0.6284 | 0.6234 | 0.6629 | 0.6695 | 0.6662 | 0.6400 | 0.6403 | 11153 | 11486 |
| Andrew Gradient Boosting | 0.6230 | 0.6189 | 0.6606 | 0.6569 | 0.6587 | 0.6262 | 0.6280 | 11576 | 11390 |
| Andrew Logistic v1-config | 0.6248 | 0.6187 | 0.6568 | 0.6757 | 0.6661 | 0.6343 | 0.6958 | 10941 | 11915 |
| Andrew XGBoost | 0.6212 | 0.6162 | 0.6564 | 0.6633 | 0.6599 | 0.6398 | 0.6929 | 11360 | 11715 |
| Andrew Logistic + poly | 0.5970 | 0.5908 | 0.6330 | 0.6481 | 0.6404 | 0.6102 | 0.6296 | 11874 | 12681 |
| Beslisboom depth 5 (baseline) | 0.5623 | 0.5719 | 0.6387 | 0.4830 | 0.5500 | 0.5694 | 0.5957 | 17446 | 9218 |
| Meerderheidsreferentie | 0.5538 | 0.5000 | 0.5538 | 1.0000 | 0.7129 | 0.5000 | 0.5538 | 0 | 27181 |

De tabel combineert alle 60.923 voorspellingen buiten training. Een gepoolde metric kan
afwijken van het ongewogen foldgemiddelde doordat foldgroottes en klasseaantallen verschillen.
ROC-AUC en AP beoordelen ranking van de p-scores; ze bewijzen geen kanskalibratie.

![Metrics op ongeziene soorten](comparison_species/metrics.png)

De meerderheidsreferentie haalt F1 p **0.7129** en recall p **1.0000**,
maar balanced accuracy **0.5000**: ze herkent geen eetbare records.
Daarom beoordelen we een configuratie niet alleen op F1 of recall. De gekozen pipeline
heeft balanced accuracy 0.6564, specificity
0.6291 en precision p 0.6959.

### Waarom de eerdere 100% geen eindconclusie was

| Configuratie | Random test accuracy | Soorten-CV accuracy | Soorten-CV F1 p |
| --- | --- | --- | --- |
| Jorre AWS RF100 (getuned) | 1.000000 | 0.628400 | 0.666175 |
| Jorre AWS RF100 (baseline) | 1.000000 | 0.635277 | 0.666627 |
| Andrew RF500 | 0.999836 | 0.659308 | 0.689709 |
| Andrew Gradient Boosting | 0.999836 | 0.623032 | 0.658742 |
| Andrew XGBoost | 0.998112 | 0.621243 | 0.659856 |
| Andrew Logistic + poly | 0.848748 | 0.596950 | 0.640436 |
| Andrew Logistic v1-config | 0.845794 | 0.624838 | 0.666131 |
| Meerderheidsreferentie | 0.553878 | 0.553847 | 0.712872 |

De random tabel gebruikt 12.185 testrecords binnen bekende soorten; groeps-CV gebruikt
60.923 records van telkens ongeziene soorten met opnieuw gefitte modellen. De opzetten
meten verschillende doelen en zijn geen gepaarde toets op één testset. Beide worden
behouden om de verandering in conclusie inzichtelijk te maken.
De dataset-auteurs rapporteerden eveneens perfecte Random Forest-scores bij hun
simulatie-evaluatie. [Wagner et al., Scientific Reports](https://www.nature.com/articles/s41598-021-87602-3).

De ongewijzigde AWS-pipeline is afzonderlijk met scikit-learn 1.7.2 gecontroleerd:
100% op de gereconstrueerde oorspronkelijke testrecords, zonder FN of FP. De audit
vond geen doelkolom in de features. Met geschudde labels wordt balanced accuracy
49,82% en ROC-AUC 0,4964. Dit past bij toeval, maar sluit niet ieder mogelijk datalek uit.
Zie [generalization_audit.json](comparison_team/generalization_audit.json) en
[aws_artifact_audit.json](comparison_team/aws_artifact_audit.json).

## 5. Foutanalyse en grenzen

FN is `p` voorspeld als `e`; FP is `e` voorspeld als `p`. De aantallen staan afzonderlijk
omdat accuracy niet vertelt welke fout wordt gemaakt.

| Configuratie | TN e→e | FP e→p | FN p→e | TP p→p |
| --- | --- | --- | --- | --- |
| Andrew RF500 | 17099 | 10082 | 10674 | 23068 |
| Andrew RF200 (API-configuratie) | 16502 | 10679 | 10034 | 23708 |
| Jorre AWS RF100 (baseline) | 16487 | 10694 | 11526 | 22216 |
| Jorre AWS RF100 (getuned) | 15695 | 11486 | 11153 | 22589 |
| Andrew Gradient Boosting | 15791 | 11390 | 11576 | 22166 |
| Andrew Logistic v1-config | 15266 | 11915 | 10941 | 22801 |
| Andrew XGBoost | 15466 | 11715 | 11360 | 22382 |
| Andrew Logistic + poly | 14500 | 12681 | 11874 | 21868 |
| Beslisboom depth 5 (baseline) | 17963 | 9218 | 17446 | 16296 |
| Meerderheidsreferentie | 0 | 27181 | 0 | 33742 |

![Confusion matrices op dezelfde soorten-folds](comparison_species/confusion_matrices.png)

De gekozen configuratie mist 10,674 van 33,742 p-records
(31.63%), verdeeld over 54 soorten met minstens één FN.
Hier zijn drie FN-records met de laagste p-score, dus voorbeelden van relatief
overtuigde verkeerde voorspellingen. De score is een modelschatting, geen bewezen kans.

| Bronrij (0-based) | Soortgroep | Werkelijk → voorspeld | Score p | Ontbrekende kenmerken |
| --- | --- | --- | --- | --- |
| 23135 | 65 | p → e | 0.0099 | 5 |
| 23220 | 65 | p → e | 0.0109 | 5 |
| 23103 | 65 | p → e | 0.0110 | 5 |

De vijf soorten met de hoogste FN-fractie bij deze configuratie:

| Primaire soort | Groep | FN / records | FN-fractie |
| --- | --- | --- | --- |
| False Panther Cap | 2 | 353 / 353 | 100.00% |
| Ivory Clitocybe | 20 | 353 / 353 | 100.00% |
| Wood Woolly-foot | 21 | 353 / 353 | 100.00% |
| Stinking Russula | 65 | 353 / 353 | 100.00% |
| Birch Russula | 67 | 353 / 353 | 100.00% |

Voor deze soorten bestaan geen trainingsvoorbeelden binnen hun evaluatiefold.
De fouten wijzen op zwakke overdracht naar die soorten; ze bewijzen niet welk kenmerk
de fout veroorzaakt. Omdat de data gesimuleerd is en soortgroepen intern verwant zijn,
worden tienduizenden records niet voorgesteld als evenveel onafhankelijke veldwaarnemingen.
Er wordt geen naïeve rijgewijze significantietoets gebruikt om een winnaar te bewijzen.
Alle fouten met twintig bronfeatures en fold/soortgroep staan in
[selected_error_records.csv](comparison_species/selected_error_records.csv).
Foutaantallen per soort en per configuratie staan in
[species_errors.csv](comparison_species/species_errors.csv).

## 6. Training, eenvoud en praktische afweging

| Configuratie | Gem. fit per fold (s) | Gem. train accuracy | Gem. validatie accuracy |
| --- | --- | --- | --- |
| Andrew RF500 | 8.18 | Niet opnieuw gemeten | 0.6591 |
| Andrew RF200 (API-configuratie) | 11.95 | 1.0000 | 0.6599 |
| Jorre AWS RF100 (baseline) | 5.81 | 1.0000 | 0.6351 |
| Jorre AWS RF100 (getuned) | 4.99 | Niet opnieuw gemeten | 0.6281 |
| Andrew Gradient Boosting | 845.42 | 1.0000 | 0.6231 |
| Andrew Logistic v1-config | 0.83 | 0.8585 | 0.6245 |
| Andrew XGBoost | 2.01 | 0.9996 | 0.6215 |
| Andrew Logistic + poly | 1.32 | 0.8725 | 0.5970 |
| Beslisboom depth 5 (baseline) | 0.37 | 0.7690 | 0.5627 |
| Meerderheidsreferentie | 0.30 | 0.5539 | 0.5539 |

Tijden zijn lokale Windows-metingen, geen AWS-kosten of Render-latencybenchmark.
AWS gebruikt twee threads; RF500/XGBoost gebruiken beschikbare threads; RF200 behoudt
de bestaande instellingen. AWS getuned en RF500 hergebruiken de geverifieerde resultaten
van de eerdere identieke soorten-folds, inclusief toen gemeten fit-tijden. Hun trainmetrics
werden niet opgeslagen en worden niet achteraf ingevuld. Andere foldruns bewaren trainmetrics.
Een train-validatiekloof beschrijft gedrag, maar bewijst niet één oorzaak.
Er zijn 0 vastgelegde waarschuwingen tijdens de nieuwe evaluatie/refit;
de volledige berichten staan in het resultaat-JSON en worden in het notebook getoond.

Voor modelkeuze wegen de groepsvalidatiescores zwaarder dan snelheid. Een eenvoudiger
model kan voldoende zijn als de score vergelijkbaar is; het type algoritme, aantal bomen
of het gebruik van AWS bewijst op zichzelf geen betere generalisatie.

## 7. Eindkeuze, deployment en resterende opdrachtvereisten

**Kies voorlopig Andrew RF500 voor het vastgelegde doel 'ongeziene gesimuleerde soorten'.**
Dit is de hoogst scorende beschikbare getrainde configuratie onder het beschreven
balanced-accuracyprotocol. Rapporteer daarbij de recall, FN en spreiding; de uitkomsten
onderbouwen geen perfecte voorspellingen of prestaties op echte paddenstoelen.

Een volledige hertraining van deze configuratie op de 60.923 unieke records is opgeslagen
als [selected_pipeline.joblib](comparison_species/selected_pipeline.joblib), met twintig
inputs en volledige preprocessing. De outputlabels zijn `0=e` en `1=p`; een API moet
deze mapping expliciet toepassen. Die fit is een overdrachtsartifact: de trainingsscore
ervan geldt niet als extra testresultaat. De versie- en hashgegevens staan in het JSON.

De live Mushroom-API gebruikt nog de eerdere Andrew RF200 met twaalf inputs.
Een omschakeling vraagt een bewuste keuze, passende serving-versies, twintig frontend/API-
velden, de juiste labelmapping en een controle van API-voorspellingen tegen dezelfde pipeline. Deze vergelijking
wijzigt de backend niet. De Citi Bike-deployment van het team blijft een afzonderlijk onderdeel.

Voor de eindinlevering volgens de opdracht:

1. Laat het team deze vergelijking en de preprocessing reviewen en zelf kunnen uitleggen.
2. Voeg eventueel nog ontbrekende AutoML- en tuningruns toe onder hetzelfde groepsprotocol.
   Houd tuning volledig binnen trainingsgroepen; verander geen drempel op de eindtest.
3. Er is geen onaangeraakte eindtest meer voor dit ontwikkeltraject. Leg vóór nieuw
   onderzoek een onafhankelijke evaluatie vast, of motiveer deze groeps-CV als de
   beschikbare ontwikkelvalidatie. Noem de huidige scores geen finale onafhankelijke test.
4. Controleer ook de overige opdrachtbestanden: EDA, definitieve voorbereiding, model-
   notebooks per dataset, AWS-notebook, Citi Bike-vergelijking en werkende deploymentpipeline.

De modelvergelijking is hiermee uitgevoerd; dit rapport verklaart niet het hele project
automatisch af. De opdracht vraagt uiteindelijk een GitHub-link en een mondelinge verdediging.

## 8. Reproduceren en bewaarde bewijzen

De nieuwe fits gebruiken de gepinde omgeving uit `comparison-requirements.txt`:

```powershell
py -3.13 -m venv .venv-analysis
.\.venv-analysis\Scripts\python.exe -m pip install -r secondary_mushroom/comparison-requirements.txt
.\.venv-analysis\Scripts\python.exe secondary_mushroom/compare_species_models.py
.\.venv-analysis\Scripts\python.exe secondary_mushroom/build_species_comparison_report.py --execute
```

De scripts downloaden de oorspronkelijke UCI-data met code en controleren soortgroepen.
De twee eerder geverifieerde configuraties worden alleen bij dezelfde data, folds en
parameters hergebruikt. De overige resultaten worden per configuratie bewaard zodat
onderbroken runs kunnen worden hervat. Het notebook leest standaard de opgeslagen
voorspellingen, berekent de metrics opnieuw en controleert de selectie en soortscheiding.
Voor een nieuwe volledige run kunnen de per-configuratie resultaatcaches worden verwijderd.
Het originele AWS-artifact wordt uitsluitend in zijn afzonderlijke 1.7.2-auditomgeving geladen.

| Bewijs | Bestand |
| --- | --- |
| Eén uitgevoerd vergelijkingsnotebook | [09_compare_models.ipynb](09_compare_models.ipynb) |
| Trainings- en vergelijkingscode | [compare_species_models.py](compare_species_models.py) |
| Rapport/notebookgenerator | [build_species_comparison_report.py](build_species_comparison_report.py) |
| Vastgelegd protocol | [protocol.json](comparison_species/protocol.json) |
| Scores, parameters, omgeving, waarschuwingen en modelkeuze | [comparison_summary.json](comparison_species/comparison_summary.json) |
| Exacte metrictabel | [model_comparison.csv](comparison_species/model_comparison.csv) |
| Alle 50 fold-evaluaties | [cv_folds.csv](comparison_species/cv_folds.csv) |
| Bronrij, soortgroep en fold van iedere unieke rij | [split_manifest.csv](comparison_species/split_manifest.csv) |
| Alle labels en p-scores buiten training | [oof_predictions.csv](comparison_species/oof_predictions.csv) |
| Volledig hertrainde gekozen pipeline | [selected_pipeline.joblib](comparison_species/selected_pipeline.joblib) |

Versies: Python 3.13.7, scikit-learn 1.9.1,
pandas 3.0.2, NumPy 2.5.3 en joblib
1.6.0. Modelhash: `dba934c55ef07b2137006f9b05f02371a5b6f763ebeb5d7e333e898a8be42f61`.
Voor het laden van het artifact moet `secondary_mushroom` op het Python-importpad staan;
de pipeline verwijst waar nodig naar de bewaarde schoonmaakfunctie in `compare_andrew_models`.
Bewaar deze bestanden en gebruik dezelfde dependencies bij reproductie.
