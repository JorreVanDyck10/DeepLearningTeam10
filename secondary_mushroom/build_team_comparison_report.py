"""Build the extended Andrew + Jorre/AWS report and executed main comparison notebook."""
import argparse
import json
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
import pandas as pd

from build_comparison_report import markdown_table
from compare_team_models import NAMES, ROOT, OUT

FOLDER = ROOT / 'secondary_mushroom'


def report_text(summary, audit, metrics, errors):
    generalization = json.loads((OUT / 'generalization_audit.json').read_text(encoding='utf-8'))
    grouped_scores = {row['model']: row for row in generalization['species_held_out_scores']}
    control_scores = generalization['shuffled_label_negative_control']['metrics']
    species_table = markdown_table(['Configuratie', 'Random test accuracy', 'Soorten apart: accuracy', 'Recall p', 'F1 p', 'FN'],
        [[NAMES[row['model']], f"{metrics.loc[row['model'], 'accuracy']*100:.4f}%",
          f"{row['accuracy']*100:.2f}%", f"{row['recall_poisonous']*100:.2f}%",
          f"{row['f1_poisonous']:.6f}", row['fn']] for row in generalization['species_held_out_scores']])
    selected = metrics.loc[summary['selected_model']]
    trained = metrics.drop(index='majority')
    ranking = metrics.sort_values(['cv_f1_mean', 'cv_recall_mean'], ascending=False, kind='stable')
    cv_table = markdown_table(['Configuratie', 'CV F1 p ± SD', 'CV recall p', 'CV accuracy'],
        [[NAMES[key], f'{row.cv_f1_mean:.6f} ± {row.cv_f1_std:.6f}',
          f'{row.cv_recall_mean:.6f}', f'{row.cv_accuracy_mean:.6f}'] for key, row in ranking.iterrows()])
    test_table = markdown_table(['Configuratie', 'Accuracy', 'Precision p', 'Recall p', 'F1 p', 'ROC-AUC', 'AP', 'FN', 'FP'],
        [[NAMES[key], *[f'{row[column]:.6f}' for column in ['accuracy', 'precision_poisonous',
          'recall_poisonous', 'f1_poisonous', 'roc_auc', 'average_precision']], int(row.fn), int(row.fp)]
         for key, row in metrics.iterrows()])
    resources = markdown_table(['Configuratie', 'Train F1 p', 'Test F1 p', 'Fit (s)', 'Proba 12.185 records (ms)', 'Pipeline (MiB)'],
        [[NAMES[key], f'{row.train_f1_poisonous:.6f}', f'{row.f1_poisonous:.6f}',
          f'{row.fit_seconds:.2f}', f'{row.predict_12185_ms:.2f}', f'{row.serialized_bytes_compress3 / 2**20:.3f}']
         for key, row in trained.iterrows()])
    worst = trained.sort_values('f1_poisonous').index[0]
    concrete = errors[(errors.model == worst) & (errors.true_class == 'p')].sort_values('probability_p').head(3)
    example_table = markdown_table(['Model', 'Unieke rij / bronrij (0-based)', 'Werkelijk → voorspeld', 'Score p', 'Ontbrekende features'],
        [[NAMES[row.model], f'{int(row.unique_row)} / {int(row.source_row)}',
          f'{row.true_class} → {row.predicted_class}', f'{row.probability_p:.6f}', int(row.missing_features)]
         for _, row in concrete.iterrows()])
    confusion = markdown_table(['Configuratie', 'TN e→e', 'FP e→p', 'FN p→e', 'TP p→p'],
        [[NAMES[key], *[int(row[column]) for column in ['tn', 'fp', 'fn', 'tp']]] for key, row in metrics.iterrows()])
    tuning = pd.read_csv(ROOT / 'SolutionJorre/MushroomDataset/aws_tuning_results.csv')
    tuning_table = markdown_table(['Bomen', 'Max depth', 'Min leaf', 'CV F1', 'SD', 'Rank'],
        [[int(row.param_model__n_estimators), 'Geen limiet' if pd.isna(row.param_model__max_depth) else int(row.param_model__max_depth),
          int(row.param_model__min_samples_leaf), f'{row.mean_test_score:.6f}', f'{row.std_test_score:.6f}', int(row.rank_test_score)]
         for _, row in tuning.iterrows()])
    chosen_record = next(row for row in summary['model_comparison'] if row['model'] == summary['selected_model'])
    tie_sentence = ('Er zijn exact gelijke beste CV F1- en recall-scores voor: **' +
        ', '.join(NAMES[key] for key in summary['exact_cv_ties']) + '**. ' +
        'De vooraf vastgelegde voorkeur kiest bij zulke gelijke scores het reeds beschikbare, '
        'geregulariseerde AWS RF100 als die configuratie ertussen staat.') if len(summary['exact_cv_ties']) > 1 else (
        'Deze configuratie heeft de hoogste gemiddelde CV F1 p; er is geen exacte gedeelde eerste plaats op F1 én recall.')
    bounds = lambda values: f'{values[0]*100:.4f}–{values[1]*100:.4f}%'
    return rf"""# Vergelijkingsrapport: Andrew + Jorre/AWS — Mushroom

**DeepLearningTeam10 — 9 oktober 2026.** Uitbreiding van de eerdere Andrew-vergelijking.
Hoofdnotebook: [09_compare_models.ipynb](09_compare_models.ipynb).
Bronmodeldefinities en exports: commit [`7fc26c2`](https://github.com/JorreVanDyck10/DeepLearningTeam10/commit/7fc26c2).

## Besluit

**Geen algemeen beste model vastgesteld.** De eerdere voorkeur voor AWS geldt alleen
voor een willekeurige rij-split binnen dezelfde gesimuleerde soorten. In een aanvullende
controle met volledige soorten buiten de training haalt de AWS-configuratie **{grouped_scores['aws_tuned']['accuracy']*100:.2f}%
accuracy**, tegenover **{grouped_scores['random_forest_500']['accuracy']*100:.2f}%** voor Andrew RF500. De oorspronkelijke 100% bewijst dus
geen perfecte generalisatie. Zie sectie 6a voor het protocol en de bewaarde voorspellingen.

**Voorlopige voorkeur binnen de oorspronkelijke random split: {NAMES[summary['selected_model']]}.**
De gemiddelde CV F1 p is **{selected.cv_f1_mean:.6f}**. Op dezelfde 12.185 testrecords
haalt de nieuwe lokale fit **{selected.accuracy*100:.4f}% accuracy**, met **{int(selected.fn)}
giftige records als eetbaar** en **{int(selected.fp)} eetbare records als giftig**.
{tie_sentence}

Het **ongewijzigde opgeslagen AWS-model** is daarnaast apart gecontroleerd in scikit-learn
1.7.2: de oorspronkelijke 100%-scores en confusion matrix worden gereproduceerd, met
0 FN en 0 FP. De nieuwe gezamenlijke vergelijking bestaat uit **lokale hertrainingen van
modelconfiguraties**. Andrew's oorspronkelijke 5.000-rijenmodellen krijgen daarmee geen
nieuwe scores toegeschreven: hun input is voor dit experiment aangepast naar dezelfde
twintig oorspronkelijke UCI-features. De historisch gemelde 81% en AWS 100% worden dus
niet rechtstreeks als gelijke experimenten gerangschikt.

De gedeelde testset was eerder in het AWS-notebook bekeken. Deze conclusie is een
ontwikkelbesluit, geen bewijs van perfecte generalisatie naar nieuwe soorten of echte
paddenstoelen. De API wordt door dit rapport niet naar een ander model omgeschakeld.

## 1. Opdracht, scope en bijdragen

De opdracht vraagt één vergelijkingsnotebook per dataset met verzamelde modelmetrics,
grondige vergelijking, foutanalyse en conclusies. Dit hoofdnotebook bevat nu zowel Andrew's
configuraties als Jorre's AWS-baseline en getunede configuratie, op één dataversie en split.
De [historische Andrew-deelvergelijking](09_andrew_vergelijkingsrapport.md) blijft beschikbaar
voor de reproductie van zijn originele notebookoutputs.

| Onderdeel | Bijdrage |
| --- | --- |
| Oorspronkelijke Andrew-modeldefinities, notebook en opgeslagen pipelines | Andrew Noeyens |
| Oorspronkelijke AWS-training, tuning en gedownloade exports | Jorre Van Dyck |
| Controle van het AWS-artifact, lokale hertraining, soortgebonden audit, vergelijking en rapportage | Codex op verzoek van Jorre |
| Review en eigen mondelinge verdediging | Nog door het team uit te voeren |

AI-gebruik is hiermee vermeld. Nieuwe lokale trainingsruns zijn niet uitgevoerd op AWS.
De oorspronkelijke AWS-export wordt behouden. AutoML, verdere tuning en Citi Bike zijn
geen nieuwe experimenten in deze uitbreiding; hun ontbrekende bewijs is hiermee niet aangevuld.

## 2. Oorspronkelijke resultaten en data-identiteit

| Oorspronkelijk experiment | Data / features | Testset | Gemelde testaccuracy | Status |
| --- | --- | --- | --- | --- |
| Andrew RF500 | Eigen CSV: 5.000 rijen, 12 features inclusief twee noise-velden | 900, gestratificeerd 18%, seed 42 | 81,00% | Eerder exact op bronprecisie gereproduceerd |
| Andrew Gradient Boosting | Dezelfde Andrew-data | Dezelfde 900 | 80,89% | Eerder gereproduceerd en opgeslagen pipeline gecontroleerd |
| Andrew XGBoost | Dezelfde Andrew-data | Dezelfde 900 | 78,67% | Eerder gereproduceerd met oorspronkelijke `n_jobs=-1` |
| Andrew Logistic + poly | Dezelfde Andrew-data | Dezelfde 900 | 67,11% | Eerder gereproduceerd |
| Andrew Logistic v1-config | Andere opgeslagen LR-configuratie, zelfde Andrew-data | Dezelfde 900 in gecontroleerde hertraining | 70,89% | Apart gehouden van notebook-LR |
| Jorre AWS RF100 getuned | UCI: 60.923 unieke rijen, 20 features | 12.185, gestratificeerd 20%, seed 42 | 100,00% | Opgeslagen model nu opnieuw op deze records gecontroleerd |

De eerdere Andrew RF200 in de API hoort bij een andere 80/20-split met 1.000 testrecords.
Zijn 78,2% is een historisch deploymentresultaat, geen score voor de nieuwe RF500-configuratie.

Voor de gezamenlijke vergelijking downloaden we met code het oorspronkelijke
[UCI-archief](https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset).
Er zijn 61.069 oorspronkelijke rijen en twintig kenmerken. Na het verwijderen van
146 exacte duplicaten blijven 60.923 rijen over. `p` bevat giftige én niet aanbevolen
paddenstoelen met onbekende eetbaarheid; de gegevens zijn gesimuleerd vanuit 173 soorten.

Alle kandidaten krijgen de **zelfde 48.738 trainingsrecords en 12.185 testrecords**.
De testset heeft 5.436 `e` en 6.749 `p`. De invoer bevat de twintig bronfeatures, geen doelkolom
en geen Andrew-noisevelden. De raw-CSV heeft LF-genormaliseerde SHA-256
`{summary['dataset']['sha256_lf']}`. De herkomst- en splitbestanden maken de rij-identiteit controleerbaar.

Er is **{summary['dataset']['exact_feature_overlap_train_test']} exacte feature-overlap** tussen
train en test. Deze check sluit afhankelijkheden binnen de simulatie of nabijgelegen
records niet uit. In de oorspronkelijke vergelijking werden geen soortgroepen gebruikt.
De aanvullende audit reconstrueert en controleert die groepen tegen de primaire UCI-data:
alle **173 soorten** komen in zowel de random trainingsset als de testset voor.

## 3. Gecontroleerd opgeslagen AWS-model en tuning

Het artifact in [SolutionJorre/MushroomDataset](../SolutionJorre/MushroomDataset/) bevat
preprocessing én een Random Forest: 100 bomen, maximale diepte 24, minimaal twee records
per leaf, seed 42 en twee threads. Numerieke waarden krijgen de trainingsmediaan;
categorische ontbrekende waarden worden `missing`, gevolgd door one-hot encoding.
AWS is hier de trainingsomgeving; het modelalgoritme is scikit-learn Random Forest.

De controle reconstrueert de exacte AWS-split vanaf de brondata. De opgeslagen pipeline
verwacht twintig features en maakt 128 encoded features; `class` is uitgesloten.
Er waren geen versie- of inferentiewaarschuwingen. De opnieuw berekende confusion matrix is:

```text
              voorspeld e    voorspeld p
werkelijk e          5436              0
werkelijk p             0           6749
```

Accuracy, precision p, recall p, F1 p, ROC-AUC en AP zijn opnieuw 1,0. De modelhash is
`{audit['artifact_sha256']}`. Het artifact is niet gewijzigd of opnieuw getraind.
Een afzonderlijk bewaard trainingsmanifest van de oorspronkelijke AWS-run ontbreekt;
trainingslidmaatschap van dat artifact is dus niet onafhankelijk bewezen. De gecontroleerde
nieuwe fit heeft wel een vastgelegd train/test-manifest zonder train-testoverlap.

Het AWS-notebook testte vier RandomizedSearchCV-kandidaten met drie trainingsfolds:

{tuning_table}

Alle kandidaten én de AWS-baseline hadden CV F1=1,0. De tuning levert hier dus **geen
gemeten verbetering**. De bronsearch kiest de eerste kandidaat uit de gedeelde beste
rank; uit deze tabel volgt niet dat depth 24/leaf 2 uniek optimaal is.

De auditomgeving gebruikt dezelfde scikit-learn 1.7.2 en pandas 2.3.3 als de export.
Python/NumPy verschillen van de SageMaker-omgeving (hier 3.13.7/2.2.6, oorspronkelijk
3.10.20/1.26.4); de vastgelegde inferentie-uitkomsten komen overeen. Voor de gezamenlijke
nieuwe fits gebruiken alle modellen dezelfde scikit-learn 1.9.1-omgeving.

## 4. Eerlijke gezamenlijke vergelijking

De featurekolommen van Andrew's configuraties worden uitgebreid naar de twintig UCI-features;
de oorspronkelijke noise-kolommen worden niet verzonnen of ingevuld. Zijn hyperparameters
blijven gelijk. De LR-v1 wordt gekloond zodat opgeslagen aangeleerde toestand verdwijnt,
met behoud van missing indicators, scaling en classifierinstellingen; de kolomselectie
wordt aangepast. Het getrainde v1-artifact zelf wordt niet voor deze UCI-scores gebruikt.

| Configuratie | Preprocessing en belangrijkste instellingen |
| --- | --- |
| AWS RF100 getuned | Numerieke mediaan; categorisch `missing` + OHE; 100 bomen, depth 24, leaf 2 |
| AWS RF100 baseline | Dezelfde AWS-preprocessing; 100 bomen, geen depth-limiet, leaf 1 |
| Andrew RF500 | Nullen in stemmetingen → ontbrekend; numerieke mediaan; categorische modus + OHE; 500 bomen, depth 20, split 3, leaf 1, balanced |
| Andrew Gradient Boosting | Dezelfde Andrew-preprocessing; 350 bomen, learning rate 0,08, depth 8, split 4, leaf 2 |
| Andrew XGBoost | Dezelfde Andrew-preprocessing; 500 bomen, learning rate 0,05, depth 5, min child weight 2, subsample/colsample 0,8 |
| Andrew Logistic + poly | Andrew-schoonmaak; mediaan + numerieke polynomial features graad 2 + scaling; C=1, balanced, max iter 3.000 |
| Andrew Logistic v1-config | Andrew-schoonmaak; mediaan + missing indicators + scaling; categorische modus + OHE; C=1, geen class weights, max iter 2.000 |
| Meerderheidsreferentie | Voorspelt steeds de meerderheidsklasse `p` in deze UCI-versie |

We vergelijken **volledige pipelineconfiguraties**, dus verschillen kunnen uit preprocessing
én classifierinstellingen komen. Dit is geen geïsoleerde causaliteitsproef van één algoritme.
Andrew's nulmetingen-schoonmaak is overgenomen voor reproductie; haar inhoudelijke noodzaak
op de volledige UCI-data is hiermee niet bewezen en vraagt nog een afzonderlijke EDA/ablation.

Op de gedeelde trainingsset gebruiken we drie identieke gestratificeerde folds met shuffle
en seed 42, zoals in de AWS-opzet. Imputatie, encoding, scaling en modeltraining worden
uitsluitend op de trainingsrecords van elke fold gefit. Er is geen nieuwe hyperparametersearch.
De selectie gebruikt gemiddelde CV F1 p, daarna CV recall p. Bij exacte gelijke beste scores
staat de beschikbare, geregulariseerde AWS RF100 vooraf eerst in de voorkeurvolgorde.
De keuze wordt vastgelegd **vóór** het berekenen van de gemeenschappelijke testmetrics.

F1 balanceert precision en recall; FN en recall p blijven afzonderlijk zichtbaar omdat
giftig als eetbaar een ander fouttype is dan eetbaar als giftig. F1 is geen veiligheidsnorm.
De standaard `predict`-drempel rond 0,5 blijft behouden. Er is geen tuning of drempelkeuze
op test gedaan. Zie het [validatieprotocol](https://scikit-learn.org/stable/modules/cross_validation.html).

## 5. Gezamenlijke resultaten

### Kruisvalidatie: basis voor de modelkeuze

{cv_table}

SD is de spreiding over drie folds, geen betrouwbaarheidsinterval voor de rangschikking.
Afgeronde 100%-scores kunnen kleine verschillen verbergen; daarom tonen we zes decimalen
en ook de absolute foutaantallen. De nieuwe CV-scores horen bij deze volledige UCI-versie,
niet bij Andrew's eerdere vijf folds op 4.100 records.

### Test-audit op dezelfde 12.185 records

{test_table}

![Gemeenschappelijke metrics](comparison_team/shared_metrics.png)

ROC-AUC en average precision beoordelen de rangschikking van p-scores over drempels.
Ze tonen niet of deze scores gekalibreerde kansen zijn. Er is geen kalibratieonderzoek gedaan.
De referentie voorspelt altijd `p`: hierdoor kan recall 1,0 zijn terwijl specificity 0 is.
Een hoge recall of accuracy op zichzelf is dus onvoldoende.

![Gemeenschappelijke precision-recall-curves](comparison_team/shared_pr_curves.png)

De nieuwe AWS-fit en het opgeslagen AWS-artifact verschillen in
**{summary['aws_local_refit_vs_original_artifact']['label_disagreements']} testlabels**.
Het maximale verschil in p-score is
**{summary['aws_local_refit_vs_original_artifact']['maximum_probability_difference']:.8g}**.
Dit controleert consistentie op deze records, niet prestaties op nieuwe data.

## 6. Foutanalyse en onzekerheid

{confusion}

![Gemeenschappelijke confusion matrices](comparison_team/shared_confusion_matrices.png)

De gekozen configuratie maakt {int(selected.fn + selected.fp)} testfouten. Het beschrijvende
Wilson-interval van 95% voor accuracy is **{bounds(chosen_record['accuracy_ci95'])}**;
voor recall p **{bounds(chosen_record['recall_ci95'])}**. Ook nul geobserveerde fouten
bewijst geen nul foutkans in de populatie. De intervallen veronderstellen onafhankelijke
records en corrigeren niet voor eerdere experimenten, modelselectie of simulatiestructuur.

Voor concrete fouten bekijken we de zwakste getrainde kandidaat op test-F1:
**{NAMES[worst]}**. Dit is een achteraf gekozen foutanalyse, geen selectie- of tuningcriterium.
Onder de onterecht eetbaar gelabelde giftige records staan de drie laagste p-scores:

{example_table}

Alle foutrecords met bronfeatures zijn bewaard in
[shared_error_records.csv](comparison_team/shared_error_records.csv).
Eén tabelrij kan per foutmakend model terugkomen. Gemiddelde scores vertellen niet welke
combinaties problematisch zijn; daarom behoudt het notebook deze voorbeelden en het aantal
ontbrekende kenmerken. De voorbeelden alleen bewijzen geen oorzakelijk effect van imputatie.

Het accuracyverschil tussen AWS100 en Andrew RF500/Gradient Boosting is slechts **twee
records op 12.185**, ongeveer **0,0164 procentpunt**. Een exacte gepaarde McNemar-toets
geeft in beide vergelijkingen p=0,5: dit kleine testverschil bewijst niet dat AWS100
statistisch beter presteert. Het bewijst ook geen equivalentie.
De paarvergelijkingen in het JSON-bestand zijn exploratief, zonder meervoudige-toetscorrectie;
de CV-regel en praktische tie-breaker blijven het selectiecriterium.

We schrijven de betere uitkomsten op UCI niet toe aan alleen "meer training" of "AWS".
Andrew's oorspronkelijke input had minder kenmerken, twee noisevelden en een andere
klasseverdeling/ontbrekendheid. De nieuwe vergelijking houdt de rijen en bronfeatures gelijk;
een specifiek effect van datavolume, featurekeuze of noise vraagt afzonderlijke ablation.

## 6a. Waarom 100%? Controle op volledig ongeziene soorten

100% is voor deze dataset niet op zichzelf bewijs van een programmeerfout. De auteurs
rapporteren voor Random Forest eveneens vijfvoudige CV accuracy en F2 van 1,0. Hun
gegevens zijn gesimuleerd met 353 voorbeelden per soort.
[Wagner et al., Scientific Reports](https://www.nature.com/articles/s41598-021-87602-3).

**Verificatie van de groepen.** Het originele archief bevat 173 primaire soorten en
61.069 secundaire records in 173 opeenvolgende blokken van 353. De audit controleert
de klassevolgorde en alle 2.941 combinaties van soort en categorisch kenmerk tegen
de primaire data, zonder afwijkingen. Pas daarna wordt `source_row // 353` als
groepsnummer gebruikt; de oorspronkelijke bronrij blijft na deduplicatie behouden.
De groeps-ID, soortnaam, rij-index en doelkolom worden nooit aan het model gevoerd.
Deze reconstructie is specifiek voor de gecontroleerde UCI-bestandsversie.

**Protocol.** Twee vaste volledige pipelineconfiguraties worden lokaal opnieuw gefit
met vijfvoudige `StratifiedGroupKFold`, shuffle en seed 42. Alle records van één soort
zitten in dezelfde validatiefold; elke fold heeft nul soortoverlap met training.
Imputatie en encoding worden binnen elke trainingsfold gefit. De opgeslagen AWS-pipeline
zelf is hiervoor niet gebruikt: die heeft al voorbeelden van alle soorten gezien.
De metrics hieronder worden gepoold over 60.923 voorspellingen buiten de training.
[scikit-learn: grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).

{species_table}

![Random split tegenover ongeziene soorten](comparison_team/generalization_comparison.png)

De oorspronkelijke random test gebruikt 12.185 records; deze diagnostische groeps-CV
gebruikt 60.923 records met per fold opnieuw gefitte modellen. Het zijn verschillende
generalisatievragen en geen gepaarde vergelijking op één testset. Andrew RF500 scoort
in deze controle hoger dan AWS100, maar de overige kandidaten zijn niet met dit
groepsprotocol geëvalueerd. Hiermee is geen definitieve teamrangschikking vastgesteld.

**Negatieve controle.** Bij willekeurig geschudde labels haalt dezelfde AWS-configuratie
{control_scores['balanced_accuracy']*100:.2f}% balanced accuracy en ROC-AUC {control_scores['roc_auc']:.4f} op de oorspronkelijke random test.
Dat gedrag past bij toeval. Samen met het gecontroleerde featureschema levert dit
geen aanwijzing voor een rechtstreeks meegevoerde doelkolom; het sluit niet ieder
mogelijk datalek uit. De duidelijke terugval bij ongeziene soorten laat vooral zien
dat de random split de prestaties voor nieuwe soorten sterk overschat.

**Conclusie voor modelkeuze.** Voor interpolatie binnen deze 173 gesimuleerde soorten
blijft de random vergelijking bruikbaar. Voor ongeziene soorten moet het team alle
kandidaten en tuning met gescheiden soortgroepen vergelijken, en een ongebruikte
eindtest vastleggen. Deze audit is achteraf toegevoegd nadat de random scores bekend
waren en vormt geen nieuwe onaangeraakte eindtest. Ook soorten-CV test geen echte
veldmetingen. De eerdere brede conclusie dat AWS het beste model is, wordt ingetrokken.

## 7. Training, complexiteit en praktische keuze

{resources}

Train-testverschillen kunnen op overfitting wijzen, maar zijn geen bewijs van één oorzaak.
Een zwakkere LR-score op train én test wijst op beperkingen van deze representatie of
instellingen; het bewijst niet dat alle lineaire modellen slecht zijn.

Tijden zijn lokale metingen op Windows 11 met twintig logische CPU's. Fit is één training
op 48.738 records; proba-tijd is de mediaan van vijf batches van 12.185 records; pipelinegrootte
is joblib-compressie 3 inclusief preprocessing. AWS-configuraties gebruiken twee threads,
Andrew RF/XGBoost `n_jobs=-1`. Dit is geen gecontroleerde vergelijking van algoritmische
rekenefficiëntie en geen Render-latencybenchmark. Modelgrootte en beschikbaarheid zijn wel
praktische afwegingen wanneer validatiescores gelijk zijn.

{tie_sentence} Dit geldt uitsluitend binnen de oorspronkelijke random split.
De aanvullende soortcontrole ondersteunt geen algemene voorkeur voor AWS.
De keuze binnen de random split wordt niet voorgesteld als
een statistisch bewezen uniek beste model. Het AWS-notebookmodel is beschikbaar als complete
pipeline; een gedeelde prestatie zou op zichzelf geen reden zijn om een veel groter model
naar de backend te verhuizen.

## 8. Beperkingen en acties voor de eindinlevering

De opdracht vraagt ook AutoML, systematische tuning, uitleg per model, EDA en een werkende
deploymentpipeline. De gezamenlijke vergelijking vult alleen het modelvergelijkingsdeel aan.
Voor de definitieve inlevering zijn onder meer nog nodig:

1. Leg het uiteindelijke probleem en generalisatiedoel vast. Een random split binnen
   gesimuleerde soorten is geen test op nieuwe soorten of echte paddenstoelen. Gebruik
   bij het doel 'nieuwe soorten' het gecontroleerde groepsprotocol voor alle kandidaten.
2. Voeg ontbrekende AutoML/team-experimenten toe aan hetzelfde datacontract en protocol.
   Behoud tuninglogs; rapporteer ook experimenten zonder verbetering.
3. Gebruik een eindtest die niet al voor ontwikkeling is bekeken, of motiveer een passend
   alternatief. Het huidige AWS-testresultaat is eerder bekend geweest.
4. Laat het team de preprocessing, metrickeuze, tie-breaker en concrete fouten reviewen
   en zelf kunnen uitleggen. Gebruik modelbestanden met de passende versies.
5. Als het team voor het AWS-model kiest: pas API en frontend bewust van Andrew's twaalf
   velden naar de twintig AWS-features aan, en valideer voorspellingen tegen het notebook.
   Het gepubliceerde model is op dit moment nog de oudere RF200.

Het laden van een scikit-learn-pipeline uit een andere versie is geen betrouwbare migratie;
zie [model persistence](https://scikit-learn.org/stable/model_persistence.html).
Deze uitbreiding heeft de originele bronnotebooks en modelbestanden niet aangepast.

## 9. Reproduceren en bewijs

Voer vanuit de repositoryroot twee afzonderlijke omgevingen uit:

```powershell
py -3.13 -m venv .venv-aws-audit
.\.venv-aws-audit\Scripts\python.exe -m pip install -r secondary_mushroom/requirements-aws-audit.txt
.\.venv-aws-audit\Scripts\python.exe secondary_mushroom/audit_aws_model.py

py -3.13 -m venv .venv-analysis
.\.venv-analysis\Scripts\python.exe -m pip install -r secondary_mushroom/comparison-requirements.txt
.\.venv-analysis\Scripts\python.exe secondary_mushroom/compare_team_models.py
.\.venv-analysis\Scripts\python.exe secondary_mushroom/audit_species_generalization.py
.\.venv-analysis\Scripts\python.exe secondary_mushroom/build_team_comparison_report.py --execute
```

De twee omgevingen scheiden het laden van het originele artifact (1.7.2) van nieuwe fits
(1.9.1). De notebooks laden standaard de bewaarde resultaten; een volledige hertraining
gebeurt met het script of `RUN_TRAINING=True`. Op andere hardware kunnen timing en sommige
numerieke uitkomsten veranderen. Bewaar bij nieuwe runs de hashes, parameters en versies.

| Bestand | Inhoud |
| --- | --- |
| [09_compare_models.ipynb](09_compare_models.ipynb) | Uitgevoerd hoofdnotebook: Andrew + Jorre/AWS |
| [09_compare_andrew_models.ipynb](09_compare_andrew_models.ipynb) | Historische reproductie van Andrew's eigen 5.000-rijenexperiment |
| [audit_aws_model.py](audit_aws_model.py) | Controle van het ongewijzigde opgeslagen AWS-model |
| [compare_team_models.py](compare_team_models.py) | Gezamenlijke lokale training en CV-selectie |
| [audit_species_generalization.py](audit_species_generalization.py) | Groepsverificatie, vijfvoudige soorten-CV en geschudde-labelcontrole |
| [generalization_audit.json](comparison_team/generalization_audit.json) | Soortoverlap, groepsprotocol en diagnostische resultaten |
| [species_lookup.csv](comparison_team/species_lookup.csv) | Gecontroleerde koppeling van groeps-ID naar primaire soort |
| [species_cv_folds.csv](comparison_team/species_cv_folds.csv) | Alle tien model/fold-evaluaties |
| [species_model_comparison.csv](comparison_team/species_model_comparison.csv) | Gepoolde metrics bij ongeziene soorten |
| [species_oof_predictions.csv](comparison_team/species_oof_predictions.csv) | Voorspelling, bronrij, soortgroep en validatiefold van alle unieke records |
| [shuffled_label_predictions.csv](comparison_team/shuffled_label_predictions.csv) | Individuele voorspellingen van de negatieve controle |
| [comparison_data.py](comparison_data.py) | Gecodeerde UCI-download, deduplicatie en gedeelde split |
| [build_team_comparison_report.py](build_team_comparison_report.py) | Rapport en uitgevoerd notebook opbouwen |
| [aws_artifact_audit.json](comparison_team/aws_artifact_audit.json) | Modelhash, schema, omgeving en opnieuw berekende AWS-scores |
| [aws_artifact_predictions.csv](comparison_team/aws_artifact_predictions.csv) | Alle voorspellingen van het oorspronkelijke AWS-artifact |
| [shared_comparison_summary.json](comparison_team/shared_comparison_summary.json) | Gedeelde parameters, versies, scores en keuze |
| [shared_cv_folds.csv](comparison_team/shared_cv_folds.csv) | Alle 24 model/fold-evaluaties |
| [shared_split_manifest.csv](comparison_team/shared_split_manifest.csv) | Unieke rij-index, originele bronrij, train/test en fold |
| [shared_model_comparison.csv](comparison_team/shared_model_comparison.csv) | Exacte vergelijkingstabel |
| [shared_test_predictions.csv](comparison_team/shared_test_predictions.csv) | Labels en p-scores per model op dezelfde testrecords |
| [shared_error_records.csv](comparison_team/shared_error_records.csv) | Concrete foutrecords per model |
"""


def notebook_content(report):
    cells = []
    def md(source): cells.append(nbformat.v4.new_markdown_cell(source))
    def code(source): cells.append(nbformat.v4.new_code_cell(source))
    md('''# 09 — Mushroom-modelvergelijking: Andrew + Jorre/AWS

**Bijdragen:** Andrew Noeyens maakte de oorspronkelijke lokale modeldefinities en exports;
Jorre Van Dyck trainde en exporteerde het AWS-model; Codex controleerde het artifact,
trainde de configuraties lokaal op gedeelde data, controleerde generalisatie naar ongeziene
soorten en maakte deze vergelijking op verzoek van Jorre.
Teamreview en mondelinge verdediging staan nog open. AI-gebruik is expliciet vermeld.

Dit is het hoofdnotebook volgens de opdracht: metrics verzamelen, eerlijk vergelijken,
fouten onderzoeken en een keuze motiveren. De volledige uitleg staat in
[09_vergelijkingsrapport.md](09_vergelijkingsrapport.md).
Nieuwe hertrainingen zijn lokaal uitgevoerd, niet opnieuw op AWS.
''')
    md('''## 1. Omgeving en resultaten laden

Het originele AWS-model wordt alleen met scikit-learn 1.7.2 geladen via `audit_aws_model.py`.
De nieuwe fits gebruiken 1.9.1 voor alle kandidaten. Gebruik de twee omgevingen uit het rapport.
Hier lezen we standaard de opgeslagen audit en evaluaties. Zet `RUN_TRAINING=True` voor een
nieuwe volledige gemeenschappelijke training; het script legt de CV-keuze vast vóór testmetrics.
Voer daarna de rapportgenerator opnieuw uit om tekst en cijfers gelijk te houden.
''')
    code('''from pathlib import Path
import json
import sys
import pandas as pd
from IPython.display import display, Image

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / 'SolutionJorre/MushroomDataset/aws_results.json').exists())
FOLDER = ROOT / 'secondary_mushroom'
OUT = FOLDER / 'comparison_team'
sys.path.insert(0, str(FOLDER))
from compare_team_models import run, NAMES

RUN_TRAINING = False
if RUN_TRAINING:
    run()
summary = json.loads((OUT / 'shared_comparison_summary.json').read_text(encoding='utf-8'))
audit = json.loads((OUT / 'aws_artifact_audit.json').read_text(encoding='utf-8'))
metrics = pd.read_csv(OUT / 'shared_model_comparison.csv').set_index('model')
folds = pd.read_csv(OUT / 'shared_cv_folds.csv')
split = pd.read_csv(OUT / 'shared_split_manifest.csv')
predictions = pd.read_csv(OUT / 'shared_test_predictions.csv')
artifact_predictions = pd.read_csv(OUT / 'aws_artifact_predictions.csv')
errors = pd.read_csv(OUT / 'shared_error_records.csv')
display(pd.DataFrame([audit['environment'], summary['environment']], index=['AWS artifact-audit', 'gemeenschappelijke nieuwe fits']))
print('Voorlopige voorkeur binnen de random split:', NAMES[summary['selected_model']])''')
    md('## 2. Oorspronkelijke resultaten en data-identiteit\n' + report.split('## 2. Oorspronkelijke resultaten en data-identiteit', 1)[1].split('## 3. Gecontroleerd opgeslagen AWS-model', 1)[0])
    code('''assert len(split) == 60923 and split.unique_row.is_unique and split.source_row.is_unique
assert (split.split == 'train').sum() == 48738
assert (split.split == 'test').sum() == 12185
assert split.loc[split.split == 'train', 'cv_validation_fold'].notna().all()
assert split.loc[split.split == 'test', 'cv_validation_fold'].isna().all()
assert set(predictions.unique_row) == set(split.loc[split.split == 'test', 'unique_row'])
assert predictions.unique_row.tolist() == artifact_predictions.unique_row.tolist()
assert summary['dataset']['sha256_lf'] == audit['csv_sha256_lf']
display(pd.crosstab(split.split, split['class']))
display(split.loc[split.split == 'train', 'cv_validation_fold'].value_counts().sort_index())
display(pd.Series(summary['dataset']['missing_percent'], name='ontbrekend (%)'))''')
    md('## 3. Audit van het opgeslagen AWS-model\n' + report.split('## 3. Gecontroleerd opgeslagen AWS-model en tuning', 1)[1].split('## 4. Eerlijke gezamenlijke vergelijking', 1)[0])
    code('''import hashlib
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
artifact_path = ROOT / 'SolutionJorre/MushroomDataset/mushroom_pipeline.joblib'
assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == audit['artifact_sha256']
y_original = (artifact_predictions.true_class == 'p').astype(int)
labels_original = (artifact_predictions.prediction == 'p').astype(int)
original_cm = confusion_matrix(y_original, labels_original, labels=[0, 1])
assert original_cm.tolist() == [[5436, 0], [0, 6749]]
assert not audit['warnings'] and audit['matches_exported_metrics']
display(pd.Series(audit['recomputed_metrics'], name='opnieuw berekend op origineel artifact'))
display(pd.read_csv(ROOT / 'SolutionJorre/MushroomDataset/aws_tuning_results.csv')
    [['param_model__n_estimators', 'param_model__max_depth', 'param_model__min_samples_leaf', 'mean_test_score', 'std_test_score', 'rank_test_score']])''')
    md('## 4. Gezamenlijk protocol en modelkeuze\n' + report.split('## 4. Eerlijke gezamenlijke vergelijking', 1)[1].split('## 5. Gezamenlijke resultaten', 1)[0])
    code('''ranking = metrics.sort_values(['cv_f1_mean', 'cv_recall_mean'], ascending=False, kind='stable')
assert ranking.index[0] == summary['selected_model']
assert len(folds) == 24 and (folds.groupby('model').size() == 3).all()
display(ranking[['cv_f1_mean', 'cv_f1_std', 'cv_recall_mean', 'cv_accuracy_mean']].round(6))
display(folds.pivot(index='fold', columns='model', values='f1_poisonous').round(6))
print('Exact gelijke beste CV-scores:', summary['exact_cv_ties'])''')
    md('''## 5. Gedeelde test-audit en controle tegen individuele voorspellingen

Alle kandidaten zijn opnieuw gefit op dezelfde trainingsrecords. De oorspronkelijke Andrew
modellen hadden een ander schema; dit zijn scores voor aangepaste configuraties, geen nieuwe
claims over zijn opgeslagen 12-featuremodellen. De positieve klasse is p. We herberekenen
de metrics uit de bewaarde voorspellingen om te controleren dat tabel en notebook overeenkomen.
ROC-AUC/AP meten ranking; de scores zijn niet op kalibratie getest.
''')
    code('''y = (predictions.true_class == 'p').astype(int)
functions = {'accuracy': accuracy_score, 'balanced_accuracy': balanced_accuracy_score,
             'precision_poisonous': precision_score, 'recall_poisonous': recall_score, 'f1_poisonous': f1_score}
for key, row in metrics.iterrows():
    labels = (predictions[key + '_prediction'] == 'p').astype(int)
    proba = predictions[key + '_probability_p']
    for name, function in functions.items():
        actual = function(y, labels) if name in ['accuracy', 'balanced_accuracy'] else function(y, labels, zero_division=0)
        assert abs(actual - row[name]) < 1e-12
    assert abs(roc_auc_score(y, proba) - row.roc_auc) < 1e-12
    assert abs(average_precision_score(y, proba) - row.average_precision) < 1e-12
    cm = confusion_matrix(y, labels, labels=[0, 1])
    assert cm.sum() == 12185
    assert cm.ravel().tolist() == [int(row[c]) for c in ['tn', 'fp', 'fn', 'tp']]
display(metrics[['accuracy', 'balanced_accuracy', 'precision_poisonous', 'recall_poisonous', 'f1_poisonous', 'roc_auc', 'average_precision', 'fn', 'fp']].round(6))
display(Image(filename=str(OUT / 'shared_metrics.png')))
display(Image(filename=str(OUT / 'shared_pr_curves.png')))
display(pd.Series(summary['aws_local_refit_vs_original_artifact']))''')
    md('## 6. Foutanalyse en onzekerheid\n' + report.split('## 6. Foutanalyse en onzekerheid', 1)[1].split('## 6a. Waarom 100%?', 1)[0])
    code('''display(Image(filename=str(OUT / 'shared_confusion_matrices.png')))
display(metrics[['tn', 'fp', 'fn', 'tp']].astype(int))
for key, row in metrics.iterrows():
    assert len(errors[errors.model == key]) == row.fn + row.fp
worst = metrics.drop(index='majority').sort_values('f1_poisonous').index[0]
display(errors[(errors.model == worst) & (errors.true_class == 'p')].sort_values('probability_p').head(3))
display(errors[errors.model != 'majority'].groupby(['model', 'true_class', 'predicted_class']).size().rename('aantal').to_frame())
chosen = next(row for row in summary['model_comparison'] if row['model'] == summary['selected_model'])
print('Wilson 95% accuracy:', chosen['accuracy_ci95'])
print('Wilson 95% recall p:', chosen['recall_ci95'])''')
    md('''De gepaarde accuracyvergelijking kijkt naar records waarop slechts één van de modellen
correct is. Bij AWS100 tegenover RF500/Gradient Boosting gaat het om twee zulke records
en p=0,5. De verschillen tussen de beste boommodellen zijn hier te klein voor een stellige
statistische conclusie. De overige toetsen zijn exploratief en niet gecorrigeerd voor
meerdere vergelijkingen; ze bepalen de modelkeuze niet.''')
    code("display(pd.DataFrame(summary['pairwise_accuracy_audit']))")
    md('## 6a. Waarom 100%? Controle op volledig ongeziene soorten\n' + report.split('## 6a. Waarom 100%? Controle op volledig ongeziene soorten', 1)[1].split('## 7. Training, complexiteit en praktische keuze', 1)[0])
    code('''generalization = json.loads((OUT / 'generalization_audit.json').read_text(encoding='utf-8'))
oof = pd.read_csv(OUT / 'species_oof_predictions.csv', float_precision='round_trip')
species_metrics = pd.read_csv(OUT / 'species_model_comparison.csv').set_index('model')
assert len(oof) == 60923 and oof.unique_row.is_unique
assert oof.groupby('species_group').validation_fold.nunique().eq(1).all()
assert oof.species_group.nunique() == 173
assert set(oof.validation_fold) == {1, 2, 3, 4, 5}
assert (oof.species_group == oof.source_row // 353).all()
assert oof[['unique_row', 'source_row']].equals(split[['unique_row', 'source_row']])
assert generalization['dataset']['sha256_lf'] == summary['dataset']['sha256_lf']
train_groups = set(oof.loc[split.split == 'train', 'species_group'])
test_groups = set(oof.loc[split.split == 'test', 'species_group'])
assert len(train_groups & test_groups) == 173
for fold in range(1, 6):
    assert not set(oof.loc[oof.validation_fold == fold, 'species_group']) & set(oof.loc[oof.validation_fold != fold, 'species_group'])
y_group = (oof.true_class == 'p').astype(int)
for key, row in species_metrics.iterrows():
    labels = (oof[key + '_prediction'] == 'p').astype(int)
    proba = oof[key + '_probability_p']
    for name, function in functions.items():
        actual = function(y_group, labels) if name in ['accuracy', 'balanced_accuracy'] else function(y_group, labels, zero_division=0)
        assert abs(actual - row[name]) < 1e-12
    assert abs(roc_auc_score(y_group, proba) - row.roc_auc) < 1e-12
    assert abs(average_precision_score(y_group, proba) - row.average_precision) < 1e-12
    assert confusion_matrix(y_group, labels, labels=[0, 1]).ravel().tolist() == [int(row[c]) for c in ['tn', 'fp', 'fn', 'tp']]
control = pd.read_csv(OUT / 'shuffled_label_predictions.csv', float_precision='round_trip')
import numpy as np
shuffled_targets = np.random.default_rng(42).permutation(y_group.to_numpy())
assert set(control.unique_row) == set(split.loc[split.split == 'test', 'unique_row'])
assert np.array_equal(control.shuffled_target, shuffled_targets[control.unique_row])
control_metrics = generalization['shuffled_label_negative_control']['metrics']
for name, function in functions.items():
    actual = function(control.shuffled_target, control.prediction) if name in ['accuracy', 'balanced_accuracy'] else function(control.shuffled_target, control.prediction, zero_division=0)
    assert abs(actual - control_metrics[name]) < 1e-12
assert abs(roc_auc_score(control.shuffled_target, control.probability_p) - control_metrics['roc_auc']) < 1e-12
display(species_metrics[['accuracy', 'recall_poisonous', 'f1_poisonous', 'fn', 'fp']].round(6))
display(pd.DataFrame(generalization['fold_membership']))
display(pd.Series(control_metrics, name='geschudde labels'))
print('Alle 173 soorten zaten in zowel random train als test; groeps-CV houdt soorten gescheiden.')
print('Twee vaste configuraties gecontroleerd; geen algemene winnaar geselecteerd.')''')
    md('## 7. Trainingsgedrag, kosten en keuze\n' + report.split('## 7. Training, complexiteit en praktische keuze', 1)[1].split('## 8. Beperkingen en acties voor de eindinlevering', 1)[0])
    code('''practical = metrics.drop(index='majority')[['train_accuracy', 'accuracy', 'train_f1_poisonous', 'f1_poisonous', 'fit_seconds', 'predict_12185_ms']].copy()
practical['pipeline MiB (compressie 3)'] = metrics.serialized_bytes_compress3 / 2**20
display(practical.round(6))
display(pd.Series({key: sum(counts.values()) for key, counts in summary['warnings'].items()}, name='waarschuwingen per run'))''')
    md('## 8. Conclusie en resterend werk\n' + report.split('## Besluit', 1)[1].split('## 1. Opdracht', 1)[0] + '\n' + report.split('## 8. Beperkingen en acties voor de eindinlevering', 1)[1].split('## 9. Reproduceren en bewijs', 1)[0])
    md('## 9. Reproduceren en bewijsbestanden\n' + report.split('## 9. Reproduceren en bewijs', 1)[1])
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {'name': 'python3', 'display_name': 'Python 3 (comparison environment)', 'language': 'python'}
    notebook.metadata.language_info = {'name': 'python', 'version': '3.13.7'}
    return notebook


def main():
    # Keep the main notebook on the complete species comparison once it exists.
    if (ROOT / 'secondary_mushroom/comparison_species/comparison_summary.json').exists():
        from build_species_comparison_report import main as build_species_main
        return build_species_main()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    summary = json.loads((OUT / 'shared_comparison_summary.json').read_text(encoding='utf-8'))
    audit = json.loads((OUT / 'aws_artifact_audit.json').read_text(encoding='utf-8'))
    metrics = pd.read_csv(OUT / 'shared_model_comparison.csv').set_index('model')
    errors = pd.read_csv(OUT / 'shared_error_records.csv')
    report = report_text(summary, audit, metrics, errors)
    (FOLDER / '09_vergelijkingsrapport.md').write_text(report, encoding='utf-8')
    notebook = notebook_content(report)
    if args.execute:
        environment = dict(os.environ)
        environment['PATH'] = str(Path(sys.executable).parent) + os.pathsep + environment['PATH']
        NotebookClient(notebook, timeout=120, kernel_name='python3',
            resources={'metadata': {'path': str(FOLDER)}}).execute(env=environment)
    nbformat.validate(notebook)
    nbformat.write(notebook, FOLDER / '09_compare_models.ipynb')
    print('Extended report and notebook written; executed:', args.execute)


if __name__ == '__main__':
    main()
