"""Build the Dutch report and optionally execute the comparison notebook.

Run compare_andrew_models.py first, then this script with --execute.
This script reads the saved evaluation results; it does not train models.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FOLDER = ROOT / "secondary_mushroom"
RESULTS = FOLDER / "comparison_andrew"
NAMES = {
    "majority": "Meerderheidsreferentie",
    "random_forest_500": "Random Forest 500",
    "gradient_boosting": "Gradient Boosting",
    "xgboost": "XGBoost",
    "logistic_polynomial": "Logistic Regression + poly",
    "logistic_artifact_config": "Logistic Regression v1-config",
}


def markdown_table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in rows],
    ])


def build_report(summary, metrics, errors):
    selected = metrics.loc[summary["selected_model"]]
    cv_rank = metrics.sort_values("cv_f1_mean", ascending=False)
    cv_table = markdown_table(
        ["Model", "CV F1 p ± SD", "CV recall p", "CV accuracy"],
        [[NAMES[key], f"{r.cv_f1_mean:.4f} ± {r.cv_f1_std:.4f}",
          f"{r.cv_recall_mean:.4f}", f"{r.cv_accuracy_mean:.4f}"]
         for key, r in cv_rank.iterrows()],
    )
    test_table = markdown_table(
        ["Model", "Accuracy", "Balanced acc.", "Precision p", "Recall p", "F1 p", "ROC-AUC", "AP"],
        [[NAMES[key], *[f"{r[c]:.4f}" for c in ["accuracy", "balanced_accuracy",
          "precision_poisonous", "recall_poisonous", "f1_poisonous", "roc_auc", "average_precision"]]]
         for key, r in metrics.iterrows()],
    )
    confusion = markdown_table(
        ["Model", "TN: e → e", "FP: e → p", "FN: p → e", "TP: p → p"],
        [[NAMES[key], *[int(r[c]) for c in ["tn", "fp", "fn", "tp"]]] for key, r in metrics.iterrows()],
    )
    resources = markdown_table(
        ["Model", "Train accuracy", "Train F1 p", "Fit 4.100 records (s)", "Proba 900 records (ms)", "Pipeline (MiB)"],
        [[NAMES[key], f"{r.train_accuracy:.4f}", f"{r.train_f1_poisonous:.4f}",
          f"{r.fit_seconds:.3f}", f"{r.predict_900_ms:.2f}", f"{r.serialized_bytes_compress3 / 2**20:.3f}"]
         for key, r in metrics.iterrows() if key != "majority"],
    )
    recorded = markdown_table(
        ["Notebookmodel", "Accuracy in Andrew's output", "Opnieuw berekend", "Overeenkomst op bronprecisie"],
        [[NAMES[key], f"{values['accuracy_rounded']:.3f}", f"{metrics.loc[key, 'accuracy']:.6f}", "Ja"]
         for key, values in summary["notebook_recorded_scores"].items()],
    )
    missing = markdown_table(
        ["Feature", "Ontbrekend na schoonmaak"],
        [[key, f"{value:.2f}%"] for key, value in summary["dataset"]["missing_percent_after_cleaning"].items()],
    )
    subgroups = markdown_table(
        ["Ontbrekende features", "Giftige records", "FN", "Recall p"],
        [[r["missing_features"], r["poisonous_count"], r["false_negatives"], f"{r['recall']:.4f}"]
         for r in summary["selected_false_negative_missingness"]],
    )
    examples = errors[errors.true_class == "p"].sort_values("probability_p").head(3)
    example_table = markdown_table(
        ["Bronrij (0-based)", "Werkelijk → voorspeld", "Modelscore p", "Ontbrekende features", "cap-shape", "stem-surface"],
        [[int(r.source_row), "p → e", f"{r.probability_p:.4f}", int(r.missing_features_after_cleaning),
          "ontbreekt" if pd.isna(r["cap-shape"]) else r["cap-shape"],
          "ontbreekt" if pd.isna(r["stem-surface"]) else r["stem-surface"]] for _, r in examples.iterrows()],
    )
    return rf"""# Vergelijkingsrapport: Andrew's Mushroom-modellen

**Historische deelvergelijking:** zie [het uitgebreide teamrapport](09_vergelijkingsrapport.md) voor Andrew + Jorre/AWS op dezelfde UCI-data.

**DeepLearningTeam10 — 9 oktober 2026.** Analyse van de modeldefinities en bestanden uit
broncommit [`17c4a61`](https://github.com/JorreVanDyck10/DeepLearningTeam10/commit/17c4a61).
Bijbehorend uitgevoerd notebook: [09_compare_andrew_models.ipynb](09_compare_andrew_models.ipynb).

## Besluit en afbakening

**Random Forest 500 is de voorlopige keuze binnen Andrew's datasetversie.** Het behaalt de
hoogste gemiddelde F1 voor de giftige klasse in vijfvoudige kruisvalidatie: **0,6804**.
Op de bestaande testset zijn Random Forest en Gradient Boosting vrijwel gelijk: **81,00%
tegenover 80,89% accuracy**, een verschil van één correcte voorspelling op 900 records.
Beide herkennen 227 van de 341 giftige records en classificeren **114 giftige records als
eetbaar**. Dat is 33,43% van de giftige testrecords.

Dit is een vergelijking van Andrew's bestand met **5.000 records en 12 features**.
Het is geen definitieve rangschikking van alle Mushroom-modellen van het team. De volledige
UCI-data, de AWS-resultaten en de eerder gedeployde Random Forest gebruiken andere data
of een andere testsplit. De onderzochte gegevens zijn bovendien gesimuleerd; deze demo
kan geen eetbaarheid van echte paddenstoelen bepalen.

## 1. Aansluiting op de opdracht en bijdragen

De opdracht vraagt per dataset een notebook dat modelmetrics verzamelt, modellen grondig
vergelijkt en conclusies bevat. De rubric beoordeelt ook passende validatie, concrete
foutanalyse en reproduceerbaarheid. Dit rapport en notebook leveren voor Andrew's modellen
die vergelijking, inclusief een referentiemodel, opgeslagen modelcontrole, confusion
matrices, train/validatie/test-scores en een onderbouwde voorlopige keuze.

| Bijdrage | Wie / status |
| --- | --- |
| Oorspronkelijke datasetversie, notebook en modeldefinities | Andrew Noeyens; gecommitteerde bronbestanden |
| Aanvraag voor dit vergelijkingsrapport | Jorre Van Dyck |
| Reproductiescript, nieuwe CV-evaluatie, grafieken, notebook en rapporttekst | Codex, uitgevoerd in de lokale projectomgeving |
| Inhoudelijke teamreview en mondelinge verdediging | Nog door het team uit te voeren; dit rapport beweert geen afgeronde review |

AI-gebruik is hiermee expliciet vermeld. Teamleden moeten de keuzes en beperkingen zelf
kunnen uitleggen. Dit rapport vervangt de volledige EDA, AutoML, systematische tuning,
afzonderlijke modelnotebooks of Citi Bike-vergelijking niet.

## 2. Welke modellen zijn werkelijk aanwezig?

Bronmap: [SolutionAndrew/MushroomDataset](../SolutionAndrew/MushroomDataset/).
Andrew's huidige `mushroom.ipynb` bevat vier classifiers. Twee `.pkl`-bestanden bevatten
opgeslagen pipelines. Die bestanden zijn met **joblib** opgeslagen.

| Kandidaat | Instellingen die opnieuw zijn uitgevoerd | Reden voor vergelijking |
| --- | --- | --- |
| Meerderheidsreferentie | Altijd `e` voorspellen | Laat zien waarom accuracy alleen onvoldoende is |
| Random Forest 500 | 500 bomen; depth 20; min split 3; min leaf 1; max features sqrt; balanced class weights | Boomensemble kan categorische combinaties en niet-lineaire patronen modelleren |
| Gradient Boosting | 350 bomen; learning rate 0,08; depth 8; min split 4; min leaf 2; subsample 1 | Sequentiële bomen corrigeren eerdere fouten; relatief diepe bomen kunnen overfitten |
| XGBoost | 500 bomen; learning rate 0,05; depth 5; min child weight 2; subsample en colsample 0,8 | Andere boostingimplementatie met sampling en regularisatiemogelijkheden |
| Logistic Regression + poly | Numerieke polynomial features graad 2; scaling; C=1; balanced; max iter 3.000 | Vergelijking met een eenvoudiger beslisfunctie en beperkte numerieke interacties |
| Logistic Regression v1-config | Geen polynomial features; numerieke missing indicators en scaling; C=1; geen class weights; max iter 2.000 | Controle van de werkelijk opgeslagen LR-configuratie |

**De opgeslagen Logistic Regression is een ander model dan de huidige notebookvariant.**
We vergelijken daarom beide configuraties apart. De v1-configuratie wordt vanuit de
opgeslagen pipeline gekloond en opnieuw getraind; `clone` neemt geen aangeleerde parameters
mee. Hierdoor gebruikt ook deze kandidaat precies dezelfde nieuwe CV-folds en trainingsset.

De opgeslagen Gradient Boosting reproduceert de notebookconfiguratie. Voor beide opgeslagen
pipelines komen de labels én de waarschijnlijkheidsscores op deze 900 records exact overeen
met hun opnieuw getrainde tegenhanger. Hun oorspronkelijke trainingslidmaatschap is niet
apart bewezen; de hoofdvergelijking berust daarom op de gecontroleerde hertraining.

Het oudere bestand `models/random_forest.joblib` bevat **200 bomen**, gebruikt een eerdere
80/20-split en hoort bij de huidige API-configuratie. Het is niet het RF500-model uit de
nieuwste notebook. Voor RF500 en XGBoost staan in deze broncommit modeldefinities en outputs,
maar geen nieuwe afzonderlijk geëxporteerde modelbestanden. In de onderzochte notebook staat
geen vastgelegd GridSearch/RandomizedSearch-traject; gekozen hyperparameters zijn niet op
zichzelf bewijs van systematische tuning.

## 3. Data, schoonmaak en vergelijkbaarheid

Het gecommitteerde CSV-bestand heeft 5.000 rijen: 3.107 `e` en 1.893 `p` (37,86%).
De features zijn drie numerieke metingen en negen categorische velden, waaronder
`jumbled_noise_0` en `jumbled_noise_1`. De herkomst en bewerkingen van deze 5.000-rijenversie
zijn niet voldoende vastgelegd om haar als willekeurige UCI-steekproef te behandelen.

Het oorspronkelijke [UCI Secondary Mushroom-dataset](https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset)
beschrijft 61.069 gesimuleerde paddenstoelen op basis van 173 soorten met 20 features.
De aanvullende noise-kolommen en ontbrekende oorspronkelijke features maken Andrew's
probleemdefinitie verschillend van de volledige UCI-versie.

Zoals in Andrew's notebook worden nullen in `stem-height` en `stem-width` omgezet naar
ontbrekende waarden (60 records hebben minstens één zulke nul). Dit is een overgenomen
schoonmaakkeuze die nog een inhoudelijke motivatie in de uiteindelijke EDA nodig heeft.
Numerieke waarden worden met de trainingsmediaan ingevuld; categorische waarden met de
trainingsmodus en daarna one-hot encoded. Onbekende categorieën worden genegeerd.
De LR-varianten voegen hun hierboven beschreven transformaties toe.

{missing}

Er zijn geen exact gelijke volledige rijen gevonden en geen exact overeenkomende feature-rijen
tussen train en test. Dat sluit afhankelijkheid tussen nabije, verwante of gesimuleerde
records niet uit. De hoge ontbrekendheid van `spore-print-color` en `stem-surface` beperkt
hoeveel informatie deze velden voor afzonderlijke voorspellingen leveren.

## 4. Evaluatieprotocol en metrics

1. Reproduceer Andrew's gestratificeerde split: `test_size=0.18`, `random_state=42`.
   Dit geeft 4.100 trainingsrecords en 900 testrecords, waarvan 559 `e` en 341 `p`.
2. Gebruik op de 4.100 trainingsrecords vijf identieke gestratificeerde validatiefolds,
   met shuffle en seed 42. Per fold worden de volledige preprocessing en classifier
   uitsluitend op de fold-trainingsrecords gefit.
3. Selecteer de kandidaat met de hoogste gemiddelde **F1 voor p**; gemiddelde recall p
   is de tie-breaker. Deze keuze wordt in het script vastgelegd vóór de testmetrics worden berekend.
4. Fit vervolgens elke configuratie op alle 4.100 trainingsrecords en analyseer de bestaande
   900 testrecords. Gebruik de standaard `predict`-beslissing (grens rond 0,5); pas geen
   drempel of hyperparameter aan op basis van de testuitkomsten.

De positieve klasse is `p` (intern 1); `e` is 0. Accuracy telt alle correcte labels.
Recall p = TP/(TP+FN) meet hoeveel giftige records worden herkend. Precision p = TP/(TP+FP)
meet hoe vaak een giftig-voorspelling klopt. F1 combineert die twee. Balanced accuracy
weegt beide klasserecalls gelijk. ROC-AUC en average precision (AP) beoordelen de rangschikking
van modelscores over meerdere drempels; AP is geen gemeten recall bij de huidige drempel.

F1 is hier een expliciete ontwikkelkeuze om beide fouttypen mee te wegen, geen onderbouwde
veiligheidsgrens. Recall en het aantal FN blijven daarom zichtbaar. Voor een toekomstig
gebruik met asymmetrische foutkosten moet het team eerst een doel voor recall of foutkosten
vastleggen en een drempel uitsluitend op trainingsvalidatie bepalen.

**Deze testset was al in Andrew's notebook bekeken.** De modelkeuze gebruikt in deze nieuwe
run alleen CV, maar de eerdere experimenten kunnen al door testresultaten zijn beïnvloed.
Dit is een ontwikkelvergelijking, geen nieuwe onaangeroerde eindtest. De foldspreiding is
geen betrouwbaarheidsinterval voor de rangschikking. Een definitieve vergelijking vraagt
een vooraf vastgelegde gemeenschappelijke dataversie en een werkelijk nieuwe eindtest of
een passend protocol voor de beschikbare gegevens. Zie het
[scikit-learn-validatieprotocol](https://scikit-learn.org/stable/modules/cross_validation.html).

## 5. Resultaten en reproductie van Andrew's output

### Selectie op trainingsvalidatie

{cv_table}

RF500 behaalt de hoogste CV F1 en CV recall p. XGBoost heeft een iets hogere CV F1 dan
Gradient Boosting; de testvolgorde verschilt dus van de CV-volgorde. Het verschil in
gemiddelde CV F1 tussen RF500 en Gradient Boosting is ongeveer 0,0262. Vijf overlappende
foldtrainingen volstaan niet om een universele beste classifier te bewijzen.

### Audit op dezelfde 900 testrecords

{test_table}

De meerderheidsreferentie haalt 62,11% accuracy maar vindt geen enkele giftige paddenstoel.
RF500 verbetert accuracy met 18,89 procentpunt en herkent 227 giftige records. XGBoost heeft
de hoogste precision p van de getrainde kandidaten, maar mist 149 giftige records. De
LR-v1 heeft meer accuracy dan LR-poly, terwijl de recall p veel lager is (35,78% tegenover
61,88%): een concrete reden om nooit alleen accuracy te gebruiken.

![Testmetrics van de modellen](comparison_andrew/metrics_comparison.png)

{recorded}

Ook de opnieuw berekende precision, recall en F1 van de vier notebookmodellen komen op de
afgeronde precisie van Andrew's output overeen. Versies en parallelle instellingen doen
ertoe: tijdens een reproductiecontrole gaf XGBoost met twee threads 79,78% accuracy; met
Andrew's `n_jobs=-1` werd de gepubliceerde 78,67% gereproduceerd. De definitieve tabel gebruikt
`n_jobs=-1`; een vaste seed alleen garandeert geen identieke uitkomsten in elke omgeving.

![ROC en precision-recall](comparison_andrew/roc_pr_curves.png)

RF500 heeft hier de hoogste ROC-AUC (0,8517) en AP (0,8046). De curves laten zien dat de
ranking relatief bruikbaar is, terwijl de recall bij de standaardbeslissing beperkt blijft.
De scores zijn niet op kalibratie getest: een score 0,8 betekent niet automatisch een
gevalideerde kans van 80% voor echte paddenstoelen.

## 6. Foutanalyse en onzekerheid

{confusion}

![Confusion matrices](comparison_andrew/confusion_matrices.png)

Voor RF500 zijn er 171 fouten: 114 FN en 57 FP. FN betekent dat een giftig record `e` krijgt;
FP betekent dat een eetbaar record `p` krijgt. De gif-recall is 66,57%. Het beschrijvende
Wilson-interval van 95% is **61,40–71,37%**; voor accuracy **78,31–83,43%**. Deze intervallen
veronderstellen onafhankelijke testrecords en houden geen rekening met eerdere modelselectie
of de simulatiestructuur.

RF500 en Gradient Boosting hebben hetzelfde aantal FN, maar maken niet alle fouten op
dezelfde records. RF500 is als enige correct op 40 records; Gradient Boosting op 39.
De exacte gepaarde McNemar-toets geeft p=1,0 voor hun accuracyverschil. Er is daarmee in
deze test geen aanwijzing voor een accuracyvoordeel van RF500; dit bewijst geen equivalentie.
Andere paarvergelijkingen staan in het JSON-resultaat en zijn exploratief, zonder correctie
voor meerdere toetsen. Zij bepalen de modelkeuze niet.

### Concrete onjuiste voorspellingen van RF500

{example_table}

Dit zijn de drie giftige records met de laagste p-score onder de foutieve eetbaar-labels,
gekozen voor de foutanalyse ná de modelselectie. Bronrijen tellen vanaf nul en staan in
`split_manifest.csv`. Rij 3875 mist vijf features, maar rij 1106 mist slechts één feature.
Ontbrekende waarden kunnen dus meespelen; ze verklaren niet alle fouten. De p-score is
een modeluitvoer, geen bewijs van echte eetbaarheid. Alle foutrecords inclusief originele
features zijn opgeslagen in [selected_model_errors.csv](comparison_andrew/selected_model_errors.csv).

### Ontbrekendheid bij werkelijk giftige testrecords

{subgroups}

Groepen met vier of vijf ontbrekende features hebben hier een lagere recall dan de groep
met drie. Het patroon is niet monotone en de groepen met zes en zeven ontbrekende features
zijn te klein voor stellige conclusies. De tabel toont een onderzoeksspoor voor EDA; ze
bewijst niet dat imputatie deze fouten veroorzaakt.

### Welke features gebruikt het gekozen model?

![Permutation importance](comparison_andrew/permutation_importance.png)

Op de eerste CV-validatiefold zijn `stem-surface`, `cap-shape`, `stem-width` en `gill-color`
de belangrijkste velden volgens de daling in F1 na permutatie. Er zijn acht herhalingen;
foutbalken tonen hun standaardafwijking, geen interval over dataversies. Deze analyse is
op trainingsvalidatie uitgevoerd en heeft de kandidaatselectie niet aangepast. Importance
is modelafhankelijk en niet causaal; één fold en gecorreleerde features beperken de uitleg.
Zie [permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html).

`jumbled_noise_0` draagt in deze meting ook bij. De naam alleen bewijst niet dat het
onafhankelijke ruis is. Zonder vastgelegde generatiecode kunnen we geen leakage of
betekenis claimen. Een nuttig vervolgexperiment is dezelfde CV vergelijken met en zonder
de twee noise-velden, vóór evaluatie op een nieuwe eindtest.

## 7. Overfitting, complexiteit en praktische inzet

{resources}

Gradient Boosting heeft bijna perfecte trainingaccuracy (99,59%) tegenover 80,89% op test.
RF500 heeft ook een train-testverschil (95,27% tegenover 81,00%), maar een kleiner verschil
dan Gradient Boosting. Dit is een aanwijzing voor overfitting/variantie, geen bewezen enige
oorzaak van fouten. Beide LR-varianten presteren ook op hun trainingsdata beperkt; hun
representatie, class weights en instellingen passen in deze run minder goed bij de taak.

Tijden zijn lokale metingen op Windows 11 met 20 logische CPU's. Fit-tijd is één fit op
4.100 records; predict-tijd is de mediaan van zeven `predict_proba`-batches van 900 records.
Pipelinegrootte is voor alle kandidaten gemeten met joblib-compressie 3, inclusief preprocessing.
Dit zijn geen Render-latencies, opstarttijden of gegarandeerde productiebenchmarks. XGBoost
en Gradient Boosting zijn in deze batch sneller en compacter dan RF500. Dat kan bij hosting
een afweging zijn, maar is onvoldoende om de vastgelegde F1-selectie achteraf te veranderen.

## 8. Keuze voor het team en grenzen aan de eindconclusie

We kiezen **voorlopig RF500** voor Andrew's huidige probleemdefinitie: de hoogste CV F1 p,
de hoogste CV recall p en bruikbare test-rangschikkingsscores. Gradient Boosting is een
praktisch alternatief bij strengere eisen aan modelgrootte en inference-tijd. Het kleine
accuracyverschil op test is geen geldige hoofdreden voor RF500.

| Resultaat buiten deze vergelijking | Waarom niet in dezelfde ranglijst? |
| --- | --- |
| Andrew's oudere RF200: 78,2% | Andere modelconfiguratie én 80/20-split met 1.000 testrecords; geen zuivere vergelijking met RF500 op 900 records |
| Jorre's AWS RF: 100% | Volledige UCI-versie met 20 features; na deduplicatie 60.923 records en 12.185 testrecords; andere data, preprocessing en validatie |
| Baseline/AutoML op volledige UCI | Pas vergelijkbaar na gedeeld datacontract en evaluatieprotocol |

De AWS-score is dus niet als winnaar aangewezen en ook niet als fout weggezet. Eerst moeten
datasetversie, features, schoonmaak en testrecords overeenkomen. Bij gesimuleerde data moet
het team ook uitleggen welk generalisatiedoel het meet; een random split bewijst geen
prestatie op volledig nieuwe soorten of op echte paddenstoelen.

Voor een definitieve selectie en inlevering blijven deze stappen nodig:

1. Documenteer waar Andrew's CSV vandaan komt en hoe selectie, ontbrekendheid en noise-velden
   zijn gemaakt. Leg één gemeenschappelijke data-/featureversie en split vast voor de
   uiteindelijke teamvergelijking.
2. Voeg baseline, AutoML en alle relevante team-/AWS-modellen toe aan datzelfde protocol.
   Bewaar het tuningtraject, zoekruimte en validatieresultaten; tune niet op de eindtest.
3. Gebruik de foutanalyse voor vooraf gedefinieerde trainingsexperimenten (noise-ablation,
   imputatie/missing indicators, regularisatie, eventueel een recall-doel en kalibratie).
   Rapporteer daarna een onafhankelijke eindbeoordeling en motiveer de echte winnaar.
4. Bewaar één bestand/notebook per model zoals de opdracht vraagt. RF500 en XGBoost moeten
   bij uiteindelijke selectie nog als complete pipeline worden geëxporteerd. Het huidige
   RF200-artifact is daarvoor geen vervanging.
5. Laat de backend de gekozen pipeline met het juiste inputcontract en versies laden en
   controleer API/websitevoorspellingen tegen de notebook. Dit rapport wijzigt de live
   modelkeuze niet; code deployen en een nieuw model trainen/exporteren zijn verschillende stappen.

De analyse gebruikt scikit-learn **1.9.1**; de eerdere RF200-deployment gebruikt **1.8.0**.
Een modelbestand uit een andere versie zomaar laden is geen betrouwbare migratie. Bewaar
versies en valideer de geëxporteerde pipeline in de doelomgeving, zoals beschreven in
[scikit-learn model persistence](https://scikit-learn.org/stable/model_persistence.html).

## 9. Reproduceren en bewijsbestanden

Gebruik een aparte analyseomgeving vanuit de repositoryroot, zodat backenddependencies
niet door deze vergelijking worden aangepast:

```powershell
py -3.13 -m venv .venv-analysis
.\.venv-analysis\Scripts\python.exe -m pip install -r secondary_mushroom/comparison-requirements.txt
.\.venv-analysis\Scripts\python.exe secondary_mushroom/compare_andrew_models.py
.\.venv-analysis\Scripts\python.exe secondary_mushroom/build_comparison_report.py --execute
```

De vastgelegde run gebruikt Python 3.13.7, scikit-learn 1.9.1, XGBoost 3.4.1, pandas 3.0.2,
NumPy 2.5.3, SciPy 1.18.1 en joblib 1.6.0. RF/XGBoost gebruiken `n_jobs=-1`; timing en sommige
numerieke uitkomsten kunnen op een andere machine veranderen. De volledige parameters,
versies, waarschuwingen en hashes staan in het resultaatbestand.

| Bestand | Bewijs / gebruik |
| --- | --- |
| [09_compare_andrew_models.ipynb](09_compare_andrew_models.ipynb) | Uitgevoerd vergelijkingsnotebook, uitleg, tabellen, grafieken, foutanalyse en conclusie |
| [compare_andrew_models.py](compare_andrew_models.py) | Hertraining, CV-selectie, test-audit en modelbestandcontrole |
| [build_comparison_report.py](build_comparison_report.py) | Bouwt dit rapport en voert optioneel het notebook uit |
| [comparison_summary.json](comparison_andrew/comparison_summary.json) | Volledige metrics, parameters, herkomst, versies en artifact-audit |
| [model_comparison.csv](comparison_andrew/model_comparison.csv) | Metrics per kandidaat |
| [cv_folds.csv](comparison_andrew/cv_folds.csv) | Alle 30 model/fold-evaluaties |
| [split_manifest.csv](comparison_andrew/split_manifest.csv) | Rij-identiteit, train/test en CV-validatiefold |
| [test_predictions.csv](comparison_andrew/test_predictions.csv) | Alle labels en p-scores per model op exact dezelfde 900 records |
| [selected_model_errors.csv](comparison_andrew/selected_model_errors.csv) | Alle 171 foutrecords van de gekozen kandidaat |
| [permutation_importance.csv](comparison_andrew/permutation_importance.csv) | Onderliggende featureanalyse |

De CSV heeft LF-genormaliseerde SHA-256
`{summary['dataset']['sha256_lf']}`.
Dit maakt de data-identiteit controleerbaar ongeacht Windows/Git-regeluiteinden. De originele
notebook, CSV en modelbestanden van Andrew zijn voor deze vergelijking niet overschreven.
"""


def build_notebook(report):
    cells = []
    md = lambda source: cells.append(nbformat.v4.new_markdown_cell(source))
    code = lambda source: cells.append(nbformat.v4.new_code_cell(source))
    md("""# 09 — Vergelijking van Andrew's Mushroom-modellen

**Bijdragen:** Andrew Noeyens leverde de bronnotebook/modeldefinities en modelbestanden;
Jorre Van Dyck vroeg de vergelijking; Codex implementeerde en voerde de reproductie,
kruisvalidatie, foutanalyse en rapportage uit. De inhoudelijke teamreview staat nog open.

Dit notebook verzamelt en onderzoekt de gecontroleerde resultaten van dezelfde 5.000-rijenversie.
Het is **geen definitieve ranglijst voor alle UCI/AWS-modellen**. Broncommit: `17c4a61`.
Het volledige rapport staat in [09_andrew_vergelijkingsrapport.md](09_andrew_vergelijkingsrapport.md).

Elke codecel wordt vooraf uitgelegd. Opgeslagen outputs behouden het bewijs voor de inlevering.
""")
    md("""## 1. Reproduceerbare omgeving en resultaatbestanden

Voer de twee scripts uit volgens het rapport met `comparison-requirements.txt` in een aparte
omgeving. Hieronder laden we standaard de opgeslagen resultaten. Zet `RUN_TRAINING=True`
om alle pipelines opnieuw te trainen op de vijf identieke folds en de 4.100 trainingsrecords.
Het script legt de CV-keuze vast vóór het berekenen van testmetrics. Preprocessing wordt
per fold gefit; oorspronkelijke bestanden worden behouden.

**Let op:** opnieuw uitvoeren met True overschrijft de vergelijkingsoutputs. Bouw daarna
ook het Markdownrapport opnieuw met `build_comparison_report.py --execute`.
""")
    code("""from pathlib import Path
import json
import sys
import pandas as pd
from IPython.display import display, Image

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / 'SolutionAndrew/MushroomDataset/mushroom_project_dataset.csv').exists())
FOLDER = ROOT / 'secondary_mushroom'
OUT = FOLDER / 'comparison_andrew'
sys.path.insert(0, str(FOLDER))
from compare_andrew_models import run, NAMES, clean_measurements, fingerprint

RUN_TRAINING = False
if RUN_TRAINING:
    run()
summary = json.loads((OUT / 'comparison_summary.json').read_text(encoding='utf-8'))
metrics = pd.read_csv(OUT / 'model_comparison.csv').set_index('model')
folds = pd.read_csv(OUT / 'cv_folds.csv')
split = pd.read_csv(OUT / 'split_manifest.csv')
predictions = pd.read_csv(OUT / 'test_predictions.csv')
errors = pd.read_csv(OUT / 'selected_model_errors.csv')
display(pd.Series(summary['environment'], name='omgeving van de trainingsrun'))
print('Modelkeuze:', NAMES[summary['selected_model']])""")
    md("""## 2. Datasetidentiteit, labels en split

We controleren de datahash en aantallen zodat cijfers niet stilzwijgend uit verschillende
dataversies worden samengevoegd. `source_row` verwijst naar de rij-index (vanaf nul) in de CSV.
`p` is de positieve klasse, `e` de negatieve. Andrew's split is gestratificeerd 82/18 met seed 42;
op de 4.100 trainingsrecords komen vijf gestratificeerde CV-folds. Elke trainingsrij is precies
één keer validatierij. Deze bestaande testset was al bekeken in de bronnotebook, en is dus
een ontwikkeltestset. Exacte overlapchecks sluiten verwante gesimuleerde records niet uit.
""")
    code("""source = ROOT / 'SolutionAndrew/MushroomDataset/mushroom_project_dataset.csv'
data = pd.read_csv(source)
assert fingerprint(source)['sha256_lf'] == summary['dataset']['sha256_lf']
assert len(split) == 5000 and split.source_row.is_unique
assert (split.split == 'train').sum() == 4100
assert (split.split == 'test').sum() == 900
assert split.loc[split.split == 'train', 'cv_validation_fold'].notna().all()
assert split.loc[split.split == 'test', 'cv_validation_fold'].isna().all()
assert set(predictions.source_row) == set(split.loc[split.split == 'test', 'source_row'])
display(pd.crosstab(split.split, split['class']))
display(split[split.split == 'train'].cv_validation_fold.value_counts().sort_index())
print('Exacte duplicaten:', summary['dataset']['exact_duplicate_rows'])
print('Exacte feature-overlap train/test:', summary['dataset']['overlapping_exact_feature_rows_train_test'])
display(pd.Series(summary['dataset']['missing_percent_after_cleaning'], name='ontbrekend (%)'))""")
    model_description = report.split("## 2. Welke modellen zijn werkelijk aanwezig?", 1)[1].split("## 3. Data, schoonmaak", 1)[0]
    md("## 3. Kandidaten en opgeslagen modelbestanden\n" + model_description)
    code("""artifact_rows = []
for key, audit in summary['artifact_audit'].items():
    path = ROOT / audit['path']
    assert fingerprint(path)['sha256'] == audit['sha256']
    artifact_rows.append({'model': NAMES[key],
        'labels gelijk aan hertraining': audit['matches_refitted_labels'],
        'max. verschil p-score': audit['maximum_probability_difference'],
        'oorspronkelijke trainingsrijen onafhankelijk bewezen': audit['training_membership_verified']})
display(pd.DataFrame(artifact_rows))
for key in ['random_forest_500', 'gradient_boosting', 'xgboost', 'logistic_polynomial', 'logistic_artifact_config']:
    print(NAMES[key], summary['parameters'][key])""")
    md("""## 4. Modelselectie op trainingsvalidatie

Hoofdcriterium: gemiddelde F1 voor `p`; tie-breaker: gemiddelde recall voor `p`.
F1 weegt precision en recall samen; het is geen veiligheidsgrens. Testmetrics spelen geen
rol in deze scriptkeuze. Toon naast het gemiddelde de standaardafwijking over vijf folds;
die spreiding is geen betrouwbaarheidsinterval van de modelrangschikking.
""")
    code("""ranking = metrics.sort_values(['cv_f1_mean', 'cv_recall_mean'], ascending=False)
assert ranking.index[0] == summary['selected_model']
assert len(folds) == 30 and (folds.groupby('model').size() == 5).all()
display(ranking[['cv_f1_mean', 'cv_f1_std', 'cv_recall_mean', 'cv_accuracy_mean']].round(4))
display(folds.pivot(index='fold', columns='model', values='f1_poisonous').round(4))""")
    md("""## 5. Gemeenschappelijke test-audit en bronreproductie

We verzamelen accuracy, balanced accuracy, precision/recall/F1 voor `p`, ROC-AUC en average
precision. `predict` gebruikt de standaardbeslissing rond 0,5; er is geen testgestuurde
drempelwijziging. Recall telt herkende giftige records; FN telt giftig als eetbaar.
ROC-AUC/AP beoordelen ranking, geen gekalibreerde kansen. De vier notebookmodellen moeten
overeenkomen met Andrew's afgeronde bronoutputs. XGBoost gebruikt dezelfde `n_jobs=-1` als
Andrew; een controle met twee threads leverde andere scores op.
""")
    code("""columns = ['accuracy', 'balanced_accuracy', 'precision_poisonous', 'recall_poisonous',
           'f1_poisonous', 'roc_auc', 'average_precision']
display(metrics[columns].round(4))
checks = []
for key, recorded in summary['notebook_recorded_scores'].items():
    row = metrics.loc[key]
    assert round(row.accuracy, 3) == recorded['accuracy_rounded']
    for metric in ['precision_poisonous', 'recall_poisonous', 'f1_poisonous']:
        assert round(row[metric], 2) == recorded[metric + '_rounded']
    checks.append({'model': NAMES[key], 'accuracy bron': recorded['accuracy_rounded'],
                   'accuracy reproductie': row.accuracy, 'overeenkomst afgeronde scores': True})
display(pd.DataFrame(checks))
display(Image(filename=str(OUT / 'metrics_comparison.png')))
display(Image(filename=str(OUT / 'roc_pr_curves.png')))""")
    md("""## 6. Concrete fouten en onzekerheid

Controleer de confusion matrices opnieuw vanuit de individuele voorspellingen. Rijvolgorde
is werkelijk e/p; kolomvolgorde voorspeld e/p. RF500 en Gradient Boosting missen elk 114
van 341 giftige records. Hun accuracy verschilt met één geval. De gepaarde McNemar-toets
is exploratief en test accuracy, niet F1 of veiligheidsbruikbaarheid. p=1,0 bewijst geen
equivalentie; er is geen meervoudige-toetscorrectie toegepast.
""")
    code("""from sklearn.metrics import confusion_matrix, f1_score, recall_score
y = (predictions.true_class == 'p').astype(int)
for key, row in metrics.iterrows():
    labels = (predictions[key + '_prediction'] == 'p').astype(int)
    cm = confusion_matrix(y, labels, labels=[0, 1])
    assert cm.sum() == 900
    assert cm.ravel().tolist() == [int(row[c]) for c in ['tn', 'fp', 'fn', 'tp']]
    assert abs(f1_score(y, labels, zero_division=0) - row.f1_poisonous) < 1e-12
    assert abs(recall_score(y, labels, zero_division=0) - row.recall_poisonous) < 1e-12
display(metrics[['tn', 'fp', 'fn', 'tp']].astype(int))
display(Image(filename=str(OUT / 'confusion_matrices.png')))
chosen = next(r for r in summary['model_comparison'] if r['model'] == summary['selected_model'])
print('Wilson 95% accuracy:', chosen['accuracy_ci95'])
print('Wilson 95% recall p:', chosen['recall_ci95'])
display(pd.DataFrame(summary['pairwise_accuracy_audit']))""")
    md("""De intervals beschrijven steekproefonzekerheid onder onafhankelijkheid; ze corrigeren
niet voor de reeds bekeken testset of simulatiestructuur. Nu bekijken we de drie giftige
records met de laagste p-score onder de foutieve eetbaar-voorspellingen. De selectie dient
alleen als concrete foutanalyse en leidt niet tot testgestuurde parameterwijzigingen.
""")
    code("""assert len(errors) == chosen['fn'] + chosen['fp'] == 171
display(errors[errors.true_class == 'p'].sort_values('probability_p').head(3))
display(pd.DataFrame(summary['selected_false_negative_missingness']))""")
    md("""Rij 3875 mist vijf features; rij 1106 mist er slechts één. Ontbrekendheid is dus geen
afdoende verklaring voor alle FN. De ontbrekendheidsgroepen tonen geen monotone trend en
de hoogste groepen zijn klein. Een oorzaak zoals slechte imputatie is hiermee niet bewezen.

## 7. Featureanalyse op trainingsvalidatie

Permutation importance voor de gekozen RF500 wordt op de eerste CV-validatiefold berekend,
met acht herhalingen en F1 als score. Dit verandert de modelkeuze niet. De foutbalken tonen
spreiding over permutaties, geen onzekerheid over andere datasets. Importance is niet causaal
en gecorreleerde velden kunnen de uitkomst beïnvloeden. Een naam als `jumbled_noise` bewijst
niet dat het veld onafhankelijke ruis is; generatiecode en een vooraf geplande ablation zijn nodig.
""")
    code("""importance = pd.read_csv(OUT / 'permutation_importance.csv').sort_values('f1_decrease', ascending=False)
display(importance.round(4))
display(Image(filename=str(OUT / 'permutation_importance.png')))""")
    md("""## 8. Train-testverschillen en praktische kosten

Vergelijk train- met test-scores als aanwijzing voor overfitting. Gradient Boosting past de
training bijna perfect; het verschil met test is groter dan bij RF500. De LR-varianten
hebben ook beperkte trainingsscores. De timing is lokaal: één fit op 4.100 records en de
mediaan van zeven batches `predict_proba` op 900 records. De pipelinegrootte is voor iedereen
joblib-compressie 3. Dit meet geen Render-responstijd of opstarttijd.
""")
    code("""practical = metrics.loc[metrics.index != 'majority',
    ['train_accuracy', 'accuracy', 'train_f1_poisonous', 'f1_poisonous', 'fit_seconds', 'predict_900_ms']].copy()
practical['accuracy train-test verschil'] = practical.train_accuracy - practical.accuracy
practical['pipeline MiB (compressie 3)'] = metrics.serialized_bytes_compress3 / 2**20
display(practical.round(4))
assert all(not counts for counts in summary['warnings'].values())
print('Geen waarschuwingen tijdens de vastgelegde CV- en full-fit runs.')""")
    md("## 9. Onderbouwde conclusie en resterend werk\n" + report.split("## 8. Keuze voor het team en grenzen aan de eindconclusie", 1)[1].split("## 9. Reproduceren en bewijsbestanden", 1)[0])
    md("""## 10. Referenties en controleerbare bestanden

- [Volledig rapport en reproduceercommando's](09_andrew_vergelijkingsrapport.md)
- [Reproductiescript](compare_andrew_models.py)
- [Volledige audit en hashes](comparison_andrew/comparison_summary.json)
- [Alle testvoorspellingen](comparison_andrew/test_predictions.csv)
- [UCI Secondary Mushroom](https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset)
- [scikit-learn cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [scikit-learn permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
- [scikit-learn model persistence](https://scikit-learn.org/stable/model_persistence.html)
""")
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"name": "python3", "display_name": "Python 3 (comparison environment)", "language": "python"}
    notebook.metadata.language_info = {"name": "python", "version": "3.13.7"}
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Execute and retain notebook outputs")
    args = parser.parse_args()
    summary = json.loads((RESULTS / "comparison_summary.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(RESULTS / "model_comparison.csv").set_index("model")
    errors = pd.read_csv(RESULTS / "selected_model_errors.csv")
    report = build_report(summary, metrics, errors)
    (FOLDER / "09_andrew_vergelijkingsrapport.md").write_text(report, encoding="utf-8")
    notebook = build_notebook(report)
    if args.execute:
        environment = dict(os.environ)
        environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment["PATH"]
        NotebookClient(notebook, timeout=120, kernel_name="python3",
            resources={"metadata": {"path": str(FOLDER)}}).execute(env=environment)
    nbformat.validate(notebook)
    nbformat.write(notebook, FOLDER / "09_compare_andrew_models.ipynb")
    print("Report and notebook written; notebook executed:", args.execute)


if __name__ == "__main__":
    main()
