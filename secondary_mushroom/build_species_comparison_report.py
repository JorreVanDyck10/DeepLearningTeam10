"""Build the assignment's Mushroom comparison report and executed notebook."""
import argparse
import json
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
import pandas as pd

from build_comparison_report import markdown_table
from compare_species_models import NAMES, ORDER, OUT
from comparison_data import ROOT

FOLDER = ROOT / 'secondary_mushroom'


def report_text(summary, metrics, folds, errors, per_species):
    selected = summary['selected_model']
    row = metrics.loc[selected]
    rank = metrics.drop(index='majority').sort_values(
        ['cv_balanced_accuracy_mean', 'cv_f1_poisonous_mean', 'cv_recall_poisonous_mean'], ascending=False, kind='stable')
    next_key = rank.index[1]
    ref = metrics.loc['majority']
    order = [*rank.index, 'majority']
    cv_table = markdown_table(['Configuratie', 'Gem. balanced accuracy ± SD', 'Gem. F1 p ± SD', 'Gem. recall p'],
        [[NAMES[k], f"{metrics.loc[k, 'cv_balanced_accuracy_mean']:.4f} ± {metrics.loc[k, 'cv_balanced_accuracy_std']:.4f}",
          f"{metrics.loc[k, 'cv_f1_poisonous_mean']:.4f} ± {metrics.loc[k, 'cv_f1_poisonous_std']:.4f}",
          f"{metrics.loc[k, 'cv_recall_poisonous_mean']:.4f}"] for k in order])
    pooled = markdown_table(['Configuratie', 'Accuracy', 'Balanced acc.', 'Precision p', 'Recall p', 'F1 p', 'ROC-AUC', 'AP', 'FN', 'FP'],
        [[NAMES[k], *[f'{metrics.loc[k, c]:.4f}' for c in ['accuracy', 'balanced_accuracy',
          'precision_poisonous', 'recall_poisonous', 'f1_poisonous', 'roc_auc', 'average_precision']],
          int(metrics.loc[k, 'fn']), int(metrics.loc[k, 'fp'])] for k in order])
    confusion = markdown_table(['Configuratie', 'TN e→e', 'FP e→p', 'FN p→e', 'TP p→p'],
        [[NAMES[k], *[int(metrics.loc[k, c]) for c in ['tn', 'fp', 'fn', 'tp']]] for k in order])
    timing = markdown_table(['Configuratie', 'Gem. fit per fold (s)', 'Gem. train accuracy', 'Gem. validatie accuracy'],
        [[NAMES[k], f"{metrics.loc[k, 'mean_fold_fit_seconds']:.2f}",
          ('Niet opnieuw gemeten' if pd.isna(folds.loc[folds.model == k, 'train_accuracy'].mean()) else
           f"{folds.loc[folds.model == k, 'train_accuracy'].mean():.4f}"),
          f"{metrics.loc[k, 'cv_accuracy_mean']:.4f}"] for k in order])
    examples = errors.loc[errors.true_class == 'p'].sort_values('probability_p').head(3)
    example_table = markdown_table(['Bronrij (0-based)', 'Soortgroep', 'Werkelijk → voorspeld', 'Score p', 'Ontbrekende kenmerken'],
        [[int(r.source_row), int(r.species_group), f'{r.true_class} → {r.predicted_class}',
          f'{r.probability_p:.4f}', int(r.missing_features)] for _, r in examples.iterrows()])
    lookup = pd.read_csv(FOLDER / 'comparison_team/species_lookup.csv')
    worst = per_species.loc[(per_species.model == selected) & (per_species['class'] == 'p')].copy()
    worst['fn_fraction'] = worst.fn / worst.records
    worst = worst.sort_values(['fn_fraction', 'fn'], ascending=False).head(5).merge(lookup[['species_group', 'name']], on='species_group')
    worst_table = markdown_table(['Primaire soort', 'Groep', 'FN / records', 'FN-fractie'],
        [[r['name'], int(r.species_group), f'{int(r.fn)} / {int(r.records)}', f'{r.fn_fraction:.2%}'] for _, r in worst.iterrows()])
    source = pd.read_csv(FOLDER / 'comparison_team/shared_model_comparison.csv').set_index('model')
    random_table = markdown_table(['Configuratie', 'Random test accuracy', 'Soorten-CV accuracy', 'Soorten-CV F1 p'],
        [[NAMES[k], f"{source.loc[k, 'accuracy']:.6f}", f"{metrics.loc[k, 'accuracy']:.6f}",
          f"{metrics.loc[k, 'f1_poisonous']:.6f}"] for k in source.index])
    n_bad_species = int(((per_species.model == selected) & (per_species.fn > 0)).sum())
    warning_count = {k: sum(count.values()) for k, count in summary['warnings'].items()}
    return f'''# Mushroom — modelvergelijking volgens de opdracht

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

**Ontwikkelkeuze voor ongeziene gesimuleerde soorten: {NAMES[selected]}.**
Van de negen getrainde kandidaatconfiguraties heeft dit model de hoogste gemiddelde
balanced accuracy over vijf identieke soorten-folds: **{row.cv_balanced_accuracy_mean:.4f}**.
De gepoolde voorspellingen buiten training geven **{row.accuracy:.2%} accuracy**,
**{row.recall_poisonous:.2%} recall p** en **F1 p {row.f1_poisonous:.4f}**.
Er zijn **{int(row.fn):,} FN** en **{int(row.fp):,} FP** op 60.923 unieke records.

De eerstvolgende configuratie op het keuzecriterium is {NAMES[next_key]} met
{metrics.loc[next_key, 'cv_balanced_accuracy_mean']:.4f} gemiddelde balanced accuracy.
Het verschil bedraagt slechts
{(row.cv_balanced_accuracy_mean-metrics.loc[next_key, 'cv_balanced_accuracy_mean'])*100:.3f}
procentpunt. RF200 haalt gepoolde accuracy
{metrics.loc['random_forest_200', 'accuracy']:.2%} en recall p
{metrics.loc['random_forest_200', 'recall_poisonous']:.2%}; {NAMES[selected]} haalt
respectievelijk {row.accuracy:.2%} en {row.recall_poisonous:.2%}.
De keuze voor {NAMES[selected]} volgt dus het vastgelegde criterium;
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
Bronhash (LF-normalisatie): `{summary['dataset']['sha256_lf']}`.

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

{cv_table}

SD beschrijft spreiding tussen vijf folds; het is geen betrouwbaarheidsinterval van
de rangschikking. Verschillen in samenstelling van de soorten beïnvloeden de uitkomsten.
De keuze gebruikt het gemiddelde van foldmetrics, niet achteraf de hoogste losse fold.

### Gepoolde voorspellingen buiten training

{pooled}

De tabel combineert alle 60.923 voorspellingen buiten training. Een gepoolde metric kan
afwijken van het ongewogen foldgemiddelde doordat foldgroottes en klasseaantallen verschillen.
ROC-AUC en AP beoordelen ranking van de p-scores; ze bewijzen geen kanskalibratie.

![Metrics op ongeziene soorten](comparison_species/metrics.png)

De meerderheidsreferentie haalt F1 p **{ref.f1_poisonous:.4f}** en recall p **{ref.recall_poisonous:.4f}**,
maar balanced accuracy **{ref.balanced_accuracy:.4f}**: ze herkent geen eetbare records.
Daarom beoordelen we een configuratie niet alleen op F1 of recall. De gekozen pipeline
heeft balanced accuracy {row.balanced_accuracy:.4f}, specificity
{row.tn/(row.tn+row.fp):.4f} en precision p {row.precision_poisonous:.4f}.

### Waarom de eerdere 100% geen eindconclusie was

{random_table}

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

{confusion}

![Confusion matrices op dezelfde soorten-folds](comparison_species/confusion_matrices.png)

De gekozen configuratie mist {int(row.fn):,} van {int(row.fn+row.tp):,} p-records
({1-row.recall_poisonous:.2%}), verdeeld over {n_bad_species} soorten met minstens één FN.
Hier zijn drie FN-records met de laagste p-score, dus voorbeelden van relatief
overtuigde verkeerde voorspellingen. De score is een modelschatting, geen bewezen kans.

{example_table}

De vijf soorten met de hoogste FN-fractie bij deze configuratie:

{worst_table}

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

{timing}

Tijden zijn lokale Windows-metingen, geen AWS-kosten of Render-latencybenchmark.
AWS gebruikt twee threads; RF500/XGBoost gebruiken beschikbare threads; RF200 behoudt
de bestaande instellingen. AWS getuned en RF500 hergebruiken de geverifieerde resultaten
van de eerdere identieke soorten-folds, inclusief toen gemeten fit-tijden. Hun trainmetrics
werden niet opgeslagen en worden niet achteraf ingevuld. Andere foldruns bewaren trainmetrics.
Een train-validatiekloof beschrijft gedrag, maar bewijst niet één oorzaak.
Er zijn {sum(warning_count.values())} vastgelegde waarschuwingen tijdens de nieuwe evaluatie/refit;
de volledige berichten staan in het resultaat-JSON en worden in het notebook getoond.

Voor modelkeuze wegen de groepsvalidatiescores zwaarder dan snelheid. Een eenvoudiger
model kan voldoende zijn als de score vergelijkbaar is; het type algoritme, aantal bomen
of het gebruik van AWS bewijst op zichzelf geen betere generalisatie.

## 7. Eindkeuze, deployment en resterende opdrachtvereisten

**Kies voorlopig {NAMES[selected]} voor het vastgelegde doel 'ongeziene gesimuleerde soorten'.**
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
.\\.venv-analysis\\Scripts\\python.exe -m pip install -r secondary_mushroom/comparison-requirements.txt
.\\.venv-analysis\\Scripts\\python.exe secondary_mushroom/compare_species_models.py
.\\.venv-analysis\\Scripts\\python.exe secondary_mushroom/build_species_comparison_report.py --execute
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

Versies: Python {summary['environment']['python']}, scikit-learn {summary['environment']['sklearn']},
pandas {summary['environment']['pandas']}, NumPy {summary['environment']['numpy']} en joblib
{summary['environment']['joblib']}. Modelhash: `{summary['selected_full_refit_artifact']['sha256']}`.
Voor het laden van het artifact moet `secondary_mushroom` op het Python-importpad staan;
de pipeline verwijst waar nodig naar de bewaarde schoonmaakfunctie in `compare_andrew_models`.
Bewaar deze bestanden en gebruik dezelfde dependencies bij reproductie.
'''


def notebook_content(report):
    cells = [nbformat.v4.new_markdown_cell(report.split('## 1.', 1)[0])]
    def md(s): cells.append(nbformat.v4.new_markdown_cell(s))
    def code(s): cells.append(nbformat.v4.new_code_cell(s))
    md('''## Resultaten laden en hertraining kiezen

De volledige pipeline wordt per fold uitsluitend op trainingssoorten gefit. Standaard
laden we de opgeslagen evaluaties; zet `RUN_TRAINING=True` voor uitvoering van het
trainingsscript. Gebruik de gepinde omgeving. Na een nieuwe run moet de rapportgenerator
opnieuw draaien zodat tekst en resultaten dezelfde versie weergeven.''')
    code('''from pathlib import Path
import json, sys, hashlib
import numpy as np
import pandas as pd
from IPython.display import display, Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'SolutionJorre/MushroomDataset/aws_results.json').exists())
FOLDER = ROOT / 'secondary_mushroom'
sys.path.insert(0, str(FOLDER))
from compare_species_models import run, NAMES, ORDER
OUT = FOLDER / 'comparison_species'
RUN_TRAINING = False
if RUN_TRAINING:
    run()
summary = json.loads((OUT / 'comparison_summary.json').read_text(encoding='utf-8'))
metrics = pd.read_csv(OUT / 'model_comparison.csv', float_precision='round_trip').set_index('model')
folds = pd.read_csv(OUT / 'cv_folds.csv', float_precision='round_trip')
split = pd.read_csv(OUT / 'split_manifest.csv')
predictions = pd.read_csv(OUT / 'oof_predictions.csv', float_precision='round_trip')
errors = pd.read_csv(OUT / 'selected_error_records.csv', float_precision='round_trip')
species_errors = pd.read_csv(OUT / 'species_errors.csv')
display(pd.Series(summary['environment']))''')
    sections = report.split('\n## ')
    for section in sections[1:]:
        title = section.split('\n', 1)[0]
        md('## ' + section)
        if title.startswith('1.'):
            code('''ranking = metrics.drop(index='majority').sort_values(['cv_balanced_accuracy_mean', 'cv_f1_poisonous_mean', 'cv_recall_poisonous_mean'], ascending=False, kind='stable')
assert ranking.index[0] == summary['selected_model']
print('CV-ontwikkelkeuze:', NAMES[summary['selected_model']])
display(ranking[['cv_balanced_accuracy_mean', 'cv_balanced_accuracy_std', 'cv_f1_poisonous_mean', 'cv_recall_poisonous_mean']])''')
        elif title.startswith('2.'):
            code('''assert len(split) == 60923 and split.unique_row.is_unique and split.source_row.is_unique
assert split.species_group.nunique() == 173
assert split.groupby('species_group').validation_fold.nunique().eq(1).all()
assert (split.species_group == split.source_row // 353).all()
assert set(split.validation_fold) == {1, 2, 3, 4, 5}
for fold in range(1, 6):
    assert not set(split.loc[split.validation_fold == fold, 'species_group']) & set(split.loc[split.validation_fold != fold, 'species_group'])
pd.testing.assert_frame_equal(split, predictions[split.columns])
assert len(folds) == 50 and folds.groupby('model').size().eq(5).all()
assert set(metrics.index) == set(ORDER)
display(pd.DataFrame(summary['fold_membership']))
display(pd.crosstab(split.validation_fold, split.true_class))
display(pd.Series(summary['group_verification']))''')
        elif title.startswith('3.'):
            code('''assert len(summary['dataset']['features']) == 20
assert not set(summary['dataset']['features']) & {'class', 'name', 'source_row', 'species_group', 'unique_row'}
display(pd.DataFrame(summary['classifier_parameters']).T)
display(pd.read_csv(ROOT / 'SolutionJorre/MushroomDataset/aws_tuning_results.csv'))''')
        elif title.startswith('4.'):
            code('''y = (predictions.true_class == 'p').astype(int)
functions = {'accuracy': accuracy_score, 'balanced_accuracy': balanced_accuracy_score,
             'precision_poisonous': precision_score, 'recall_poisonous': recall_score, 'f1_poisonous': f1_score}
for key, row in metrics.iterrows():
    labels = (predictions[key + '_prediction'] == 'p').astype(int)
    proba = predictions[key + '_probability_p']
    for name, function in functions.items():
        value = function(y, labels) if name in ['accuracy', 'balanced_accuracy'] else function(y, labels, zero_division=0)
        assert abs(value - row[name]) < 1e-12
    assert abs(roc_auc_score(y, proba) - row.roc_auc) < 1e-12
    assert abs(average_precision_score(y, proba) - row.average_precision) < 1e-12
    assert confusion_matrix(y, labels, labels=[0, 1]).ravel().tolist() == [int(row[c]) for c in ['tn', 'fp', 'fn', 'tp']]
    for fold in range(1, 6):
        take = split.validation_fold == fold
        saved = folds[(folds.model == key) & (folds.fold == fold)].iloc[0]
        for name, function in functions.items():
            value = function(y[take], labels[take]) if name in ['accuracy', 'balanced_accuracy'] else function(y[take], labels[take], zero_division=0)
            assert abs(value - saved[name]) < 1e-12
    for name in ['accuracy', 'balanced_accuracy', 'recall_poisonous', 'f1_poisonous']:
        assert abs(folds.loc[folds.model == key, name].mean() - row['cv_' + name + '_mean']) < 1e-12
display(metrics.loc[[*ranking.index, 'majority'], ['accuracy', 'balanced_accuracy', 'precision_poisonous', 'recall_poisonous', 'f1_poisonous', 'roc_auc', 'average_precision', 'fn', 'fp']])
display(folds.pivot(index='fold', columns='model', values='balanced_accuracy'))
display(Image(filename=str(OUT / 'metrics.png')))''')
        elif title.startswith('5.'):
            code('''selected = summary['selected_model']
assert len(errors) == metrics.loc[selected, 'fn'] + metrics.loc[selected, 'fp']
wrong = predictions[predictions[selected + '_prediction'] != predictions.true_class]
assert set(errors.unique_row) == set(wrong.unique_row)
assert (errors.predicted_class != errors.true_class).all()
display(metrics[['tn', 'fp', 'fn', 'tp']].astype(int))
display(Image(filename=str(OUT / 'confusion_matrices.png')))
display(errors[errors.true_class == 'p'].sort_values('probability_p').head(3))
lookup = pd.read_csv(FOLDER / 'comparison_team/species_lookup.csv')
by_species = species_errors[species_errors.model == selected].merge(lookup[['species_group','name']], on='species_group')
assert by_species.fn.sum() == metrics.loc[selected,'fn']
assert by_species.fp.sum() == metrics.loc[selected,'fp']
by_species['fn_fraction'] = by_species.fn / by_species.records
display(by_species[by_species['class'] == 'p'].sort_values(['fn_fraction','fn'], ascending=False).head(5))''')
        elif title.startswith('6.'):
            code('''display(folds.groupby('model')[['fit_seconds','train_accuracy','accuracy','train_f1_poisonous','f1_poisonous']].mean())
display(pd.Series({k: sum(v.values()) for k,v in summary['warnings'].items()}, name='aantal waarschuwingen'))
display(summary['warnings'])''')
        elif title.startswith('7.'):
            code('''artifact = OUT / summary['selected_full_refit_artifact']['path']
assert hashlib.sha256(artifact.read_bytes()).hexdigest() == summary['selected_full_refit_artifact']['sha256']
assert summary['selected_full_refit_artifact']['training_rows'] == 60923
assert not summary['protocol']['untouched_final_test']
print('Keuze is ontwikkelvalidatie; de full-data pipeline is geen nieuwe testscore.')
display(pd.Series(summary['selected_full_refit_artifact']))''')
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {'name': 'python3', 'display_name': 'Python 3 (comparison environment)', 'language': 'python'}
    notebook.metadata.language_info = {'name': 'python', 'version': '3.13.7'}
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    summary = json.loads((OUT / 'comparison_summary.json').read_text(encoding='utf-8'))
    metrics = pd.read_csv(OUT / 'model_comparison.csv', float_precision='round_trip').set_index('model')
    folds = pd.read_csv(OUT / 'cv_folds.csv', float_precision='round_trip')
    errors = pd.read_csv(OUT / 'selected_error_records.csv', float_precision='round_trip')
    per_species = pd.read_csv(OUT / 'species_errors.csv')
    report = report_text(summary, metrics, folds, errors, per_species)
    (FOLDER / '09_vergelijkingsrapport.md').write_text(report, encoding='utf-8')
    notebook = notebook_content(report)
    if args.execute:
        env = dict(os.environ)
        env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env['PATH']
        NotebookClient(notebook, timeout=180, kernel_name='python3',
            resources={'metadata': {'path': str(FOLDER)}}).execute(env=env)
    nbformat.validate(notebook)
    nbformat.write(notebook, FOLDER / '09_compare_models.ipynb')
    print('Assignment comparison report and notebook written; executed:', args.execute)


if __name__ == '__main__':
    main()
