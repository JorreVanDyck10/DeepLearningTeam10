"""Build the final course comparison from the uploaded CSV and executed PyCaret run."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib.pyplot as plt
import nbformat
from nbclient import NotebookClient
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix
from MushroomFinal.compare_correct_dataset import OUT,NAMES,FOLDER,load_data,digest,write_json


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
        *['| '+' | '.join(str(x) for x in row)+' |' for row in rows]])


def build_report():
    summary=json.loads((OUT/'comparison_summary.json').read_text(encoding='utf-8'))
    metrics=pd.read_csv(OUT/'model_comparison.csv',float_precision='round_trip').set_index('model')
    pc=pd.read_csv(OUT/'pycaret_all.csv',index_col=0)
    three=pd.read_csv(OUT/'pycaret_three.csv',index_col=0)
    pctest=pd.read_csv(OUT/'pycaret_test_metrics.csv')
    pc_audit=json.loads((OUT/'pycaret_audit.json').read_text(encoding='utf-8'))
    aws=json.loads((OUT/'aws_artifact_audit.json').read_text(encoding='utf-8')) if (OUT/'aws_artifact_audit.json').exists() else None
    if aws:
        assert aws['dataset']==summary['protocol']['dataset']['sha256_lf']
        assert aws['cv_macro_f1']<=metrics.loc[summary['selected_model'],'cv_macro_f1_mean']
        tuning=pd.read_csv(ROOT/'SolutionJorre/MushroomDataset/correct_dataset/correct_aws_tuning_results.csv')
        aws_table=table(['Bomen','Max depth','Min leaf','CV macro F1','SD','Rank'],
            [[int(r.param_classifier__n_estimators),'Geen limiet' if pd.isna(r.param_classifier__max_depth) else int(r.param_classifier__max_depth),
              int(r.param_classifier__min_samples_leaf),f'{r.mean_test_score:.6f}',f'{r.std_test_score:.6f}',int(r.rank_test_score)]
             for _,r in tuning.sort_values('rank_test_score').iterrows()])
        am=aws['recomputed_metrics']
        aws_section=f'''### Nieuwe uitgevoerde SageMaker-run op het juiste bestand

Jorre heeft de training en vierkandidatentuning opnieuw op SageMaker uitgevoerd.
De [ongewijzigde exports](../SolutionJorre/MushroomDataset/correct_dataset/) bevatten
het uitgevoerde notebook, de volledige pipeline, tuninglogs en resultaten.
De bronhash, elf features, 4.100/900-verdeling en scikit-learn-versie zijn gecontroleerd.
Het model is apart geladen en alle testmetrics zijn uit zijn voorspellingen herberekend.

{aws_table}

SD is hier de populatiestandaardafwijking over vijf folds, zoals GridSearchCV exporteert.
De beste instelling is **100 bomen, geen depth-limiet en leaf 1**: de baseline wint.
De search toont dus geen verbetering boven die baseline; de eerder getunede instelling
depth 24/leaf 2 scoort lager. CV macro F1 is **{aws['cv_macro_f1']:.6f}**.

Op dezelfde 900 testrecords: **{am['accuracy']:.2%} accuracy**, macro F1
**{am['macro_f1']:.4f}**, **{am['recall_p']:.2%} recall p**, **{am['fn']} FN** en **{am['fp']} FP**.
De confusion matrix is `[[{am['tn']}, {am['fp']}], [{am['fn']}, {am['tp']}]]`, met e/p als rij- en kolomvolgorde.
De opgeslagen SageMaker-pipeline en de lokale AWS-baseline verschillen in
**{aws['local_baseline_label_disagreements']} testlabels**; het maximale p-scoreverschil is
{aws['local_baseline_max_probability_difference']:.3g}. Ook alle vijf CV macro-F1-scores
komen overeen met de lokale baseline. Dat is bevestiging van dezelfde configuratie,
geen extra onafhankelijke modelverbetering of tiende configuratie in de ranking.

SageMaker: Python {aws['aws_versions']['python']}, sklearn {aws['aws_versions']['sklearn']};
de lokale audit gebruikt Python {aws['audit_versions']['python']} met dezelfde sklearn,
pandas, NumPy en joblib. Het exportbestand is niet opnieuw getraind of gewijzigd.
Het rijlidmaatschap is uit de uitgevoerde splitcode gereconstrueerd; een afzonderlijk
oorspronkelijk cloud-trainingsmanifest is niet geëxporteerd.
'''
    else:
        aws_section='De nieuwe SageMaker-export moet nog worden toegevoegd; de AWS-rijen zijn lokale hertrainingen.'
    exclusion_note=('LightGBM is niet in de ranking opgenomen: deze Windows-omgeving geeft bij '
        'fitten een native OSError/access violation. Dit is afzonderlijk gecontroleerd en '
        'vastgelegd in pycaret_exclusions.json; het is geen slechte modelscore. '
        'De modelkeuze geldt voor de succesvol geëvalueerde kandidaten.') if pc_audit.get('not_returned_by_compare_models') else ''
    predictions=pd.read_csv(OUT/'test_predictions.csv',float_precision='round_trip')
    selected=summary['selected_model']; row=metrics.loc[selected]
    rank=metrics.drop(index='dummy').sort_values(['cv_macro_f1_mean','cv_recall_p_mean'],ascending=False,kind='stable')
    assert row.cv_macro_f1_mean>float(pc.iloc[0]['Macro F1']), 'Reconsider overall choice: AutoML scores higher.'
    write_json(OUT/'overall_selection.json',{'selected_model':selected,'source':'fixed candidate pipelines',
        'mean_cv_macro_f1':float(row.cv_macro_f1_mean),'best_pycaret_cv_macro_f1':float(pc.iloc[0]['Macro F1']),
        'criterion':'training CV macro F1; test not used in the automatic ranking',
        'test_previously_known':True,'selection_is_provisional':True,
        'correct_aws_export_included':bool(aws),'correct_aws_cv_macro_f1':aws['cv_macro_f1'] if aws else None})
    ordered=[*rank.index,'dummy']
    cvtable=table(['Configuratie','CV macro F1 ± SD','CV recall p','CV balanced acc.'],
        [[NAMES[k],f'{metrics.loc[k,"cv_macro_f1_mean"]:.4f} ± {metrics.loc[k,"cv_macro_f1_std"]:.4f}',
          f'{metrics.loc[k,"cv_recall_p_mean"]:.4f}',f'{metrics.loc[k,"cv_balanced_accuracy_mean"]:.4f}'] for k in ordered])
    testtable=table(['Configuratie','Accuracy','Macro F1','Precision p','Recall p','F1 p','ROC-AUC','AP','FN','FP'],
        [[NAMES[k],*[f'{metrics.loc[k,c]:.4f}' for c in ['accuracy','macro_f1','precision_p','recall_p','f1_p','roc_auc','average_precision']],
            int(metrics.loc[k,'fn']),int(metrics.loc[k,'fp'])] for k in ordered])
    pctable=table(['PyCaret-model','CV accuracy','CV macro F1','CV recall p'],
        [[r['Model'],f'{r.Accuracy:.4f}',f'{r["Macro F1"]:.4f}',f'{r.Recall:.4f}'] for _,r in pc.iterrows()])
    times=table(['Configuratie','Train accuracy','Test accuracy','Gem. fit/fold (s)'],
        [[NAMES[k],f'{metrics.loc[k,"train_accuracy"]:.4f}',f'{metrics.loc[k,"accuracy"]:.4f}',
          f'{metrics.loc[k,"mean_fold_fit_seconds"]:.2f}'] for k in ordered])
    errors=pd.read_csv(OUT/'selected_errors.csv')
    bad=errors[errors.true_class==1].sort_values('probability_p').head(3)
    errortable=table(['Bronrij','Werkelijk → voorspeld','Score p','Cap diameter','Steelhoogte','Gill color'],
        [[int(r.source_row),'p → e',f'{r.probability_p:.4f}',r.cap_diameter,r.stem_height,r.gill_color] for _,r in bad.iterrows()])
    own=(predictions[selected+'_prediction']==predictions.true_class)
    other=(predictions.rf_500_previous_prediction==predictions.true_class)
    only_own=int((own&~other).sum());only_other=int((other&~own).sum())
    pvalue=float(binomtest(only_own,only_own+only_other,.5).pvalue) if only_own+only_other else 1.
    fig,ax=plt.subplots(figsize=(10,5))
    positions=np.arange(len(ordered))
    for off,col,label,color in [(-.23,'macro_f1','Macro F1','#315e91'),(0,'recall_p','Recall p','#b85c4a'),(.23,'accuracy','Accuracy','#4b8062')]:
        ax.barh(positions+off,metrics.loc[ordered,col],height=.21,label=label,color=color)
    ax.set_yticks(positions,[NAMES[k] for k in ordered]);ax.invert_yaxis();ax.set_xlim(0,1)
    ax.set_title('Juiste cursusbestand: dezelfde 900 testrecords');ax.legend(loc='lower right')
    fig.tight_layout();fig.savefig(OUT/'metrics.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(11,9))
    for ax,k in zip(axes.flat,metrics.index):
        cm=confusion_matrix(predictions.true_class,predictions[k+'_prediction'],labels=[0,1])
        ax.imshow(cm,cmap='Blues',vmin=0,vmax=560)
        for i in range(2):
            for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>280 else 'black')
        ax.set_xticks([0,1],['e','p']);ax.set_yticks([0,1],['e','p']);ax.set_title(NAMES[k],fontsize=8)
        ax.set_xlabel('Voorspeld');ax.set_ylabel('Werkelijk')
    fig.tight_layout();fig.savefig(OUT/'confusion_matrices.png',dpi=150);plt.close(fig)
    pcrow=pctest.iloc[0]
    report=f'''# Definitieve Mushroom-vergelijking — het juiste cursusbestand

**DeepLearningTeam10 · bijgewerkt 14 oktober 2026**

Bijdragen: Andrew Noeyens maakte het finale Mushroom-notebook, de modelkeuzes en
tuning. Jorre Van Dyck voerde de gecorrigeerde AWS-training en tuning uit. Codex heeft op verzoek van Jorre
de dataset gecorrigeerd, de gezamenlijke hertraining uitgevoerd, PyCaret gerepareerd
en daadwerkelijk uitgevoerd, en deze vergelijking gemaakt. Het team moet de inhoud
reviewen en zelf kunnen uitleggen. Hiermee is het gebruik van generatieve AI vermeld.

## 1. Besluit en correctie van het eerdere rapport

**Voorlopige keuze op de juiste data: {NAMES[selected]}.** Gemiddelde trainings-CV
macro F1 is **{row.cv_macro_f1_mean:.4f}**, met SD {row.cv_macro_f1_std:.4f}.
Op de 900 gedeelde testrecords: **{row.accuracy:.2%} accuracy**, **macro F1
{row.macro_f1:.4f}**, **recall p {row.recall_p:.2%}**, **{int(row.fn)} FN** en
**{int(row.fp)} FP**. De hoogste PyCaret-score is {float(pc.iloc[0]['Macro F1']):.4f}
macro F1 bij {pc.iloc[0]['Model']}; dit zijn standaardinstellingen, geen getunede RF300.

De eerdere vergelijking op 61.069 UCI-rijen en de daarop gebaseerde soorten-CV zijn
**vervallen als grondslag voor deze inlevering**. Hun modelkeuze en scores mogen niet
als eindresultaat worden gebruikt. Het geüploade bestand met 5.000 rijen is de bron.
De AWS-score van 100% op UCI zegt niets over deze aangeleverde dataset.

Dit is ontwikkelvalidatie: de testset is in Andrew's notebook al bekeken. De CV-keuze
wordt hier automatisch vastgelegd voordat nieuwe testvoorspellingen worden berekend,
maar de hele onderzoeksprocedure is daardoor niet alsnog geblindeerd.

## 2. Dataset, features en gelijke splits

Bron: [MushroomFinal/mushroom_project_dataset.csv](../MushroomFinal/mushroom_project_dataset.csv),
5.000 rijen, 3.107 `e` en 1.893 `p`, zonder exacte duplicaten. Het CSV-bestand kan met
Excel worden geopend; de repository bevat hiervoor geen afzonderlijk xlsx-bestand.
LF-genormaliseerde SHA-256: `{summary['protocol']['dataset']['sha256_lf']}`.

`jumbled_noise_0` wordt vóór training verwijderd. **`jumbled_noise_1` blijft behouden**,
zoals in de finale modelcode. De drie numerieke en acht categorische invoervelden zijn:
`{', '.join(summary['protocol']['features'])}`. Namen gebruiken underscores. De doelkolom
en rij-index worden niet meegegeven als feature. Dezelfde CSV staat ook in de oudere
Andrew-map; het probleem was de overschakeling naar de volledige UCI-download.

Alle handmatige kandidaten en PyCaret gebruiken exact dezelfde 4.100 trainingsrijen
en 900 testrijen: 82/18, gestratificeerd, seed 42. De testset heeft 559 `e` en 341 `p`.
Trainings-CV gebruikt vijf identieke gestratificeerde folds, shuffle en seed 42.
Het manifest bewaart de oorspronkelijke rij, trainingsvolgorde en validatiefold.
Er is geen betrouwbare soort-ID: bronrij delen door 353 zou voor deze aangepaste
dataset ongeldig zijn en wordt niet gedaan.

**Keuzeregel:** gemiddelde CV macro F1; bij exacte gelijkstand CV recall p en vaste
kandidaatvolgorde. Macro F1 weegt beide klassen gelijk. Precision, recall, FN, FP,
balanced accuracy, ROC-AUC en AP blijven zichtbaar. De dummy is alleen een referentie.
Er is geen nieuwe hyperparameter- of drempelkeuze op de testset.

## 3. Welke modellen zijn vergeleken?

| Configuratie | Belangrijkste instellingen en preprocessing |
| --- | --- |
| Final RF300 | Uit finale notebook: 300 bomen, depth 20, max features None, split 4, leaf 1, balanced; mediaan/modus + one-hot |
| Final Gradient Boosting | 200 bomen, learning rate 0,1, depth 3, subsample 0,8; extra proximity-imputatie voor spore_print_color |
| Final Logistic Regression | C=10, L1, liblinear, balanced; proximity-imputatie, mediaan/modus, scaling + one-hot |
| Eerdere RF500 | 500 bomen, depth 20, split 3, balanced; nulmetingen steel → ontbrekend, mediaan/modus + one-hot |
| RF200 API-configuratie | 200 bomen, geen depth-limiet, balanced; dezelfde steel-schoonmaak |
| AWS-configuratie getuned | Lokale nieuwe fit: 100 bomen, depth 24, leaf 2; numerieke mediaan, categorisch missing + one-hot |
| AWS-configuratie baseline | Lokale nieuwe fit: 100 bomen, geen depth-limiet, leaf 1; dezelfde AWS-preprocessing |
| Beslisboom baseline | Depth 5, mediaan/modus + one-hot |
| Meerderheidsreferentie | Voorspelt steeds de meest voorkomende trainingsklasse e |

De finale hyperparameters komen uit de bestaande notebookoutputs; de oorspronkelijke
grids zijn niet opnieuw doorzocht. De nu gerapporteerde scores horen bij **nieuwe
fits in één gemeenschappelijke omgeving**, niet bij de drie bestaande pkl-bestanden.
Oudere configuraties die door finale varianten zijn vervangen, blijven in de historische
notebooks beschikbaar. We vergelijken complete pipelines, geen geïsoleerd causaal
effect van één classifier of schoonmaakstap.

**Correctie bij proximity-imputatie.** In het oorspronkelijke notebook werd een
unsupervised forest op de volledige trainingsset geleerd vóór de CV. Daardoor kon
een validatiefold de imputatie beïnvloeden. Hier zit de volledige proximity-imputatie
binnen de pipeline en wordt zij opnieuw op uitsluitend de trainingsfold geleerd.
Referentiekleuren en vergelijkingsbomen zien geen test- of validatierecords.
De RF300 gebruikt gewone imputatie, omdat de finale RF in de broncode op `X_train`
wordt gefit, terwijl GB/LR op de extra geïmputeerde invoer worden gefit.

## 4. Gezamenlijke resultaten

### Trainings-CV: basis van de keuze

{cvtable}

De RF300 ligt dicht bij de eerdere RF500. SD is spreiding over vijf folds, geen
betrouwbaarheidsinterval of bewijs dat de eerste rij statistisch superieur is.

### Dezelfde 900 testrecords

{testtable}

![Gedeelde testmetrics](../MushroomFinal/comparison_correct/metrics.png)

De meerderheid voorspellen haalt 62,11% accuracy maar mist alle 341 p-records.
Dit maakt duidelijk waarom alleen accuracy onvoldoende is. De gekozen RF300 mist
{int(row.fn)} van 341 p-records ({1-row.recall_p:.2%}). De ROC-AUC/AP beoordelen ranking
van modelscores; een onderzoek naar kanskalibratie is niet uitgevoerd.

{aws_section}

Het oorspronkelijke notebook meldde 82,00% / 78,11% / 68,67% voor RF/GB/LR.
De gezamenlijke nieuwe fits geven respectievelijk {metrics.loc['rf_final','accuracy']:.2%},
{metrics.loc['gb_final','accuracy']:.2%} en {metrics.loc['lr_final','accuracy']:.2%}.
Dat verschil wordt behouden. De omgeving en voor GB/LR de validatieprocedure verschillen;
zonder afzonderlijke ablation schrijven we niet ieder verschil aan één oorzaak toe.

## 5. PyCaret: daadwerkelijk uitgevoerd

PyCaret **3.3.2** werkt in de aparte **Python 3.11**-omgeving met scikit-learn 1.4.2.
Het oorspronkelijke `positive_label`-argument is verwijderd; labels zijn expliciet
`0=e, 1=p`. Een eigen metricfunctie dwingt macro-averaging af: alleen
`add_metric(..., average='macro')` bleek hier een binaire F1-kolom te produceren.
De aangepaste scorer is gecontroleerd met een meerderheidsreferentie.

Eerst zijn LR, RF en GB vergeleken; **{three.iloc[0]['Model']}** scoort het hoogst.
Daarna zijn {len(pc)} snelle modelfamilies met de standaardinstellingen succesvol
vergeleken op dezelfde trainingsrijen en dezelfde folds:

{pctable}

{exclusion_note}

De hoogste PyCaret-familie haalt op de gedeelde testset {pcrow.accuracy:.2%} accuracy,
macro F1 {pcrow.macro_f1:.4f}, recall p {pcrow.recall_p:.2%} en {int(pcrow.fn)} FN.
De keuze is gemaakt op CV, niet op deze testscore. PyCaret-defaults zijn geen vervanging
van de al getunede modelinstellingen; beide experimenten staan daarom afzonderlijk.
Het notebook [mushroom_project.ipynb](../MushroomFinal/mushroom_project.ipynb) bevat nu
uitgevoerde PyCaret-cellen, resultaten en de ontbrekende conclusie.

## 6. Foutanalyse, training en praktische keuze

![Confusion matrices](../MushroomFinal/comparison_correct/confusion_matrices.png)

FN is p voorspeld als e; FP is e voorspeld als p. Drie relatief overtuigde FN-gevallen
van het gekozen model, met bronkenmerken:

{errortable}

Alle fouten, met alle elf kenmerken, staan in
[selected_errors.csv](../MushroomFinal/comparison_correct/selected_errors.csv).
Dit zijn concrete probleemgevallen; zij bewijzen geen oorzakelijk effect van een
afzonderlijk kenmerk. De index wordt alleen gebruikt om de rij terug te vinden.

Tegenover de eerdere RF500 zijn {only_own} records alleen door RF300 correct en
{only_other} alleen door RF500 correct. Een exploratieve exacte McNemar-toets geeft
p={pvalue:.4f}; dit is geen correctie voor modelselectie of herhaald testen.
We claimen geen algemeen bewezen winnaar op basis van een klein verschil.

{times}

Tijden zijn lokale metingen, geen AWS-kosten of website-latency. De proximity-stap wordt
voor identieke trainingsfolds gecachet; de timings zijn dus geen zuivere vergelijking
van algoritmische complexiteit. RF300 heeft een duidelijke train-testkloof; de
test-/CV-uitkomsten blijven de relevante prestaties, niet de trainingsaccuracy.

## 7. Modelbestand, AWS en website

De gekozen complete pipeline staat in
[selected_pipeline.joblib](../MushroomFinal/comparison_correct/selected_pipeline.joblib).
Zij is op de 4.100 trainingsrijen gefit en gebruikt elf underscore-veldnamen, met
labels `0=e, 1=p`. De testrecords zijn niet in dit artifact meegetraind.
Gebruik de gepinde omgeving en bewaar `MushroomFinal/final_model_support.py` op het
Python-importpad voor pipelines met extra schoonmaak/imputatie.

API en website zijn gekoppeld aan een verliesvrije export van de gekozen RF300.
De export bevat dezelfde preprocessing, splits en bladwaarden; alle 900 opgeslagen
testvoorspellingen komen overeen. Het formulier gebruikt elf underscore-velden,
zonder `jumbled_noise_0`, en controleert de model-ID `mushroom_final_rf300`.
Zie [RF300-deployment en exportaudit](../docs/mushroom_rf300.md).
De migratie is afzonderlijk uitgevoerd na de modelvergelijking; de preprocessing
en labelmapping van het gekozen artifact blijven daarbij behouden.

**AWS-status:** de nieuwe SageMaker-training en tuning op het juiste bestand zijn
uitgevoerd en de exports zijn gecontroleerd. Zie de nieuwe AWS-sectie hierboven en het
[uitgevoerde notebook](../SolutionJorre/MushroomDataset/correct_dataset/04_aws_correct_dataset.ipynb).
De bestaande twee AWS-rijen blijven herkenbaar als lokale fits; de echte SageMaker-
uitkomst is apart gecontroleerd en komt overeen met de baseline-rij. De oude 100%-run
op UCI is uitsluitend historisch. RF300 blijft de voorlopige keuze op CV macro F1.

Er is nog geen onaangeraakte eindtest, noch een test op volledig ongeziene soorten of
echte veldmetingen. Teamreview en controle van de hosting
blijven afzonderlijke onderdelen voor de eindinlevering.

## 8. Reproduceren en bewijs

```powershell
py -3.11 -m venv .venv-comparison
.\\.venv-comparison\\Scripts\\python.exe -m pip install -r MushroomFinal/requirements-comparison.txt
.\\.venv-comparison\\Scripts\\python.exe MushroomFinal/compare_correct_dataset.py --stage manual
.\\.venv-comparison\\Scripts\\python.exe MushroomFinal/execute_pycaret_notebook.py
.\\.venv-comparison\\Scripts\\python.exe MushroomFinal/audit_correct_aws.py
.\\.venv-comparison\\Scripts\\python.exe MushroomFinal/build_correct_comparison.py --execute
```

| Bestand | Bewijs |
| --- | --- |
| [09_compare_models.ipynb](09_compare_models.ipynb) | Hoofdvergelijking volgens de opdracht, met uitgevoerde controles |
| [protocol.json](../MushroomFinal/comparison_correct/protocol.json) | Juiste bronhash, features en vooraf vastgelegde keuzecriteria |
| [split_manifest.csv](../MushroomFinal/comparison_correct/split_manifest.csv) | Iedere bronrij, split en CV-fold |
| [cv_folds.csv](../MushroomFinal/comparison_correct/cv_folds.csv) | 45 handmatige model/fold-evaluaties |
| [model_comparison.csv](../MushroomFinal/comparison_correct/model_comparison.csv) | Alle gezamenlijke metrics |
| [oof_predictions.csv](../MushroomFinal/comparison_correct/oof_predictions.csv) | Validatievoorspellingen op de 4.100 trainingsrijen |
| [test_predictions.csv](../MushroomFinal/comparison_correct/test_predictions.csv) | Alle negen kandidaten op dezelfde 900 testrecords |
| [pycaret_all.csv](../MushroomFinal/comparison_correct/pycaret_all.csv) | De uitgevoerde AutoML-vergelijking |
| [pycaret_audit.json](../MushroomFinal/comparison_correct/pycaret_audit.json) | Exacte PyCaret-rijen, versies, bronhash en scorercontrole |
| [comparison_summary.json](../MushroomFinal/comparison_correct/comparison_summary.json) | Parameters, omgeving, artifacthash en waarschuwingen |
| [overall_selection.json](../MushroomFinal/comparison_correct/overall_selection.json) | Keuze inclusief controle tegen de beste PyCaret-score |
| [aws_artifact_audit.json](../MushroomFinal/comparison_correct/aws_artifact_audit.json) | Controle van de nieuwe echte AWS-export |
| [aws_artifact_predictions.csv](../MushroomFinal/comparison_correct/aws_artifact_predictions.csv) | Alle 900 opnieuw berekende AWS-voorspellingen |

Oude UCI-resultaten blijven voor herkomstcontrole in `comparison_team` en
`comparison_species`, maar zijn expliciet historisch en tellen niet als eindvergelijking.
'''
    return report


def notebook_content(report):
    cells=[nbformat.v4.new_markdown_cell(report)]
    cells.append(nbformat.v4.new_markdown_cell('''## Uitgevoerde controle van bron, splits en metrics

Deze cellen laden de opgeslagen werkelijke trainingsresultaten. Ze controleren de
bronhash, de verwijderde feature, de gedeelde splits en herberekenen alle CV- en
testmetrics uit individuele voorspellingen. Hertraining gebeurt met de scripts hierboven.'''))
    cells.append(nbformat.v4.new_code_cell('''from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
from IPython.display import display,Image
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'MushroomFinal/mushroom_project_dataset.csv').exists())
sys.path.insert(0,str(ROOT))
from MushroomFinal.compare_correct_dataset import OUT,load_data,evaluate,NAMES,digest
raw,X,y,train,test,folds=load_data()
summary=json.loads((OUT/'comparison_summary.json').read_text(encoding='utf-8'))
protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
manifest=pd.read_csv(OUT/'split_manifest.csv')
metrics=pd.read_csv(OUT/'model_comparison.csv',float_precision='round_trip').set_index('model')
cv=pd.read_csv(OUT/'cv_folds.csv',float_precision='round_trip')
oof=pd.read_csv(OUT/'oof_predictions.csv',float_precision='round_trip')
predictions=pd.read_csv(OUT/'test_predictions.csv',float_precision='round_trip')
assert protocol['dataset']['sha256_lf']==digest(ROOT/protocol['source'])['sha256_lf']
assert X.shape==(5000,11) and 'jumbled_noise_0' not in X and 'jumbled_noise_1' in X
assert not set(train)&set(test)
assert predictions.source_row.tolist()==test.tolist() and oof.source_row.tolist()==train.tolist()
assert manifest.source_row.is_unique and len(manifest)==5000
assert manifest.loc[manifest.split=='test','cv_validation_fold'].isna().all()
assert len(cv)==45
for fold,(tr,val) in enumerate(folds,1):
    assert not set(train[tr])&set(train[val])
    assert set(manifest.loc[manifest.cv_validation_fold==fold,'source_row'])==set(train[val])
display(pd.crosstab(manifest.split,manifest['class']))
display(pd.Series(summary['environment']))'''))
    cells.append(nbformat.v4.new_markdown_cell('''De volgende controle herberekent de metrics voor iedere kandidaat, zowel op test
als per CV-fold. Zo wordt de tabel gecontroleerd tegen de werkelijke voorspellingen.'''))
    cells.append(nbformat.v4.new_code_cell('''for key,row in metrics.iterrows():
    actual=evaluate(predictions.true_class,predictions[key+'_prediction'],predictions[key+'_probability_p'])
    for name,value in actual.items(): assert abs(value-row[name])<1e-12,(key,name)
    for fold,(_,val) in enumerate(folds,1):
        actual=evaluate(oof.true_class.iloc[val],oof[key+'_prediction'].iloc[val],oof[key+'_probability_p'].iloc[val])
        saved=cv[(cv.model==key)&(cv.fold==fold)].iloc[0]
        for name,value in actual.items(): assert abs(value-saved[name])<1e-12,(key,fold,name)
    assert abs(cv.loc[cv.model==key,'macro_f1'].mean()-row.cv_macro_f1_mean)<1e-12
ranking=metrics.drop(index='dummy').sort_values(['cv_macro_f1_mean','cv_recall_p_mean'],ascending=False,kind='stable')
assert ranking.index[0]==summary['selected_model']
display(metrics[['cv_macro_f1_mean','cv_macro_f1_std','accuracy','macro_f1','recall_p','fn','fp']])
display(Image(filename=str(OUT/'metrics.png')))
display(Image(filename=str(OUT/'confusion_matrices.png')))'''))
    cells.append(nbformat.v4.new_markdown_cell('''Controleer dat PyCaret exact dezelfde bron en rijen gebruikt, en dat macro F1
daadwerkelijk beide klassen weegt. De automatische keuze mag niet alleen op testaccuracy rusten.'''))
    cells.append(nbformat.v4.new_code_cell('''audit=json.loads((OUT/'pycaret_audit.json').read_text(encoding='utf-8'))
pc=pd.read_csv(OUT/'pycaret_all.csv',index_col=0)
assert audit['train_source_rows']==train.tolist() and audit['test_source_rows']==test.tolist()
assert audit['dataset']['sha256_lf']==protocol['dataset']['sha256_lf']
assert pc.loc['dummy','Macro F1']>.3 and pc.loc['dummy','F1']==0
assert abs(pc.loc['dummy','Macro F1']-metrics.loc['dummy','cv_macro_f1_mean'])<1e-9
assert ranking.iloc[0].cv_macro_f1_mean>pc.iloc[0]['Macro F1']
display(pd.read_csv(OUT/'pycaret_three.csv',index_col=0))
display(pc)
display(pd.read_csv(OUT/'pycaret_test_metrics.csv'))
errors=pd.read_csv(OUT/'selected_errors.csv')
key=summary['selected_model']
assert len(errors)==metrics.loc[key,'fn']+metrics.loc[key,'fp']
display(errors[errors.true_class==1].sort_values('probability_p').head(3))
print('Keuze:',NAMES[key],'; de echte SageMaker-export is daarnaast afzonderlijk gecontroleerd.')'''))
    if (OUT/'aws_artifact_audit.json').exists():
        cells.append(nbformat.v4.new_markdown_cell('''## Controle van het echte AWS-artifact

Deze cel laadt de ontvangen SageMaker-pipeline opnieuw en controleert alle 900 labels,
p-scores, metrics, tuningfolds en de vergelijking met de lokale baseline. Er vindt
geen nieuwe modeltraining plaats.'''))
        cells.append(nbformat.v4.new_code_cell('''import joblib
aws=json.loads((OUT/'aws_artifact_audit.json').read_text(encoding='utf-8'))
aws_predictions=pd.read_csv(OUT/'aws_artifact_predictions.csv',float_precision='round_trip')
path=ROOT/'SolutionJorre/MushroomDataset/correct_dataset/correct_mushroom_pipeline.joblib'
assert digest(path)['sha256']==aws['artifact']['sha256']
aws_model=joblib.load(path)
assert list(aws_model.feature_names_in_)==X.columns.tolist()
assert list(aws_model.classes_)==[0,1]
assert aws_predictions.source_row.tolist()==test.tolist()
assert np.array_equal(aws_model.predict(X.iloc[test]),aws_predictions.prediction)
assert np.allclose(aws_model.predict_proba(X.iloc[test])[:,1],aws_predictions.probability_p,rtol=0,atol=1e-12)
actual=evaluate(aws_predictions.true_class,aws_predictions.prediction,aws_predictions.probability_p)
for name,value in actual.items(): assert abs(value-aws['recomputed_metrics'][name])<1e-12
tuning=pd.read_csv(ROOT/'SolutionJorre/MushroomDataset/correct_dataset/correct_aws_tuning_results.csv',float_precision='round_trip')
best=tuning[tuning.rank_test_score==1].iloc[0]
assert abs(best.mean_test_score-aws['cv_macro_f1'])<1e-12
assert np.allclose(best[[f'split{i}_test_score' for i in range(5)]].astype(float),cv[cv.model=='aws_baseline_local'].sort_values('fold').macro_f1,rtol=0,atol=1e-12)
assert aws['local_baseline_label_disagreements']==0 and not aws['warnings']
display(pd.Series(aws['recomputed_metrics'],name='Echte SageMaker-export'))
display(tuning[['param_classifier__n_estimators','param_classifier__max_depth','param_classifier__min_samples_leaf','mean_test_score','rank_test_score']])
print('SageMaker-run toegevoegd; RF300 blijft gekozen op CV macro F1.')'''))
    cells.append(nbformat.v4.new_markdown_cell('''Controleer het herlaadbare artifact en zijn elf invoervelden. Deze pipeline bevat
de volledige preprocessing; de labelmapping is 0=e, 1=p.'''))
    cells.append(nbformat.v4.new_code_cell('''import joblib
artifact=OUT/'selected_pipeline.joblib'
assert digest(artifact)['sha256']==summary['artifact']['sha256']
model=joblib.load(artifact)
assert list(model.feature_names_in_)==X.columns.tolist() and list(model.classes_)==[0,1]
assert np.array_equal(model.predict(X.iloc[test]),predictions[summary['selected_model']+'_prediction'])
assert np.allclose(model.predict_proba(X.iloc[test])[:,1],predictions[summary['selected_model']+'_probability_p'],rtol=0,atol=1e-12)
print('Artifact, schema, labels en alle 900 inferenties gecontroleerd.')'''))
    nb=nbformat.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec={'name':'python3','display_name':'Python 3.11 (correct dataset)','language':'python'}
    return nb


def main():
    import sklearn
    assert sklearn.__version__=='1.4.2', 'Use Python 3.11 and MushroomFinal/requirements-comparison.txt'
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    report=build_report();(ROOT/'secondary_mushroom/09_vergelijkingsrapport.md').write_text(report,encoding='utf-8')
    nb=notebook_content(report)
    if args.execute:
        env=dict(os.environ);env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env['PATH']
        NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(ROOT/'secondary_mushroom')}}).execute(env=env)
    nbformat.validate(nb);nbformat.write(nb,ROOT/'secondary_mushroom/09_compare_models.ipynb')
    print('Correct-dataset report and main notebook saved; executed:',args.execute)


if __name__=='__main__':main()
