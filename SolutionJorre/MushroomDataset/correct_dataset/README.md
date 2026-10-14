# Jorre — uitgevoerde SageMaker-training op de juiste cursusdata

Jorre Van Dyck voerde deze training en tuning uit op AWS SageMaker. De vier ontvangen
exports zijn inhoudelijk ongewijzigd overgenomen. Alleen de gedownloade notebooknaam
`04_aws_correct_dataset (1).ipynb` is genormaliseerd. Hashes staan in
[import_receipt.json](import_receipt.json). Codex controleerde het artifact en de
metrics op verzoek van Jorre.

| Bestand | Inhoud |
| --- | --- |
| [04_aws_correct_dataset.ipynb](04_aws_correct_dataset.ipynb) | Notebook met uitgevoerde training, tuning en resultaten |
| [correct_aws_results.json](correct_aws_results.json) | Datasetidentiteit, parameters, metrics en versies |
| [correct_aws_tuning_results.csv](correct_aws_tuning_results.csv) | Vier kandidaten × vijf trainingsfolds |
| [correct_mushroom_pipeline.joblib](correct_mushroom_pipeline.joblib) | Complete opgeslagen preprocessing en Random Forest |

De aangeleverde notebook bevat bovenaan nog de oude voorbereidingsmelding "Nog niet
op SageMaker uitgevoerd". De ingevulde uitvoercellen en exports horen bij de inmiddels
uitgevoerde run; de originele inhoud is hier bewust bewaard.

Bron: `MushroomFinal/mushroom_project_dataset.csv`, 5.000 rijen. Elf underscore-features:
`jumbled_noise_0` is uitgesloten; `jumbled_noise_1` blijft behouden. Gestratificeerde
82/18-split met seed 42: 4.100 training, 900 test. Vijf trainingsfolds met shuffle/seed 42.
Keuzemetric: macro F1. Labels: `0=e`, `1=p`.

De beste van vier instellingen is RF100, geen maximale diepte, leaf 1. De baseline
wint; de onderzochte complexiteitsvarianten geven geen verbetering:

| Metric | Resultaat |
| --- | ---: |
| Gemiddelde CV macro F1 | 0,753124 |
| Testaccuracy | 80,11% |
| Test macro F1 | 0,772534 |
| Giftige recall | 58,94% |
| Giftige gevallen als eetbaar (FN) | 140 van 341 |
| Eetbare gevallen als giftig (FP) | 39 van 559 |

Confusion matrix, werkelijk e/p × voorspeld e/p: `[[520, 39], [140, 201]]`.
De lokale audit herberekent deze metrics op hetzelfde gereconstrueerde testlidmaatschap,
zonder opnieuw te trainen. Alle 900 labels en alle vijf CV macro-F1-scores komen
overeen met de eerdere lokale AWS-baseline. De gedeelde testset was al bekend;
dit is geen nieuwe onafhankelijke eindtest.

AWS: Python 3.10.20, scikit-learn 1.4.2, pandas 2.1.4, NumPy 1.26.4, joblib 1.3.2.
Gebruik dezelfde libraryversies bij het laden. De audit is lokaal met Python 3.11
uitgevoerd; er waren geen inferentie- of versiewaarschuwingen.

De [gezamenlijke vergelijking](../../../secondary_mushroom/09_vergelijkingsrapport.md)
behoudt RF300 als voorlopige voorkeur: CV macro F1 0,761876 en testaccuracy 81,33%.
Deze upload vervangt het gepubliceerde websitemodel niet automatisch.
