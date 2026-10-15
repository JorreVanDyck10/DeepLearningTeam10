# Jorre — Mushroom op AWS SageMaker

De nieuwe training en tuning op het **juiste cursusbestand van 5.000 rijen** zijn
uitgevoerd. De ontvangen notebook, pipeline en resultaten staan overzichtelijk in
[correct_dataset](correct_dataset/README.md).

- [Uitgevoerd notebook](correct_dataset/04_aws_correct_dataset.ipynb)
- [Resultaten](correct_dataset/correct_aws_results.json)
- [Tuninglogs](correct_dataset/correct_aws_tuning_results.csv)
- [Complete pipeline](correct_dataset/correct_mushroom_pipeline.joblib)

Beste instelling: 100 bomen, geen depth-limiet, leaf 1. CV macro F1 **0,753124**,
accuracy **80,11%**, giftige recall **58,94%**, 140 FN en 39 FP op dezelfde 900 testrecords.
`jumbled_noise_0` is uitgesloten; `jumbled_noise_1` blijft in de elf invoervelden.

Codex heeft op verzoek van Jorre de ongewijzigde exports opgenomen en de pipeline
opnieuw geladen om de metrics te verifiëren. De artifact-audit en voorspellingen staan
in [MushroomFinal/comparison_correct](../../MushroomFinal/comparison_correct/).
De [gezamenlijke vergelijking](../../secondary_mushroom/09_vergelijkingsrapport.md)
behoudt RF300 als voorlopige modelkeuze. De website is nog niet omgeschakeld.

## Oude run — uitsluitend historisch

`03_aws_mushrooms.ipynb`, `aws_results.json`, `aws_tuning_results.csv` en
`mushroom_pipeline.joblib` in deze bovenliggende map gebruiken de verkeerde UCI-data
met 61.069 oorspronkelijke rijen. Hun 100%-scores tellen niet voor de inlevering.
Ze blijven behouden voor herkomstcontrole. Gebruik voortaan de nieuwe correct_dataset-map.

`04_aws_correct_dataset.ipynb` in deze bovenliggende map is de herbruikbare template;
de daadwerkelijk uitgevoerde versie staat in correct_dataset.

## Bijdrage en GenAI

Jorre Van Dyck voerde de AWS-training en tuning op SageMaker uit en leverde de
exports aan. Hij stelde ook de gezamenlijke pipeline op en ontwikkelde backend
en UI met substantiële hulp van Codex. Zie [bijdragen](../../Final/BIJDRAGEN.md) en
[GenAI Disclosure](../../Final/GENAI_DISCLOSURE.md).
