# Jorre - Mushroom-model op AWS SageMaker

Random Forest getraind en getuned in een AWS SageMaker-notebookomgeving,
op de UCI Secondary Mushroom Dataset.

## Bestanden

| Bestand | Inhoud |
| --- | --- |
| `03_aws_mushrooms.ipynb` | Uitgevoerd AWS-notebook met code, resultaten en confusion matrix |
| `mushroom_pipeline.joblib` | Opgeslagen preprocessing en het getunede Random Forest samen |
| `aws_results.json` | Cross-validatiescores, gekozen hyperparameters en testresultaten |
| `aws_tuning_results.csv` | Alle vier kandidaten van de RandomizedSearchCV |

De vier aangeleverde bestanden zijn ongewijzigd overgenomen uit de AWS-export.

## Data en werkwijze

Het notebook downloadt het oorspronkelijke UCI-archief en leest
`secondary_data.csv` met puntkomma als scheidingsteken. Na het verwijderen van
exacte duplicaten blijven 60.923 rijen met 20 kenmerken over.
De doelklasse is `0 = eetbaar` en `1 = giftig/onbekend`.

Een gestratificeerde split met `random_state=42` gebruikt 48.738 rijen voor
training en 12.185 rijen voor de aparte testset. De preprocessing vult ontbrekende
numerieke waarden met de trainingsmediaan. Categorische waarden krijgen de
waarde `missing` en worden daarna met one-hot encoding verwerkt.

De RandomizedSearchCV vergelijkt vier kandidaten met drie gestratificeerde
cross-validatiefolds op de trainingsset. De testset wordt daarna gebruikt voor
de eindbeoordeling. De gekozen instellingen zijn:

- `n_estimators=100`
- `min_samples_leaf=2`
- `max_depth=24`

## Opgeslagen resultaten

Volgens de meegeleverde resultaten zijn de baseline-CV-F1 en getunede CV-F1
beide **1,0**. Tuning verbeterde de gemeten CV-score dus niet.
Accuracy, precision, recall, F1 en ROC-AUC op de testset zijn allemaal **1,0**.
De confusion matrix, in klassevolgorde `[0, 1]`, is:

```text
[[5436,    0],
 [   0, 6749]]
```

Dit zijn de opgeslagen AWS-resultaten; bij het toevoegen aan GitHub is het model
niet opnieuw getraind of de evaluatie opnieuw uitgevoerd. De dataset bevat
gesimuleerde paddenstoelen. Deze scores bewijzen geen betrouwbaarheid voor het
beoordelen van echte paddenstoelen.

## Omgeving en opnieuw uitvoeren

De versie-uitvoer in het notebook vermeldt:

- Python `3.10.20`
- pandas `2.3.3`
- scikit-learn `1.7.2`

Gebruik dezelfde scikit-learn-versie voor het laden van het opgeslagen artifact.
Open het notebook vanuit deze map in Jupyter en voer de cellen in volgorde uit
om training en tuning te herhalen. Het notebook downloadt de dataset en schrijft
de drie resultaatbestanden in de huidige werkmap.
