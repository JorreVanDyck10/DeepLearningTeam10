# Voorlopig gepubliceerd model

De website gebruikt voorlopig het Random Forest uit `mushroom.ipynb`:

- Dataset: `mushroom_project_dataset.csv`, 5.000 records.
- Split: 80% training / 20% test, gestratificeerd, random state 42.
- Model: 200 bomen, `class_weight="balanced"`, random state 42.
- Numerieke ontbrekende waarden: mediaan van de trainingsdata.
- Categorische ontbrekende waarden: meest voorkomende trainingswaarde.
- Categorische encoding: one-hot, onbekende categorieën negeren binnen de pipeline.
- Nulmetingen voor steelhoogte en steeldikte worden, net als in cel 8,
  als ontbrekende waarden behandeld. Deze stap zit ook in de opgeslagen pipeline.

De twaalf oorspronkelijke invoervelden, inclusief `jumbled_noise_0` en
`jumbled_noise_1`, blijven behouden. De twee ruisvelden zijn geen biologische
eigenschappen en zijn optioneel in de API; bij ontbrekende waarden gebruikt de
pipeline dezelfde imputatie als bij training.

## Reproduceren en publiceren

Voer vanuit de repository-root uit in de omgeving uit `backend/requirements.txt`:

```powershell
python -m backend.mushroom_model
python frontend/export_metadata.py
python -m unittest discover -s tests -v
```

Dit schrijft in deze map:

- `models/random_forest.joblib`: de volledige, gecomprimeerde pipeline.
- `metrics/random_forest.json`: resultaten, dataset-hash en instellingen.
- `example_mushroom.json`: een voorbeeld uit de testset.

De websitevelden en getoonde scores komen uit `frontend/metadata.json`.
Commit het model, de resultaten en de vernieuwde frontendmetadata samen naar
`main`. Render laadt de nieuwe pipeline bij de volgende automatische deployment.

De API kiest standaard dit model. Een afwijkend vertrouwd artifactpad kan via
`ANDREW_MUSHROOM_MODEL_PATH` worden ingesteld. De oude variabele
`MUSHROOM_MODEL_PATH` selecteert dit model niet meer.

## Controle van de resultaten

De geëxporteerde pipeline is vergeleken met een letterlijke uitvoering van de
trainingscellen in Andrew's notebook in dezelfde omgeving: labels en kansen
zijn identiek. De opnieuw berekende accuracy is **78,2%**, tegenover **78,7%**
in de al opgeslagen notebookuitvoer. De website toont de opnieuw berekende
resultaten; de oorspronkelijke notebookuitvoer is behouden.

Giftige recall is circa **57,0%**: 163 van de 379 giftige/onbekende testgevallen
worden als eetbaar voorspeld. Dit is een voorlopige schooldemo op de aangeleverde
dataset, geen beoordeling van echte paddenstoelen.

Codex heeft de bestaande modelcode reproduceerbaar geëxporteerd en de API,
invoerinterface en tests hierop aangesloten. Er zijn geen hyperparameters
gekozen of aangepast op basis van de testresultaten.
