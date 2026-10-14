# Finale Mushroom-uitwerking — juiste cursusdata

Gebruik **mushroom_project_dataset.csv: 5.000 rijen**. De oudere download met 61.069
UCI-records is niet het bedoelde cursusbestand. `jumbled_noise_0` wordt verwijderd;
`jumbled_noise_1` blijft behouden. Alle elf modelvelden gebruiken underscores.

- [mushroom_project.ipynb](mushroom_project.ipynb): Andrew's EDA en modellen, met nu
  werkelijk uitgevoerde PyCaret-cellen en een ingevulde conclusie.
- [Hoofdvergelijking](../secondary_mushroom/09_compare_models.ipynb) en
  [rapport](../secondary_mushroom/09_vergelijkingsrapport.md): dezelfde 4.100/900 rijen,
  vijf gedeelde CV-folds, fold-lokale preprocessing, metrics, fouten en modelkeuze.
- [comparison_correct](comparison_correct/): bronhash, splits, voorspellingen,
  PyCaret-tabellen en nieuwe complete pipeline. De oorspronkelijke `final_*.pkl`
  bestanden blijven ongewijzigd; nieuwe scores horen bij nieuwe fits.

Reproduceren in een aparte omgeving:

```powershell
py -3.11 -m venv .venv-comparison
.\.venv-comparison\Scripts\python.exe -m pip install -r MushroomFinal/requirements-comparison.txt
.\.venv-comparison\Scripts\python.exe MushroomFinal/compare_correct_dataset.py --stage manual
.\.venv-comparison\Scripts\python.exe MushroomFinal/execute_pycaret_notebook.py
.\.venv-comparison\Scripts\python.exe MushroomFinal/build_correct_comparison.py --execute
```

Voor interactief gebruik in Jupyter, registreer en selecteer de juiste kernel:

```powershell
.\.venv-comparison\Scripts\python.exe -m ipykernel install --user --name mushroom-pycaret --display-name "Mushroom Python 3.11"
```

Kies daarna **Mushroom Python 3.11** in Jupyter. De volledig vastgelegde uitgevoerde
omgeving staat ook in `requirements-lock.txt`.

Jorre heeft de AWS-training en tuning opnieuw uitgevoerd op de juiste CSV. De
[uitgevoerde SageMaker-exports](../SolutionJorre/MushroomDataset/correct_dataset/README.md)
zijn opgenomen en de opgeslagen pipeline is afzonderlijk gecontroleerd: 80,11%
accuracy, 58,94% giftige recall en CV macro F1 0,753124. De uitkomst komt overeen met
de lokale AWS-baseline; RF300 blijft voorlopig gekozen. Herhaal de audit met:

```powershell
.\.venv-comparison\Scripts\python.exe MushroomFinal/audit_correct_aws.py
```

De gekozen pipeline verwacht elf velden en geeft `0=e`, `1=p`. API en website zijn
nu gekoppeld aan een verliesvrije RF300-export; de API vertaalt naar `e` en `p`.
Zie [deploymentdocumentatie](../docs/mushroom_rf300.md) voor de exportaudit,
versiecontrole en hosting.
Het [uitgevoerde deploymentnotebook](../secondary_mushroom/10_deploy.ipynb) legt
deze stap van de opdracht vast, inclusief live controles en automatische updates.

Bijdragen: Andrew maakte de originele uitwerking. Codex voerde op verzoek van Jorre
de datasetcorrectie, gedeelde evaluatie, PyCaret-uitvoering en rapportage uit.
Het team blijft verantwoordelijk voor review en mondelinge verdediging.
