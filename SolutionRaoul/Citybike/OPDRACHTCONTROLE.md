# Citi Bike — controle tegenover de officiële opdracht

Bron: `Discussion topics/project assignment.md` en de uitgewerkte cursusnotebooks.
Hoofdversie: `Citybike_full`; inleverversie: `DeepLearningTeam10/SolutionRaoul/Citybike`.
Datum: 14 oktober 2026. De oorspronkelijke studie en bekeken eindtest blijven behouden.

| Vereiste | Status | Bestand |
|---|---|---|
| Automatische bronverwerking, cleaning en aggregatie | ✅ klaar | `citibike/prepare.py`, `citibike/assembly.py`, `01a_dataonderzoek_en_cleaning.ipynb` |
| Missing values, duplicaten, foutieve waarden, outliers en categorieën | ✅ klaar | `01a_dataonderzoek_en_cleaning.ipynb` |
| EDA met tijd, gebruikers, fietsen en stations | ✅ klaar | `01b_EDA_en_hypothesen.ipynb` |
| Statistisch bewijs vóór doelkeuze | ✅ klaar | `01b_EDA_en_hypothesen.ipynb`: hoofdvraag, H0/H1, bloktoets, effect, interpretatie |
| Zelfstandige definitieve voorbereiding zonder grafieken | ✅ klaar | `02_definitieve_voorbereiding.ipynb`, `reports/preparation_independence.json` |
| Dezelfde modelinvoer, chronologische splits en leakagebescherming | ✅ klaar | `citibike/model_data.py`, `reports/prepared_model_data.json`, `tests/test_model_data.py` |
| Baseline en Ridge | ✅ klaar | `03_baseline_en_Ridge.ipynb` |
| PyCaret met werkelijke uitvoering en gecontroleerde folds | ✅ klaar | `04_PyCaret_vergelijking.ipynb`, `reports/notebook_reproduction/pycaret_audit.json` |
| Meerdere eigen modellen en beperkte chronologische tuning | ✅ klaar | `05_Random_Forest.ipynb`, `06_LightGBM.ipynb`, oorspronkelijke tuningrapporten |
| Centrale modelvergelijking, generalisatie en praktische complexiteit | ✅ klaar | `07_eindvergelijking.ipynb` |
| Residuals en concrete grootste fouten | ✅ klaar | `07_eindvergelijking.ipynb`, `reports/validation_residuals.csv` |
| Notebookuitleg en bijdragen voor het mondeling | ✅ klaar | Negen genummerde notebooks, beginnend bij `00_projectoverzicht.ipynb` |
| Bestaande frontend/API en gedeelde offline inference | ✅ klaar | Groepsrepo `frontend/`, `backend/`; `reports/inference_parity.json` |
| Lokale vaste hertraining, serialisatiecontrole en artefact | ✅ klaar | `tools/retrain_pipeline.py`, `reports/retraining_pipeline_latest.json` |
| Automatische hertraining bij push op GitHub | ⚠️ nog handmatig nodig | Groepsrepo `.github/workflows/citibike-retrain.yml`; runner registreren en geslaagde GitHub-run aantonen |
| Automatische omschakeling van het live model | ⚠️ nog handmatig nodig | `DEPLOYMENT.md`; de API gebruikt nog de vastgezette bestaande release |
| AWS-model getraind én getuned, lokaal notebook aanwezig | ⚠️ nog handmatig nodig | `SolutionJorre/MushroomDataset/03_aws_mushrooms.ipynb`, tuningexports; cloudprovenance bevestigen |
| README, dependencies, .gitignore en gecontroleerde groepskopie | ✅ klaar | `README.md`, requirements, `EXPORT_MANIFEST.json` in groepskopie |
| Definitieve GitHub-upload en docenttoegang | ⚠️ nog handmatig nodig | Groepsrepo `JorreVanDyck10/DeepLearningTeam10`; deze ronde pusht niet |

De status **klaar** verwijst naar lokale uitvoering en het aanwezige bewijs. De
machineleesbare eindcontrole staat in `reports/implementation_checks.json` en
`reports/delivery_check.json`. Bronherbouw gebruikte gecontroleerde echte caches;
een volledige herbouw van alle ruwe ritten is niet opnieuw uitgevoerd.

De negen notebooks fitten de vaste modellen werkelijk en bewaren nieuwe outputs.
De historische tuning is behouden en niet opnieuw gestart. Oorspronkelijke
train/validatiescores worden gecontroleerd; bestaande testresultaten worden alleen
beschreven. De eindtest wordt niet gebruikt om modellen, features of regels te wijzigen.

**OPENSTAAND: AWS / SageMaker model — cloudprovenance bevestigen.** Het geëxporteerde
Mushroom-notebook bevat uitgevoerde training/tuning, `conda_python3` en een
`/home/ec2-user/`-runtimepad. Dat ondersteunt de gemelde SageMaker-herkomst, maar
er is geen afzonderlijke verificatie tegen AWS gedaan. De generieke AWS-slots
zijn placeholders; zij bewijzen geen voltooid model. Geen nieuwe AWS-kosten gemaakt.

Mushroom-inhoud, groepspresentatie en eventuele onbekende bijdragen van andere
teamleden worden in deze Citi Bike-ronde niet aangepast.
