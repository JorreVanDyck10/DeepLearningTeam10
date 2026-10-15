# Final — DeepLearningTeam10

**Start hier voor de beoordeling.** Dit is de definitieve leesmap binnen de
volledige GitHub-repository. De eerdere bijdragen zijn behouden: er is niets uit
`MushroomFinal`, `SolutionAndrew`, `SolutionMilan`, `SolutionJorre` of `SolutionRaoul`
verplaatst of verwijderd. Deze map is geen losstaand project; download de hele
repository zodat de gedeelde code, data en onderzoeksrapporten beschikbaar zijn.

Team: Jorre Van Dyck, Milan Wouters, Raoul Buffels en Andrew Noeyens.

- [Website](https://10messi10.netlify.app/)
- [API en Swagger](https://deeplearningteam10-api.onrender.com/docs)
- [Voortgang en eindcontrole](INLEVERCHECK.md)
- [Presentatie en spreekplan](Presentatie/README.md)
- [Hertraining en publicatie](AUTOMATISERING.md)
- [Bronmanifest van de leesversies](source_manifest.json)

## 1. Mushroom

Alle definitieve vergelijking gebruikt het afgesproken cursusbestand van
**5.000 rijen**, niet de historische UCI-download van 61.069. `jumbled_noise_0`
is verwijderd, `jumbled_noise_1` blijft behouden. De elf kenmerken gebruiken
underscores. Splits: 4.100 train, 900 ontwikkeltest; vijf stratified trainingfolds.

| Volgorde | Bestand | Inhoud |
| --- | --- | --- |
| 1 | [01_EDA](Mushroom/01_EDA.ipynb) | Missing values, verdelingen, klasseverhouding en grafieken |
| 2 | [02_voorbereiding](Mushroom/02_voorbereiding.ipynb) | Definitieve voorbereiding zonder grafieken; geen leakage door globale imputatie |
| 3 | [03_baseline](Mushroom/03_baseline.ipynb) | Kleine beslisboom, opnieuw gefit en gecontroleerd |
| 4 | [04_AutoML](Mushroom/04_AutoML.ipynb) | Controle en resultaten van de werkelijk uitgevoerde PyCaret-run |
| 5 | [05_Random_Forest](Mushroom/05_Random_Forest.ipynb) | RF300, oorspronkelijke tuningcontext en herhaalde vijf-fold-evaluatie |
| 6 | [06_Gradient_Boosting](Mushroom/06_Gradient_Boosting.ipynb) | Eigen modelnotebook met fold-lokale proximity-imputatie |
| 7 | [07_Logistic_Regression](Mushroom/07_Logistic_Regression.ipynb) | Eigen modelnotebook met schalen en regularisatie |
| 8 | [08_AWS](Mushroom/08_AWS.ipynb) | Aangeleverde SageMaker-uitvoer, juiste bron en tuning |
| 9 | [09_vergelijking](Mushroom/09_vergelijking.ipynb) | Gedeelde metrics, foutenanalyse, AWS-audit en conclusie |
| 10 | [10_deployment](Mushroom/10_deployment.ipynb) | Artifactketen, API, website, beslisregel en echte live controles |

De gekozen RF300 heeft CV macro F1 0,761876. De ontwikkeltest geeft 81,33%
accuracy en 64,52% giftige recall; 121 giftige voorbeelden krijgen de ruwe klasse e.
Deze testset was eerder bekeken. Dit is geen onaangeraakte eindtest en geen
bewezen detector voor echte paddenstoelen. De website bevestigt daarom geen
eetbaarheid: zij toont giftig of onzeker / mogelijk giftig. Zie het
[beslisrapport](../docs/mushroom_decisions.md).

Het [vergelijkingsnotebook](Mushroom/09_vergelijking.ipynb) bevat daarnaast een
duidelijk gemarkeerde aanvulling met featureselectie en voting uit
[Milans ontwikkelstudie](https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/d8fe8dc5516372316e7d6d09e5e2ecdab457ce1b/solutions/milan/RESULTATEN_V3.md). Die cijfers komen
uit vijf buitenfolds op 4.100 trainingsrijen, niet uit de 900 ontwikkeltestrecords.
Het huidige model wordt door deze rapportaanvulling niet vervangen.
De [bevroren resultatensamenvatting](Mushroom/reports/milan_development_summary_v3.csv)
is bij Final opgenomen; deze aanvulling vereist geen lokale Milan-omgeving.

De huidige herlaadbare pipeline staat in
[`Mushroom/artifacts/current_pipeline.joblib`](Mushroom/artifacts/current_pipeline.joblib).
De oorspronkelijke wetenschappelijke vergelijking en pipeline blijven onveranderd
in [`MushroomFinal/comparison_correct`](../MushroomFinal/comparison_correct/).
De [trainingsreceipt](Mushroom/artifacts/training_run.json) legt iedere automatische
refit vast. Een refit is geen nieuwe vergelijking of onbewezen prestatieverbetering.

## 2. Citi Bike

Raouls oorspronkelijke, beschermde studie wordt behouden. De leesversies maken
baseline en Ridge afzonderlijk vindbaar. De grootste brondatasets en modellen
blijven volgens de opdracht buiten Git. De cijfers in de bronnotebooks zijn
de werkelijk uitgevoerde oorspronkelijke resultaten; kopiëren is geen nieuwe run.

| Volgorde | Bestand | Inhoud |
| --- | --- | --- |
| 1 | [01_EDA_en_hypothesen](CitiBike/01_EDA_en_hypothesen.ipynb) | NYC/JC-scheiding, aggregaties, grafieken en statistische hypothesen |
| 2 | [02_voorbereiding](CitiBike/02_voorbereiding.ipynb) | Reproduceerbare voorbereiding zonder verkennende grafieken |
| 3 | [03_baseline](CitiBike/03_baseline.ipynb) | Weekbaseline; eigen rapportnotebook |
| 4 | [04_Ridge](CitiBike/04_Ridge.ipynb) | Ridge en eigen chronologische CV-rapportage |
| 5 | [05_AutoML](CitiBike/05_AutoML.ipynb) | Uitgevoerde PyCaret-vergelijking |
| 6 | [06_Random_Forest](CitiBike/06_Random_Forest.ipynb) | Random Forest en tuning |
| 7 | [07_LightGBM](CitiBike/07_LightGBM.ipynb) | LightGBM en tuning |
| 9 | [09_vergelijking](CitiBike/09_vergelijking.ipynb) | Vastgelegde selectie, foutenanalyse en bestaande eindtest |
| 10 | [10_deployment](CitiBike/10_deployment.ipynb) | Werkelijke inference op de team-API |

Het gekozen model is RF320. De vastgelegde juli–augustus-eindtest heeft RMSE
1.979,51 tegenover 2.608,78 voor de weekbaseline: 24,12% lager. MAE is 1.076,23.
Voorspellen vereist de 14 volledige voorafgaande lokale NYC-dagen. De demo
gebruikt beschikbare historische tellingen en vormt geen nieuwe eindtest.

Er is geen tweede AWS-run voor Citi Bike verzonnen. De expliciete modelsectie
van de opdracht vraagt minimaal één getraind en getuned AWS-model voor het
project; de uitgevoerde Mushroom-run is dat bewijs. De bronstudie vermeldt
terecht dat een afzonderlijke Citi Bike-AWS-run niet is uitgevoerd.

## Uitvoeren

Voor Mushroom-voorbereiding, modelreproducties en rapportcontroles:

```powershell
py -3.11 -m venv .venv-final
.\.venv-final\Scripts\python.exe -m pip install -r Final/requirements-notebooks.txt
.\.venv-final\Scripts\python.exe Final/tools/execute_notebooks.py
```

Dit herhaalt de vier Mushroom-modelconfiguraties en controleert de opgeslagen
referenties; het draait geen nieuwe volledige PyCaret-gridsearch en opent geen
nieuwe Citi Bike-eindtest. Voor de oorspronkelijke PyCaret-run gebruik je de
aparte [vergelijkingsomgeving](../MushroomFinal/README.md). Voor de Citi Bike-
bronverwerking gebruik je [Raouls instructies](../SolutionRaoul/Citybike/README.md).

## Wat blijven historische bestanden?

De lege starttemplates onder `secondary_mushroom` en `nyc_citi_bike` zijn geen
ontbrekende eindresultaten: deze Final-index verwijst naar de werkelijk ingevulde
uitwerkingen. De oude UCI-baselines, 100%-AWS-run en vergelijkingen op 61k rijen
blijven voor herkomstcontrole bewaard en tellen niet als definitieve Mushroom-
evaluatie. Oudere individuele notebooks blijven onderzoeksarchief; hun cijfers
mogen niet worden gecombineerd met de nieuwe gedeelde split.

**Stap 5 is gedeeltelijk afgerond:** Jorres vier bijdragen zijn bevestigd en de
GenAI-verantwoording is uitgewerkt tegen de aangeleverde policy. De andere
teamleden controleren nog hun eigen bijdrage/AI-inzet; iedereen moet de inhoud
zelfstandig kunnen verdedigen. De [inlevercheck](INLEVERCHECK.md) houdt die
persoonlijke controles afzonderlijk open.

## GenAI Disclosure

De eindoplevering sluit af met de [volledige GenAI Disclosure](GENAI_DISCLOSURE.md).
Deze beschrijft AI-code versus menselijke bijdragen, proza en structuur, vijf echte
prompts, reproduceerbare controles en concrete bijsturing. De [bijdragen](BIJDRAGEN.md)
vermelden Jorres pipeline, backend, UI en AWS-training. De individuele review en
zelfstandige beheersing door alle teamleden blijven menselijke eindcontroles.
