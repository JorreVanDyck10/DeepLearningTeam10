# Spreekplan bij de definitieve groepspresentatie

Dit spreekplan hoort bij [Team10_presentatie.pptx](Team10_presentatie.pptx), de op 15 oktober 2026 door Raoul aangeleverde definitieve PowerPoint. De presentatie bevat 27 dia’s. Dia 26 en 27 zijn verborgen reservedia’s. De dianummers hieronder zijn de posities in de volledige PowerPoint. De kleine nummers 2–9 op de Citi Bike-dia’s horen bij Raouls oorspronkelijke deelpresentatie.

Oefen ongeveer twintig minuten inclusief demo en houd ongeveer vijf minuten vrij voor vragen binnen het slot van 25 minuten. Raouls hoofddeel staat op dia 14–22 en duurt volgens de richttijden ongeveer 7 minuten en 15 seconden. Spreek onderling af wie de gezamenlijke introductie, deployment en demo presenteert.

Raouls [afzonderlijke spreektekst](SPREEKPLAN.md) bevat ook zijn bijdrageoverzicht en voorbereiding op vragen. De [GenAI-verantwoording](../GENAI_DISCLOSURE.md) hoort bij de oplevering. Deze PowerPoint heeft geen aparte GenAI-dia. Benoem de AI-hulp mondeling zoals in de tekst en gebruik de volledige verantwoording als naslag.

## Dia 1 — Machine Learning / Cloud AI Challenge

Richttijd: 20 seconden. Twee problemen: Mushroom-classificatie en Citi Bike-uurvraag. Deze presentatie is met Codex bijgewerkt vanuit Andrews bronpresentatie en gecontroleerde projectresultaten. Het team controleert de GenAI-policy en kan de keuzes zelf toelichten.

## Dia 2 — Table of contents

Richttijd: 15 seconden.

Overzicht. Ongeveer 19,5 minuten inclusief demo; houd 5,5 minuten vrij voor vragen.

Bronnen:

## Dia 3 — Team 10

Richttijd: 25 seconden. Stel het team voor en licht de echte taakverdeling mondeling toe. De definitieve bijdragen en GenAI-vermelding moeten door het team worden bevestigd; deze dia verzint geen rollen.

## Dia 4 — Mushroom Challenge

Richttijd: 40 seconden.

Het juiste cursusbestand heeft 5.000 rijen; de oude UCI-download met 61.069 rijen is een andere studie. Drie numerieke en acht categorische modelvelden. jumbled_noise_0 verwijderd, jumbled_noise_1 behouden. Hypothetische cursusdata.

Bronnen:

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/MushroomFinal/README.md

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/MushroomFinal/mushroom_project_dataset.csv

## Dia 5 — Mushroom: Distribution

Richttijd: 45 seconden.

3.107 eetbare en 1.893 giftige gevallen. De meerderheid voorspellen levert ongeveer 62% accuracy. Daarom gebruiken we dummy en beslisboom als baselines, macro F1 en recall van de giftige klasse.

Bronnen:

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/MushroomFinal/mushroom_project_dataset.csv

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/MushroomFinal/comparison_correct/model_comparison.csv

## Dia 6 — Mushroom: Missing values

Richttijd: 60 seconden.

Spore print ontbreekt bij 94,8%; stem surface bij 67,18%; cap diameter bij 40,64%. Niet alle rijen weggooien. RF300 gebruikt mediaan voor getallen, modus en one-hot voor categorieën. Alles leert alleen uit de betreffende trainingsfold. De oorspronkelijke proximity-imputatie vóór CV gaf leakage-risico; de gecorrigeerde fits gebruiken fold-lokale preprocessing.

Bronnen:

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/MushroomFinal/mushroom_project_dataset.csv

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/MushroomFinal/final_model_support.py

## Dia 7 — Mushroom: Outliers

Richttijd: 40 seconden.

De stamhoogteverdelingen overlappen sterk. Eén kenmerk kan dus niet beslissen. Extreme waarden zijn niet automatisch meetfouten. We verwijderen ze niet blind; bij RF300 blijven nulmetingen nul. De grafiek komt uit Andrews EDA op dezelfde cursusdata.

Bronnen:

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/MushroomFinal/mushroom_project.ipynb

## Dia 8 — Mushroom: Validation

Richttijd: 60 seconden.

Stratified split seed42: 4.100 train, 900 ontwikkeltest. Vijf gedeelde stratified trainingsfolds. Fit preprocessing en model opnieuw binnen elke fold. Selecteer op CV macro F1 en rapporteer giftige recall. De ontwikkeltest is eerder bekeken, dus geen onaangeraakte eindtest. Gecorrigeerde fits hebben andere scores dan oorspronkelijke screenshots.

Bronnen:

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/MushroomFinal/final_model_support.py

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/MushroomFinal/comparison_correct/protocol.json

## Dia 9 — Mushroom: Random Forest

Richttijd: 65 seconden.

RF300 is geselecteerd op de gemiddelde macro F1 over de vijf trainingsfolds: 0,761876. De 300 bomen hebben max_depth=20, max_features=None, min_samples_split=4, min_samples_leaf=1, class_weight=balanced en random_state=42. De matrix en de scores horen bij dezelfde gecorrigeerde pipeline. Accuracy is (512+220)/900 = 81,33%. Giftige recall is 220/341 = 64,52%. De 121 giftige gevallen die als eetbaar geclassificeerd worden, tonen waarom accuracy onvoldoende is. De website gebruikt deze RF300, niet de oorspronkelijke RF200 uit Andrews eerdere screenshot. De preprocessing gebruikt medianen en modi, geen vooraf op alle trainingsrijen geleerde proximity-matrix.

Bronnen:

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/Final/config/mushroom.json

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/MushroomFinal/comparison_correct/model_comparison.csv

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/backend/artifacts/mushroom_rf300.json

## Dia 10 — Mushroom: Other models

Richttijd: 55 seconden.

Gradient Boosting leert opeenvolgende bomen die fouten van eerdere bomen corrigeren. Logistic Regression is een lineaire referentie met scaling. In de gecontroleerde vergelijking wordt ook de proximity-imputatie van GB binnen elke fold geleerd. GB heeft TN536, FP23, FN171 en TP170. LR heeft TN410, FP149, FN136 en TP205. De oude GB-matrix en het oude rapport spraken elkaar tegen en zijn vervangen door cijfers uit dezelfde gecontroleerde fit. De gemiddelde fitduur per trainingsfold was ongeveer 0,31 seconde voor LR, 3,07 seconden voor GB en 6,36 seconden voor RF300. Dit zijn gemeten looptijden op die omgeving, geen garanties voor andere computers. LR is het snelste; RF300 heeft hier de hoogste CV macro F1.

Bronnen:

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/MushroomFinal/final_model_support.py

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/MushroomFinal/comparison_correct/model_comparison.csv

## Dia 11 — Mushroom: AutoML

Richttijd: 45 seconden.

PyCaret 3.3.2 is daadwerkelijk uitgevoerd op het juiste cursusbestand. De brede vergelijking retourneerde dertien modellen. De tabel toont de aparte vergelijking van drie standaardmodellen met dezelfde trainingsrijen en vijf folds. Het betreft geen handmatig getunede eindmodellen. Macro F1 is expliciet als macro-gemiddelde gecontroleerd. De getoonde recall betreft de giftige klasse. RF scoort hier beter dan GB en LR. Deze cijfers moeten niet rechtstreeks verwisseld worden met de volgende tabel met de uiteindelijke fits.

Bronnen:

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/MushroomFinal/comparison_correct/pycaret_audit.json

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/MushroomFinal/comparison_correct/pycaret_three.csv

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/MushroomFinal/comparison_correct/pycaret_all.csv

## Dia 12 — Mushroom: Final comparison

Richttijd: 70 seconden.

Alle kandidaten gebruiken dezelfde 4.100 trainingsrijen, 900 ontwikkeltestrecords en vijf trainingsfolds. De primaire modelkeuze gebruikt gemiddelde CV macro F1. De beslisboom is de snelle eerste baseline; de dummy voorspelt de meerderheidsklasse. De AWS-rij komt uit de afzonderlijk gecontroleerde echte SageMaker-pipeline. Haar voorspellingen komen overeen met de lokale AWS-baseline. RF300 blijft de geselecteerde keuze, met een kleine voorsprong op AWS. De verschillen zijn geen bewijs van statistische significantie. De ontwikkeltest is eerder bekeken en is dus niet meer onaangeraakt. De tabel toont de belangrijkste modelfamilies en baselines; de volledige vergelijking in GitHub bevat ook andere RF-configuraties.

Bronnen:

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/MushroomFinal/comparison_correct/model_comparison.csv

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/MushroomFinal/comparison_correct/aws_artifact_audit.json

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/MushroomFinal/comparison_correct/cv_selection.json

## Dia 13 — AWS SageMaker: Training and tuning

Richttijd: 50 seconden.

Jorre heeft in AWS SageMaker opnieuw getraind op het correcte cursusbestand met 5.000 rijen. Er zijn vier kandidaatconfiguraties onderzocht. Het uitgevoerde notebook, de herlaadbare pipeline, het resultatenbestand en de tuningtabel zijn gedownload naar de repository. Een aparte artifactaudit controleert de voorspellingen van de werkelijk gedownloade pipeline: 80,11% accuracy en 58,94% recall voor de giftige klasse. De oude 100%-run gebruikte een ander databestand en hoort niet in deze vergelijking. De AWS-training is uitgevoerd, maar RF300 blijft het gekozen gehoste model.

Bronnen:

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/SolutionJorre/MushroomDataset/correct_dataset/04_aws_correct_dataset.ipynb

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/SolutionJorre/MushroomDataset/correct_dataset/correct_aws_tuning_results.csv

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/MushroomFinal/comparison_correct/aws_artifact_audit.json

## Dia 14 — Citi Bike: mijn deel

Richttijd: 25 seconden.

Ik ben Raoul Buffels. Mijn deel gaat over Citi Bike. Ik werkte de bronverwerking, cleaning, EDA, hypothesen en modelvergelijking uit. Daarnaast maakte ik de oorspronkelijke Citi Bike-inference en applicatieonderdelen. Jorre hielp met deployment op Render en verzorgde de gezamenlijke backend, UI en centrale pipeline. Ik bespreek eerst de data, daarna de modellen en ten slotte het resultaat. Codex hielp bij implementatie, uitleg en controles.

Bronnen: README.md en SolutionRaoul/Citybike/notebooks/00_projectoverzicht.ipynb; opdracht: CloudAI/Discussion topics/project assignment.md (bestaande gecontroleerde uitwerking, 14 oktober 2026).

## Dia 15 — Wat ik in de brondata tegenkwam

Richttijd: 45 seconden.

Ik had grote bestanden verwacht. De lastigste stap was bepalen welke records echte bronkopieën waren. De 175 ZIPs bevatten NYC en Jersey City en verschillende oude en nieuwe schema’s. Na verwerking blijven circa 323,8 miljoen NYC-records over. Dat is technische dekking tot augustus 2026, geen EDA op de eindtest. Mijn EDA gebruikt alleen ontwikkeling vóór mei. Een ontbrekend eindstation hoeft een geldige start niet onbruikbaar te maken. Extreem lange duur kan een registratieprobleem of een laat afgesloten rit zijn. Voor uurvolume tel ik de start en behoud ik kwaliteitsflags. Alleen bewezen bronkopieën verwijder ik. Identieke historische ritten kunnen echte afzonderlijke ritten zijn.

Bronnen: reports/data_manifest.json; 01a_dataonderzoek_en_cleaning.ipynb; reports/NYC_duration_sample_audit.json (bestaande gecontroleerde uitwerking, 14 oktober 2026).

## Dia 16 — Gebruik volgt het uur en de weekdag

Richttijd: 45 seconden.

Deze figuur gebruikt echte gemiddelde uurtellingen vanaf 2023 tot eind april 2026. Ik vergelijk werkdagen met weekends. Werkdagen hebben een ander dagprofiel en vooral een duidelijke avondpiek. Het hoogste gemiddelde uur ligt hier om 17 uur op werkdagen en om 15 uur in het weekend. Een verschil tussen dagtypes past bij woon-werkgebruik, maar de data bewijzen het reisdoel niet. Ik verwachtte herhaling in tijd. De grafiek toont dat de patronen veranderen met dagtype en dat één algemeen gemiddelde onvoldoende is. Dat motiveert kalenderkenmerken en historische tellingen in het model.

Bronnen: data/NYC_hourly_development.parquet, dezelfde aggregatie als citibike/plots.py hour_weekday (bestaande gecontroleerde uitwerking, 14 oktober 2026).

## Dia 17 — Weekherhaling ondersteunt het uurdoel

Richttijd: 55 seconden.

Mijn hoofdhypothese is dat een uurtelling samenhangt met hetzelfde uur vorige week. H0 zegt dat de gemiddelde correlatie over tijdsblokken nul is. H1 zegt dat die verschilt van nul. Ik behandel miljoenen ritten niet als onafhankelijke observaties. Ik bereken effecten per week en toets met 43 vierweekblokken. De bestaande sign-fliptoets veronderstelt voldoende onafhankelijke blokken en symmetrie onder H0. De correlatie is ongeveer 0,856 met een interval van 0,837 tot 0,876. De Holm-p is ongeveer 0,00015. Het patroon is dus sterk in ontwikkeling. De vraag kwam uit EDA en blijft exploratief. Correlatie bewijst geen oorzaak of modelprestatie. Pas daarna kies ik het bestaande doel: geregistreerd uurvolume één dag vooruit.

Bronnen: reports/hypotheses.json; reports/hypothesis_protocol.json; reports/target_decision.json (bestaande gecontroleerde uitwerking, 14 oktober 2026).

## Dia 18 — Alle modellen krijgen dezelfde modeldata

Richttijd: 45 seconden.

Notebook 02 maakt zelfstandig de definitieve modeldata vanaf de bronnen of gecontroleerde caches. Het target is geregistreerde vertrekken per lokaal uurvak. Ik gebruik veertien features: kalenderkenmerken en historische tellingen. Voorbeelden zijn het uur en de weekdag, hetzelfde uur gisteren en vorige week, en een gemiddelde van zeven afgesloten dagen. Rolling wordt eerst verschoven. Niets van de voorspeldag komt als historische input binnen. De eerste veertien dagen vallen weg omdat historie ontbreekt. Ik split chronologisch: training vóór mei, validatie mei en juni, test juli en augustus. Binnen training gebruik ik vijf chronologische folds van 28 dagen. Ridge leert zijn scaler alleen op de trainingsrijen van iedere fold.

Bronnen: 02_definitieve_voorbereiding.ipynb; reports/prepared_model_data.json; citibike/features.py (bestaande gecontroleerde uitwerking, 14 oktober 2026).

## Dia 19 — Random Forest valideert het beste

Richttijd: 65 seconden.

De weekbaseline voorspelt hetzelfde uur van vorige week. Ridge is een eenvoudige lineaire referentie. Random Forest en LightGBM leren niet-lineaire verbanden. PyCaret vergelijkt dezelfde vier basismodellen met dezelfde chronologische folds. Daarna behoud ik de beperkte bestaande RF- en LightGBM-tuning. De finale getunede RF heeft een validatie-RMSE van ongeveer 1940,49 ritten per uurvak tegenover 2826,16 voor de baseline. Dat is circa 31,34 procent lager. LightGBM-tuning verbeterde trainings-CV, maar niet de mei-juni-validatie. De verschillen tussen de sterke modellen zijn klein. De keuze lag vóór de test vast en gebruikte de bestaande 1%-regel met een eenvoudigheidsrang per modeltype.

Bronnen: reports/model_comparison.csv; reports/model_selection.json; reports/pycaret_comparison.csv; tuningrapporten (bestaande gecontroleerde uitwerking, 14 oktober 2026).

## Dia 20 — Snelle training en generalisatie verschillen

Richttijd: 55 seconden.

Ridge is het eenvoudigst uit te leggen en fit in deze reproductie in ongeveer 0,05 seconde. LightGBM fit in minder dan een seconde, terwijl RF tientallen seconden nodig heeft. Die tijden zijn echte afzonderlijke fits op deze computer en niet de volledige tuningduur. De getunede RF heeft een trainings-RMSE van ongeveer 536 tegenover 1940 op validatie. Dat is een grote kloof. Overfit kan meespelen, maar de vroege trainingsjaren hebben ook minder intensiteit dan 2026. Ik kan die kloof dus niet uitsluitend aan overfit toeschrijven. RF-tuning bracht maar circa 0,27 procent extra validatiewinst en een groot modelbestand. Dat bespreek ik naast de score. Ik wijzig de keuze niet nadat de test al bekeken is.

Bronnen: reports/notebook_reproduction/*.json, fit_seconds; reports/model_comparison.csv; reports/model_artifact.json (bestaande gecontroleerde uitwerking, 14 oktober 2026).

## Dia 21 — De bestaande eindtest bevestigt winst

Richttijd: 45 seconden.

Ik bespreek de bestaande eindtest juli en augustus 2026. Alleen de vooraf geselecteerde RF en de baseline zijn daarop vergeleken. De model-RMSE is ongeveer 1979,51 tegenover 2608,78 voor de baseline. Dat is 24,12 procent lager. De MAE is ongeveer 1076 ritten per uurvak en R² is 0,828. Deze scores horen bij de finale refit tot eind juni, niet bij de ontwikkelingsfit op de vorige dia. De test was al bekeken. Ik gebruik hem nu niet om features of instellingen te veranderen. Twee zomermaanden garanderen geen prestaties voor een heel toekomstig jaar. Weer en evenementen ontbreken, en recente historische tellingen moeten op tijd beschikbaar zijn.

Bronnen: reports/final_evaluation.json; reports/MODEL_CARD.md (bestaande gecontroleerde uitwerking, 14 oktober 2026).

## Dia 22 — Mijn resultaat en de volgende stap

Richttijd: 55 seconden.

Mijn Citi Bike-uitwerking eindigt in een voorspelfunctie die ook via de teamwebsite en API wordt gebruikt. Om een dag te voorspellen zijn de veertien volledige voorgaande dagen nodig. De API gebruikt dezelfde featureberekening en predict_day als offline. Ik controleerde de overeenkomst van de voorspellingen, ook bij de overgang naar zomer- en wintertijd. De screenshot toont een ontwikkelingsdemonstratie en is geen onafhankelijke test.

Daarnaast werkte ik een lokale hertrainingspipeline uit. Die fit dezelfde vaste Random Forest opnieuw, controleert de voorspellingen na herladen en bewaart een apart modelartefact. Jorres centrale GitHub-workflow traint en publiceert Mushroom. Citi Bike gebruikt een gepinde RF320-release. Automatische nieuwe Citi Bike-training en omschakeling van het live model zijn daarmee nog geen onderdeel van die centrale workflow. Mijn conclusie is dat historische tellingen helpen om geregistreerde vertrekken per uur te voorspellen, zolang de benodigde historie op tijd beschikbaar is.

Bronnen: docs/citibike_raoul_frontend.jpg; DEPLOYMENT.md; reports/inference_parity.json; reports/retraining_pipeline_latest.json; OPDRACHTCONTROLE.md (bestaande gecontroleerde uitwerking, 14 oktober 2026).

Statusupdate: Final/AUTOMATISERING.md en Final/VALIDATIE.md.

## Dia 23 — Deployment: API and website

Richttijd: 30 seconden.

De frontend gebruikt HTML, CSS en JavaScript op Netlify. De backend is Python met FastAPI op Render. De API valideert invoer en gebruikt het geselecteerde model. Mushroom gebruikt een verliesvrije NumPy-export van RF300; paritycontroles vergelijken de uitkomsten met de getrainde pipeline. Citi Bike laadt een gepinde modelrelease met checksum en memory mapping. Beide voorspeltaken gebruiken dezelfde API. Voor Mushroom controleert de frontend ook het model-id en de exacte modelrevisie. Dit voorkomt dat frontend en backend ongemerkt verschillende modelversies gebruiken. Het is een eigen website met API, geen Streamlit-app.

Bronnen:

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/backend/main.py

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/backend/mushroom_model.py

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/backend/raoul_citibike.py

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/frontend/README.md

## Dia 24 — GitHub: Automated publication

Richttijd: 40 seconden.

Een relevante code-, configuratie- of datapush op GitHub start de Mushroom-workflow. Python 3.11 voert vijfvoudige cross-validatie en een refit van de vastgelegde RF300-configuratie uit. De CV macro F1 en giftige recall moeten de referentiedrempels halen. Een aparte Python 3.13-job controleert het servingexport en de API met beide modellen. De export wordt op alle 5.000 rijen vergeleken met de gefitte kandidaat. Pas na geslaagde controles worden model en metadata gepubliceerd. Een inmiddels verouderde broncommit wordt geweigerd. Render en Netlify volgen de gevalideerde publicatie. Dit is automatisch herfitten van de gekozen configuratie, geen nieuwe automatische hyperparameterzoektocht. Voor Citi Bike is er momenteel een gepinde modelrelease en geen automatische hertraining.

Bronnen:

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/.github/workflows/model-training.yml

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/Final/AUTOMATISERING.md

## Dia 25 — Conclusion and live demo

Richttijd: 75 seconden.

Beide taken hebben een reproduceerbare keten van brondata en preprocessing tot vergelijking en gehoste inference. RF300 wint de Mushroom-vergelijking op CV, maar mist nog 121 giftige gevallen in de ontwikkeltest. Een grens van 95% eetbare modelscore is onvoldoende: van 55 toegelaten gevallen in out-of-fold-training waren er zeven giftig. Dat is geen schatting van het risico van echte paddenstoelen en de scores zijn niet aantoonbaar gekalibreerd. Daarom toont de website giftig of onzeker/mogelijk giftig en bevestigt zij geen eetbaarheid. Deze berichtgeving verbetert de statistische classifiermetrics niet. RF320 verlaagt bij Citi Bike de test-RMSE met 24,12%, maar pieken en drift blijven beperkingen. Demonstreer kort het Mushroom-voorbeeld en de historische Citi Bike-demo met de volgende 24 uur. Warm de backend vooraf op. Houd ongeveer zeven minuten vrij voor vragen. Het team controleert de werkelijke bijdragen en de GenAI-vermelding volgens de opleidingspolicy.

Bronnen:

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/docs/mushroom_decisions.md

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/SolutionRaoul/Citybike/reports/final_evaluation.json

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/Final/INLEVERCHECK.md

## Dia 26 — Grootste fouten blijven rond drukke uren (verborgen reserve)

Reservedia voor vragen.

Reservedia. Residual is actual minus prediction. Positief betekent onderschatting. Ik onderzocht de verdeling, fouten tegenover voorspellingen, fouten over tijd en de grootste missers. Deze figuur toont gemiddelde absolute validatiefout per uur. Een normale verdeling betekent niet dat het model perfect is. Systematische fouten kunnen wijzen op ontbrekende informatie, veranderde omstandigheden of beperkingen van het model. Weer en evenementen zijn mogelijke factoren, zonder dat ik daarmee één fout causaal verklaar. De analyse gebruikt mei-juni-validatie. Ze rechtvaardigt geen nieuwe keuze op basis van de reeds bekeken test.

Bronnen: reports/validation_errors_by_hour.csv; reports/validation_residuals.csv; 07_eindvergelijking.ipynb (bestaande gecontroleerde uitwerking, 14 oktober 2026).

## Dia 27 — Keuzes die ik mondeling kan uitleggen (verborgen reserve)

Reservedia voor vragen.

Reservedia. De vaste selectieregel vergelijkt modeltypen: weekbaseline, Ridge, RF en LightGBM. Modellen die strikt minder dan één procent boven de minimale validatiefout zitten mogen winnen als ze eenvoudiger zijn volgens die rang. Binnen dezelfde RF-familie geldt de lagere validatiefout. Daarom behoud ik de getunede RF, ook al is de praktische extra winst klein. Daglags passen bij voorspellen om 00:00 en kennen geen uur van de voorspeldag. De scaler van Ridge wordt per trainfold gefit. Ik gebruik blokken omdat ritten en naburige uren samenhangen. Geen enkele normale residualverdeling garandeert goede toekomstvoorspellingen. Voor nieuwe keuzes na deze studie is een nieuwe onafhankelijke testperiode nodig.

Bronnen: citibike/modeling.py choose_model; citibike/features.py; hypotheseprotocol; model_selection.json (bestaande gecontroleerde uitwerking, 14 oktober 2026).

## Demo voorbereiden

1. Open de website vooraf zodat de Render-backend kan opstarten.
2. Toon kort het Mushroom-voorbeeld met de modelmelding.
3. Toon de historische Citi Bike-demo, bijvoorbeeld 30 april 2026 om 8 uur, en de 24 uurvoorspellingen.
4. Leg bij Citi Bike uit dat de demonstratie ontwikkeling gebruikt en geen nieuwe onafhankelijke test is.
5. Gebruik bij netwerkproblemen de opgeslagen notebookuitvoer en de screenshot in de presentatie.
