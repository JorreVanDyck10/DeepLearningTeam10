# Spreekplan bij de actuele groepspresentatie

Oefen circa twintig minuten inclusief korte demo; het volledige slot is 25 minuten inclusief vragen. Raouls bronvolgorde en spreektekst staan op dia 14–22. Alleen de oude status is bijgewerkt. Dia 27–28 zijn verborgen reserves. Jorres bijdrage staat op dia 3; de GenAI Disclosure sluit de zichtbare presentatie af op dia 26.

## Dia 1

Richttijd: 20 seconden. Twee problemen: Mushroom-classificatie en Citi Bike-uurvraag. Deze presentatie is met Codex bijgewerkt vanuit Andrews bronpresentatie en gecontroleerde projectresultaten. Het team controleert de GenAI-policy en kan de keuzes zelf toelichten.

## Dia 2

Richttijd: 15 seconden.

Overzicht. Ongeveer 19,5 minuten inclusief demo; houd 5,5 minuten vrij voor vragen.

Bronnen:

## Dia 3

Richttijd: 35 seconden.

Jorre Van Dyck bevestigde op 15 oktober zijn bijdragen: de pipeline opgesteld, de gezamenlijke backend ontwikkeld, de UI gebouwd en een model op AWS SageMaker getraind. Deze onderdelen zijn met substantiële hulp van Codex uitgewerkt. De bronbijdragen van Andrew, Raoul en Milan blijven behouden. De precieze eigen taken en AI-inzet van de andere leden worden door henzelf gecontroleerd.

Bronnen: Final/BIJDRAGEN.md en Final/GENAI_DISCLOSURE.md; foto door Jorre aangeleverd.

## Dia 4

Richttijd: 40 seconden.

Het juiste cursusbestand heeft 5.000 rijen; de oude UCI-download met 61.069 rijen is een andere studie. Drie numerieke en acht categorische modelvelden. jumbled_noise_0 verwijderd, jumbled_noise_1 behouden. Hypothetische cursusdata.

Bronnen:

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/MushroomFinal/README.md

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/MushroomFinal/mushroom_project_dataset.csv

## Dia 5

Richttijd: 45 seconden.

3.107 eetbare en 1.893 giftige gevallen. De meerderheid voorspellen levert ongeveer 62% accuracy. Daarom gebruiken we dummy en beslisboom als baselines, macro F1 en recall van de giftige klasse.

Bronnen:

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/MushroomFinal/mushroom_project_dataset.csv

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/MushroomFinal/comparison_correct/model_comparison.csv

## Dia 6

Richttijd: 60 seconden.

Spore print ontbreekt bij 94,8%; stem surface bij 67,18%; cap diameter bij 40,64%. Niet alle rijen weggooien. RF300 gebruikt mediaan voor getallen, modus en one-hot voor categorieën. Alles leert alleen uit de betreffende trainingsfold. De oorspronkelijke proximity-imputatie vóór CV gaf leakage-risico; de gecorrigeerde fits gebruiken fold-lokale preprocessing.

Bronnen:

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/MushroomFinal/mushroom_project_dataset.csv

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/MushroomFinal/final_model_support.py

## Dia 7

Richttijd: 40 seconden.

De stamhoogteverdelingen overlappen sterk. Eén kenmerk kan dus niet beslissen. Extreme waarden zijn niet automatisch meetfouten. We verwijderen ze niet blind; bij RF300 blijven nulmetingen nul. De grafiek komt uit Andrews EDA op dezelfde cursusdata.

Bronnen:

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/MushroomFinal/mushroom_project.ipynb

## Dia 8

Richttijd: 60 seconden.

Stratified split seed42: 4.100 train, 900 ontwikkeltest. Vijf gedeelde stratified trainingsfolds. Fit preprocessing en model opnieuw binnen elke fold. Selecteer op CV macro F1 en rapporteer giftige recall. De ontwikkeltest is eerder bekeken, dus geen onaangeraakte eindtest. Gecorrigeerde fits hebben andere scores dan oorspronkelijke screenshots.

Bronnen:

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/MushroomFinal/final_model_support.py

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/MushroomFinal/comparison_correct/protocol.json

## Dia 9

Richttijd: 65 seconden.

RF300 is geselecteerd op de gemiddelde macro F1 over de vijf trainingsfolds: 0,761876. De 300 bomen hebben max_depth=20, max_features=None, min_samples_split=4, min_samples_leaf=1, class_weight=balanced en random_state=42. De matrix en de scores horen bij dezelfde gecorrigeerde pipeline. Accuracy is (512+220)/900 = 81,33%. Giftige recall is 220/341 = 64,52%. De 121 giftige gevallen die als eetbaar geclassificeerd worden, tonen waarom accuracy onvoldoende is. De website gebruikt deze RF300, niet de oorspronkelijke RF200 uit Andrews eerdere screenshot. De preprocessing gebruikt medianen en modi, geen vooraf op alle trainingsrijen geleerde proximity-matrix.

Bronnen:

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/Final/config/mushroom.json

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/MushroomFinal/comparison_correct/model_comparison.csv

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/backend/artifacts/mushroom_rf300.json

## Dia 10

Richttijd: 55 seconden.

Gradient Boosting leert opeenvolgende bomen die fouten van eerdere bomen corrigeren. Logistic Regression is een lineaire referentie met scaling. In de gecontroleerde vergelijking wordt ook de proximity-imputatie van GB binnen elke fold geleerd. GB heeft TN536, FP23, FN171 en TP170. LR heeft TN410, FP149, FN136 en TP205. De oude GB-matrix en het oude rapport spraken elkaar tegen en zijn vervangen door cijfers uit dezelfde gecontroleerde fit. De gemiddelde fitduur per trainingsfold was ongeveer 0,31 seconde voor LR, 3,07 seconden voor GB en 6,36 seconden voor RF300. Dit zijn gemeten looptijden op die omgeving, geen garanties voor andere computers. LR is het snelste; RF300 heeft hier de hoogste CV macro F1.

Bronnen:

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/MushroomFinal/final_model_support.py

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/MushroomFinal/comparison_correct/model_comparison.csv

## Dia 11

Richttijd: 45 seconden.

PyCaret 3.3.2 is daadwerkelijk uitgevoerd op het juiste cursusbestand. De brede vergelijking retourneerde dertien modellen. De tabel toont de aparte vergelijking van drie standaardmodellen met dezelfde trainingsrijen en vijf folds. Het betreft geen handmatig getunede eindmodellen. Macro F1 is expliciet als macro-gemiddelde gecontroleerd. De getoonde recall betreft de giftige klasse. RF scoort hier beter dan GB en LR. Deze cijfers moeten niet rechtstreeks verwisseld worden met de volgende tabel met de uiteindelijke fits.

Bronnen:

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/MushroomFinal/comparison_correct/pycaret_audit.json

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/MushroomFinal/comparison_correct/pycaret_three.csv

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/MushroomFinal/comparison_correct/pycaret_all.csv

## Dia 12

Richttijd: 70 seconden.

Alle kandidaten gebruiken dezelfde 4.100 trainingsrijen, 900 ontwikkeltestrecords en vijf trainingsfolds. De primaire modelkeuze gebruikt gemiddelde CV macro F1. De beslisboom is de snelle eerste baseline; de dummy voorspelt de meerderheidsklasse. De AWS-rij komt uit de afzonderlijk gecontroleerde echte SageMaker-pipeline. Haar voorspellingen komen overeen met de lokale AWS-baseline. RF300 blijft de geselecteerde keuze, met een kleine voorsprong op AWS. De verschillen zijn geen bewijs van statistische significantie. De ontwikkeltest is eerder bekeken en is dus niet meer onaangeraakt. De tabel toont de belangrijkste modelfamilies en baselines; de volledige vergelijking in GitHub bevat ook andere RF-configuraties.

Bronnen:

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/MushroomFinal/comparison_correct/model_comparison.csv

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/MushroomFinal/comparison_correct/aws_artifact_audit.json

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/MushroomFinal/comparison_correct/cv_selection.json

## Dia 13

Richttijd: 50 seconden.

Jorre heeft in AWS SageMaker opnieuw getraind op het correcte cursusbestand met 5.000 rijen. Er zijn vier kandidaatconfiguraties onderzocht. Het uitgevoerde notebook, de herlaadbare pipeline, het resultatenbestand en de tuningtabel zijn gedownload naar de repository. Een aparte artifactaudit controleert de voorspellingen van de werkelijk gedownloade pipeline: 80,11% accuracy en 58,94% recall voor de giftige klasse. De oude 100%-run gebruikte een ander databestand en hoort niet in deze vergelijking. De AWS-training is uitgevoerd, maar RF300 blijft het gekozen gehoste model.

Bronnen:

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/SolutionJorre/MushroomDataset/correct_dataset/04_aws_correct_dataset.ipynb

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/SolutionJorre/MushroomDataset/correct_dataset/correct_aws_tuning_results.csv

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/MushroomFinal/comparison_correct/aws_artifact_audit.json

## Dia 14

Richttijd: 25 seconden.

Ik ben Raoul Buffels. Ons team heet Messi10. In dit deel bespreek ik mijn Citi Bike-uitwerking. Ik werkte aan de bronverwerking, cleaning, EDA, hypothesen, modellen en vergelijking. Ik maakte ook de Citi Bike-frontend en backend. Jorre hielp met deployment op Render. De andere teamleden presenteren hun eigen werk. Ik volg de opdrachtvolgorde: EDA, modellen en conclusie. Codex hielp bij implementatie en uitleg, zoals vermeld in de repository. Ik moet de keuzes zelf kunnen verdedigen.

Bronnen: README.md en SolutionRaoul/Citybike/notebooks/00_projectoverzicht.ipynb; opdracht: CloudAI/Discussion topics/project assignment.md (bestaande gecontroleerde uitwerking, 14 oktober 2026).

## Dia 15

Richttijd: 45 seconden.

Ik had grote bestanden verwacht. De lastigste stap was bepalen welke records echte bronkopieën waren. De 175 ZIPs bevatten NYC en Jersey City en verschillende oude en nieuwe schema’s. Na verwerking blijven circa 323,8 miljoen NYC-records over. Dat is technische dekking tot augustus 2026, geen EDA op de eindtest. Mijn EDA gebruikt alleen ontwikkeling vóór mei. Een ontbrekend eindstation hoeft een geldige start niet onbruikbaar te maken. Extreem lange duur kan een registratieprobleem of een laat afgesloten rit zijn. Voor uurvolume tel ik de start en behoud ik kwaliteitsflags. Alleen bewezen bronkopieën verwijder ik. Identieke historische ritten kunnen echte afzonderlijke ritten zijn.

Bronnen: reports/data_manifest.json; 01a_dataonderzoek_en_cleaning.ipynb; reports/NYC_duration_sample_audit.json (bestaande gecontroleerde uitwerking, 14 oktober 2026).

## Dia 16

Richttijd: 45 seconden.

Deze figuur gebruikt echte gemiddelde uurtellingen vanaf 2023 tot eind april 2026. Ik vergelijk werkdagen met weekends. Werkdagen hebben een ander dagprofiel en vooral een duidelijke avondpiek. Het hoogste gemiddelde uur ligt hier om 17 uur op werkdagen en om 15 uur in het weekend. Een verschil tussen dagtypes past bij woon-werkgebruik, maar de data bewijzen het reisdoel niet. Ik verwachte herhaling in tijd. De grafiek toont dat de patronen veranderen met dagtype en dat één algemeen gemiddelde onvoldoende is. Dat motiveert kalenderkenmerken en historische tellingen in het model.

Bronnen: data/NYC_hourly_development.parquet, dezelfde aggregatie als citibike/plots.py hour_weekday (bestaande gecontroleerde uitwerking, 14 oktober 2026).

## Dia 17

Richttijd: 55 seconden.

Mijn hoofdhypothese is dat een uurtelling samenhangt met hetzelfde uur vorige week. H0 zegt dat de gemiddelde correlatie over tijdsblokken nul is. H1 zegt dat die verschilt van nul. Ik behandel miljoenen ritten niet als onafhankelijke observaties. Ik bereken effecten per week en toets met 43 vierweekblokken. De bestaande sign-fliptoets veronderstelt voldoende onafhankelijke blokken en symmetrie onder H0. De correlatie is ongeveer 0,856 met een interval van 0,837 tot 0,876. De Holm-p is ongeveer 0,00015. Het patroon is dus sterk in ontwikkeling. De vraag kwam uit EDA en blijft exploratief. Correlatie bewijst geen oorzaak of modelprestatie. Pas daarna kies ik het bestaande doel: geregistreerd uurvolume één dag vooruit.

Bronnen: reports/hypotheses.json; reports/hypothesis_protocol.json; reports/target_decision.json (bestaande gecontroleerde uitwerking, 14 oktober 2026).

## Dia 18

Richttijd: 45 seconden.

Notebook 02 maakt zelfstandig de definitieve modeldata vanaf de bronnen of gecontroleerde caches. Het target is geregistreerde vertrekken per lokaal uurvak. Ik gebruik veertien features: kalenderkenmerken en historische tellingen. Voorbeelden zijn het uur en de weekdag, hetzelfde uur gisteren en vorige week, en een gemiddelde van zeven afgesloten dagen. Rolling wordt eerst verschoven. Niets van de voorspeldag komt als historische input binnen. De eerste veertien dagen vallen weg omdat historie ontbreekt. Ik split chronologisch: training vóór mei, validatie mei en juni, test juli en augustus. Binnen training gebruik ik vijf chronologische folds van 28 dagen. Ridge leert zijn scaler alleen op de trainingsrijen van iedere fold.

Bronnen: 02_definitieve_voorbereiding.ipynb; reports/prepared_model_data.json; citibike/features.py (bestaande gecontroleerde uitwerking, 14 oktober 2026).

## Dia 19

Richttijd: 65 seconden.

De weekbaseline voorspelt hetzelfde uur van vorige week. Ridge is een eenvoudige lineaire referentie. Random Forest en LightGBM leren niet-lineaire verbanden. PyCaret vergelijkt dezelfde vier basismodellen met dezelfde chronologische folds. Daarna behoud ik de beperkte bestaande RF- en LightGBM-tuning. De finale getunede RF heeft een validatie-RMSE van ongeveer 1940,49 ritten per uurvak tegenover 2826,16 voor de baseline. Dat is circa 31,34 procent lager. LightGBM-tuning verbeterde trainings-CV, maar niet de mei-juni-validatie. De verschillen tussen de sterke modellen zijn klein. De keuze lag vóór de test vast en gebruikte de bestaande 1%-regel met een eenvoudigheidsrang per modeltype.

Bronnen: reports/model_comparison.csv; reports/model_selection.json; reports/pycaret_comparison.csv; tuningrapporten (bestaande gecontroleerde uitwerking, 14 oktober 2026).

## Dia 20

Richttijd: 55 seconden.

Ridge is het eenvoudigst uit te leggen en fit in deze reproductie in ongeveer 0,05 seconde. LightGBM fit in minder dan een seconde, terwijl RF tientallen seconden nodig heeft. Die tijden zijn echte afzonderlijke fits op deze computer en niet de volledige tuningduur. De getunede RF heeft een trainings-RMSE van ongeveer 536 tegenover 1940 op validatie. Dat is een grote kloof. Overfit kan meespelen, maar de vroege trainingsjaren hebben ook minder intensiteit dan 2026. Ik kan die kloof dus niet uitsluitend aan overfit toeschrijven. RF-tuning bracht maar circa 0,27 procent extra validatiewinst en een groot modelbestand. Dat bespreek ik naast de score. Ik wijzig de keuze niet nadat de test al bekeken is.

Bronnen: reports/notebook_reproduction/*.json, fit_seconds; reports/model_comparison.csv; reports/model_artifact.json (bestaande gecontroleerde uitwerking, 14 oktober 2026).

## Dia 21

Richttijd: 45 seconden.

Ik bespreek de bestaande eindtest juli en augustus 2026. Alleen de vooraf geselecteerde RF en de baseline zijn daarop vergeleken. De model-RMSE is ongeveer 1979,51 tegenover 2608,78 voor de baseline. Dat is 24,12 procent lager. De MAE is ongeveer 1076 ritten per uurvak en R² is 0,828. Deze scores horen bij de finale refit tot eind juni, niet bij de ontwikkelingsfit op de vorige dia. De test was al bekeken. Ik gebruik hem nu niet om features of instellingen te veranderen. Twee zomermaanden garanderen geen prestaties voor een heel toekomstig jaar. Weer en evenementen ontbreken, en recente historische tellingen moeten op tijd beschikbaar zijn.

Bronnen: reports/final_evaluation.json; reports/MODEL_CARD.md (bestaande gecontroleerde uitwerking, 14 oktober 2026).

## Dia 22

Richttijd: 55 seconden.

Mijn uitwerking eindigt in een eigen webpagina en API. Een gebruiker kiest een datum en levert veertien volledige voorgaande dagen aan. De API gebruikt dezelfde preprocessing en predict_day als offline. De screenshot toont een ontwikkelingsdemonstratie en is geen onafhankelijk testresultaat. Ik heb voorspellingsovereenkomst gecontroleerd, ook bij beide DST-overgangen. Mijn lokale pipeline traint dezelfde RF opnieuw, controleert herladen en bewaart een apart artefact. De groepsintegratie is inmiddels bijgewerkt: er zijn geslaagde GitHub Actions-runs voor Mushroom-training en modelpublicatie. Render en Netlify volgen de gevalideerde publicatie. Het gedownloade Mushroom-SageMaker-model is afzonderlijk gecontroleerd. Deze centrale workflow traint Citi Bike niet automatisch opnieuw: die API gebruikt de gepinde modelrelease. De lokale Citi Bike-hertraining blijft een afzonderlijk traject. Mijn conclusie is dat historische tellingen bruikbaar zijn voor geregistreerd uurgebruik, onder een duidelijk invoercontract.

Bronnen: docs/citibike_raoul_frontend.jpg; DEPLOYMENT.md; reports/inference_parity.json; reports/retraining_pipeline_latest.json; OPDRACHTCONTROLE.md (bestaande gecontroleerde uitwerking, 14 oktober 2026).

Statusupdate: Final/AUTOMATISERING.md en Final/VALIDATIE.md.

## Dia 23

Richttijd: 30 seconden.

De frontend gebruikt HTML, CSS en JavaScript op Netlify. De backend is Python met FastAPI op Render. De API valideert invoer en gebruikt het geselecteerde model. Mushroom gebruikt een verliesvrije NumPy-export van RF300; paritycontroles vergelijken de uitkomsten met de getrainde pipeline. Citi Bike laadt een gepinde modelrelease met checksum en memory mapping. Beide voorspeltaken gebruiken dezelfde API. Voor Mushroom controleert de frontend ook het model-id en de exacte modelrevisie. Dit voorkomt dat frontend en backend ongemerkt verschillende modelversies gebruiken. Het is een eigen website met API, geen Streamlit-app.

Bronnen:

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/backend/main.py

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/backend/mushroom_model.py

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/backend/raoul_citibike.py

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/frontend/README.md

## Dia 24

Richttijd: 40 seconden.

Een relevante code-, configuratie- of datapush op GitHub start de Mushroom-workflow. Python 3.11 voert vijfvoudige cross-validatie en een refit van de vastgelegde RF300-configuratie uit. De CV macro F1 en giftige recall moeten de referentiedrempels halen. Een aparte Python 3.13-job controleert het servingexport en de API met beide modellen. De export wordt op alle 5.000 rijen vergeleken met de gefitte kandidaat. Pas na geslaagde controles worden model en metadata gepubliceerd. Een inmiddels verouderde broncommit wordt geweigerd. Render en Netlify volgen de gevalideerde publicatie. Dit is automatisch herfitten van de gekozen configuratie, geen nieuwe automatische hyperparameterzoektocht. Voor Citi Bike is er momenteel een gepinde modelrelease en geen automatische hertraining.

Bronnen:

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/.github/workflows/model-training.yml

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/Final/AUTOMATISERING.md

## Dia 25

Richttijd: 75 seconden.

Beide taken hebben een reproduceerbare keten van brondata en preprocessing tot vergelijking en gehoste inference. RF300 wint de Mushroom-vergelijking op CV, maar mist nog 121 giftige gevallen in de ontwikkeltest. Een grens van 95% eetbare modelscore is onvoldoende: van 55 toegelaten gevallen in out-of-fold-training waren er zeven giftig. Dat is geen schatting van het risico van echte paddenstoelen en de scores zijn niet aantoonbaar gekalibreerd. Daarom toont de website giftig of onzeker/mogelijk giftig en bevestigt zij geen eetbaarheid. Deze berichtgeving verbetert de statistische classifiermetrics niet. RF320 verlaagt bij Citi Bike de test-RMSE met 24,12%, maar pieken en drift blijven beperkingen. Demonstreer kort het Mushroom-voorbeeld en de historische Citi Bike-demo met de volgende 24 uur. Warm de backend vooraf op. Houd ongeveer zeven minuten vrij voor vragen. Het team controleert de werkelijke bijdragen en de GenAI-vermelding volgens de opleidingspolicy.

Bronnen:

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/docs/mushroom_decisions.md

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/SolutionRaoul/Citybike/reports/final_evaluation.json

https://github.com/JorreVanDyck10/DeepLearningTeam10/blob/main/Final/INLEVERCHECK.md

## Dia 26

Richttijd: 25 seconden.

Codex is een hulpmiddel, geen teamlid. Jorres bevestigde bijdragen zijn pipeline, backend, UI en het AWS-model. De AI-hulp omvat ook substantiële code en proza. De uitvoer en referentievoorspellingen zijn gecontroleerd; menselijke review en zelfstandige beheersing moeten door ieder teamlid worden bevestigd. De vijf echte prompts, concrete bijsturing en bronverwijzingen staan hieronder.

# GenAI Disclosure

Deze verantwoording hoort bij de volledige eindoplevering van DeepLearningTeam10.

Zij is op 15 oktober 2026 uitgewerkt tegen de door Jorre aangeleverde

`GenAI_Policy.pdf` (twee pagina's). De eerdere korte disclaimers blijven als

historische bronvermeldingen bestaan; dit document geeft de actuele, bredere scope.

## 1. Functionele scope: AI-code en menselijke bijdragen

De gebruikte assistent in deze chat is **OpenAI Codex**. Codex is een hulpmiddel,

geen teamlid. Jorre bevestigde dat hij de pipeline heeft opgesteld, de backend

heeft ontwikkeld, de UI heeft gebouwd en een model op AWS heeft getraind.

Dit beschrijft zijn projectbijdrage en verantwoordelijkheid; het is geen claim

dat alle bijbehorende code zonder AI of volledig vanaf nul is geschreven.

| Onderdeel | Aantoonbare hulp van Codex in deze chat | Menselijke inbreng / afbakening |

| --- | --- | --- |

| Structuur en notebooks | Structuur, templates, downloadcode, aparte Final-leesversies en toelichtingen gegenereerd of aangepast. | Jorre bepaalde de opdrachtcontext en vroeg om een aparte Final-map met behoud van eerder werk. Oorspronkelijke onderzoeken behouden hun bestaande auteursvermelding. |

| Mushroom-evaluatie | Vergelijkingscode, gedeelde splits/folds, fold-lokale preprocessing, PyCaret-reparatie en -uitvoering, AWS-artifactaudit en foutenanalyse uitgewerkt en uitgevoerd. | Jorre wees de 100%-score af als verdacht en corrigeerde de voorgeschreven bron naar het cursusbestand van 5.000 rijen. Andrew leverde de oorspronkelijke Mushroom-uitwerking. |

| AWS | Notebookcode voorbereid, foutmeldingen toegelicht en de gedownloade pipeline en resultaten geaudit. | Jorre voerde de training op SageMaker uit en leverde het uitgevoerde notebook, model en tuningresultaten aan. |

| Pipeline | Trainings-, validatie-, export-, test- en publicatiecode en GitHub Actions-configuratie gegenereerd en aangepast. | Jorre stelde de pipeline op met deze hulp en vroeg expliciet om automatische updates na een GitHub-push. Nieuwe training geldt hier voor Mushroom; Citi Bike gebruikt een vaste release. |

| Backend en UI | FastAPI-routes, modelkoppelingen, lossless exports, invoercontrole, JavaScript/HTML/CSS, tests en versiecontroles gegenereerd of aangepast. | Jorre ontwikkelde en stuurde de gezamenlijke backend en UI met deze hulp. Raouls oorspronkelijke Citi Bike-inference en applicatieonderdelen blijven als bronbijdrage vermeld. |

| Beslismelding Mushroom | Drempelaudit en de voorzichtige API/UI-melding uitgewerkt. | Jorre gaf aan dat onterecht eetbaar melden ernstiger is en vroeg om een strengere beslisregel. Een onbewezen grens van 95% werd na controle niet als veilige oplossing gebruikt. |

| Presentatie | Tekst, structuur, tabellen, grafieken, opmaak en spreeknotities samengesteld of bijgewerkt. | Andrew leverde de bronpresentatie. Op Jorres aanwijzing zijn Raouls bestaande volgorde en spreektekst behouden; Jorre leverde zijn foto aan. De eerdere twaalfdia-versie heeft een AI-coverillustratie; de huidige groepsversie gebruikt de aangeleverde foto's en bestaande bronbeelden. |

De AI-hulp was dus substantieel en omvat ook inhoudelijke implementatie,

evaluatiehelpers en rapportage. Er wordt geen lijst van 'volledig zelfgeschreven'

codeblokken verzonnen: dat onderscheid is voor de oudere teamonderdelen niet

volledig uit deze chat vast te stellen. Ieder teamlid moet de eigen bijdrage en

eventueel aanvullend AI-gebruik zelf preciseren. Codex heeft de oorspronkelijke

datasetonderzoeken niet onafhankelijk van de bronbijdragen geclaimd.

## 2. Proza en structuur

Codex hielp met het schrijven en herschikken van README-bestanden, Markdown in

notebooks, modelvergelijkingen, conclusies, deploymentdocumentatie, inlevercheck,

presentaties en spreeknotities. Ook deze verantwoording is door Codex opgesteld

op basis van de echte chat, bestanden en policy. Het is geen automatisch bewijs

dat alle teamleden alle teksten al hebben gelezen of begrepen.

## 3. Vijf echte prompts uit de ontwikkelworkflow

Onderstaande prompts zijn letterlijk overgenomen uit Jorres berichten in deze

chat, inclusief oorspronkelijke spelling. Het zijn geen achteraf verzonnen

voorbeeldprompts. Zij tonen ook bijsturing en controle, naast verzoeken om code.

1. Automatisering:

   > Maak het nu dat als er op github gepushed word dat de backend automatisch geupdate word en dat de frontend automatisch de backend ophaald

2. Vergelijking aan de opdracht toetsen:

   > Maak nu volgens de taak wat we moeten doen de vergelijking tussen de modellen

3. Een verdachte uitkomst kritisch bevragen:

   > Het kan toch niet dat dat aws model het beste is. 100% juist is wel heel verdacht

4. De databron corrigeren:

   > En dit er nog bij: Zorg er ook voor dat je het excel bestand gebruikt dat ik heb geupload naar github. De vorige die we gebruikten met 61k rijen was de foute.

5. De foutkosten en beslisregel sturen:

   > Iegenlijk zou het pas eetbaar mogen tonen als het 95% eetbaar is ofzo niet? Of iets anders wat jij denkt. want we willen niet dat het eetbaar toont als het giftig is. giftig tonen als het eetbaar is is minder erg

## 4. Verificatie en reproduceerbaarheid

De gerapporteerde projectmetrics komen uit bewaarde uitvoer en uitgevoerde code,

niet uit door AI bedachte scores. De technische bewijsstukken zijn:

- De [uitvoeringsregistratie](../execution_checks.json): twaalf opnieuw uitgevoerde

  Final-notebooks. Andere Citi Bike-resultaten zijn expliciet bewaarde bronuitvoer,

  geen fictieve nieuwe training en geen opnieuw geopende eindtest.

- Het [bronmanifest](../source_manifest.json) en de vaste bronhash/splits: dezelfde

  5.000 cursusrijen en gedeelde trainingfolds voor de Mushroom-vergelijking.

- De [Mushroom-vergelijking](../../MushroomFinal/comparison_correct/model_comparison.csv),

  [AWS-artifactaudit](../../MushroomFinal/comparison_correct/aws_artifact_audit.json) en

  [drempelaudit](../../MushroomFinal/comparison_correct/decision_threshold_audit.json).

- De [Citi Bike-eindbeoordeling](../../SolutionRaoul/Citybike/reports/final_evaluation.json)

  en het [evaluatieslot](../../SolutionRaoul/Citybike/reports/evaluation_lock.json).

- Tests op backend, frontend, modelpublicatie en referentievoorspellingen, plus

  uitgevoerde Actions-runs en live controles. De aantallen en runlinks staan in

  [VALIDATIE.md](../VALIDATIE.md). Een nieuwe documentatiewijziging is geen nieuwe

  training of extra prestatiemeting.

Dit zijn technische controles, deels uitgevoerd met Codex. Zij vervangen de

persoonlijke inhoudelijke review en mondelinge verdediging niet.

## 5. Concrete correcties en grenzen

| Aanleiding | Bijsturing en controle |

| --- | --- |

| Verdachte 100%-score en verkeerde bron | Jorre bevroeg de uitkomst en gaf aan dat het 61k-bestand fout was. De vergelijking is vervangen door dezelfde gecontroleerde 5.000 cursusrijen. Historische scores worden niet als finale resultaten verkocht. |

| Leakage-risico bij vooraf geleerde preprocessing | De complete preprocessing wordt per trainingsfold geleerd. De gecorrigeerde fits hebben eigen referentievoorspellingen; oude screenshots worden niet met nieuwe scores gemengd. |

| Eetbaar tonen ondanks een giftigheidsscore | De 95%-regel is onderzocht. In trainings-OOF bleken zeven van 55 toegelaten gevallen giftig. De UI bevestigt daarom geen eetbaarheid; dit wordt niet als hogere classifieraccuracy of 100% recall gepresenteerd. |

| Tegenstrijdige presentatiecijfers | In Andrews presentatie paste de GB-confusion matrix niet bij het classificatierapport. Die combinatie is vervangen door de cijfers van één gecontroleerde fit; RF200 is niet verward met de gehoste RF300. |

| Geleerde verhaallijn van Raoul | Jorre vroeg om diens eigen dia's te gebruiken. De volgorde en spreektekst zijn behouden; alleen de verouderde projectstatus is bijgewerkt. |

Scopekeuzes: originele bestanden en lokale bijdragen behouden; de eerder bekeken

Mushroom-test als ontwikkeltest benoemen; geen nieuwe modelkeuze op de vergrendelde

Citi Bike-eindtest; en geen automatische Citi Bike-hertraining claimen die de

centrale workflow niet uitvoert. De publicatie weigert een kandidaat die de

kwaliteitscontroles niet haalt of op een verouderde broncommit gebaseerd is.

## 6. Intellectueel eigenaarschap en nog te bevestigen review

Het team draagt de eindverantwoordelijkheid voor code, cijfers, beveiliging,

interpretatie en conclusies. Ieder lid moet zelfstandig kunnen uitleggen waarom

de split en preprocessing zo gekozen zijn, wat macro F1/recall/RMSE meten, waarom

het geselecteerde model gekozen werd en hoe model, API en website samenwerken.

Een antwoord als 'de AI stelde dit voor' is geen technische onderbouwing.

Jorres vier genoemde bijdragen zijn bevestigd. De precieze bijdragen en AI-inzet

van de overige leden en hun zelfstandige beheersing zijn hiermee niet automatisch

bevestigd. Die menselijke eindcontrole blijft open in de

[inlevercheck](../INLEVERCHECK.md). De aanwezigheid van deze tekst is evenmin bewijs

van een Canvas-indiening.

## Dia 27 — reserve

Reservedia voor vragen.

Reservedia. Residual is actual minus prediction. Positief betekent onderschatting. Ik onderzocht de verdeling, fouten tegenover voorspellingen, fouten over tijd en de grootste missers. Deze figuur toont gemiddelde absolute validatiefout per uur. Een normale verdeling betekent niet dat het model perfect is. Systematische fouten kunnen wijzen op ontbrekende informatie, veranderde omstandigheden of beperkingen van het model. Weer en evenementen zijn mogelijke factoren, zonder dat ik daarmee één fout causaal verklaar. De analyse gebruikt mei-juni-validatie. Ze rechtvaardigt geen nieuwe keuze op basis van de reeds bekeken test.

Bronnen: reports/validation_errors_by_hour.csv; reports/validation_residuals.csv; 07_eindvergelijking.ipynb (bestaande gecontroleerde uitwerking, 14 oktober 2026).

## Dia 28 — reserve

Reservedia voor vragen.

Reservedia. De vaste selectieregel vergelijkt modeltypen: weekbaseline, Ridge, RF en LightGBM. Modellen die strikt minder dan één procent boven de minimale validatiefout zitten mogen winnen als ze eenvoudiger zijn volgens die rang. Binnen dezelfde RF-familie geldt de lagere validatiefout. Daarom behoud ik de getunede RF, ook al is de praktische extra winst klein. Daglags passen bij voorspellen om 00:00 en kennen geen uur van de voorspeldag. De scaler van Ridge wordt per trainfold gefit. Ik gebruik blokken omdat ritten en naburige uren samenhangen. Geen enkele normale residualverdeling garandeert goede toekomstvoorspellingen. Voor nieuwe keuzes na deze studie is een nieuwe onafhankelijke testperiode nodig.

Bronnen: citibike/modeling.py choose_model; citibike/features.py; hypotheseprotocol; model_selection.json (bestaande gecontroleerde uitwerking, 14 oktober 2026).
