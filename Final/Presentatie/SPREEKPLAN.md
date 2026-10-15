# Raoul Buffels — spreektekst Citi Bike

Bij [de definitieve PowerPoint](Team10_presentatie.pptx). Mijn hoofddeel staat op dia 14–22: ongeveer 7 minuten en 15 seconden volgens de richttijden. Dia 26 en 27 zijn verborgen reserves voor vragen. De nummers hieronder verwijzen naar de volledige groepspresentatie.

## Mijn bijdrage

Mijn Citi Bike-deel omvat de verwerking van de bronarchieven, cleaning en kwaliteitscontroles, EDA en hypothesetoetsen, de voorbereiding van de modeldata, chronologische modelvergelijking en tuning, en de interpretatie van de vastgelegde eindtest. Ik werkte ook de oorspronkelijke Citi Bike-inference en applicatieonderdelen uit, controleerde de overeenkomst tussen offline voorspellingen en de API en maakte een lokale hertrainingspipeline voor dezelfde vaste Random Forest.

De gezamenlijke backend, UI, centrale GitHub-pipeline en Mushroom-training op AWS horen bij Jorres bijdrage. Mijn Citi Bike-onderdelen sluiten daarop aan. Codex hielp substantieel met implementatie, uitleg, tests en teksten. De code en de spreektekst worden niet als volledig zonder AI gemaakt voorgesteld. Het centrale [bijdragenoverzicht](../BIJDRAGEN.md) en de [GenAI-verantwoording](../GENAI_DISCLOSURE.md) beschrijven die afbakening.

## Korte versie als ik mijn bijdrage moet toelichten

“Ik heb het Citi Bike-gedeelte uitgewerkt, vanaf de grote bronbestanden en datakwaliteitscontrole tot de EDA, hypothesen en modelvergelijking. Ik koos en evalueerde een Random Forest voor de 24 uurvakken van een volgende dag. Daarnaast werkte ik de oorspronkelijke Citi Bike-voorspelfunctie en applicatieonderdelen uit en controleerde ik dat de API dezelfde voorspellingen geeft als offline. Ik maakte ook een lokale hertrainingspipeline voor dezelfde vaste configuratie. Jorre verzorgde de gezamenlijke integratie en hosting. Codex hielp bij de uitwerking en controles.”

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

## Dia 26 — Grootste fouten blijven rond drukke uren (verborgen reserve)

Reservedia voor vragen.

Reservedia. Residual is actual minus prediction. Positief betekent onderschatting. Ik onderzocht de verdeling, fouten tegenover voorspellingen, fouten over tijd en de grootste missers. Deze figuur toont gemiddelde absolute validatiefout per uur. Een normale verdeling betekent niet dat het model perfect is. Systematische fouten kunnen wijzen op ontbrekende informatie, veranderde omstandigheden of beperkingen van het model. Weer en evenementen zijn mogelijke factoren, zonder dat ik daarmee één fout causaal verklaar. De analyse gebruikt mei-juni-validatie. Ze rechtvaardigt geen nieuwe keuze op basis van de reeds bekeken test.

Bronnen: reports/validation_errors_by_hour.csv; reports/validation_residuals.csv; 07_eindvergelijking.ipynb (bestaande gecontroleerde uitwerking, 14 oktober 2026).

## Dia 27 — Keuzes die ik mondeling kan uitleggen (verborgen reserve)

Reservedia voor vragen.

Reservedia. De vaste selectieregel vergelijkt modeltypen: weekbaseline, Ridge, RF en LightGBM. Modellen die strikt minder dan één procent boven de minimale validatiefout zitten mogen winnen als ze eenvoudiger zijn volgens die rang. Binnen dezelfde RF-familie geldt de lagere validatiefout. Daarom behoud ik de getunede RF, ook al is de praktische extra winst klein. Daglags passen bij voorspellen om 00:00 en kennen geen uur van de voorspeldag. De scaler van Ridge wordt per trainfold gefit. Ik gebruik blokken omdat ritten en naburige uren samenhangen. Geen enkele normale residualverdeling garandeert goede toekomstvoorspellingen. Voor nieuwe keuzes na deze studie is een nieuwe onafhankelijke testperiode nodig.

Bronnen: citibike/modeling.py choose_model; citibike/features.py; hypotheseprotocol; model_selection.json (bestaande gecontroleerde uitwerking, 14 oktober 2026).

## Begrippen voor de vragenronde

- **RMSE:** de wortel van de gemiddelde gekwadrateerde fout. Grote missers wegen zwaarder mee. Hier is de eenheid ritten per lokaal uurvak.
- **MAE:** de gemiddelde absolute fout, in dezelfde eenheid.
- **R²:** de fout van het model vergeleken met steeds het gemiddelde voorspellen op dezelfde evaluatierijen. Een score van 0,828 betekent geen 82,8% juiste voorspellingen.
- **Weekbaseline:** voor elk uur de telling van hetzelfde uur vorige week voorspellen.
- **Leakage:** informatie gebruiken die tijdens de echte voorspelling nog niet beschikbaar zou zijn. Daarom gebruik ik afgesloten historie, een chronologische split en preprocessing per trainingsfold.
- **Validatie en eindtest:** mei–juni 2026 diende voor modelkeuze. De vastgelegde eindtest van juli–augustus 2026 beoordeelde daarna de gekozen RF en baseline. Een nieuwe modelkeuze vraagt een nieuwe onafhankelijke testperiode.
- **Lokale hertraining:** dezelfde configuratie opnieuw fitten en herladen controleren. Dat is geen nieuwe modelselectie en bewijst geen automatische live modelupdate.

## Resultaten om te onthouden

| Onderdeel | Resultaat |
| --- | --- |
| Model | Getunede Random Forest, 320 bomen |
| Voorspeldoel | Geregistreerde NYC-vertrekken per lokaal uurvak |
| Moment | Om 00:00 New York-tijd de 24 uurvakken van die dag |
| Invoerhistorie | Veertien volledige voorafgaande dagen, 336 uurvakken |
| Features | 14 kalender- en historievariabelen |
| Validatie-RMSE RF / weekbaseline | 1.940,49 / 2.826,16 |
| Eindtest-RMSE RF / weekbaseline | 1.979,51 / 2.608,78 |
| Lagere eindtest-RMSE | 24,12% |
| Eindtest-MAE / R² | 1.076,23 / 0,8280 |
| Eindtestomvang | 1.488 lokale uurvakken, juli–augustus 2026 |

De voorspelling gaat over geregistreerde vertrekken. Stationbeschikbaarheid en onvervulde vraag zijn geen gemeten modeldoelen. Twee zomermaanden geven geen garantie voor een volledig toekomstig jaar. Een tijdige tellingenfeed blijft nodig voor operationele voorspellingen.

Bronnen: [modelkaart](../../SolutionRaoul/Citybike/reports/MODEL_CARD.md), [eindtest](../../SolutionRaoul/Citybike/reports/final_evaluation.json), [hypothesen](../../SolutionRaoul/Citybike/reports/hypotheses.json), [modelvergelijking](../../SolutionRaoul/Citybike/reports/model_comparison.csv), [Citi Bike-inference en API](../../docs/citibike_raoul.md) en [centrale automatisering](../AUTOMATISERING.md).
