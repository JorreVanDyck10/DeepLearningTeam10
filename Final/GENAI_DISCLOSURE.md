# GenAI Disclosure

Deze verantwoording hoort bij de volledige eindoplevering van DeepLearningTeam10.
Zij is op 15 oktober 2026 uitgewerkt tegen de door Jorre aangeleverde
`GenAI_Policy.pdf` (twee pagina's). De eerdere korte disclaimers blijven als
historische bronvermeldingen bestaan; dit document geeft de actuele, bredere scope.
Raoul heeft op dezelfde datum zijn eigen bijdrage en AI-gebruik aangevuld en bevestigd.

## 1. Functionele scope: AI-code en menselijke bijdragen

De gebruikte assistent bij de hieronder beschreven werkzaamheden is **OpenAI Codex**. Codex is een hulpmiddel,
geen teamlid. Jorre bevestigde dat hij de pipeline heeft opgesteld, de backend
heeft ontwikkeld, de UI heeft gebouwd en een model op AWS heeft getraind.
Dit beschrijft zijn projectbijdrage en verantwoordelijkheid; het is geen claim
dat alle bijbehorende code zonder AI of volledig vanaf nul is geschreven.
Raoul bevestigde zijn Citi Bike-onderzoek, datavoorbereiding, modellen en
vergelijking, oorspronkelijke inference- en applicatieonderdelen, documentatie
en presentatieonderdeel. Ook die bijdrage is met substantiële AI-hulp uitgewerkt.

| Onderdeel | Hulp van Codex bij de beschreven werkzaamheden | Menselijke inbreng / afbakening |
| --- | --- | --- |
| Structuur en notebooks | Structuur, templates, downloadcode, aparte Final-leesversies en toelichtingen gegenereerd of aangepast. | Jorre bepaalde de opdrachtcontext en vroeg om een aparte Final-map met behoud van eerder werk. Oorspronkelijke onderzoeken behouden hun bestaande auteursvermelding. |
| Citi Bike | Hulp bij bronverwerking, cleaning, aggregaties, EDA, hypothesen, features, chronologische splits, baseline/Ridge, PyCaret, Random Forest, LightGBM, tuning, foutenanalyse, inference, tests en notebookuitleg. | Raoul werkte het Citi Bike-onderdeel uit, beoordeelde voorstellen en resultaten kritisch en stuurde bij aan de hand van de opdracht, brondata en uitvoer. Hij onderbouwde de modelkeuze op validatie; de oorspronkelijke eindtest bleef beschermd. Jorre hielp met deployment op Render. |
| Mushroom-evaluatie | Vergelijkingscode, gedeelde splits/folds, fold-lokale preprocessing, PyCaret-reparatie en -uitvoering, AWS-artifactaudit en foutenanalyse uitgewerkt en uitgevoerd. | Jorre wees de 100%-score af als verdacht en corrigeerde de voorgeschreven bron naar het cursusbestand van 5.000 rijen. Andrew leverde de oorspronkelijke Mushroom-uitwerking. |
| AWS | Notebookcode voorbereid, foutmeldingen toegelicht en de gedownloade pipeline en resultaten geaudit. | Jorre voerde de training op SageMaker uit en leverde het uitgevoerde notebook, model en tuningresultaten aan. |
| Pipeline | Trainings-, validatie-, export-, test- en publicatiecode en GitHub Actions-configuratie gegenereerd en aangepast. | Jorre stelde de pipeline op met deze hulp en vroeg expliciet om automatische updates na een GitHub-push. Nieuwe training geldt hier voor Mushroom; Citi Bike gebruikt een vaste release. |
| Backend en UI | FastAPI-routes, modelkoppelingen, lossless exports, invoercontrole, JavaScript/HTML/CSS, tests en versiecontroles gegenereerd of aangepast. | Jorre ontwikkelde en stuurde de gezamenlijke backend en UI met deze hulp. Raouls oorspronkelijke Citi Bike-inference en applicatieonderdelen blijven als bronbijdrage vermeld. |
| Beslismelding Mushroom | Drempelaudit en de voorzichtige API/UI-melding uitgewerkt. | Jorre gaf aan dat onterecht eetbaar melden ernstiger is en vroeg om een strengere beslisregel. Een onbewezen grens van 95% werd na controle niet als veilige oplossing gebruikt. |
| Presentatie | Tekst, structuur, tabellen, grafieken, opmaak en spreeknotities samengesteld of bijgewerkt. | Andrew leverde de bronpresentatie. Op Jorres aanwijzing zijn Raouls bestaande volgorde en spreektekst behouden; Jorre leverde zijn foto aan. De eerdere twaalfdia-versie heeft een AI-coverillustratie; de huidige groepsversie gebruikt de aangeleverde foto's en bestaande bronbeelden. |

De AI-hulp was dus substantieel en omvat ook inhoudelijke implementatie,
evaluatiehelpers en rapportage. Er wordt geen lijst van 'volledig zelfgeschreven'
codeblokken verzonnen: dat onderscheid is voor de oudere teamonderdelen niet
volledig uit de beschikbare chats en bestanden vast te stellen. Jorre en Raoul
hebben hun eigen bijdrage en AI-gebruik bevestigd; Andrew en Milan preciseren
hun eigen bijdrage en eventueel aanvullend AI-gebruik nog. Codex heeft de oorspronkelijke
datasetonderzoeken niet onafhankelijk van de bronbijdragen geclaimd.

## 2. Proza en structuur

Codex hielp met het schrijven en herschikken van README-bestanden, Markdown in
notebooks, modelvergelijkingen, conclusies, deploymentdocumentatie, inlevercheck,
presentaties en spreeknotities. Ook deze verantwoording is door Codex opgesteld
op basis van de echte chat, bestanden en policy. Het is geen automatisch bewijs
dat alle teamleden alle teksten al hebben gelezen of begrepen.

Voor Raouls onderdeel hielp Codex ook met de Nederlandstalige notebookuitleg,
de beschrijving van modelkeuzes en beperkingen, de documentatie en zijn
presentatietekst. Raoul beoordeelde de inhoud en interpretaties; de teksthulp
neemt zijn verantwoordelijkheid voor wat hij presenteert niet over.

### Raouls gebruik van AI als leidraad en ondersteuning

Voor mijn Citi Bike-onderdeel gebruikte ik **OpenAI Codex als leidraad en
ondersteuning** tijdens het uitwerken van het project. Ik vroeg uitleg over
begrippen, mogelijke aanpakken en foutmeldingen, zodat ik de stappen en
modelresultaten beter kon begrijpen en beoordelen. De voorstellen hielpen mij
om mijn werk te structureren en keuzes af te wegen. Daarnaast hielp Codex
substantieel met het genereren en aanpassen van code, het uitvoeren van
controles en het formuleren van teksten.

Die ondersteuning had betrekking op de volgende onderdelen van mijn werk:

| Onderdeel van mijn bijdrage | Waarvoor ik AI als leidraad en ondersteuning gebruikte |
| --- | --- |
| Projectaanpak en notebooks | De opdracht vertalen naar een werkvolgorde, notebooks structureren en onderzoeksstappen toelichten. |
| Bronverwerking en cleaning | Download- en verwerkingscode uitwerken, grote ZIP-bestanden en verschillende schema's verwerken, NYC en Jersey City apart behandelen en uitleg krijgen over bronoverlap, ontbrekende waarden, afwijkende ritduren en aggregatie naar uurtellingen. |
| EDA en hypothesen | Grafieken opstellen, patronen onderzoeken, onderzoeksvragen en statistische hypothesen formuleren en de uitkomsten en beperkingen van toetsen interpreteren. |
| Datavoorbereiding en features | Het voorspeldoel en voorspeltijdstip uitwerken, kalenderkenmerken en historische tellingen voorbereiden en chronologische splits, folds, preprocessing en mogelijke datalekken controleren. |
| Modellen en tuning | De weekbaseline, Ridge, PyCaret, Random Forest en LightGBM uitwerken en vergelijken, hyperparameters begrijpen, tuningcode aanpassen en technische fouten oplossen. |
| Evaluatie en modelkeuze | Uitleg krijgen over RMSE, MAE en R², validatie-uitkomsten en fouten analyseren en de modelkeuze onderbouwen. De oorspronkelijke eindtest werd niet opnieuw gebruikt om modellen te kiezen; latere experimenten bleven verkennend. |
| Inference en applicatie | De oorspronkelijke Citi Bike-voorspelfunctie, lokale Streamlit-demo en applicatieonderdelen uitwerken, invoercontroles begrijpen en nagaan of offline code en de API dezelfde voorspellingen geven. Jorre hielp met de deployment op Render en verzorgde de verdere gezamenlijke backend en UI. |
| Reproduceerbaarheid en lokale hertraining | Tests en controles uitwerken, modelopslag en herladen controleren en een lokale hertrainingspipeline voor dezelfde vaste Random Forest opzetten. De voorbereide Citi Bike-workflow bewijst op zichzelf geen geslaagde automatische GitHub-run of automatische live modelupdate. |
| Documentatie en presentatie | Notebookuitleg, README-bestanden, conclusies, beperkingen, deploymentdocumentatie en mijn presentatieonderdeel met spreektekst en voorbereiding op vragen formuleren en verduidelijken. |

Ik nam de antwoorden niet zomaar over. Ik beoordeelde of de voorstellen
aansloten op de opdracht en de Citi Bike-data, vergeleek conclusies met de
opgeslagen resultaten en liet fouten of onduidelijke uitleg bijsturen. AI was
daarbij een hulpmiddel om tot een onderbouwde uitwerking te komen, met hulp
bij zowel begrip en aanpak als code, uitvoering en teksten. Ik blijf
verantwoordelijk voor de keuzes, interpretatie en conclusies van mijn onderdeel.

De technische controles zijn vastgelegd in onder meer
[de leakage-audit](../SolutionRaoul/Citybike/reports/leakage_audit.json),
[de modelverificatie](../SolutionRaoul/Citybike/reports/verification.json)
en [de deploymentcontrole](../SolutionRaoul/Citybike/reports/frontend_deployment_verification.json).
Die controles behandelen onder andere reproduceerbare modeluitvoer en dezelfde
voorspellingen in offline code en de API. Codex hielp ook bij deze controles;
ze worden niet voorgesteld als volledig zonder AI uitgevoerd. Raoul blijft
verantwoordelijk voor de inhoudelijke beoordeling en uitleg van zijn onderdeel.

## 3. Voorbeelden van prompts uit de ontwikkelworkflow

### Jorre — vijf voorbeelden

De vijf onderstaande voorbeelden zijn gebaseerd op Jorres werkelijke berichten
in deze chat. Punt 2 en 5 zijn leesbaar geformuleerde samenvattingen, geen
letterlijke citaten. De voorbeelden tonen ook bijsturing en controle, naast
verzoeken om code.

1. Hulp bij het opstellen en trainen van het AWS-model:

   > Zeg me stap voor stap hoe ik het doe in aws sagemaker

   Ik vroeg begeleiding bij het opstellen en trainen van mijn Mushroom-model
   in AWS SageMaker. AI hielp met de notebookcode, het oplossen van foutmeldingen
   en het opslaan van het getrainde model en de resultaten.

2. Een vergelijkingsdocument als basis voor beoordeling en bijsturing:

   Ik vroeg om een document waarin de resultaten van de verschillende modellen
   naast elkaar staan, zodat ik ze kan vergelijken, de keuzes kan beoordelen en
   kan bijsturen waar dat nodig is.

3. Een verdachte uitkomst kritisch bevragen:

   > Het kan toch niet dat dat aws model het beste is. 100% juist is wel heel verdacht

4. De databron corrigeren:

   > En dit er nog bij: Zorg er ook voor dat je het excel bestand gebruikt dat ik heb geupload naar github. De vorige die we gebruikten met 61k rijen was de foute.

5. Een voorzichtigere beslisregel voor de website:

   Ik vroeg om het label 'eetbaar' alleen bij een hoge modelscore te tonen, met
   95% als mogelijke grens om te onderzoeken. Een giftige paddenstoel onterecht
   als eetbaar aanduiden weegt voor mij zwaarder dan een eetbare paddenstoel
   afwijzen. Daarom vroeg ik om de beslisregel daarop af te stemmen.

### Raoul — twee voorbeeldprompts voor uitleg

Deze voorbeeldprompts gaan over uitleg bij mijn Citi Bike-onderdeel. Ze zijn
ter illustratie geformuleerd en zijn geen letterlijke citaten uit de chat.

1. Uitleg over de verdeling van tijdreeksdata:

   > Kan je uitleggen waarom ik de Citi Bike-data chronologisch moet splitsen in plaats van willekeurig, en wat datalekken daarbij betekenen? Geef een eenvoudig voorbeeld.

2. Uitleg over de evaluatie van mijn modellen:

   > Kan je in eenvoudige woorden uitleggen wat RMSE, MAE en R² betekenen en hoe ik die moet interpreteren bij het vergelijken van mijn Citi Bike-modellen?

## 4. Verificatie en reproduceerbaarheid

De gerapporteerde projectmetrics komen uit bewaarde uitvoer en uitgevoerde code,
niet uit door AI bedachte scores. De technische bewijsstukken zijn:

- De [uitvoeringsregistratie](execution_checks.json): twaalf opnieuw uitgevoerde
  Final-notebooks. Andere Citi Bike-resultaten zijn expliciet bewaarde bronuitvoer,
  geen fictieve nieuwe training en geen opnieuw geopende eindtest.
- Het [bronmanifest](source_manifest.json) en de vaste bronhash/splits: dezelfde
  5.000 cursusrijen en gedeelde trainingfolds voor de Mushroom-vergelijking.
- De [Mushroom-vergelijking](../MushroomFinal/comparison_correct/model_comparison.csv),
  [AWS-artifactaudit](../MushroomFinal/comparison_correct/aws_artifact_audit.json) en
  [drempelaudit](../MushroomFinal/comparison_correct/decision_threshold_audit.json).
- De [Citi Bike-eindbeoordeling](../SolutionRaoul/Citybike/reports/final_evaluation.json)
  en het [evaluatieslot](../SolutionRaoul/Citybike/reports/evaluation_lock.json).
- Tests op backend, frontend, modelpublicatie en referentievoorspellingen, plus
  uitgevoerde Actions-runs en live controles. De aantallen en runlinks staan in
  [VALIDATIE.md](VALIDATIE.md). Een nieuwe documentatiewijziging is geen nieuwe
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

## 6. Intellectueel eigenaarschap

Het team draagt de eindverantwoordelijkheid voor code, cijfers, beveiliging,
interpretatie en conclusies. Ieder lid moet zelfstandig kunnen uitleggen waarom
de split en preprocessing zo gekozen zijn, wat macro F1/recall/RMSE meten, waarom
het geselecteerde model gekozen werd en hoe model, API en website samenwerken.
Een antwoord als 'de AI stelde dit voor' is geen technische onderbouwing.

Jorre en Raoul hebben hun eigen bijdragen en AI-gebruik bevestigd. Raouls
bevestiging is verwerkt in de [bijdragen](BIJDRAGEN.md) en in dit document.
De [inlevercheck](INLEVERCHECK.md) houdt alleen de nog openstaande teamacties bij.
