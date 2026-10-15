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

## 3. Vijf voorbeelden uit de ontwikkelworkflow

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

## 6. Intellectueel eigenaarschap en nog te bevestigen review

Het team draagt de eindverantwoordelijkheid voor code, cijfers, beveiliging,
interpretatie en conclusies. Ieder lid moet zelfstandig kunnen uitleggen waarom
de split en preprocessing zo gekozen zijn, wat macro F1/recall/RMSE meten, waarom
het geselecteerde model gekozen werd en hoe model, API en website samenwerken.
Een antwoord als 'de AI stelde dit voor' is geen technische onderbouwing.

Jorres vier genoemde bijdragen zijn bevestigd. De precieze bijdragen en AI-inzet
van de overige leden en hun zelfstandige beheersing zijn hiermee niet automatisch
bevestigd. Die menselijke eindcontrole blijft open in de
[inlevercheck](INLEVERCHECK.md). De aanwezigheid van deze tekst is evenmin bewijs
van een Canvas-indiening.
