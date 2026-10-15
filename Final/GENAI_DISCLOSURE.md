# GenAI Disclosure

Deze verantwoording hoort bij de volledige eindoplevering van DeepLearningTeam10.
Zij is op 15 oktober 2026 uitgewerkt tegen de door Jorre aangeleverde
`GenAI_Policy.pdf` (twee pagina's). De eerdere korte disclaimers blijven als
historische bronvermeldingen bestaan; dit document geeft de actuele, bredere scope.
Milans persoonlijke aanvulling hieronder is op dezelfde datum toegevoegd op basis
van zijn eigen bevestiging en de opdrachten in zijn ontwikkelchat.

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
| Milans review en Final-aanvullingen | Alternatieven in `solutions/milan`, rapportaanvulling, linkreparaties, compatibele notebookversies, testcode en bewijsnotebooks uitgewerkt en uitgevoerd. | Milan bevestigde zijn aandeel in cleaning/EDA, de herbouwde frontend en deze recente verbeteringen. Hij vroeg om lesconforme methoden, behoud van de werkende keten en zichtbaar uitvoeringsbewijs; de AI-implementatie wordt expliciet vermeld. |

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

Jorre, Raoul en Milan hebben hun eigen bijdragen en beschreven AI-inzet
bevestigd; zie [bijdragen](BIJDRAGEN.md). Andrew preciseert zijn eigen aandeel
en AI-inzet nog. De zelfstandige beheersing en gezamenlijke eindgoedkeuring
worden hiermee niet automatisch bevestigd. Die menselijke eindcontrole blijft open in de
[inlevercheck](INLEVERCHECK.md). De aanwezigheid van deze tekst is evenmin bewijs
van een Canvas-indiening.

## Milan Wouters — eigen GenAI-inzet

Ik heb meegewerkt aan de data cleaning en EDA van de Mushroom-data, de frontend
herbouwd en de recente verbeteringen en eindcontrole van Final met hulp van
**OpenAI Codex** laten uitwerken. De eerdere bijdragen van de andere teamleden
blijven als bron vermeld. Mijn oorspronkelijke cleaning- en
voorbereidingsnotebooks vermelden ook Codex-hulp bij code, structuur en uitleg.

Voor mijn recente werk hielp Codex substantieel met het lezen en vergelijken
van de cursus- en projectbestanden, het ontwikkelen van alternatieven in
`solutions/milan`, de aanvullende vergelijking in Final, het herstellen van
links, het oplossen van notebookcompatibiliteitsproblemen en het schrijven en
uitvoeren van controles. Codex hielp ook met de Markdown-toelichtingen,
resultatenrapporten, dit bijdragenoverzicht en deze persoonlijke disclosure.
Dit is dus geen claim dat ik alle code en teksten zonder AI heb geschreven.

Mijn eigen bijsturing bestond uit de scope en de eisen: eerst alleen een
alternatief in `solutions/milan`, aansluiten op hoe we het geleerd hebben,
kritisch zoeken naar echte verbeteringen en pas daarna gerichte veranderingen
in Final toestaan. Ik vroeg bovendien om opnieuw te testen en de uitgevoerde
notebookuitvoer te tonen. De AI-uitvoering en mijn opdrachtgeving worden
hier afzonderlijk benoemd; een technische geslaagde test is geen bewijs dat
ik of alle teamleden de volledige inhoud al mondeling hebben verdedigd.

### Vijf illustratieve voorbeeldprompts

Deze vijf voorbeelden zijn nieuw geformuleerd op basis van mijn beschreven
bijdragen. Het zijn geen bewaarde chatberichten of bewijs van de precieze
verdeling tussen eigen werk en AI-hulp. De daadwerkelijk gebruikte, substantiële
hulp van Codex staat hierboven en blijft afzonderlijk verantwoord.

1. De frontend en CSS verbeteren:

   > ik ben de frontend aan het herwerken kun je eens kijken hoe ik die beter kan maken met andere css en een duidelijkere indeling? zeg er ook bij wa ik kan aanpassen en waarom zodat ik zelf kan kiezen wat ik gebruik

2. Een alternatief in mijn eigen solutions-map vergelijken:

   > kun je in solutions/milan een alternatief maken om te vergelijken met wat we nu hebben? doe het zoals we het in de lessen hebben geleerd ma laat de final voorlopig staan zodat ik eerst kan zien wat er beter is en wat ik eventueel wil overnemen

3. Meedenken over data cleaning en EDA:

   > ik werk mee aan de data cleaning en eda van de mushroom data kun je eens meekijken naar de ontbrekende waarden en uitschieters? leg uit wat ik best controleer en wat de voor en nadelen zijn zodat ik snap waarom we da zo doen

4. De resultaten van de vergelijking begrijpen:

   > kun je efkes uitleggen wat die verschillen in macro f1 en recall juist betekenen en welke fouten de modellen maken? ik wil zelf kunnen uitleggen waar het alternatief beter is en waar we nog op moeten letten

5. Gerichte aanpassingen beoordelen en controleren:

   > welke kleine dingen zou ik makkelijk in de final kunnen verbeteren zonder dat de rest ineens niet meer werkt? toon welke stukken ik moet aanpassen en hoe ik dat kan controleren dan kan ik eerst bekijken wa ik wil veranderen

### Controle van mijn recente wijzigingen

Het [uitgevoerde controle-notebook](../solutions/milan/11_Final_eindcontrole.ipynb)
en de [bijbehorende bewijsmap](../solutions/milan/eindcontrole/README.md) bewaren
de echte uitvoer: twaalf Final-notebooks, 28 Python-tests, 19 frontendtests,
pipeline/exportcontrole op 5.000 rijen en live voorspellingen voor beide datasets.
Tijdens de controle bleek een incompatibiliteit tussen de oude Matplotlib-versie
en de nieuwste IPython-versie. De notebookversies zijn vastgezet; na problemen
met de Windows-kernel is een volledige geslaagde bewijsrun bewaard.

Deze tests zijn met Codex uitgevoerd. Er zijn geen modelprestaties verzonnen en
de actieve modellen zijn niet vervangen door de experimentele alternatieven.
De aanvullende modelvergelijking gebruikt trainingsfolds; de eerder bekeken
900 ontwikkeltestrecords zijn geen nieuwe onafhankelijke eindtest. De zeven
overige oorspronkelijke AWS-/Citi Bike-bronnotebooks zijn bij de eindcontrole
niet opnieuw uitgevoerd. De automatische controle vervangt mijn inhoudelijke
verantwoordelijkheid voor de gemaakte keuzes niet.
