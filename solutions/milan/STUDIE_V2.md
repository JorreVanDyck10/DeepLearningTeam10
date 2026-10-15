# Verdere verbetering — onderzoeksafspraken

Deze uitbreiding volgt op de eerste, marginaal betere RF150-proef. De gebruiker
vroeg om verder te werken, uitsluitend binnen `solutions/milan`, met lesmethoden
en een beperkte latere integratie. De oorspronkelijke bestanden blijven onaangeroerd.

## Wat wordt onderzocht?

1. **RF300 binnen het bestaande servingcontract:** dezelfde elf invoervelden,
   300 bomen, maximaal diepte 20, mediaan en categorische one-hot encoding. Tuning
   van modus/missingcategorie, class weights en minimale bladgrootte: 8 combinaties.
2. **RF300 met eenvoudige voorbereiding:** expliciete ontbrekend-indicatoren en
   twee afmetingsverhoudingen. Acht configuraties. Voor deze extra numerieke
   kolommen is later een aangepaste encoder nodig; dit is geen blinde artifactswap.
3. **Gewone Gradient Boosting:** twaalf combinaties met twee leerstappen, drie
   dieptes en wel/geen afmetingsverhoudingen. Geen proximity-imputatie.
4. **SVM met RBF-kernel:** zes combinaties van C en gamma, met numerieke scaling.

Elke familie heeft vijf buitenfolds en drie binnenfolds voor selectie. Ook een
optionele macro-F1-drempel wordt alleen op binnen-OOF voorspellingen gekozen.
De standaardvoorspelling blijft apart zichtbaar. De exacte zoekruimtes worden
vóór scoring opgeslagen in [het protocol](results_v2/protocol.json).

De bouwstenen komen uit de lessen over feature engineering, imputatie, one-hot
encoding, Random Forest, boosting, SVM, scaling en modelkwaliteit. De geneste
evaluatie past de scheiding tussen kiezen en beoordelen strikter toe. Er wordt
geen nieuwe AutoML- of neuralenetwerkoplossing ingevoerd.

## Wanneer noem ik iets een bruikbare verbetering?

Niet alleen de hoogste tabelrij. Een kandidaat voor beperkte integratie moet:

- minstens 0,005 gemiddelde macro-F1-winst tonen tegenover RF300;
- minstens dezelfde gemiddelde giftige recall halen;
- op ten minste drie van de vijf gepaarde buitenfolds hoger scoren;
- binnen het huidige invoercontract passen en met identieke exportvoorspellingen
  gecontroleerd zijn.

Dit zijn vooraf vastgelegde ontwikkelcriteria, geen statistisch significantietoets
of bewijs van veiligheid. Varianten die deze grens niet halen blijven als experiment
zichtbaar en worden niet als bewezen betere vervanging aanbevolen. Lagere artifact-
of rekengrootte kan een afzonderlijke praktische verbetering zijn, mits prestaties
niet materieel verslechteren en die claim ook werkelijk gemeten is.

## Grenzen van de evaluatie

Alle 4.100 ontwikkelingsrijen en eerdere foldresultaten zijn eerder in dit project
gebruikt. De tweede studie is dus geïnformeerd door eerdere ontwikkeling. De
geneste folds voorkomen rechtstreeks lekken binnen een run; zij herstellen geen
nieuwe, prospectieve eindtest. De 900 bekende ontwikkeltestrecords worden niet
voor deze nieuwe selectie of scoring gebruikt. Zo voorkomen we extra selectie
op die al vaak bekeken testset.

## Aansluiten zonder het bestaande project open te breken

De bestaande API kan een ander RF300-manifest al via
`MUSHROOM_RF300_MANIFEST` lezen. Een compatibele RF met gewone medianen en
categorische one-hot encoding kan daardoor via een apart modelpad worden
gecontroleerd. Voor een werkelijke latere omschakeling moeten ook de frontend-
metadata, de tekst over de evaluatieset en de trainingsconfiguratie hetzelfde
model/protocol beschrijven. Alleen een modelbestand vervangen zou oude cijfers
bij nieuwe voorspellingen tonen.

De model-ID, artifacthash, velden, backend- en frontendbeleid worden daarom
afzonderlijk gecontroleerd. De prototypecode schrijft uitsluitend onder deze
map. Er is geen automatische installatie, cloudrun, wijziging van main of live
modelupdate onderdeel van deze uitbreiding.

## Technische bronnen

De lokale lesinhoud en oefeningen blijven leidend. Implementatiedetails worden
getoetst aan de officiële scikit-learn-documentatie voor
[Random Forest](https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.RandomForestClassifier.html),
[Gradient Boosting](https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
en [imputatie](https://scikit-learn.org/1.6/modules/impute.html).
De versies blijven gepind op de gemeten omgeving; de actuele stable-documentatie
is geen reden om de werkomgeving tijdens een lopende vergelijking te upgraden.
