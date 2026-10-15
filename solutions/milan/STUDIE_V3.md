# Derde studie: gerichte verbetering vanuit de lessen

De gebruiker vraagt om een sterker resultaat. Deze studie onderzoekt drie nieuwe
richtingen die uit de cursus volgen, uitsluitend binnen `solutions/milan`.

- **Feature selection:** RF300 met wel/geen `jumbled_noise_1` en wel/geen vrijwel
  volledig ontbrekende sporenkleur. De publieke invoer houdt elf velden; een model
  mag intern een veld negeren. Acht combinaties inclusief bladgrootte.
- **LightGBM:** acht combinaties van bladcomplexiteit, minimale bladpopulatie en
  wel/geen ruisveld. Categorische velden worden expliciet als categorie gemarkeerd.
- **CatBoost:** acht combinaties van diepte, regularisatie en wel/geen ruisveld.
  Categorische missings krijgen een expliciete categorie; numerieke NaN blijven.
- **Gewogen voting:** zeven vooraf vastgelegde combinaties van bovenstaande drie
  modellen. Gewichten worden uitsluitend op binnen-OOF voorspellingen gekozen.

De referentie is de eerdere RF300 met afgestemde beslispolicy: macro F1 0,7691,
515 gemiste giftige gevallen en 356 valse alarmen. Het doel is hogere macro F1 en
minder gemiste giftige gevallen zonder dit aantal valse alarmen te overschrijden.
Alle uitkomsten worden gerapporteerd, ook als een richting dit doel niet haalt.

We gebruiken vijf bestaande buitenfolds en drie binnenfolds voor selectie. Naast
standaardlabels en een binnen training afgestemde macro-F1-drempel wordt een
vooraf gekozen 14%-false-positivebudget onderzocht. De buitenfold meet vervolgens
of dat doel generaliseert; het budget wordt daar niet opnieuw afgedwongen.

De lesbasis is **4 — Models**, dia 36 (ensembles), 38 (bagging), 42 (boosting,
inclusief LightGBM en CatBoost) en 49 (geoptimaliseerde boosters en regularisatie),
aangevuld met de lessen over voorbereiding, modelkwaliteit en ROC-drempels.
De extra bibliotheken zijn toepassingen van die aangeleerde modelfamilie.

Dit is verdere ontwikkeling op eerder gebruikte projectdata. Het protocol kan
binnen een run lekken vermijden, maar maakt de hele onderzoeksreeks niet alsnog
onafhankelijk. De 900 bekende ontwikkeltestrecords worden opnieuw uitgesloten.
Meer zoeken geeft geen garantie op meer voorspellingskracht; een klein verschil
wordt niet als een grote verbetering voorgesteld.

De exacte zoekruimtes, versies en hashes staan na de start in
`results_v3/protocol.json`. De bestaande studies, artifacts en integratie blijven
behouden tot het nieuwe resultaat en de aansluitbaarheid gecontroleerd zijn.

Implementatiebronnen: de officiële documentatie van
[LightGBM](https://lightgbm.readthedocs.io/en/v4.6.0/Python-API.html) en
[CatBoostClassifier](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier).
