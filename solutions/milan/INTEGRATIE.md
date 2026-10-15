# Later aansluiten: beperkte en gecontroleerde wijziging

**Nu is alleen `solutions/milan` gewijzigd.** De API, website, Final en centrale
trainingsworkflow zijn niet door dit concept aangepast. De preview is een
afzonderlijke lokale controle, geen tweede gepubliceerde teamsite.

## Wat is al voorbereid?

- `artifacts/current_pipeline.joblib`: gewone sklearn-pipeline, zonder eigen
  transformerklassen. Alleen bedoeld voor de gepinde trainingsomgeving.
- `artifacts/mushroom_rf300.npz` en `.json`: verliesvrije export die door de
  bestaande `backend.mushroom_model.PortableMushroom` gelezen kan worden.
- Dezelfde elf invoervelden, dezelfde labelmapping, 300 bomen en maximale diepte 20.
- `proposed_decision_threshold`: apart geleerde beslisgrens. Dit metadataveld is
  voor de huidige modellezer bewust inert: de ruwe classificatie blijft ongewijzigd.
- Afzonderlijke ruwe classifiermetrics en beslispolicymetrics. Er worden geen oude
  testcijfers bij het nieuwe artifact gezet.
- `preview.py`: hergebruikt het echte invoerschema en de echte Mushroom-controller,
  kopieert het bestaande formulier alleen naar een genegeerde map onder Milan en
  past daar expliciet de voorgestelde beslisregel toe.

De preview gebruikt voor de beslissing dezelfde modelscore en de voorgestelde
drempel. Hij behoudt de ruwe `label`-waarde en voegt de beleidsuitkomst toe. Een
waarschuwing bij een score onder 0,5 is dus een bewuste voorzichtigere beslissing,
geen heimelijke wijziging van het wetenschappelijke classifierlabel.

## Eerst zelf bekijken

Gebruik de bestaande Milan-omgeving en installeer de aanvullende previewpackages
alleen daarin:

```powershell
.\solutions\milan\.venv\Scripts\python.exe -m pip install -r solutions/milan/requirements-preview.txt
.\solutions\milan\.venv\Scripts\python.exe solutions/milan/preview.py
```

Open daarna `http://127.0.0.1:8011`. Het formulier en de voorbeeldinvoer werken met
de kandidaat. De pagina is als concept gemarkeerd. **Citi Bike is in deze lokale
Mushroom-preview niet aangesloten**; de bestaande gepubliceerde Citi Bike blijft
buiten dit experiment.

Op deze computer zijn de aanvullende packages in `.generated/preview_packages`
geïnstalleerd om de eerdere ML-omgeving niet te veranderen. De scripts herkennen
die lokale map. Een verse installatie in de venv gebruikt de gewone route hierboven.

## Kleine wijzigingen voor een latere gezamenlijke overname

1. **Artifact en configuratie samen kiezen.** Neem de pipeline, array-export,
   metadata en het bron-/versierecord als één pakket over. De API ondersteunt al
   `MUSHROOM_RF300_MANIFEST` voor een alternatieve manifestlocatie. Alleen die
   instelling veranderen activeert nog niet de voorgestelde beslisregel.
2. **Backendbeslisregel configureerbaar maken.** Laat
   `backend/mushroom_policy.py` een gecontroleerde `threshold` accepteren, standaard
   0,5. Laat de route in `backend/main.py` de kandidaatgrens uit metadata doorgeven.
   Geef die grens en een nieuwe policyversie ook in de response terug. De bestaande
   ruwe `label` en `probability_poisonous` blijven behouden.
3. **Frontend dezelfde beslissing laten controleren.** Laat
   `frontend/mushroom.js` die responsegrens en policyversie controleren in plaats
   van de grens 0,5 hard te coderen. Behoud de uitkomsten giftig of onzeker; voeg
   geen bevestiging van eetbaarheid toe. `preview.py` bevat de werkende variant.
4. **Metadata en evaluatietekst vernieuwen.** Genereer het formulier met dezelfde
   artifacthash. Toon de policy-CV-metrics uit het kandidaatmanifest als
   ontwikkelings-CV op 4.100 rijen; vervang de hardgecodeerde tekst over 900 records.
   De interne missingcategorie hoeft niet als een nieuw meetbaar kenmerk in het
   formulier te verschijnen. Het contract van elf velden blijft gelijk.
5. **Automatische hertraining op dezelfde voorbereiding zetten.** De bestaande
   centrale workflow kiest nog de oorspronkelijke imputatie en RF-instellingen.
   Neem de gekozen classifierparameters én categorische imputatiestrategie over
   in haar configuratie en `fit_model`. Bewaar ook de gekozen beslisgrens en
   bereken bij hertraining nieuwe policy-OOF-metrics; kopieer geen oude policycijfers.
   Pas de publicatiecontrole aan de bijbehorende nieuwe referenties aan.

Dit zijn enkele gerichte koppelpunten, geen nieuwe frontend- of backendarchitectuur.
**Het hele pakket is nog niet klaar voor de bestaande automatische publicatiejob.**
Die job verwacht oude bestandslocaties, oude referenties en de oorspronkelijke
trainingsconfiguratie. Een losse modelwissel zonder stap 5 zou later door een
automatische refit kunnen worden overschreven of een versieverschil veroorzaken.
Deze map installeert daarom niets automatisch.

## Wat moet na de latere koppeling slagen?

- De oorspronkelijke formulierpayloads, nullen en nulmetingen blijven geldig;
  ongeldige of onbekende codes worden geweigerd.
- API en frontend tonen dezelfde modelhash, policyversie en beslisdrempel.
- Array-export en sklearn-pipeline geven dezelfde scores en ruwe labels.
- Beslispolicymetrics worden herberekend op buiten-OOF voorspellingen; de
  export-equivalentiecheck op trainingsinputs wordt niet als accuracytest gebruikt.
- Bestaande Citi Bike-controles blijven slagen; de gedeelde backend behoudt beide
  routes en ongewijzigde Citi Bike-artifacts.
- De automatische training reproduceert dezelfde voorbereiding en kan een
  afgekeurde of corrupte kandidaat niet publiceren.

De al uitgevoerde lokale controles staan in
[test_results.json](results_v2/test_results.json) en
[candidate_package.json](results_v2/candidate_package.json).
De preview bevestigt Mushroom-compatibiliteit. Een volledige latere integratie
vereist aanvullend de bestaande groepssuite; er wordt niet beweerd dat de
onaangepaste centrale workflow deze kandidaat nu al correct publiceert.
