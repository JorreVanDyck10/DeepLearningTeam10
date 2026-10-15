# Status van de verbeterde versie

Uitgevoerd op 15 oktober 2026, uitsluitend binnen `solutions/milan`.

## Gekozen richting

**RF300 met eenvoudige missingcategorie en een afgestemde beslispolicy.** De
pipeline accepteert dezelfde elf velden. De uiteindelijke full-trainingselectie
koos balanced class weights, minimale bladgrootte 1 en categorische constante
imputatie. De beslisgrens uit training-OOF is circa **0,45**; dit is geen veiligheidsnorm.

Op dezelfde buitenfolds als de oude RF300:

| Metriek | Oude RF300 | Nieuwe RF + beslispolicy |
|---|---:|---:|
| Macro F1 | 0,7619 | 0,7691 |
| Recall giftige klasse | 60,44% | 66,82% |
| Giftig gemist | 614 | 515 |
| Vals alarm | 256 | 356 |
| Accuracy | 78,78% | 78,76% |

Dit verbetert de gekozen afweging tussen fouttypen. Het verlaagt het totale
aantal fouten niet. Vier van de vijf gepaarde folds verbeteren in macro F1.
De classifier zonder nieuwe policy heeft macro F1 0,7646; de grotere verbetering
hoort dus bij de **complete procedure inclusief beslisdrempel**.

## Wat is werkelijk afgerond?

- Vier modelprocedures, 34 vaste configuraties en 20 volledige buitenfoldruns;
  510 grid-CV-fits, naast refits en drempelberekeningen.
- Alle nieuwe selectie/scoring gebruikt uitsluitend de oorspronkelijke 4.100
  trainingsrijen. De bekende 900 ontwikkeltestrecords zijn niet opnieuw gescoord.
- Kandidaatpipeline plus array-export. De bestaande modellezer geeft op alle
  4.100 gecontroleerde invoerrijen exact dezelfde scores en ruwe labels.
- Zeven Python-tests geslaagd: geometrie, fold-lokale voorbereiding, buiten-OOF
  rijlidmaatschap, echt API-invoerschema/controller, versiecontract, foutafhandeling
  en gescheiden raw-/policymetrics.
- Vier frontendtests geslaagd, waaronder grensgevallen, versie- en drempelcontrole
  en het behoud van de elf velden.
- Vier nieuwe rapportnotebooks uitgevoerd en schema-gevalideerd. Hier gebeurde
  dat van boven naar beneden via IPython met verse naamruimtes; geen afzonderlijke
  Jupyter-kernelrun geclaimd. De uitvoeringsmethode staat in de receipt. De bouwer
  ondersteunt daarnaast gewone Jupyter-kernels voor een normale lokale omgeving.
- [Integratie-instructies](INTEGRATIE.md) en een Mushroom-preview voorbereid.

## Praktische grenzen

Er is geen onaangeraakte eindtest teruggewonnen en geen sterke algemene veiligheid
aangetoond. De foutanalyse toont bovendien 55,1% giftige recall bij vijf of meer
ontbrekende velden, tegenover 75,6% bij nul tot twee ontbrekende velden. Dit zijn
beschrijvende subgroepen, geen bewezen oorzakelijk effect. Vul bekende kenmerken
zorgvuldig in; de uitkomst bevestigt nooit eetbaarheid.

De lokale API is via de testclient gecontroleerd. Een aparte webserver op
`127.0.0.1:8011` is niet gestart: de automatische goedkeuringscontrole kon die
actie op dat moment niet beoordelen wegens een gebruikslimiet. Dat was een
reviewfout, geen oordeel dat de server onveilig is. Er is geen browserdemo als
uitgevoerd voorgesteld. De voorbereide startinstructies blijven beschikbaar.

De bestaande automatische trainings-/publicatieketen en volledige Citi Bike-
integratie zijn niet omgeschakeld. Daarvoor moeten de vijf kleine koppelpunten
uit het integratiedocument samen worden overgenomen en met de groepssuite worden
gecontroleerd. Dit concept schrijft of installeert niets buiten de Milan-map.

Lees [RESULTATEN_V2.md](RESULTATEN_V2.md) voor alle modellen en
[de testreceipt](results_v2/test_results.json) voor de werkelijke testuitvoer.
