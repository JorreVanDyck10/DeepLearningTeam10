# Citi Bike-modelkaart

Gekozen model: **tuned_random_forest**. Het voorspelt geregistreerde NYC-vertrekken
per lokaal uurvak, voor één volledige kalenderdag vanaf 00:00 New York-tijd.
Veertien volledige voorafgaande dagen zijn nodig. Jersey City is uitgesloten.

| Metriek | Eindmodel | Vorig weekuur |
|---|---:|---:|
| RMSE, ritten/uur | 1979.51 | 2608.78 |
| MAE, ritten/uur | 1076.23 | 1453.53 |
| R² | 0.8280 | 0.7013 |

RMSE-verbetering: 24.12%. Eindtest juli–augustus 2026,
1488 klokvakken. Selectie en tuning gebruikten deze uitkomsten niet.
Het model blijft vast; eerder gemeten testdagen worden daarna historische invoer.
Het artefact is na de eindtest niet opnieuw getraind.

Dit is geregistreerd gebruik, geen onvervulde vraag of beschikbaarheid per station.
Weer, evenementen en capaciteit ontbreken. Feature importance toont geen causaliteit.
Historische schema's en gebruik veranderen; twee testmaanden dekken niet alle seizoenen.
Het weekbootstrapinterval bevat weinig blokken en geeft beperkte zekerheid.
De actuele maandarchieven leveren niet automatisch de recente historie voor morgen.
De backtest veronderstelt een externe bron met vóór 00:00 beschikbare uurtellingen.
Publicatie-, ingestie- en rekenlatentie zijn niet gemodelleerd. De prestaties gelden
onder dit invoercontract; een operationele tellingenfeed moet apart worden gerealiseerd.

Hertrain offline met de README-commando's. Modelbestanden staan lokaal en buiten Git.
Bronnen, code, packageversies, seeds, splits en fingerprints zijn vastgelegd.
AWS-training blijft open voor de volledige schoolopdracht.
