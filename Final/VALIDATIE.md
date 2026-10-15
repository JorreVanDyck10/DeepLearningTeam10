# Uitgevoerde eindcontrole — 14 oktober 2026

## Behoud en notebooks

De Final-leesmap bevat 19 notebooks. Twaalf daarvan zijn hier opnieuw uitgevoerd:
Mushroom-EDA, definitieve voorbereiding, vier modelreproducties, de AutoML-
audit, vergelijking en deployment, plus Citi Bike-baseline-, Ridge- en deployment-
rapportage. De overige kopieën bewaren expliciet de echte oorspronkelijke
SageMaker- en Raoul-uitvoer; zij claimen geen nieuwe training op deze computer.
Alle notebooks zijn schema-valide en bevatten geen foutuitvoer.

`source_manifest.json` controleert de oorspronkelijke bronbestanden via
SHA-256 met genormaliseerde regeleinden. Ze zijn niet verplaatst of vervangen.
De bestaande lokale, niet-gepushte werkzaamheden zijn vooraf afzonderlijk
gekopieerd en worden bij het bijwerken van de hoofdcheckout behouden.

## Werkelijke automatische training en publicatie

- [Eerste volledige train/valideer/publiceer-run](https://github.com/JorreVanDyck10/DeepLearningTeam10/actions/runs/38603700337): alle drie jobs geslaagd.
- [Volledige run met controle van de exacte modelrevisie](https://github.com/JorreVanDyck10/DeepLearningTeam10/actions/runs/38604443497): alle drie jobs geslaagd.
- [Website- en API-controles](https://github.com/JorreVanDyck10/DeepLearningTeam10/actions/runs/38604443445): geslaagd.

De workflow heeft zelf de gevalideerde refit op main gepubliceerd, eerst als
commit `823e5a8` en daarna als `fec7704`. De actuele trainingsreceipt staat naast
de actieve pipeline. Python 3.11 doet echte fitting; Python 3.13 controleert de
serving-export. De oorspronkelijke onderzoeksvergelijking blijft ongewijzigd.

De actieve pipeline heeft SHA-256
`6410b277e49c3706fa4d70db548e2545066a78bcaa6406aa83fef418c489c626`.
De eerste lokale Windows-refit was bytegelijk aan het bronartifact. De Linux-
publicatie heeft een andere bestandshash en minieme floating-pointverschillen;
de scores en labels blijven binnen de gecontroleerde tolerantie overeenkomen.
Dit is geen nieuwe onafhankelijke prestatietest en geen claim van verbetering.

## API en browser

28 Python-tests en 19 JavaScript-tests slagen, waaronder weigering van een corrupt
kandidaatbestand en een kandidaat onder de minimale CV-score. Acht publieke
Mushroom-voorspellingen zijn gecontroleerd tegen de onderzoeksreferentie met
tolerantie 10⁻¹². Netlify-metadata, Render-health en voorspellingen melden dezelfde
exacte modelrevisie. De API gebruikt versie 0.6.0.

In een browser zijn **Voorbeeld invullen → Voorspellen**, de projectmetrics en
het Citi Bike-formulier bediend. Beide geven een geldige respons zonder
JavaScript-fouten. De verwijderde UI-teksten en verbindingsinstellingen blijven
weg. De Citi Bike-historiecontrole geeft dezelfde 24 voorspellingen als de
bestaande referentiefixture, en de Netlify-CORS-preflight slaagt.

Tijdens een herstart trad een tijdelijke 502 op. Na opstart zijn de publieke
controles opnieuw uitgevoerd en geslaagd. Gratis hosting kent een koude start;
de opgeslagen notebookuitvoer en slides zijn de presentatiefallback.

## Nog menselijke eindacties

Jorres en Raouls bijdragen en AI-gebruik zijn bevestigd en de GenAI-verantwoording is uitgewerkt tegen de
aangeleverde policy. Andrew en Milan moeten hun eigen bijdrage/AI-inzet en
zelfstandige beheersing nog bevestigen; zie [bijdragen](BIJDRAGEN.md) en
[GenAI Disclosure](GENAI_DISCLOSURE.md). Sprekers verdelen en oefenen,
de uiteindelijke Canvas-indiening en de mondelinge presentatie zijn niet door
deze technische controles afgevinkt. De uploadtekst staat gereed; er is geen
Canvas-indieningsbevestiging gemaakt of verondersteld.
