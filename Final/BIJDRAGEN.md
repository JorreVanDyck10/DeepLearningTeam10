# Bijdragen en verantwoordelijkheid

## Jorre Van Dyck — bevestigd op 15 oktober 2026

Jorre bevestigde de volgende eigen bijdragen:

- De pipeline opgesteld voor de gezamenlijke training, validatie en modelpublicatie.
- De gezamenlijke backend ontwikkeld en de modellen aan de API gekoppeld.
- De UI gebouwd en de verbinding met de backend uitgewerkt.
- Een Mushroom-model op AWS SageMaker getraind; de correcte cursusrun en tuningexports aangeleverd.

Deze bijdragen zijn met substantiële hulp van Codex uitgewerkt. De code en
documentatie worden niet voorgesteld als volledig zonder AI geschreven.
De [GenAI Disclosure](GENAI_DISCLOSURE.md) beschrijft de functionele AI-hulp,
echte prompts, uitgevoerde controles en menselijke bijsturing.

De gedeelde backend en UI bouwen voort op de aangeleverde modellen en
datasetonderdelen. Jorres bijdrage vervangt geen bestaande bronvermelding:
Andrew blijft vermeld bij het oorspronkelijke Mushroom-onderzoek en de
modellen; Raoul bij het Citi Bike-onderzoek, de modellen en de oorspronkelijke
Citi Bike-inference en applicatieonderdelen. Milan heeft een oorspronkelijke
Mushroom-voorbereiding aangeleverd. Hun precieze taakverdeling en eigen
AI-gebruik zijn hiermee niet opnieuw of exclusief door Jorre vastgesteld.

## Raoul Buffels — bevestigd op 15 oktober 2026

Raoul bevestigde de volgende eigen bijdragen aan Citi Bike:

- De bronverwerking, cleaning en aggregatie van de ritgegevens uitgewerkt, met afzonderlijke verwerking van NYC en Jersey City.
- De EDA, grafieken, onderzoeksvragen en statistische hypothesen uitgewerkt en geïnterpreteerd.
- De definitieve datavoorbereiding en feature-engineering opgesteld, met chronologische splits en controles op datalekken.
- De weekbaseline, Ridge, PyCaret-vergelijking, Random Forest en LightGBM uitgewerkt, inclusief tuning, modelvergelijking en foutenanalyse.
- De modelkeuze onderbouwd op validatieresultaten en de oorspronkelijke eindtest behouden zonder die voor nieuwe keuzes te gebruiken.
- De oorspronkelijke Citi Bike-inference en applicatieonderdelen voor frontend en backend uitgewerkt. Jorre hielp met deployment op Render en ontwikkelde de gezamenlijke backend en UI verder.
- De Citi Bike-notebooks, documentatie en het eigen presentatieonderdeel met spreektekst voorbereid.

Raoul gebruikte **OpenAI Codex** voor hulp bij code, debugging, uitleg,
notebookstructuur, grafieken, controles en documentatie. Die hulp was substantieel;
dit werk wordt niet voorgesteld als volledig zonder AI geschreven. Hij nam de
voorstellen niet zomaar over: hij beoordeelde de aanpak en interpretaties kritisch,
controleerde ze aan de hand van de opdracht, brondata en opgeslagen uitvoer en
stuurde bij waar nodig. De uitgevoerde technische controles omvatten onder meer
datakwaliteit, chronologische splits, reproduceerbare modeluitvoer en gelijke
voorspellingen tussen offline inference en de API.

Raoul bevestigt hiermee zijn eigen bijdrage en AI-gebruik en blijft
verantwoordelijk voor de keuzes, interpretatie en conclusies van zijn onderdeel.
De [GenAI Disclosure](GENAI_DISCLOSURE.md) licht deze AI-hulp en controle verder toe.

Raouls hoofddeel staat in de definitieve PowerPoint op dia 14–22, met verborgen reserves op dia 26–27. De [persoonlijke spreektekst](Presentatie/SPREEKPLAN.md) bevat zijn bijdrage, tekst per dia en voorbereiding op vragen. Zijn lokale hertrainingspipeline fit dezelfde vaste RF320 opnieuw en controleert voorspellingen na herladen. Dit staat los van de centrale Mushroom-workflow en bewijst geen automatische live Citi Bike-modelupdate.

## Andrew Noeyens - bevestigd op 15 oktober 2026

Andrew bevestigde de volgende bijdragen aan de Mushroom Challenge:

- Voorbewerking van de dataset en het uitwerken van de EDA.
- Random Forest, Gradient Boosting en Logistic Regression modellen getraind, geëvalueerd en geoptimaliseerd.
- Eerste versie PyCaret ontwikkeld om modellen te vergelijken op basis van verschillende evaluatiemetrics.
- De notebooks en documentatie rondom de Mushroom Challenge voorbereid

Bij de uitwerking van het Mushroom Notebook werd generatieve AI gebruikt als ondersteuning bij het schrijven en verbeteren van code,
het uitleggen van programmeerconcepten, het structureren en verwerken van de documentatie, en hulp bij het interpreteren van modelresultaten.
De AI-ondersteuning wordt niet voorgesteld als volledig zelfstandig geschreven werk.

Andrew bevestigt hiermee zijn eigen bijdrage en AI-gebruik en blijft
verantwoordelijk voor de keuzes, interpretatie en conclusies van zijn onderdeel.
De [GenAI Disclosure](GENAI_DISCLOSURE.md) licht deze AI-hulp en controle verder toe.

## Afbakening van de pipeline

De centrale GitHub Actions-workflow traint en controleert Mushroom en publiceert
een goedgekeurde modelversie. De Citi Bike-API gebruikt de gepinde RF320-release;
deze workflow claimt geen automatische nieuwe Citi Bike-training.

## Nog door het team te bevestigen

- Milan controleren en preciseren nog de eigen bijdrage en AI-inzet.
- Iedereen kan de code, ontwerpkeuzes en conclusies zelfstandig uitleggen.
- Sprekers en de uiteindelijke mondelinge taakverdeling worden samen afgesproken.

Deze menselijke controles worden niet op basis van geautomatiseerde tests als
afgerond gemarkeerd.
