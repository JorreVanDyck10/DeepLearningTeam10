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

## Milan Wouters — bevestigd op 15 oktober 2026

Milan bevestigde de volgende eigen bijdragen:

- Meegewerkt aan de data cleaning en EDA van de Mushroom-data. Zijn oorspronkelijke uitwerking staat in [clean en explore](../SolutionMilan/MushroomDataset/01_clean_en_explore.ipynb) en [data preparation](../SolutionMilan/MushroomDataset/02_data_preparation.ipynb). Deze oorspronkelijke notebooks horen bij de historische databron; de definitieve vergelijking gebruikt het cursusbestand met 5.000 rijen.
- De frontend herbouwd voor het gezamenlijke project. Dit bouwt voort op de eerdere gezamenlijke UI van Jorre en de oorspronkelijke Citi Bike-applicatieonderdelen van Raoul; hun bronbijdragen blijven behouden.
- Final kritisch laten controleren aan de hand van de aangeleverde lesmaterialen en verbeteringen laten uitwerken in `solutions/milan`. Milan bepaalde daarbij dat de aanpak bij de geleerde methoden moest blijven passen en dat de werkende keten behouden moest blijven.
- De gerichte Final-verbeteringen met Codex aangebracht: twee notebooklinks hersteld, de aanvullende ontwikkelvergelijking met bevroren resultaten toegevoegd en de notebookvereisten compatibel vastgezet. De actieve modellen zijn hiermee niet vervangen.
- De uitgebreide technische eindcontrole met Codex laten uitvoeren en als [uitgevoerd bewijsnotebook](../solutions/milan/11_Final_eindcontrole.ipynb) laten bewaren: twaalf notebookuitvoeringen, 28 Python-tests, 19 frontendtests, vergelijking van pipeline en export op 5.000 rijen en echte controles van beide publieke voorspelformulieren. De [bewijsmap](../solutions/milan/eindcontrole/README.md) vermeldt de precieze grenzen.

Voor de recente review, alternatieven, Final-aanpassingen, testcode en rapportage
gebruikte Milan **OpenAI Codex**. De implementatie en uitvoering met deze hulp
waren substantieel. Zijn oorspronkelijke voorbereidingsnotebooks vermelden
eveneens hulp van Codex bij code, structuur en uitleg. Dit wordt niet voorgesteld
als volledig zelfstandig geschreven code of als handmatig door Milan uitgevoerde
tests. Milan gaf de opdrachten, bepaalde de gewenste scope en vroeg om concrete
verbeteringen en controleerbaar bewijs. Ook deze bijdragenvermelding is met
Codex opgesteld op basis van zijn eigen bevestiging.

Dit bevestigt Milans aandeel en de beschreven AI-inzet. Het betekent niet dat
de technische tests zijn persoonlijke beheersing of de eindgoedkeuring van het
hele team bewijzen. Hij blijft verantwoordelijk voor het kunnen toelichten van
zijn onderdeel. De [GenAI Disclosure](GENAI_DISCLOSURE.md#milan-wouters--eigen-genai-inzet)
beschrijft de hulp, echte opdrachten en controles voor zijn bijdrage.

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

- Iedereen kan de code, ontwerpkeuzes en conclusies zelfstandig uitleggen.
- Sprekers en de uiteindelijke mondelinge taakverdeling worden samen afgesproken.

Deze menselijke controles worden niet op basis van geautomatiseerde tests als
afgerond gemarkeerd.
