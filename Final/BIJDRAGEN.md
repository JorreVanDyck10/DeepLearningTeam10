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

## Afbakening van de pipeline

De centrale GitHub Actions-workflow traint en controleert Mushroom en publiceert
een goedgekeurde modelversie. De Citi Bike-API gebruikt de gepinde RF320-release;
deze workflow claimt geen automatische nieuwe Citi Bike-training.

## Nog door het team te bevestigen

- Ieder ander teamlid controleert en preciseert de eigen bijdrage en AI-inzet.
- Iedereen kan de code, ontwerpkeuzes en conclusies zelfstandig uitleggen.
- Sprekers en de uiteindelijke mondelinge taakverdeling worden samen afgesproken.

Deze menselijke controles worden niet op basis van geautomatiseerde tests als
afgerond gemarkeerd.
