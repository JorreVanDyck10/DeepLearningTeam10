# Van trainingscode naar gepubliceerd model

`model-training.yml` is een echte train/valideer/publiceer-workflow voor het
gekozen Mushroom-model. Zij draait bij relevante code-, configuratie- en databron-
wijzigingen op main en kan ook handmatig via GitHub Actions worden gestart.
Documentatie- en presentatiewijzigingen veroorzaken geen onnodige modeltraining.

1. Python 3.11 met gepinde trainingdependencies leest de gecontroleerde cursusbron.
2. Het model leert preprocessing opnieuw binnen elk van de vijf trainingsfolds.
3. CV macro F1 en giftige recall moeten minstens de vastgelegde referenties halen.
   De eerder bekeken 900 ontwikkeltestlabels beslissen niet over acceptatie.
4. De geaccepteerde configuratie wordt op alle 4.100 trainingsrijen gefit.
5. De volledige sklearn-pipeline, verliesvrije NumPy-export, referentievoorspellingen
   en trainingsreceipt worden als kandidaat opgeslagen. Alle 5.000 rijen controleren
   dat de export dezelfde labels en kansen geeft.
6. Een aparte Python 3.13-job test de kandidaat in de werkelijke servingomgeving,
   inclusief Citi Bike, frontendmetadata en de voorzichtige Mushroom-beslisregel.
7. Alleen daarna commit de publish-job de gedeclareerde modelbestanden en metadata.
   Hij weigert te publiceren als iemand ondertussen nieuwe commits naar main pushte.
   Render en Netlify volgen de main-branch en laden de gevalideerde versie.

De botcommit bevat alleen artifacts en metadata en activeert geen nieuwe training.
De API meldt de werkelijke geladen modelhash in health en iedere voorspelling.
De website vergelijkt die hash met haar metadata; bij een verschil wacht zij op
de juiste deployment in plaats van oude voorspellingen bij nieuwe metrics te tonen.
GitHub voorkomt bovendien dat GITHUB_TOKEN-pushes zomaar een nieuwe Actions-keten
starten ([GitHub-documentatie](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow)).
De servingcontroles zijn daarom al vóór publicatie onderdeel van dezelfde
workflow. Een mislukte training, kwaliteitscontrole, test of stale-commitcontrole
publiceert niets en laat de bestaande websiteversie behouden.

## Onderzoeksarchief blijft ongewijzigd

De originele vergelijking, PyCaret-resultaten, AWS-export en selected_pipeline
onder `MushroomFinal/comparison_correct` worden niet overschreven. De actieve
refit staat onder `Final/Mushroom/artifacts`, met bronhash en trainingsreceipt.
Een identieke hertraining mag dezelfde scores opleveren; dit is bewijs van
reproduceerbaarheid, geen nieuwe verbetering of onafhankelijke eindtest.

De automatische configuratie is bewust RF300 met maximale diepte 20 en elf velden:
dat is het ondersteunde servingcontract. Verander je modeltype, featurecontract,
bronbestand of complexiteit buiten dit contract, dan moet je eerst de vergelijking,
export en API expliciet aanpassen. De pipeline gaat zulke wijzigingen niet stil
accepteren. De oude 61k-bron wordt geweigerd door de bronhash.

## Lokaal dezelfde stappen uitvoeren

```powershell
py -3.11 -m venv .venv-final
.\.venv-final\Scripts\python.exe -m pip install -r Final/requirements-training.txt
.\.venv-final\Scripts\python.exe Final/tools/model_workflow.py train --output Final/.generated/candidate
# Met de Python 3.13-servingomgeving:
python Final/tools/promote_candidate.py Final/.generated/candidate
python frontend/export_metadata.py
python -m unittest discover -s tests -v
```

Publicatie via Actions vereist dat GitHub de expliciete `contents: write`-permissie
voor de publish-job toestaat en dat branchbeveiliging botcommits niet blokkeert.
Er worden geen Render-wachtwoorden, AWS-credentials of GitHub-tokens in bestanden
opgenomen. De workflow wordt na de eerste push werkelijk gecontroleerd.

## Citi Bike

De bestaande RF320 blijft gekoppeld aan zijn gevalideerde GitHub Release en
checksumcontract. De workflow test deze inference naast Mushroom en publiceert
dezelfde backend opnieuw. Zij claimt geen automatische nieuwe Citi Bike-training:
de grote brondatasets blijven buiten Git en de oorspronkelijke eindtest is
vergrendeld. Voor een nieuwe studie moeten de brondata en een nieuw evaluatie-
protocol volgens Raouls instructies worden opgebouwd. Bestaande lokale refit-
commando's staan in zijn README. Dit onderscheid voorkomt dat een gewone deploy
ten onrechte als nieuwe modeltraining wordt voorgesteld.

## Bijdrage en GenAI

Jorre Van Dyck stelde de centrale pipeline op, met substantiële hulp van Codex
voor trainingscode, GitHub Actions, controles en publicatie. Zie [bijdragen](BIJDRAGEN.md)
en [GenAI Disclosure](GENAI_DISCLOSURE.md).
