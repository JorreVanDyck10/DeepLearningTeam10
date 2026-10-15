# Later aansluiten: twee concrete mogelijkheden

Alleen `solutions/milan` is gewijzigd. Kies eerst de gewenste foutafweging uit
[RESULTATEN_V3.md](RESULTATEN_V3.md). De bestaande v2-kandidaat blijft bewaard.
Een pakket is pas beschikbaar als de bijbehorende package-receipt `packaged: true`
vermeldt. Geen van deze scripts publiceert of wijzigt de gezamenlijke applicatie.

## RF: de kleinste aansluiting

`artifacts_v3` bevat de featureselectievariant. Hij houdt het bestaande formulier
met elf velden en de bestaande `PortableMushroom`-modellezer. Welke velden intern
worden genegeerd staat in het manifest. De geleerde encodings staan eerst;
genegeerde categorievelden worden als ongebruikte encodings achteraan bewaard.
Daardoor veranderen de indices van de getrainde boomsplits niet.

Lees `results_v3/candidate_package.json`: het volledige verbeterdoel tegenover v2
kan **false** zijn, ook als deze RF beide foutaantallen van het oorspronkelijke
model verlaagt. Hogere F1 alleen maakt hem geen betere keuze bij elke voorkeur.
De variant met standaardgrens behoudt de grens 0,5; bij die variant hoeven de
bestaande backend- en frontendbeslisgrenzen dus niet te veranderen.

Na een bewuste keuze moeten samen worden overgenomen:

1. Het artifactpakket, bijbehorende hash en nieuwe ontwikkelmetrics. De bestaande
   manifestinstelling kan naar het nieuwe `.json` wijzen.
2. Formuliermetadata en uitleg: vijf buitenfolds op 4.100 ontwikkelrijen. Oude
   cijfers en hardgecodeerde tekst over 900 testrecords horen niet bij dit model.
3. Het trainings-/exportrecept: geselecteerde kolommen, vaste instellingen,
   imputatie, de correcte encodingvolgorde en nieuwe referentiecontroles. Anders
   maakt een automatische refit weer de oorspronkelijke pipeline.

De export is vergeleken met de volledige getrainde pipeline op 4.100 invoerrijen
en op nullen/onbekende categorieën. Dat bewijst uitvoergelijkheid, geen accuracy.
De echte API valideert onbekende codes nog steeds als foutieve invoer.

Lokale conceptpreview, als het RF-pakket beschikbaar is:

```powershell
.\solutions\milan\.venv\Scripts\python.exe -B solutions/milan/preview_v3.py
```

Open daarna `http://127.0.0.1:8012`. Dit is een voorbereide startinstructie, geen
claim dat hier een aparte server/browserrun is uitgevoerd. De API-controle staat
in de testreceipt. Citi Bike is in deze Mushroom-preview niet aangesloten.

## Voting: afzonderlijke uitdager met meer aansluitwerk

`voting_artifacts_v3/voting.json` beschrijft native modelbestanden en de gewichten
die op binnen-OOF van de 4.100 trainingsrijen zijn gekozen. Componenten met gewicht
nul hoeven niet te worden geladen. De bijbehorende modellezer is
`voting_model_v3.VotingMushroom`; de oude RF-modellezer kan dit manifest niet lezen.
Er wordt bij laden geen pickle uitgevoerd of model opnieuw getraind.

De lezer heeft dezelfde `predict_record`, `predict_proba`, elf publieke velden en
klassen `[0,1]`. Native preprocessing wordt expliciet uit metadata hersteld.
Checksums controleren het recept en elk actief modelbestand; de interne RF-lezer
controleert ook de checksum van zijn arrays. Het bestaande model-id blijft voor
het huidige API-contract behouden; `model_name`, `serving_format` en de nieuwe
artifactrevision onderscheiden de votingkandidaat. Het is inhoudelijk een ensemble.

Naast bovenstaande metadata- en trainingsaanpassingen vraagt voting om:

1. De andere modellezer en de actieve native afhankelijkheden uit
   `requirements-boosting.txt`. De bestaande oorspronkelijke servingomgeving is
   niet met deze extra bibliotheken getest; dit concept gebruikt de vastgelegde
   Python 3.13-omgeving. Native bestanden vermijden sklearn-picklecompatibiliteit.
2. Expliciete toepassing van de gekozen policydrempel. Geef dezelfde grens en
   policyversie terug aan het formulier; de ruwe klasse blijft de 0,5-uitkomst.
   Het bestaande [integratieplan](INTEGRATIE.md) beschrijft deze koppeling.
3. Hertraining die de componentselectie, binnen-OOF votinggewichten, drempel en
   nieuwe metrics samen opnieuw berekent. Een RF-only workflow reproduceert
   deze procedure niet.

De wrapper `preview_voting_v3.py` toont de werkende lokale verbinding met het
echte invoerschema en de echte Mushroom-controller, zonder gedeelde bestanden
te wijzigen. Start hem desgewenst op dezelfde manier; hij gebruikt poort 8013.
De pakketten zijn bewust afzonderlijk: alleen een nieuw manifest bij de oude
RF-lezer neerzetten sluit voting niet aan.

## Controles en opnieuw uitvoeren

Gebruik de volledige repository en een aparte venv. Installeer `requirements.txt`,
`requirements-preview.txt`, `requirements-preview-lock.txt` en
`requirements-boosting.txt` in die omgeving. Node.js is nodig voor de frontendtests.

```powershell
# Nieuwe complete trainingsrun naar een nieuwe, genegeerde uitvoermap:
.\solutions\milan\.venv\Scripts\python.exe -B solutions/milan/reproduce_v3.py --name mijn_controle_v3

# De gepubliceerde studie controleren en de fulltrain-kandidaten opnieuw maken:
.\solutions\milan\.venv\Scripts\python.exe -B solutions/milan/package_v3.py
.\solutions\milan\.venv\Scripts\python.exe -B solutions/milan/package_voting_v3.py
.\solutions\milan\.venv\Scripts\python.exe -B solutions/milan/run_v3_checks.py
.\solutions\milan\.venv\Scripts\python.exe -B solutions/milan/run_v3_checks.py --packages
.\solutions\milan\.venv\Scripts\python.exe -B solutions/milan/check_frontends_v3.py
.\solutions\milan\.venv\Scripts\python.exe -B solutions/milan/build_v3_reports.py --inline
```

De eerste opdracht herhaalt alle 360 gridfits en aanvullende fits; de overige
opdrachten gebruiken de opgeslagen complete studie en maken/controleren de
fulltrain-artifacts. Voer rapportbouw vanuit de Milan-map uit als de notebook-
werkmap niet automatisch daar wordt ingesteld. `--inline` voert notebookcellen
in IPython met verse naamruimtes uit, zonder aparte Jupyter-kernels te claimen.
Zonder die vlag gebruikt de bouwer gewone Jupyter-kernels.

De tests bewijzen lokale voorbereiding, rijlidmaatschap, herberekende metrics,
votingberekening, exportgelijkheid, API-schema en formulierlogica. De volledige
latere groepssuite, Citi Bike-koppeling en automatische publicatie moeten na
de uiteindelijke gezamenlijke aansluiting nog worden gecontroleerd.
