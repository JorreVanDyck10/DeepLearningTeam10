# Milan — kritisch reviewconcept en alternatief

**Derde studie afgerond:** gewogen voting met een binnen training afgestemde
F1-drempel haalt **macro F1 0,7788**, tegenover **0,7691** voor kandidaat v2.
Op dezelfde 4.100 buitenfoldvoorspellingen zijn er **15 minder gemiste giftige
gevallen én 22 minder valse alarmen**. Vier van vijf folds verbeteren in macro F1.
Dit is een bescheiden verdere winst op ontwikkeldata, geen onafhankelijke eindtest.

Daarnaast haalt RF300 met interne featureselectie **macro F1 0,7749** bij de
standaardgrens. Tegenover de oorspronkelijke RF300 zijn er 38 minder gemiste
giftige gevallen en vijf minder valse alarmen. Deze eenvoudiger optie behoudt de
bestaande modellezer en elf invoervelden, maar mist meer giftige gevallen dan v2.
Voting heeft extra afhankelijkheden en een andere modellezer nodig. De website,
gedeelde API, Final en automatische training zijn nog niet omgeschakeld.

Deze map is op verzoek van Milan met substantiële hulp van Codex gemaakt.
Milan bevestigde op 15 oktober 2026 zijn aandeel in cleaning/EDA, de herbouwde
frontend en de recente Final-verbeteringen en eindcontrole met Codex. Zie de
[bijdragen](../../Final/BIJDRAGEN.md) en [GenAI Disclosure](../../Final/GENAI_DISCLOSURE.md).
Een bevestigde bijdrage staat los van persoonlijke inhoudelijke beheersing.
De map is een concept
naast de bestaande teamoplossing, op branch `codex/milan-final-review`.

## Eerst lezen

Begin bij de [actuele status](STATUS.md) voor resultaat, controles en grenzen.

1. [Nieuwe resultaten en oordeel](RESULTATEN_V3.md): alle procedures en foutaantallen.
2. [Later aansluiten: RF of voting](INTEGRATIE_V3.md): bestanden, instellingen en controles.
3. [Onderzoeksafspraken v3](STUDIE_V3.md): de vaste zoekruimtes en verbeterdoelen.
4. [Featureselectie en boosting](08_featureselectie_en_boosting.ipynb),
   [voting en vergelijking](09_voting_en_vergelijking.ipynb) en
   [foutanalyse](10_foutanalyse_v3.ipynb).
5. [Oorspronkelijke kritische review van Final](REVIEW_FINAL.md).

Studie v2 blijft als gecontroleerde referentie beschikbaar:
[resultaten](RESULTATEN_V2.md), [integratie](INTEGRATIE.md),
[RF](04_RF_verbeterd.ipynb), [Gradient Boosting](05_Gradient_Boosting.ipynb),
[SVM](06_SVM.ipynb) en [foutanalyse](07_foutanalyse_en_integratie.ipynb).

De eerste proef blijft als onderzoeksarchief staan: [resultaten v1](RESULTATEN.md),
[voorbereiding](01_voorbereiding.ipynb), [RF150-proef](02_random_forest.ipynb) en
[eerste vergelijking](03_vergelijking.ipynb). De conclusie van v1 betreft die
specifieke proef, niet de inmiddels uitgevoerde tweede studie.

## Studie v3 opnieuw uitvoeren

Installeer de bestaande requirements plus `requirements-boosting.txt` in de aparte
Milan-venv. `reproduce_v3.py` herhaalt de volledige vaste studie naar een nieuwe
map onder `.generated`, ook in een verse clone zonder lokale checkpoints.
Gebruik deze route om het bewaarde onderzoeksbewijs niet te overschrijven.
De complete opdrachten voor training, pakketten, tests en notebooks staan in
[INTEGRATIE_V3.md](INTEGRATIE_V3.md#controles-en-opnieuw-uitvoeren).

De nieuwe vergelijking omvat 360 gridfits, featureselectie, LightGBM, CatBoost
en vooraf vastgelegde votinggewichten. Voorbereiding, instellingen, gewichten en
drempels worden binnen training geleerd. Alle nieuwe procedures gebruiken alleen
de oorspronkelijke 4.100 trainingsrijen. De bekende 900 records zijn niet gescoord.

## Studie v2 opnieuw uitvoeren

Gebruik de volledige repository en de aparte Milan-omgeving. Vanaf de repositoryroot:

```powershell
.\solutions\milan\.venv\Scripts\python.exe -m pip install -r solutions/milan/requirements.txt
.\solutions\milan\.venv\Scripts\python.exe -m pip install -r solutions/milan/requirements-preview.txt
.\solutions\milan\.venv\Scripts\python.exe solutions/milan/improve.py
.\solutions\milan\.venv\Scripts\python.exe solutions/milan/package_candidate.py
.\solutions\milan\.venv\Scripts\python.exe solutions/milan/run_checks.py
.\solutions\milan\.venv\Scripts\python.exe solutions/milan/build_v2_reports.py
node --test solutions/milan/test_preview.cjs
```

Maak zo nodig eerst de venv met `python -m venv solutions/milan/.venv`. De searches
hervatten alleen reeds afgeronde folds met exact hetzelfde protocol en dezelfde
codehash. Een andere zoekruimte vereist een nieuwe studiemap. De preview- en
notebookdependencies zijn aanvullend vastgelegd in
[requirements-preview-lock.txt](requirements-preview-lock.txt).

De getrainde sklearn-pipeline en array-export staan onder `artifacts`. De ruwe
classifier houdt zijn oorspronkelijke grens. De apart voorgestelde beleidsgrens
staat in metadata en wordt alleen door de expliciete preview toegepast. Er zijn
**twee metrieksets**; raw- en policyresultaten worden niet verwisseld.

De onderstaande historische uitvoeringsroute herhaalt uitsluitend de eerste proef.

## Waarom dit bij de lessen past

De implementatie gebruikt `Pipeline`, `SimpleImputer`, `OneHotEncoder`,
`RandomForestClassifier`, `StratifiedKFold`, `GridSearchCV`, confusion matrices
en ROC-drempels. Dat sluit aan op de cursus en de opgeloste oefeningen over
Winetrees, Random Forest en threshold tuning. De extra scheiding met binnen- en
buitenfolds voorkomt dat dezelfde validatie tegelijk kiest en beoordeelt.
De [review](REVIEW_FINAL.md) vermeldt de precieze dia's.

Het experiment gebruikt dezelfde elf kenmerken en dezelfde 4.100 oorspronkelijke
trainingsrijen. De bekende 900 ontwikkeltestrecords tellen niet mee voor nieuwe
selectie of scoring. Er wordt geen nieuwe onafhankelijke eindtest geclaimd.
De drempelproef is afzonderlijk zichtbaar naast standaardlabels; hij maakt het
model niet aantoonbaar veilig en vervangt de bestaande websitebeslissing niet.

## Reproduceren

Gebruik de volledige repository en een aparte Python 3.13-omgeving. Vanaf de
repositoryroot:

```powershell
python -m venv solutions/milan/.venv
.\solutions\milan\.venv\Scripts\python.exe -m pip install -r solutions/milan/requirements.txt
.\solutions\milan\.venv\Scripts\python.exe solutions/milan/audit_final.py
.\solutions\milan\.venv\Scripts\python.exe solutions/milan/mushroom_alternative.py
.\solutions\milan\.venv\Scripts\python.exe solutions/milan/build_notebooks.py
```

De trainingscode doet de volledige nieuwe search. De notebooks tonen standaard de
werkelijk berekende resultaten; in het RF-notebook kan `RERUN_TRAINING=True` de
training opnieuw starten. De rapportbouwer voert rapportcellen in verse Python-
namespaces uit. Jupyter/VS Code kunnen dezelfde notebooks met de eigen omgeving
openen. De gemeten packageversies en de bron-/codehash staan in
[het protocol](results/protocol.json).

Deze machine gebruikte een lokale omgeving met toegang tot reeds geïnstalleerde,
hier gepinde packages. Een nieuwe installatie via bovenstaande instructies is
hier niet opnieuw vanaf lege packagecache getest. De bestaande Final-training
blijft zijn eigen Python 3.11/sklearn 1.4.2-omgeving gebruiken.

## Wat is gecontroleerd?

[audit_final.py](audit_final.py) herberekent opgeslagen metrics, controleert
bronhashes en notebooklinks, vergelijkt de serving-export met de RF300-referentie
en test met synthetische data of Citi Bike-features informatie uit de voorspeldag
of toekomst gebruiken. De [audituitvoer](results/audit_final.json) bevat de concrete
uitkomst, inclusief de twee kapotte notebooklinks.

De oorspronkelijke PyCaret-, AWS- en volledige Citi Bike-training en publieke
hosting zijn hier niet opnieuw uitgevoerd. De cursusinventaris met hashes staat
in [course_sources.json](results/course_sources.json). Er zijn geen lesbestanden,
grote modellen, omgevingen of externe credentials meegecommit.

## GenAI-verantwoording van dit concept

De echte opdracht van de gebruiker was om Final kritisch met de PowerPoints te
vergelijken, verbeteringen te melden en eventueel een alternatief in
`solutions/milan` te maken. Codex las de lokale cursus en projectcode, schreef de
review, trainingscode, audits en rapportnotebooks en voerde die controles uit.
Ook de tweede en derde studie, kandidaatpakketten, previews en tests zijn met substantiële
hulp van Codex gemaakt. Milans eigen bijdrage en beschreven AI-inzet zijn nu
vastgelegd in de centrale verantwoording. Dat is geen automatische goedkeuring
van alle onderzoekscijfers of verklaring van mondelinge beheersing.
De uitkomsten komen uit de bewaarde berekeningen; een hogere score is niet
verzonnen. De beperkte winst wordt expliciet vermeld. Er is geen persoonlijke
menselijke review, Canvas-indiening of nieuwe cloudrun als afgerond aangemerkt.
