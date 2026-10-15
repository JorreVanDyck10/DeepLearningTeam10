# Milan — kritisch reviewconcept en alternatief

**Vervolgstudie afgerond:** de bruikbaarste beperkte verbetering is RF300 met
een binnen training afgestemde beslisregel. Macro F1 stijgt van **0,7619 naar
0,7691**, giftige recall van **60,44% naar 66,82%**. Op dezelfde 4.100
buitenfoldvoorspellingen zijn er **99 minder gemiste giftige gevallen**, tegenover
**100 extra valse alarmen**. Dit is ontwikkelbewijs; geen nieuwe onafhankelijke
eindtest of bewijs van veiligheid.

Dezelfde elf invoervelden en bestaande modellezer blijven bruikbaar. Er staat een
gecontroleerd kandidaatpakket met lokale preview klaar. De gezamenlijke website,
API en automatische training zijn nog niet omgeschakeld.

Deze map is op verzoek van de gebruiker door Codex gemaakt. Milan moet de inhoud,
eigen bijdrage en AI-inzet zelf beoordelen en bevestigen. De map is een concept
naast de bestaande teamoplossing, op branch `codex/milan-final-review`.

## Eerst lezen

Begin bij de [actuele status](STATUS.md) voor resultaat, controles en grenzen.

1. [Nieuwe resultaten en oordeel](RESULTATEN_V2.md): alle vier modelprocedures,
   de werkelijke winst en de kosten in valse alarmen.
2. [Later aansluiten](INTEGRATIE.md): de kleine koppelpunten die samen moeten
   worden overgenomen; geen losse artifactwissel met oude metrics.
3. [Onderzoeksafspraken](STUDIE_V2.md): vooraf vastgelegde searches en criteria.
4. [Verbeterde Random Forest](04_RF_verbeterd.ipynb),
   [Gradient Boosting](05_Gradient_Boosting.ipynb) en [SVM](06_SVM.ipynb).
5. [Foutanalyse en integratiecontrole](07_foutanalyse_en_integratie.ipynb).
6. [Oorspronkelijke kritische review van Final](REVIEW_FINAL.md).

De eerste proef blijft als onderzoeksarchief staan: [resultaten v1](RESULTATEN.md),
[voorbereiding](01_voorbereiding.ipynb), [RF150-proef](02_random_forest.ipynb) en
[eerste vergelijking](03_vergelijking.ipynb). De conclusie van v1 betreft die
specifieke proef, niet de inmiddels uitgevoerde tweede studie.

## De nieuwe studie opnieuw uitvoeren

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
Ook de tweede studie, het kandidaatpakket, de preview en tests zijn met substantiële
hulp van Codex gemaakt. Een persoonlijke goedkeuring door Milan is niet verondersteld.
De uitkomsten komen uit de bewaarde berekeningen; een hogere score is niet
verzonnen. De beperkte winst wordt expliciet vermeld. Er is geen persoonlijke
menselijke review, Canvas-indiening of nieuwe cloudrun als afgerond aangemerkt.
