# Milan — kritisch reviewconcept en alternatief

**Conclusie:** Final is uitgebreid en de opgeslagen cijfers kloppen. De belangrijkste
beperkingen zijn de al bekeken Mushroom-test en de niet opnieuw uitgevoerde tuning
op de gecorrigeerde pipeline. Het alternatief hieronder is werkelijk getraind:
macro F1 verbetert slechts van **0,7619 naar 0,7629**. Dat rechtvaardigt geen
automatische vervanging van het gehoste model.

Deze map is op verzoek van de gebruiker door Codex gemaakt. Milan moet de inhoud,
eigen bijdrage en AI-inzet zelf beoordelen en bevestigen. De map is een concept
naast de bestaande teamoplossing, op branch `codex/milan-final-review`.

## Eerst lezen

1. [Kritische beoordeling tegenover de PowerPoints](REVIEW_FINAL.md): bevindingen,
   prioriteiten, les-/dianummers, bewijs en concrete voorstellen.
2. [Gemeten uitkomst en advies](RESULTATEN.md): wat wel en nauwelijks verbetert,
   inclusief het effect van een andere beslisdrempel.
3. [Voorbereiding](01_voorbereiding.ipynb): werkelijke invoervelden, broncontrole,
   folds en voorbereiding zonder grafieken.
4. [Random Forest en tuning](02_random_forest.ipynb): eenvoudige pipeline, echte
   nieuwe grids en uitleg van binnen- en buitenfolds.
5. [Vergelijking](03_vergelijking.ipynb): herberekende metrics, figuur en beperkingen.

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
De uitkomsten komen uit de bewaarde berekeningen; een hogere score is niet
verzonnen. De beperkte winst wordt expliciet vermeld. Er is geen persoonlijke
menselijke review, Canvas-indiening of nieuwe cloudrun als afgerond aangemerkt.
