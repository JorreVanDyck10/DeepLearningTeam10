# Citi Bike: eerste baseline

Voorspeltaak: het aantal ritstarts per uur voor het volledige NYC-netwerk,
in lokale tijd van New York. Deze eerste versie gebruikt januari 2025 uit de
bestaande automatisch gedownloade CSV en de downloadmanifest.

```powershell
.\.venv\Scripts\python.exe nyc_citi_bike/train_baseline.py
```

Of voer `04_predict_baseline.ipynb` uit met de datasetmap als werkdirectory.
Het script leest alleen `started_at`, in stukken van 200.000 rijen. Het verwijdert
ongeldige starttijden en starttijden buiten januari. De 744 kalenderuren vormen
de voorbereide dataset; uren zonder starts krijgen nul. De manifest moet bevestigen
dat de volledige gekozen maand gedownload is. Rit-ID-deduplicatie en volledige EDA
blijven aanvullend werk.

Vóór het trainen toetsen we op de trainingsdagen de vooraf geformuleerde hypothese:
het aandeel ritstarts tussen 07:00 en 09:00 is hoger op werkdagen dan in het weekend.
De permutatietoets gebruikt één waarneming per dag. De steekproef is klein en dagen
kunnen afhankelijk zijn; weer en feestdagen worden niet gecontroleerd.

Train: 1–24 januari (576 uur). Test: 25–31 januari (168 uur). Geen willekeurige split.
Een beslisboom met vaste instellingen (`max_depth=5`, `min_samples_leaf=8`) leert uur
en weekdag. We vergelijken met een vast trainingsgemiddelde en het trainingsgemiddelde
per weekdag en uur. Geen hyperparameterselectie op de testweek.

De beslisboom heeft MAE 831,30; de vaste referentie 2009,39; het weekdag/uur-gemiddelde
728,77. De laatste is beter op deze testweek. Deze deployment is een eerste baseline,
geen claim dat de beslisboom het beste eindmodel is. Verdere modelselectie vereist
chronologische validatie en een nieuwe onaangeroerde eindtestperiode.

Het artifact in `models/baseline_hourly_tree.joblib` bevat de boom en metadata, zonder
ritgegevens. Het is klein genoeg om met deze eerste deployment in Git mee te nemen.
De API `POST /predict/citibike` verwacht `{"date":"2025-01-25","hour":8}` en geeft
de gekozen uurvoorspelling plus 24 dagvoorspellingen terug. Buiten de bronmaand staat
er een waarschuwing: het model kent geen jaarlijkse seizoenen, weer of netwerkveranderingen.

Modelmetrics, cleaningaantallen, hypothesetoets en bronhash staan in `metrics/`.
De frontend krijgt een kopie van de metrics zodat de getoonde vergelijking reproduceerbaar is.
De overige notebooks, AutoML, tuning, AWS en automatische updates zijn nog niet afgerond.

Codex maakte en voerde deze baseline, notebook en API/frontend-koppeling uit.
De teamleden moeten de stappen reviewen en de GenAI-disclosure vervolledigen.
