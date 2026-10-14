# Waarom het formulier geen eetbaarheid bevestigt

Jorre gaf aan dat een giftige paddenstoel als eetbaar tonen ernstiger is dan een
eetbare paddenstoel afwijzen. Daarom is de statistische e/p-klasse gescheiden van
de boodschap op de website. RF300, preprocessing, kansen en vergelijkingsmetrics
blijven gelijk; er is geen nieuw model getraind.

Een grens van 95% modelscore voor de eetbare klasse blijkt onvoldoende. De
bestaande out-of-fold-voorspellingen op de 4.100 trainingsrijen laten dit zien:

| Minimale modelscore eetbare klasse | Gevallen onder die beslisregel als eetbaar | Daarvan werkelijk giftig |
| --- | ---: | ---: |
| 50% | 2.906 | 614 |
| 90% | 200 | 21 |
| 95% | 55 | 7 |
| 97,5% | 9 | 2 |
| 99% | 4 | 1 |
| 99,5% | 2 | 1 |
| 100% | 0 | 0 |

Zelfs een hogere score maakt de uitspraak niet betrouwbaar. Bij 95% zijn 7 van
de 55 toegelaten gevallen giftig: circa 12,7%, ondanks hun lage giftigheidsscore.
Dit is beschrijvende trainingsvalidatie, geen schatting van het risico van echte
paddenstoelen. De modellen zijn niet gekalibreerd en de cursusdata zijn geen
bewijs voor veilig gebruik buiten de dataset.

De reeds bekeken ontwikkeltest geeft bij 95% 13 toegelaten gevallen, waarvan één
giftig. Deze test is alleen beschrijvend gecontroleerd, niet gebruikt om een
gunstige grens te kiezen. Het standaardvoorbeeld is zelfs dit giftige testgeval:
de eetbare score is circa 96,53%. Alleen de 50%-grens naar 95% verplaatsen zou dit
concrete verkeerde “Eetbaar”-resultaat dus laten bestaan.

## Nieuwe beslisregel

- Giftigheidsscore vanaf 50%: **Model voorspelt: giftig**.
- Giftigheidsscore onder 50%, ook bij een heel hoge eetbare score:
  **Onzeker / mogelijk giftig**.

De tweede groep wordt niet bewezen giftig genoemd: het model onthoudt zich van
een uitspraak over eetbaarheid. Er is dus geen groene goedkeuring. De
giftigheidsscore blijft zichtbaar als modelscore, niet als bewezen reële kans.
Deze regel garandeert geen correcte classificatie; hij voorkomt dat deze demo
eetbaarheid bevestigt wanneer de onderliggende classificatie fout kan zijn.

De API behoudt `label` (ruwe modelklasse e/p) en `probability_poisonous` voor de
wetenschappelijke vergelijking. `prediction` bevat de voorzichtige boodschap;
`decision` is `uncertain` of `suspected_poisonous`, met
`decision_policy_id="mushroom_no_edibility_confirmation_v1"`. De website
controleert zowel policy als score. Een oude of tegenstrijdige API-respons wordt
afgewezen, zodat een gedeeltelijke deployment geen oude “Eetbaar”-melding toont.

De projectmetrics blijven expliciet die van de oorspronkelijke 50%-classificatie.
Onzekere gevallen zijn geen correcte giftig-voorspellingen; we claimen met deze
onthouding dus geen 100% recall of accuracy.

## Reproduceren

```powershell
python MushroomFinal/audit_decision_thresholds.py
python -m unittest discover -s tests -v
node --test tests/test_mushroom_frontend.cjs
```

Het [auditbestand](../MushroomFinal/comparison_correct/decision_threshold_audit.json)
bevat de volledige vaste-grenzenanalyse en bronhashes. Zie ook de primaire
documentatie over [beslisdrempels](https://scikit-learn.org/stable/modules/classification_threshold.html)
en [kanskalibratie](https://scikit-learn.org/stable/modules/calibration.html).

Op 14 oktober 2026 zijn 26 Python-tests en 19 JavaScript-tests geslaagd, evenals
de [GitHub-controles van deze wijziging](https://github.com/JorreVanDyck10/DeepLearningTeam10/actions/runs/38593879069).
De publieke backend geeft de nieuwe policy-ID terug. Een browsercontrole op
Netlify toont voor het standaardvoorbeeld **Onzeker / mogelijk giftig**, met
de oorspronkelijke 3,5% giftigheidsscore. Acht publieke referentievoorspellingen
behouden hun oorspronkelijke labels en kansen en gebruiken de nieuwe boodschap.
Ook het Citi Bike-formulier werkt nog. Het
[deploymentnotebook](../secondary_mushroom/10_deploy.ipynb) is opnieuw uitgevoerd
en bevat de nieuwe beslisregel en live uitvoer.
