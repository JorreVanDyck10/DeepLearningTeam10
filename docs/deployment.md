# Automatische deployment

De bestaande Render Web Service `deeplearningteam10-api` serveert nu zowel de
website uit `frontend/` als de FastAPI-backend. De openbare website is:

https://deeplearningteam10-api.onrender.com/

## Wat gebeurt er bij een push?

1. Push of merge de gewijzigde code naar `main` op GitHub.
2. GitHub Actions controleert of de website, API-routes en modelvoorspelling werken.
3. Render bouwt en deployt de gekoppelde `main`-branch automatisch bij elke commit.
4. De website gebruikt automatisch de API op hetzelfde domein. Er is geen losse
   Static Site of handmatig ingestelde backend-URL nodig.
5. Na het openen/vernieuwen van de pagina valideert de browser de websitebestanden
   opnieuw; voorspellingen worden rechtstreeks bij de actuele backend opgevraagd.

Een geopende pagina wordt niet vanzelf herladen. Vernieuw die na een deployment.
De API laadt het meegeleverde model opnieuw bij de start van de nieuwe deployment.
Automatische hertraining op nieuwe data is een afzonderlijke ML-pipeline.

## Eenmalige Render-instellingen

De bestaande service moet met jullie GitHub-account gekoppeld zijn aan:

- Repository: `JorreVanDyck10/DeepLearningTeam10`
- Branch: `main`
- Root Directory: leeg
- Build Command: `pip install -r backend/requirements.txt`
- Start Command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- Health Check Path: `/health`
- Auto-Deploy: **On Commit**

`render.yaml` beschrijft deze instellingen. Voor een service die handmatig is
aangemaakt veranderen dashboardinstellingen niet automatisch door alleen dit
bestand te pushen: controleer Auto-Deploy op de Settings-pagina. Voor een service
die door een Blueprint wordt beheerd past Render de YAML toe tijdens een sync.

GitHub Actions geeft controles; Render **On Commit** wacht daar niet op. Kies
desgewenst **After CI Checks Pass** als alleen geslaagde commits mogen deployen.

## Lokaal uitvoeren

Vanuit de repository-root:

```powershell
python -m uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000/`. Dezelfde server serveert de website en API.
De aparte ontwikkelserver op poort 5173 blijft ook werken; `config.js` kiest dan
automatisch de backend op poort 8000. Verbindingsinstellingen op de website zijn
alleen een optionele override voor deze browser.

## Controleren

- `/`: website
- `/health`: gereedheid van het paddenstoelenmodel
- `/docs`: API-documentatie
- `POST /predict/mushroom`: paddenstoelenvoorspelling

De gepubliceerde backend ondersteunt momenteel alleen paddenstoelen. De Citi Bike-
interface staat klaar, maar vereist nog publicatie van de bijbehorende API-route.

Lokale controle:

```powershell
python -m pip install -r backend/requirements-dev.txt
python -m unittest discover -s tests -v
```
