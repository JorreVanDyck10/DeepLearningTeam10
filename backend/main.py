"""Start from the repository root: python -m uvicorn backend.main:app --reload."""

from contextlib import asynccontextmanager
from datetime import date
import logging
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from backend.mushroom_model import DEFAULT_MODEL, MODEL_ID, PortableMushroom
from backend.mushroom_policy import apply_policy
from backend.raoul_citibike import ensure_artifact, predict as predict_city, model_metadata, MODEL_ID as CITY_MODEL_ID

logger = logging.getLogger(__name__)
DISCLAIMER = "Educatieve voorspelling op hypothetische data; niet gebruiken om echte paddenstoelen te eten."


class MushroomInput(BaseModel):
    """The selected pipeline's eleven features; null represents missing data."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)

    cap_diameter: float | None = Field(ge=0)
    cap_shape: str | None
    gill_color: str | None
    stem_height: float | None = Field(ge=0)
    stem_width: float | None = Field(ge=0)
    stem_surface: str | None
    ring_type: str | None
    spore_print_color: str | None
    habitat: str | None
    season: str | None
    jumbled_noise_1: str | None = None


class MushroomPrediction(BaseModel):
    model_id: str
    label: str
    prediction: str
    probability_poisonous: float
    disclaimer: str
    decision: Literal["uncertain", "suspected_poisonous"]
    decision_policy_id: str


class CityHistory(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)
    timestamp: str
    rides: float = Field(ge=0)
    area: Literal["NYC"]


class CityInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    date: date
    hour: int = Field(ge=0, le=23, strict=True)
    mode: Literal["demo", "history"] = "demo"
    history: list[CityHistory] | None = Field(default=None, min_length=336, max_length=336)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # A new key prevents existing Render settings from selecting the old RF200 pickle.
    model_path = Path(os.getenv("MUSHROOM_RF300_MANIFEST", str(DEFAULT_MODEL)))
    app.state.mushroom_model = None
    try:
        model = PortableMushroom(model_path)
        required = set(MushroomInput.model_fields[field].alias or field
                       for field in MushroomInput.model_fields)
        if set(model.feature_names_in_) != required or model.classes_.tolist() != [0, 1]:
            raise ValueError("Model is incompatible with the Mushroom API contract.")
        app.state.mushroom_model = model
    except Exception:
        # Keep docs available even if the artifact is absent; readiness returns 503.
        logger.exception("Mushroom model could not be loaded from %s", model_path)
    app.state.city_model = None
    try:
        # Legacy CITIBIKE_MODEL_PATH cannot silently select the January baseline.
        app.state.city_model = ensure_artifact()
    except Exception:
        logger.exception("Citi Bike model could not be loaded")
    yield
    app.state.mushroom_model = None
    app.state.city_model = None


app = FastAPI(
    title="DeepLearningTeam10 API",
    description="Voorspellingen met de geselecteerde Mushroom RF300 en Raouls getunede Citi Bike Random Forest.",
    version="0.5.1",
    lifespan=lifespan,
)

# Comma-separated frontend URLs. Never add paths or a trailing slash.
origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
allowed_origins = [origin.strip() for origin in origins.split(",") if origin.strip()]
# Keep the published team frontend working with existing Render env settings.
allowed_origins.append("https://10messi10.netlify.app")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health(request: Request):
    """Readiness: only report success if the model is available."""
    mushroom_ready = request.app.state.mushroom_model is not None
    city_ready = request.app.state.city_model is not None
    ready = mushroom_ready and city_ready
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"status": "ok" if ready else "not_ready",
                 "mushroom_model_loaded": mushroom_ready, "citibike_model_loaded": city_ready,
                 "mushroom_model_id": MODEL_ID,
                 "citibike_model_id": CITY_MODEL_ID if city_ready else None},
    )


@app.post("/predict/mushroom", response_model=MushroomPrediction)
def predict_mushroom(payload: MushroomInput, request: Request):
    model = request.app.state.mushroom_model
    if model is None:
        raise HTTPException(status_code=503, detail="Model niet beschikbaar. Controleer de serverlogs.")
    record = payload.model_dump()
    try:
        prediction = model.predict_record(record)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {**apply_policy(prediction), "model_id": MODEL_ID, "disclaimer": DISCLAIMER}


@app.post("/predict/citibike")
def predict_citibike(payload: CityInput, request: Request):
    artifact = request.app.state.city_model
    if artifact is None:
        raise HTTPException(status_code=503, detail="Citi Bike-model niet beschikbaar.")
    try:
        history = [row.model_dump() for row in payload.history] if payload.history is not None else None
        return predict_city(artifact, payload.date, payload.hour, payload.mode, history)
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/citibike/model")
def citibike_metadata(request: Request):
    artifact = request.app.state.city_model
    if artifact is None:
        raise HTTPException(status_code=503, detail="Raouls Citi Bike-model niet beschikbaar.")
    return model_metadata(artifact)


class FrontendFiles(StaticFiles):
    """Revalidate website files so a new deployment reaches returning visitors."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


# Register this last: /docs, /health and prediction routes take precedence.
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", FrontendFiles(directory=FRONTEND_DIR, html=True), name="frontend")
