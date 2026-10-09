"""Start from the repository root: python -m uvicorn backend.main:app --reload."""

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from secondary_mushroom.predict import predict_records
from backend.mushroom_model import DEFAULT_MODEL, MODEL_ID

logger = logging.getLogger(__name__)
DISCLAIMER = "Educatieve voorspelling op hypothetische data; niet gebruiken om echte paddenstoelen te eten."


class MushroomInput(BaseModel):
    """Andrew's twelve features; null represents missing data."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)

    cap_diameter: float | None = Field(alias="cap-diameter", ge=0)
    cap_shape: str | None = Field(alias="cap-shape")
    gill_color: str | None = Field(alias="gill-color")
    stem_height: float | None = Field(alias="stem-height", ge=0)
    stem_width: float | None = Field(alias="stem-width", ge=0)
    stem_surface: str | None = Field(alias="stem-surface")
    ring_type: str | None = Field(alias="ring-type")
    spore_print_color: str | None = Field(alias="spore-print-color")
    habitat: str | None
    season: str | None
    jumbled_noise_0: str | None = None
    jumbled_noise_1: str | None = None


class MushroomPrediction(BaseModel):
    label: str
    prediction: str
    probability_poisonous: float
    disclaimer: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Only load model artifacts produced by our own training pipeline.
    # Separate key prevents an old Render baseline setting selecting the old model.
    model_path = Path(os.getenv("ANDREW_MUSHROOM_MODEL_PATH", str(DEFAULT_MODEL)))
    app.state.mushroom_model = None
    try:
        model = joblib.load(model_path)
        required = set(MushroomInput.model_fields[field].alias or field
                       for field in MushroomInput.model_fields)
        if set(model.feature_names_in_) != required or "p" not in model.classes_:
            raise ValueError("Model is incompatible with the Mushroom API contract.")
        app.state.mushroom_model = model
    except Exception:
        # Keep docs available even if the artifact is absent; readiness returns 503.
        logger.exception("Mushroom model could not be loaded from %s", model_path)
    yield
    app.state.mushroom_model = None


app = FastAPI(
    title="DeepLearningTeam10 API",
    description="Voorspellingen met Andrew's Random Forest. Citi Bike volgt later.",
    version="0.2.0",
    lifespan=lifespan,
)

# Comma-separated frontend URLs. Never add paths or a trailing slash.
origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in origins.split(",") if origin.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health(request: Request):
    """Readiness: only report success if the model is available."""
    ready = request.app.state.mushroom_model is not None
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"status": "ok" if ready else "not_ready", "mushroom_model_loaded": ready,
                 "mushroom_model_id": MODEL_ID},
    )


@app.post("/predict/mushroom", response_model=MushroomPrediction)
def predict_mushroom(payload: MushroomInput, request: Request):
    model = request.app.state.mushroom_model
    if model is None:
        raise HTTPException(status_code=503, detail="Model niet beschikbaar. Controleer de serverlogs.")
    record = payload.model_dump(by_alias=True)
    # Reject unseen codes rather than silently predicting with unknown categories.
    preprocessing = model.named_steps["preprocessor"]
    for name, transformer, columns in preprocessing.transformers_:
        if name == "remainder" or not hasattr(transformer, "steps"):
            continue
        encoder = transformer.steps[-1][1]
        if hasattr(encoder, "categories_"):
            for column, categories in zip(columns, encoder.categories_):
                value = record[column]
                if value is not None and value not in categories:
                    raise HTTPException(status_code=422, detail=f"Onbekende datasetcode voor {column}.")
    try:
        prediction = predict_records(record, model=model)[0]
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {**prediction, "disclaimer": DISCLAIMER}


class FrontendFiles(StaticFiles):
    """Revalidate website files so a new deployment reaches returning visitors."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


# Register this last: /docs, /health and prediction routes take precedence.
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", FrontendFiles(directory=FRONTEND_DIR, html=True), name="frontend")
