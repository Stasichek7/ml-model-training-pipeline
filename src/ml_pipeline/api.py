"""FastAPI inference with lifespan-managed, cached model loading."""

import math
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
)

from ml_pipeline.prediction import PredictionService


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instances: list[dict[str, StrictFloat | StrictInt | StrictStr | StrictBool | None]] = Field(
        min_length=1,
        max_length=1000,
    )

    @field_validator("instances")
    @classmethod
    def reject_nonfinite(cls, rows):
        for row in rows:
            if any(isinstance(value, float) and not math.isfinite(value) for value in row.values()):
                raise ValueError("Feature values must be finite or null")
        return rows


def create_app(model_path: str | Path | None = None) -> FastAPI:
    path = Path(model_path or os.environ.get("ML_MODEL_PATH", "models/best_model.joblib"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = None
        if path.is_file():
            # Invalid existing artifacts fail startup instead of serving misleading predictions.
            app.state.service = PredictionService(path)
        yield

    application = FastAPI(title="ML Model Training Pipeline", version="1.0.0", lifespan=lifespan)

    @application.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError):
        # Do not echo invalid values: overflowed JSON numbers can be infinity,
        # which cannot itself be serialized in a standards-compliant response.
        errors = [
            {"type": error["type"], "loc": list(error["loc"]), "msg": error["msg"]}
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": errors})

    def service(request: Request) -> PredictionService:
        loaded = getattr(request.app.state, "service", None)
        if loaded is None:
            raise HTTPException(status_code=503, detail="Model unavailable. Run training first.")
        return loaded

    @application.get("/health")
    def health(request: Request):
        ready = getattr(request.app.state, "service", None) is not None
        return {"status": "ok", "model_loaded": ready}

    @application.get("/model/info")
    def model_info(request: Request):
        return service(request).metadata

    @application.post("/predict")
    def predict(payload: PredictRequest, request: Request):
        try:
            return service(request).predict(payload.instances)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return application


app = create_app()
