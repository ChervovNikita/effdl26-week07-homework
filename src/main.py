import threading

from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from prometheus_client import Counter, generate_latest
from fastapi.responses import PlainTextResponse

from src.config import Settings
from src.model import ToxicityModel


inference_count = Counter(
    "app_http_inference_count",
    "Number of requests",
)


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    is_toxic: bool


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(title="Toxicity Service")

    app.state.settings = settings
    app.state.model = ToxicityModel()
    app.state.ready = False

    def _load_model() -> None:
        app.state.model.load()
        app.state.ready = True

    @app.on_event("startup")
    def on_startup() -> None:
        thread = threading.Thread(target=_load_model, daemon=True)
        thread.start()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ready" if app.state.ready else "not ready"}

    def _predict(text: str) -> PredictResponse:
        if not app.state.ready:
            raise HTTPException(status_code=503, detail="Model is still loading.")
        inference_count.inc()
        is_toxic = app.state.model.predict(text)
        return PredictResponse(is_toxic=is_toxic)

    @app.post("/predict", response_model=PredictResponse)
    def predict_post(payload: PredictRequest) -> PredictResponse:
        return _predict(payload.text)

    @app.get("/predict", response_model=PredictResponse)
    def predict_get(text: str = Query(...)) -> PredictResponse:
        return _predict(text)

    @app.get("/metrics")
    def metrics():
        return PlainTextResponse(content=generate_latest())

    return app


app = create_app()
