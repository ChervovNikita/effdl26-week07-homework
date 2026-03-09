import threading
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import Settings
from src.model import ToxicityModel


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

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest) -> PredictResponse:
        if not app.state.ready:
            raise HTTPException(status_code=503, detail="Model is still loading.")

        is_toxic = app.state.model.predict(payload.text)

        return PredictResponse(is_toxic=is_toxic)

    return app

app = create_app()
