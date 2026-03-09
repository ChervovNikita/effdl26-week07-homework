import threading
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import Settings
from src.model import ToxicityModel

import grpc
import inference_pb2_grpc
import inference_pb2


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    is_toxic: bool


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(title="Toxicity Service")

    app.state.settings = settings

    @app.get("/health")
    def health() -> dict:
        return {"status": "ready"}

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest) -> PredictResponse:
        with grpc.insecure_channel(f'[::]:{app.state.settings.grpc_inference_port}') as channel:
            stub = inference_pb2_grpc.TextClassifierStub(channel)
            response = stub.Predict(inference_pb2.TextClassificationInput(text=payload.text))
            return PredictResponse(is_toxic=response.is_toxic)

    return app

app = create_app()
