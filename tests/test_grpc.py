from unittest.mock import MagicMock

import inference_pb2
import pytest
from fastapi.testclient import TestClient

from src.grpc_api import InferenceClassifier

import threading
import grpc
import inference_pb2_grpc
from src.grpc_api import serve
import time


def test_inference_classifier_predict_toxic() -> None:
    servicer = InferenceClassifier()
    request = inference_pb2.TextClassificationInput(text="you are stupid")
    context = MagicMock()
    response = servicer.Predict(request, context)
    assert response.is_toxic is True


def test_inference_classifier_predict_not_toxic() -> None:
    servicer = InferenceClassifier()
    request = inference_pb2.TextClassificationInput(
        text="thank you and have a nice day"
    )
    context = MagicMock()
    response = servicer.Predict(request, context)
    assert response.is_toxic is False


@pytest.mark.usefixtures("grpc_client")
class TestGrpcClient:
    def test_health(self, grpc_client: TestClient) -> None:
        r = grpc_client.get("/health")
        r.raise_for_status()
        assert r.json()["status"] == "ready"

    def test_predict_toxic(self, grpc_client: TestClient) -> None:
        r = grpc_client.post("/predict", json={"text": "you are stupid"})
        r.raise_for_status()
        assert r.json()["is_toxic"] is True

    def test_predict_not_toxic(self, grpc_client: TestClient) -> None:
        r = grpc_client.post("/predict", json={"text": "thank you and have a nice day"})
        r.raise_for_status()
        assert r.json()["is_toxic"] is False


def test_serve_starts_and_accepts_requests() -> None:
    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    time.sleep(15)
    with grpc.insecure_channel("127.0.0.1:50051") as channel:
        stub = inference_pb2_grpc.TextClassifierStub(channel)
        response = stub.Predict(
            inference_pb2.TextClassificationInput(text="you are stupid")
        )
        assert response.is_toxic is True
