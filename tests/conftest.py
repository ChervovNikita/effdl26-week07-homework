import random

import grpc
import inference_pb2_grpc
import pytest
from concurrent import futures
from fastapi.testclient import TestClient

from src.config import Settings
from src.grpc_api import InferenceClassifier
from src.grpc_client import create_app
from src.main import app


@pytest.fixture(autouse=True)
def fixed_seed() -> None:
    random.seed(12345)


@pytest.fixture(scope="session")
def http_client() -> TestClient:
    app.state.model.load()
    app.state.ready = True
    return TestClient(app)


@pytest.fixture(scope="session")
def grpc_server():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    inference_pb2_grpc.add_TextClassifierServicer_to_server(
        InferenceClassifier(), server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    yield 50051
    server.stop(grace=1)


@pytest.fixture(scope="session")
def grpc_client(grpc_server: int) -> TestClient:
    settings = Settings(
        grpc_inference_port=grpc_server,
    )
    grpc_app = create_app(settings)
    return TestClient(grpc_app)
