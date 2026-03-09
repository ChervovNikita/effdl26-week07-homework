import pytest
from fastapi.testclient import TestClient


@pytest.mark.usefixtures("http_client")
class TestHTTP:
    def test_health(self, http_client: TestClient) -> None:
        r = http_client.get("/health")
        r.raise_for_status()
        assert r.json()["status"] == "ready"

    def test_predict_toxic(self, http_client: TestClient) -> None:
        r = http_client.post("/predict", json={"text": "you are stupid"})
        r.raise_for_status()
        assert r.json()["is_toxic"] is True

    def test_predict_not_toxic(self, http_client: TestClient) -> None:
        r = http_client.post("/predict", json={"text": "thank you and have a nice day"})
        r.raise_for_status()
        assert r.json()["is_toxic"] is False

    def test_predict_get_toxic(self, http_client: TestClient) -> None:
        r = http_client.get("/predict", params={"text": "you are stupid"})
        r.raise_for_status()
        assert r.json()["is_toxic"] is True

    def test_predict_get_not_toxic(self, http_client: TestClient) -> None:
        r = http_client.get(
            "/predict", params={"text": "thank you and have a nice day"}
        )
        r.raise_for_status()
        assert r.json()["is_toxic"] is False
