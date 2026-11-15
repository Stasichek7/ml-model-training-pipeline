"""Exercise cached model serving and HTTP/Pydantic error contracts."""

from fastapi.testclient import TestClient

from ml_pipeline.api import create_app


def test_health_without_model_and_unavailable_endpoints(tmp_path):
    with TestClient(create_app(tmp_path / "missing.joblib")) as client:
        assert client.get("/health").json() == {"status": "ok", "model_loaded": False}
        assert client.get("/model/info").status_code == 503
        assert client.post("/predict", json={"instances": [{"x": 1}]}).status_code == 503


def test_api_inference_loads_once(trained_run, monkeypatch):
    from ml_pipeline import prediction

    root, _, dataset = trained_run
    original = prediction.load_artifact
    calls = []

    def counted(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(prediction, "load_artifact", counted)
    payload = {"instances": dataset.X.head(2).to_dict(orient="records")}
    with TestClient(create_app(root / "models/best_model.joblib")) as client:
        assert client.get("/health").json()["model_loaded"] is True
        info = client.get("/model/info").json()
        assert len(info["feature_names"]) == 30
        for _ in range(2):
            response = client.post("/predict", json=payload)
            assert response.status_code == 200
            assert len(response.json()["predictions"]) == 2
        assert len(calls) == 1


def test_api_rejects_invalid_requests(trained_run):
    root, _, dataset = trained_run
    row = dataset.X.iloc[0].to_dict()
    with TestClient(create_app(root / "models/best_model.joblib")) as client:
        for payload in [
            {},
            {"instances": []},
            {"instances": [row], "extra": 1},
            {"instances": [{"unknown": 1}]},
            {"instances": [[1, 2]]},
            {"instances": [{**row, "mean_radius": "12"}]},
            {"instances": [{**row, "mean_radius": True}]},
            {"instances": [{**row, "mean_radius": {"value": 1}}]},
        ]:
            assert client.post("/predict", json=payload).status_code == 422
