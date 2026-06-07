from fastapi.testclient import TestClient

from src.api.main import app

VALID_WINE = {
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.7,
    "citric_acid": 0.0,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0,
    "density": 0.9978,
    "ph": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
}


def test_healthcheck() -> None:
    with TestClient(app) as client:
        response = client.get("/healthcheck")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_available"] is True


def test_model_info() -> None:
    with TestClient(app) as client:
        response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["model_name"] == "random_forest"
    assert "metrics" in response.json()
    assert "features" in response.json()


def test_predict() -> None:
    with TestClient(app) as client:
        response = client.post("/predict", json=VALID_WINE)

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "rounded_quality" in response.json()


def test_predict_validation_error() -> None:
    invalid_wine = VALID_WINE.copy()
    invalid_wine.pop("alcohol")

    with TestClient(app) as client:
        response = client.post("/predict", json=invalid_wine)

    assert response.status_code == 422
