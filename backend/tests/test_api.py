from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_root_contract():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["system"] == "VisionShield AI Backend"


def test_health_is_honest_without_required_weights():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in {"healthy", "degraded"}


def test_dashboard_health_contract():
    response = client.get("/pipeline/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["api_status"] == "connected"
    assert "models" in payload
    assert "artifact_store" in payload
    assert {route["id"] for route in payload["weather_routes"]} == {"fog_its", "fog_ots", "rain", "snow", "low_light"}


def test_empty_history_is_valid():
    response = client.get("/pipeline/history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_empty_benchmark_summary_is_measured_only():
    response = client.get("/benchmark/summary")
    assert response.status_code == 200
    assert "runs" in response.json()


def test_invalid_weather_route_is_rejected():
    response = client.post("/benchmark/image?weather=hail", files={"file": ("test.jpg", b"not-an-image", "image/jpeg")})
    assert response.status_code == 400
    assert "Unsupported weather route" in response.json()["detail"]


def test_missing_artifact_returns_404():
    response = client.get("/artifacts/does-not-exist")
    assert response.status_code == 404
