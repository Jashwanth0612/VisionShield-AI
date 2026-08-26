from fastapi.testclient import TestClient

from main import app


def test_video_rejects_non_video_upload():
    client = TestClient(app)
    response = client.post(
        "/video/analyze",
        files={"file": ("image.jpg", b"not-a-video", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "Unsupported video format" in response.json()["detail"]
