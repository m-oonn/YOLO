"""Tests for FastAPI endpoints."""

import os
import sys
from unittest.mock import MagicMock, patch

COURSE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if COURSE_DIR not in sys.path:
    sys.path.insert(0, COURSE_DIR)

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def mock_store():
    """Auto-use fixture to mock the store for all tests."""
    with patch("backend.api.events.get_store") as mock_get_store:
        mock_instance = MagicMock()
        mock_instance.query_with_total.return_value = ([], 0)
        mock_instance.get_stats.return_value = {
            "total_events": 0,
            "by_type": {},
            "first_event": None,
            "last_event": None,
        }
        mock_instance.count.return_value = 0
        mock_instance.delete_events.return_value = 5
        mock_instance.clear_all.return_value = 10
        mock_get_store.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def client(mock_store):
    from backend.main import app

    with TestClient(app) as c:
        yield c


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data


def test_list_events(client, mock_store):
    response = client.get("/api/events/")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "page_size" in data
    assert "total_pages" in data
    assert "has_next" in data
    assert "has_prev" in data


def test_event_stats(client, mock_store):
    response = client.get("/api/events/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_events" in data
    assert "by_type" in data


def test_event_types(client, mock_store):
    response = client.get("/api/events/types")
    assert response.status_code == 200
    data = response.json()
    assert "event_types" in data


def test_detection_status_no_pipeline(client):
    response = client.get("/api/detection/status")
    assert response.status_code == 200
    data = response.json()
    assert data["running"] is False


def test_start_detection_already_active(client):
    with (
        patch("backend.api.detection._detection_active", True),
        patch("backend.api.detection._pipeline") as mock_pipeline,
    ):
        mock_pipeline.running = True
        response = client.post("/api/detection/start", json={"source": "0"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"


def test_stop_detection_no_active(client):
    with patch("backend.api.detection._detection_active", False):
        response = client.post("/api/detection/stop")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"


def test_list_cameras(client):
    response = client.get("/api/cameras/")
    assert response.status_code == 200
    data = response.json()
    assert "devices" in data


def test_event_query_with_filters(client, mock_store):
    mock_store.query_with_total.return_value = (
        [
            {
                "id": 1,
                "event_type": "running",
                "timestamp_s": 1000.0,
                "frame_index": 100,
                "track_id": 1,
                "description": "running (track 1)",
            }
        ],
        1,
    )
    response = client.get("/api/events/?event_type=running&limit=10&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert len(data["items"]) == 1


def test_delete_events(client, mock_store):
    response = client.delete("/api/events/?event_type=running")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "deleted"
    assert data["count"] == 5


def test_clear_all_events(client, mock_store):
    response = client.delete("/api/events/all")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "cleared"
    assert data["count"] == 10
