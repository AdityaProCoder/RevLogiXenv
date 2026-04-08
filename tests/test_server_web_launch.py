from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.app import build_app


def test_build_app_returns_fastapi_instance() -> None:
    app = build_app()
    assert isinstance(app, FastAPI)


def test_build_app_web_enabled_mounts_web(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_WEB_INTERFACE", "true")
    app = build_app()
    client = TestClient(app)

    response = client.get("/web/", follow_redirects=True)
    assert response.status_code != 404


def test_build_app_default_non_web_still_serves_health(monkeypatch) -> None:
    monkeypatch.delenv("ENABLE_WEB_INTERFACE", raising=False)
    app = build_app()
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200

    web = client.get("/web/", follow_redirects=True)
    assert web.status_code == 404
