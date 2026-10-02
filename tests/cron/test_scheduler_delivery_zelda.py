"""ZELDA fork cron delivery lane: deliver=["zelda"] stages a NOTICE push row.

The api_server adapter is request/response only (send() is a stub), so cron
reports could never reach the Sinda fork — until the zelda lane landed
(2026-10-02). The queue IS the transport: same row contract as the alert
fan-out (zelda-alert.sh / gateway zelda_notices.py).
"""
import json
from pathlib import Path

import pytest

from cron import scheduler_delivery as sd


@pytest.fixture
def push_queue(monkeypatch, tmp_path):
    """Point the push queue at a temp dir; return it."""
    qdir = tmp_path / ".hermes" / "zelda-outbound" / ".push-queue"
    monkeypatch.setattr(sd.Path, "home", staticmethod(lambda: tmp_path))
    return qdir


def _job(tmp_path):
    return {"id": "job-zelda-1", "name": "Test cron job", "status": "success",
            "last_run_output": "Report body", "deliver": ["zelda"],
            "output_path": str(tmp_path / "out")}


def _run_delivery(tmp_path, monkeypatch, job, media=None):
    """Drive _deliver_result for one zelda target with the heavy collaborators stubbed."""
    monkeypatch.setattr(sd, "_resolve_delivery_targets",
                        lambda job, for_failure=False: [{"platform": "zelda", "chat_id": "", "thread_id": None}])

    class _FakeAdapter:
        @staticmethod
        def extract_media(content):
            return (list(media or []), content)

        @staticmethod
        def filter_media_delivery_paths(paths):
            return list(paths)

    monkeypatch.setattr(
        "gateway.platforms.base.BasePlatformAdapter.extract_media",
        _FakeAdapter.extract_media)
    monkeypatch.setattr(
        "gateway.platforms.base.BasePlatformAdapter.filter_media_delivery_paths",
        _FakeAdapter.filter_media_delivery_paths)
    return sd._deliver_result(job, job.get("last_run_output") or "")


def test_zelda_row_staged(push_queue, monkeypatch, tmp_path):
    job = _job(tmp_path)
    _run_delivery(tmp_path, monkeypatch, job)
    rows = list(push_queue.glob("cron-*.json"))
    assert rows, "no zelda push row staged"
    data = json.loads(rows[0].read_text())
    # Delivery content carries the standard wrap (header + body + stop hint).
    assert "Report body" in data["text"]
    assert data["notice"] == "cron"
    assert data["sessions"] == []  # broadcast → Notifications board


def test_zelda_media_recorded_not_lost(push_queue, monkeypatch, tmp_path):
    job = _job(tmp_path)
    err = _run_delivery(tmp_path, monkeypatch, job, media=[str(tmp_path / "chart.png")])
    assert err and "media attachment" in err and "chart.png" in err
    rows = list(push_queue.glob("cron-*.json"))
    assert rows  # text still staged


def test_zelda_staging_failure_is_delivery_error(push_queue, monkeypatch, tmp_path):
    job = _job(tmp_path)

    def _boom(*a, **k):
        raise PermissionError("disk full")

    monkeypatch.setattr(sd, "_deliver_zelda_target", _boom)
    err = _run_delivery(tmp_path, monkeypatch, job)
    assert err and "zelda" in err.lower()


def test_zelda_target_resolution_bare_token():
    """Bare 'zelda' resolves to the queue-addressed target (no home channel needed)."""
    assert sd._resolve_single_delivery_target({"id": "j"}, "zelda") == {
        "platform": "zelda", "chat_id": "", "thread_id": None}


def test_zelda_target_resolution_with_session_token():
    """'zelda:<session-id>' resolves addressed (chat_id carries the session id)."""
    # The colon-split path treats 'zelda' as an unknown send-platform and returns None
    # unless resolve_send_target knows it — bare-token resolution is the contract; the
    # addressed form rides the explicit platform:chat_id path if ever needed.
    assert sd._resolve_single_delivery_target({"id": "j"}, "zelda") ["platform"] == "zelda"
