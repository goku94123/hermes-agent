"""Zelda fork /steer with media caption (hf49): the gate must see '/steer' riding a
PARTS-ARRAY user_message (image + caption), not only plain-string bodies.

Pre-hf49 the isinstance(str) guard made image+/steer fall through as a normal turn —
the steer never injected (found live 2026-10-02). Semantics now:
- idle session → normal media turn with the '/steer' token stripped (image whole);
- running session → steer carries caption + staged image paths (agent.steer() text-only).
"""
import base64
import json

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.platforms.test_api_server_zelda_clarify import (
    _make_app, _seed_sinda_session, API_KEY, SINDA_ID, store, zelda_env, _AUTH,  # noqa: F401
)

pytestmark = pytest.mark.allow_real_home_io

_PNG_B64 = base64.b64encode(
    b"\x89PNG\r\n\x1a\n" + b"0" * 64).decode()


def _media_body(caption: str) -> dict:
    """OpenAI-shape media turn: content = [text part, image_url data URI]."""
    content = [{"type": "text", "text": caption},
               {"type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_PNG_B64}"}}]
    return {"model": "test-model",
            "messages": [{"role": "user", "content": content}], "stream": False}


def _seed_run_status(adapter, session_id: str, run_id: str = "run-1", status: str = "running"):
    adapter._run_statuses[run_id] = {"session_id": session_id, "status": status,
                                     "updated_at": 1}


@pytest.mark.asyncio
async def test_media_caption_steer_gate_sees_token(zelda_env, monkeypatch):
    """A parts-array '/steer ...' caption reaches the steer handler (not the LLM)."""
    store, adapter, runner = zelda_env
    _seed_sinda_session(store)
    _seed_run_status(adapter, SINDA_ID)
    steered = {}

    async def _fake_steer(request, *, session_id, gateway_session_key, chat_steer_text, stream):
        steered["text"] = chat_steer_text
        steered["session"] = session_id
        from aiohttp import web as _web
        return _web.json_response({"object": "hermes.run.steer", "accepted": True})

    monkeypatch.setattr(adapter, "_handle_zelda_steer", _fake_steer)
    async with TestClient(TestServer(_make_app(adapter, runner))) as client:
        resp = await client.post("/v1/chat/completions", json=_media_body("/steer look at this"),
                                 headers=_AUTH)
        assert resp.status == 200
    assert steered.get("session") == SINDA_ID
    assert "look at this" in (steered.get("text") or "")
    assert "staged" in (steered.get("text") or "")  # image path carried


@pytest.mark.asyncio
async def test_media_caption_steer_idle_becomes_normal_turn(zelda_env, monkeypatch, tmp_path):
    """Idle session + image+/steer → the turn runs as a NORMAL media turn with the
    token stripped (no 409, no lost image)."""
    store, adapter, runner = zelda_env
    _seed_sinda_session(store)
    seen = {}

    async def _fake_run_agent(**kwargs):
        seen["user_message"] = kwargs.get("user_message")
        return ({"final_response": "ok"}, {"input_tokens": 1, "output_tokens": 1})

    monkeypatch.setattr(adapter, "_run_agent", _fake_run_agent)
    # _resolve_live_session_id must not blow up on the fake store: idle → returns the id.
    async def _resolve(self, sid):
        return sid
    monkeypatch.setattr(
        "gateway.platforms.api_server_runs._resolve_live_session_id", _resolve)
    async with TestClient(TestServer(_make_app(adapter, runner))) as client:
        resp = await client.post("/v1/chat/completions", json=_media_body("/steer describe this"),
                                 headers=_AUTH)
        assert resp.status == 200
    msg = seen.get("user_message")
    # The image part survives; the text part no longer carries the command token.
    assert isinstance(msg, list)
    assert any(isinstance(p, dict) and p.get("type") == "image_url" for p in msg)
    text_part = next((p for p in msg if isinstance(p, dict) and p.get("type") == "text"), {})
    assert "/steer" not in (text_part.get("text") or "")
    assert "describe this" in (text_part.get("text") or "")
