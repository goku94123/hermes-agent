"""Zelda fork: the generic slash gate in the fork's chat path (Task 2).

A ``X-Zelda-Client: 1`` completion whose leading ``/token`` resolves in the live
COMMAND_REGISTRY must dispatch to the real gateway handler BEFORE reaching the LLM, on the
caller's REAL id-addressed session — exactly the ``slash.exec`` shape (parse at the door,
route through the shared engine). Unknown ``/tokens`` are plain text (LLM turn). Existing
fork shims (``/new`` ``/reset`` ``/steer``) keep their faster dedicated paths; ``/branch``
gets its own Task 3 handler.

Contracts:
- ``/title <name>`` from the fork path renames the CALLER's session (no phantom row).
- A structured ``hermes.command.result`` metadata block rides the OpenAI-compatible body
  (status ok/unhandled; errors teach the alternative).
- Busy mid-turn: structured ``rejected`` (never a hang).
- Unknown ``/notacommand`` falls through to the LLM path (mocked here — assert the gate
  did NOT intercept, i.e. the LLM runner was reached with the raw text).
- A ``cli_only`` command (``/handoff``) is NOT intercepted by the gate (registry filter).
"""

from __future__ import annotations

import json
import re

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import GatewayConfig, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware
from gateway.session import SessionStore

API_KEY = "sk-zelda-test"
SINDA_ID = "20260925_120000_bbbb02"
ROOT_PRED = "zelda:20260925_110000_root0"


@pytest.fixture()
def store(tmp_path, monkeypatch):
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    return SessionStore(sessions_dir=tmp_path, config=GatewayConfig())


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra = {}
    if api_key:
        extra["key"] = api_key
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _seed_sinda_session(store: SessionStore, session_id: str = SINDA_ID, title: str = "orig title"):
    db = store._db
    db.create_session(
        session_id, "api_server",
        user_id=ROOT_PRED, chat_id=ROOT_PRED, chat_type="dm", thread_id=None,
        display_name="Sinda",
    )
    db.append_message(session_id, role="user", content="hello from sinda")
    db.append_message(session_id, role="assistant", content="hi")
    db.set_session_title(session_id, title)
    return session_id


def _fake_runner(store):
    from gateway.run import GatewayRunner
    from gateway.session import AsyncSessionStore
    from hermes_state import AsyncSessionDB

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._profile_adapters = {}
    runner.config = {}
    runner._background_tasks = set()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_ack_ts = {}
    runner._pending_approvals = {}
    runner._update_prompt_pending = {}
    runner._agent_cache_lock = None
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    runner._session_db = AsyncSessionDB(store._db)
    runner._pending_skills_reload_notes = {}
    runner._sessions = {}
    return runner


def _make_app(adapter: APIServerAdapter, runner) -> web.Application:
    app = web.Application(middlewares=[cors_middleware])
    app["api_server_adapter"] = adapter
    app["gateway_runner"] = runner
    # The generic gate lives inside the completions handler; register it so the test
    # drives the REAL path (the gate fires before the LLM runner inside the handler).
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    return app


@pytest.fixture()
def zelda_env(store, monkeypatch):
    adapter = _make_adapter(api_key=API_KEY)
    runner = _fake_runner(store)

    async def _fake_db_async():
        return store._db

    monkeypatch.setattr(adapter, "_ensure_session_db_async", _fake_db_async)
    adapter.gateway_runner = runner
    return store, adapter, runner


_AUTH = {"Authorization": f"Bearer {API_KEY}", "X-Zelda-Client": "1",
         "X-Hermes-Session-Id": SINDA_ID}


def _chat_body(text: str) -> dict:
    return {"model": "test-model", "messages": [{"role": "user", "content": text}], "stream": False}


class TestGenericGate:
    @pytest.mark.asyncio
    async def test_slash_title_dispatches_on_callers_session(self, zelda_env):
        """/title via the completions path renames the CALLER's row and returns the
        command result as the assistant bubble with structured metadata."""
        store, adapter, runner = zelda_env
        sid = _seed_sinda_session(store)
        async with TestClient(TestServer(_make_app(adapter, runner))) as client:
            resp = await client.post("/v1/chat/completions", json=_chat_body("/title gate-test"),
                                     headers=_AUTH)
            assert resp.status == 200
            data = await resp.json()
        # The bubble carries the command output, OpenAI-compatible.
        content = data["choices"][0]["message"]["content"]
        assert isinstance(content, str) and content
        meta = data["hermes_command"]
        assert meta["object"] == "hermes.command.result"
        assert meta["handled"] is True
        assert meta["status"] == "ok"
        # The delta: the CALLER's row was renamed; no phantom row appeared.
        assert store._db.get_session_title(sid) == "gate-test"

    @pytest.mark.asyncio
    async def test_unknown_slash_falls_through_to_llm(self, zelda_env, monkeypatch):
        """An unknown /token is plain text: the gate must NOT intercept."""
        store, adapter, runner = zelda_env
        _seed_sinda_session(store)

        async def _fake_run_agent(**kwargs):
            return ({"final_response": "LLM-REACHED"}, {"input_tokens": 1, "output_tokens": 1})

        monkeypatch.setattr(adapter, "_run_agent", _fake_run_agent)
        async with TestClient(TestServer(_make_app(adapter, runner))) as client:
            resp = await client.post("/v1/chat/completions", json=_chat_body("/definitely-not-a-command"),
                                     headers=_AUTH)
            assert resp.status == 200
            data = await resp.json()
        assert data["choices"][0]["message"]["content"] == "LLM-REACHED"
        assert "hermes_command" not in data

    @pytest.mark.asyncio
    async def test_cli_only_command_not_intercepted(self, zelda_env, monkeypatch):
        """cli_only registry entries (/handoff) never ride the gateway gate."""
        store, adapter, runner = zelda_env
        _seed_sinda_session(store)

        async def _fake_run_agent(**kwargs):
            return ({"final_response": "LLM-REACHED"}, {"input_tokens": 1, "output_tokens": 1})

        monkeypatch.setattr(adapter, "_run_agent", _fake_run_agent)
        async with TestClient(TestServer(_make_app(adapter, runner))) as client:
            resp = await client.post("/v1/chat/completions", json=_chat_body("/handoff telegram"),
                                     headers=_AUTH)
            assert resp.status == 200
            data = await resp.json()
        assert data["choices"][0]["message"]["content"] == "LLM-REACHED"
        assert "hermes_command" not in data

    @pytest.mark.asyncio
    async def test_gate_silent_without_zelda_header(self, zelda_env, monkeypatch):
        """Other API clients (no X-Zelda-Client) keep the plain pass-through today."""
        store, adapter, runner = zelda_env
        _seed_sinda_session(store)

        async def _fake_run_agent(**kwargs):
            return ({"final_response": "LLM-REACHED"}, {"input_tokens": 1, "output_tokens": 1})

        monkeypatch.setattr(adapter, "_run_agent", _fake_run_agent)
        headers = {"Authorization": f"Bearer {API_KEY}", "X-Hermes-Session-Id": SINDA_ID}
        async with TestClient(TestServer(_make_app(adapter, runner))) as client:
            resp = await client.post("/v1/chat/completions", json=_chat_body("/title nope"),
                                     headers=headers)
            assert resp.status == 200
            data = await resp.json()
        assert data["choices"][0]["message"]["content"] == "LLM-REACHED"
        assert "hermes_command" not in data
        # ...and the row was NOT renamed (the text went to the LLM, not the handler).
        assert store._db.get_session_title(SINDA_ID) == "orig title"


class TestGateMembershipParity:
    """The gate's membership rule must mirror the native gateway early gate
    (``GATEWAY_KNOWN_COMMANDS``): non-cli_only commands PLUS cli_only commands that carry a
    ``gateway_config_gate`` (their handler enforces the config gate at runtime). A blanket
    ``cli_only`` exclusion silently dropped commands Telegram answers."""

    @pytest.mark.asyncio
    async def test_plain_table_command_not_unhandled(self, zelda_env, monkeypatch):
        """/status (plain table, no idle entry) dispatches instead of 'not available'."""
        store, adapter, runner = zelda_env
        sid = _seed_sinda_session(store)

        async def _stub_status(event):
            return "STATUS-STUB-OK"

        monkeypatch.setattr(runner, "_handle_status_command", _stub_status)
        async with TestClient(TestServer(_make_app(adapter, runner))) as client:
            resp = await client.post("/v1/chat/completions", json=_chat_body("/status"),
                                     headers=_AUTH)
            assert resp.status == 200
            data = await resp.json()
        assert data["choices"][0]["message"]["content"] == "STATUS-STUB-OK"
        meta = data["hermes_command"]
        assert meta["handled"] is True
        assert meta["status"] == "ok"
        assert meta["sessionId"] == sid

    @pytest.mark.asyncio
    async def test_config_gated_cli_command_intercepted(self, zelda_env, monkeypatch):
        """/skills is cli_only BUT carries gateway_config_gate — the native gateway admits
        it, so the fork gate must too (the handler applies the gate at runtime)."""
        store, adapter, runner = zelda_env
        sid = _seed_sinda_session(store)

        seen = []

        async def _stub_skills(event):
            seen.append(event.get_command())
            return "SKILLS-STUB-OK"

        monkeypatch.setattr(runner, "_handle_skills_command", _stub_skills)
        async with TestClient(TestServer(_make_app(adapter, runner))) as client:
            resp = await client.post("/v1/chat/completions", json=_chat_body("/skills"),
                                     headers=_AUTH)
            assert resp.status == 200
            data = await resp.json()
        assert data["choices"][0]["message"]["content"] == "SKILLS-STUB-OK"
        assert data["hermes_command"]["handled"] is True
        assert seen == ["skills"]

    @pytest.mark.asyncio
    async def test_bare_cli_only_command_still_falls_through(self, zelda_env, monkeypatch):
        """/handoff is cli_only WITHOUT a config gate — no native gateway dispatch, so the
        fork gate must keep treating it as plain text."""
        store, adapter, runner = zelda_env
        _seed_sinda_session(store)

        async def _fake_run_agent(**kwargs):
            return ({"final_response": "LLM-REACHED"}, {"input_tokens": 1, "output_tokens": 1})

        monkeypatch.setattr(adapter, "_run_agent", _fake_run_agent)
        async with TestClient(TestServer(_make_app(adapter, runner))) as client:
            resp = await client.post("/v1/chat/completions", json=_chat_body("/handoff telegram"),
                                     headers=_AUTH)
            assert resp.status == 200
            data = await resp.json()
        assert data["choices"][0]["message"]["content"] == "LLM-REACHED"
        assert "hermes_command" not in data

    @pytest.mark.asyncio
    async def test_streaming_command_frames_carry_metadata(self, zelda_env, monkeypatch):
        """Streaming gate command (stream=true): content rides the second frame, the
        structured ``hermes.command.result`` rides the FINAL frame, [DONE] terminates.
        Regression: the streaming branch once referenced _sse_frame without importing it,
        500-ing with an empty body before a single frame was written."""
        store, adapter, runner = zelda_env
        sid = _seed_sinda_session(store)

        async def _stub_title(event):
            return "TITLE-STUB-OK"

        monkeypatch.setattr(runner, "_handle_title_command", _stub_title)
        body = _chat_body("/title stream-check")
        body["stream"] = True
        async with TestClient(TestServer(_make_app(adapter, runner))) as client:
            resp = await client.post("/v1/chat/completions", json=body, headers=_AUTH)
            assert resp.status == 200
            raw = await resp.read()
        frames = [f for f in raw.decode().split("\n\n") if f.strip()]
        payloads = []
        for frame in frames:
            m = re.match(r"^data: (.*)$", frame.strip())
            assert m, f"non-data SSE frame: {frame!r}"
            payload = m.group(1)
            if payload == "[DONE]":
                continue
            payloads.append(json.loads(payload))
        # role frame, content frame, final frame with metadata
        assert payloads[0]["choices"][0]["delta"] == {"role": "assistant"}
        assert payloads[1]["choices"][0]["delta"]["content"] == "TITLE-STUB-OK"
        assert "hermes_command" in payloads[-1]
        assert payloads[-1]["hermes_command"]["status"] == "ok"
        assert payloads[-1]["hermes_command"]["sessionId"] == sid
        assert frames[-1].strip() == "data: [DONE]"
