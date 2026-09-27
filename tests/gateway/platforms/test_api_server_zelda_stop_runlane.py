"""Zelda fork: /stop must reach the run lane Sinda turns actually execute on.

The Sep-2026 generic gate dispatched /stop into the runner's session-KEY lane
(``GatewayRunner._running_agents``) while Sinda's live turns run on the adapter's own
run lane (``_active_run_agents``, keyed by run id) — an id-addressed fork session never
holds a runner slot, so /stop answered "No active task to stop." while the turn kept
running (live 2026-09-27 ~10:02, session 0393cacf…, API call #59 mid-flight).

Contracts:
- /stop on a session with a live run-lane turn hard-interrupts that run's agent and
  returns the real "⚡ Stopped." text (never the idle answer).
- Streaming and non-streaming both work (the 2026-09-26 _sse_frame lesson: a passing
  non-streaming battery proves nothing about the SSE path).
- After the turn ends (terminal status row), /stop falls through to the standard gate —
  the runner-lane handler answers, so idle gets the true "No active task to stop."
- Streaming chat turns register their agent under the completion id (active_run_id), so
  /steer, POST /v1/runs/{id}/stop and the fork's /stop all find the agent behind the
  "running" status row, and the row is retired when the turn finishes (no phantom).
"""

from __future__ import annotations

import json
import time

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

# Module-level: this box's layout (repo inside ~/.hermes + a pre-update backup
# manifest.json at the home root) makes the import-time PM bootstrap probe the real
# home; done at collection (before the home-I/O guard fixture installs) it is the same
# unguarded path every module-level gateway import in this suite takes. A mid-test
# first import would trip the guard — which is also why a bare-pytest run of the
# sibling zelda files errors on this box (their canonical runner needs `activate`).
from gateway.config import GatewayConfig, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware
from gateway.run import GatewayRunner
from gateway.session import AsyncSessionStore, SessionStore
from hermes_state import AsyncSessionDB

API_KEY = "test-key-123"
SINDA_ID = "20260925_120000_cccc03"
ROOT_PRED = "zelda:20260925_110000_root0"
RUN_ID = "chatcmpl-test000000000000000000001"


# ---------------------------------------------------------------------------
# Harness (same shape as test_api_server_zelda_generic_gate.py)
# ---------------------------------------------------------------------------


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


def _seed_sinda_session(store: SessionStore, session_id: str = SINDA_ID):
    db = store._db
    db.create_session(
        session_id, "api_server",
        user_id=ROOT_PRED, chat_id=ROOT_PRED, chat_type="dm", thread_id=None,
        display_name="Sinda",
    )
    db.append_message(session_id, role="user", content="hello from sinda")
    db.append_message(session_id, role="assistant", content="hi")
    return session_id


def _fake_runner(store):
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


def _chat_body(text: str, stream: bool = False) -> dict:
    return {"model": "test-model", "messages": [{"role": "user", "content": text}],
            "stream": stream}


class _FakeAgent:
    """Records hard-interrupts; exposes the ABI request_hard_interrupt prefers."""

    def __init__(self):
        self.calls = []

    def hard_interrupt(self, message=None, *, tool_reason=None):
        self.calls.append((message, tool_reason))


def _seed_live_run(adapter: APIServerAdapter, status: str = "running",
                   with_agent: bool = True) -> _FakeAgent:
    adapter._run_statuses[RUN_ID] = {
        "object": "hermes.run", "run_id": RUN_ID, "status": status,
        "session_id": SINDA_ID, "updated_at": time.time()}
    agent = _FakeAgent()
    if with_agent:
        adapter._active_run_agents[RUN_ID] = agent
    return agent


def _sse_events(raw: bytes) -> list[dict]:
    events = []
    for line in raw.decode("utf-8", "replace").splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            events.append(json.loads(line[len("data: "):]))
    return events


# ---------------------------------------------------------------------------
# /stop parity — run lane
# ---------------------------------------------------------------------------


class TestZeldaStopRunLane:
    @pytest.mark.asyncio
    async def test_stop_interrupts_live_runlane_agent(self, zelda_env):
        store, adapter, runner = zelda_env
        _seed_sinda_session(store)
        agent = _seed_live_run(adapter)
        async with TestClient(TestServer(_make_app(adapter, runner))) as client:
            resp = await client.post("/v1/chat/completions", json=_chat_body("/stop"),
                                     headers=_AUTH)
            assert resp.status == 200
            data = await resp.json()
        content = data["choices"][0]["message"]["content"]
        assert "Stopped" in content
        assert "No active task" not in content
        meta = data["hermes_command"]
        assert meta["handled"] is True
        assert meta["status"] == "ok"
        assert meta["runId"] == RUN_ID
        # The delta: the live run's agent was hard-interrupted and the row marked stopping.
        assert agent.calls == [("Stop requested via /stop", "stop command")]
        assert adapter._run_statuses[RUN_ID]["status"] == "stopping"
        assert RUN_ID in adapter._stopping_run_ids

    @pytest.mark.asyncio
    async def test_stop_interrupts_live_runlane_agent_stream_mode(self, zelda_env):
        """The SSE twin of the non-streaming stop (2026-09-26 _sse_frame lesson)."""
        store, adapter, runner = zelda_env
        _seed_sinda_session(store)
        agent = _seed_live_run(adapter)
        async with TestClient(TestServer(_make_app(adapter, runner))) as client:
            resp = await client.post("/v1/chat/completions", json=_chat_body("/stop", stream=True),
                                     headers=_AUTH)
            assert resp.status == 200
            raw = await resp.read()
        events = _sse_events(raw)
        final = [e for e in events if e["choices"][0].get("finish_reason")]
        assert final, f"no terminal SSE frame in {raw[:400]!r}"
        assert final[-1]["hermes_command"]["handled"] is True
        assert final[-1]["hermes_command"]["runId"] == RUN_ID
        contents = "".join(e["choices"][0]["delta"].get("content", "") for e in events)
        assert "Stopped" in contents and "No active task" not in contents
        assert agent.calls == [("Stop requested via /stop", "stop command")]

    @pytest.mark.asyncio
    async def test_stop_after_turn_falls_through_to_gate(self, zelda_env, monkeypatch):
        """A terminal run row must not be interrupted: the standard gate answers instead."""
        store, adapter, runner = zelda_env
        sid = _seed_sinda_session(store)
        _seed_live_run(adapter, status="completed", with_agent=False)

        seen = []

        async def _stub_stop(event):
            seen.append(event.get_command())
            return "IDLE-STOP-STUB"

        monkeypatch.setattr(runner, "_handle_stop_command", _stub_stop)
        async with TestClient(TestServer(_make_app(adapter, runner))) as client:
            resp = await client.post("/v1/chat/completions", json=_chat_body("/stop"),
                                     headers=_AUTH)
            assert resp.status == 200
            data = await resp.json()
        assert data["choices"][0]["message"]["content"] == "IDLE-STOP-STUB"
        assert data["hermes_command"]["handled"] is True
        assert seen == ["stop"]
        assert "runId" not in data["hermes_command"]
        # The terminal row was left alone.
        assert adapter._run_statuses[RUN_ID]["status"] == "completed"


# ---------------------------------------------------------------------------
# Streaming runs must register their agent under the completion id
# ---------------------------------------------------------------------------


class TestStreamingRunRegistration:
    @pytest.mark.asyncio
    async def test_streaming_turn_registers_active_run_id_and_retires_row(self, zelda_env,
                                                                          monkeypatch):
        store, adapter, runner = zelda_env
        _seed_sinda_session(store)
        seen_kwargs = {}
        seen_status_rows = []

        async def _fake_run_agent(**kwargs):
            seen_kwargs.update(kwargs)
            rid = kwargs.get("active_run_id")
            # The status row the control paths match on must exist while the turn runs.
            seen_status_rows.append(
                (rid, adapter._run_statuses.get(rid, {}).get("session_id")))
            return ({"final_response": "LLM-OK"}, {"input_tokens": 1, "output_tokens": 1})

        monkeypatch.setattr(adapter, "_run_agent", _fake_run_agent)
        async with TestClient(TestServer(_make_app(adapter, runner))) as client:
            resp = await client.post(
                "/v1/chat/completions", json=_chat_body("plain text turn", stream=True),
                headers=_AUTH)
            assert resp.status == 200
            raw = await resp.read()
        events = _sse_events(raw)
        assert events, f"no SSE events: {raw[:400]!r}"
        # The fake LLM bypasses stream callbacks, so content deltas are empty here — what
        # matters is the control-plane contract below.
        # The agent registered under the completion id whose status row names THIS session…
        assert seen_kwargs.get("active_run_id"), "streaming turn passed no active_run_id"
        rid, row_session = seen_status_rows[-1]
        assert rid == seen_kwargs["active_run_id"]
        assert row_session == SINDA_ID
        # …and the row is retired on completion, so a later /stop falls through cleanly.
        assert rid not in adapter._run_statuses
        assert rid not in adapter._active_run_agents
