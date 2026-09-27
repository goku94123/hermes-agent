"""Zelda fork: interactive components — the approval surface (P2.3).

A Sinda chat turn (``X-Zelda-Client: 1``, streaming) whose agent hits a dangerous
command must reach the phone: the gateway ALREADY streams ``event: approval.request``
for every client (``_register_stream_approval`` is unconditional) and resolves
decisions on ``POST /v1/runs/{id}/approval``. This file covers the fork deltas:

1. **Session-keyed registry.** ``_run_agent`` registers ``register_gateway_notify``
   under the RUN id (``approval_session_key=completion_id``) by default, so a typed
   ``/approve``-style chat turn can never find the waiter. When the caller is the
   fork, the notify is ALSO registered under the caller's session id — that is the
   key the fork's typed answers can address.

2. **Typed-answer intercept.** While a gateway approval blocks the session, the
   fork's next plain chat turn is consumed as the answer (mirrors the clarify
   intercept): approval words (``approve``/``yes``/``deny``/... — the same map the
   messaging pipeline uses) resolve via ``resolve_gateway_approval`` and get a
   ZERO-content ack; ``/deny <reason>`` carries the reason. Non-approval prose and
   slash commands fall through untouched (no steering into the blocked turn).

Contracts (behavior, never snapshots):
- Header-gated: only ``X-Zelda-Client: 1`` turns register the session-key alias.
- A fork chat turn while blocked resolves the oldest pending approval and returns
  a zero-content ack WITHOUT spawning an agent.
- ``deny`` variants reject; ``/deny <reason>`` relays the reason to the agent.
- Approval words resolve; ordinary prose falls through to a normal turn.
- Session-boundary clear cancels pending approvals so a blocked worker wakes.

Module-level ``allow_real_home_io``: importing the runner pulls hermes_bootstrap →
pm's dependency activation, which stats ``<repo-parent>/manifest.json`` — a REAL-home
touch on a default install. Same false positive as the clarify suite (tmp stores only).
"""

from __future__ import annotations

import json
import re
import threading

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import GatewayConfig, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware
from gateway.session import SessionStore

pytestmark = pytest.mark.allow_real_home_io

API_KEY = "«redacted:sk-…»"
SINDA_ID = "20260927_130000_appr01"
ROOT_PRED = "zelda:20260927_120000_apprroot"


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
    return {"model": "test-model", "messages": [{"role": "user", "content": text}], "stream": stream}


def _seed_approval(session_key: str, command: str = "sudo reboot", request_id: str = "req01") -> threading.Event:
    """Register a blocking gateway approval exactly the way the wired callback does."""
    from tools import approval as appr

    entry_event = threading.Event()

    class _Entry:
        pass

    entry = _Entry()
    entry.event = entry_event
    entry.data = {"command": command, "description": "restart the machine",
                  "pattern_key": "sudo", "pattern_keys": ["sudo"], "request_id": request_id,
                  "allow_session": True, "allow_permanent": True}
    entry.acknowledged = True
    entry.result = None
    entry.reason = None
    entry.cancelled = None
    entry.settle = None
    with appr._lock:
        appr._gateway_queues.setdefault(session_key, []).append(entry)
    return entry_event


@pytest.fixture(autouse=True)
def _clean_approval_registry():
    from tools import approval as appr

    with appr._lock:
        appr._gateway_queues.clear()
    yield
    with appr._lock:
        appr._gateway_queues.clear()


class TestSessionKeyRegistration:
    @pytest.mark.asyncio
    async def test_zelda_turn_passes_alias_key(self, zelda_env, monkeypatch):
        """A Sinda streaming turn passes ``zelda_approval_alias_key`` (= the session id)
        into _run_agent; that is what makes _run_agent register the approval notify
        under the SESSION id (the key its typed answers can address) beside the run id."""
        store, adapter, runner = zelda_env
        sid = _seed_sinda_session(store)
        seen = {}

        async def _fake_run_agent(**kwargs):
            seen.update(kwargs)
            return ({"final_response": "ok"}, {"input_tokens": 1, "output_tokens": 1})

        monkeypatch.setattr(adapter, "_run_agent", _fake_run_agent)
        async with TestClient(TestServer(_make_app(adapter, runner))) as client:
            resp = await client.post("/v1/chat/completions", json=_chat_body("hi", stream=True),
                                     headers=_AUTH)
            assert resp.status == 200
            await resp.read()
        assert seen.get("zelda_approval_alias_key") == sid

    @pytest.mark.asyncio
    async def test_non_zelda_client_gets_no_alias_key(self, zelda_env, monkeypatch):
        store, adapter, runner = zelda_env
        _seed_sinda_session(store)
        seen = {}

        async def _fake_run_agent(**kwargs):
            seen.update(kwargs)
            return ({"final_response": "ok"}, {"input_tokens": 1, "output_tokens": 1})

        monkeypatch.setattr(adapter, "_run_agent", _fake_run_agent)
        headers = {"Authorization": f"Bearer {API_KEY}", "X-Hermes-Session-Id": SINDA_ID}
        async with TestClient(TestServer(_make_app(adapter, runner))) as client:
            resp = await client.post("/v1/chat/completions", json=_chat_body("hi", stream=True),
                                     headers=headers)
            assert resp.status == 200
            await resp.read()
        assert "zelda_approval_alias_key" not in seen

    def test_unregister_cleans_session_alias(self):
        """_unregister_approval_notify pops BOTH keys — the run id and the fork's
        session alias — so no stale notify callback outlives the turn."""
        from tools import approval as appr
        from gateway.platforms.api_server_runs import _unregister_approval_notify

        calls = []

        def _fake_unregister(key):
            calls.append(key)

        orig = appr.unregister_gateway_notify
        appr.unregister_gateway_notify = _fake_unregister
        try:
            _unregister_approval_notify("run-1", extra_keys=["sess-1"])
        finally:
            appr.unregister_gateway_notify = orig
        assert sorted(calls) == ["run-1", "sess-1"]


class TestApprovalWordIntercept:
    @pytest.mark.asyncio
    async def test_approval_word_resolves_without_agent(self, zelda_env, monkeypatch):
        """``approve`` while a dangerous-command approval blocks resolves the oldest
        pending approval and returns a zero-content ack — no agent spawn."""
        from tools import approval as appr

        store, adapter, runner = zelda_env
        sid = _seed_sinda_session(store)
        ev = _seed_approval(sid, request_id="req-word")
        spawned = []

        async def _fail_run_agent(**kwargs):
            spawned.append(kwargs)
            return ({"final_response": "SHOULD-NOT-RUN"}, {})

        monkeypatch.setattr(adapter, "_run_agent", _fail_run_agent)
        async with TestClient(TestServer(_make_app(adapter, runner))) as client:
            resp = await client.post("/v1/chat/completions", json=_chat_body("approve"),
                                     headers=_AUTH)
            assert resp.status == 200
            data = await resp.json()
        assert spawned == []
        assert data["choices"][0]["message"]["content"] == ""
        assert data["hermes_command"]["status"] == "ok"
        assert data["hermes_command"]["command"] == "clarify"  # shared zero-content ack
        assert ev.is_set()

    @pytest.mark.asyncio
    async def test_deny_word_rejects(self, zelda_env, monkeypatch):
        from tools import approval as appr

        store, adapter, runner = zelda_env
        sid = _seed_sinda_session(store)
        ev = _seed_approval(sid, request_id="req-deny")

        async def _fail_run_agent(**kwargs):
            return ({"final_response": "SHOULD-NOT-RUN"}, {})

        monkeypatch.setattr(adapter, "_run_agent", _fail_run_agent)
        async with TestClient(TestServer(_make_app(adapter, runner))) as client:
            resp = await client.post("/v1/chat/completions", json=_chat_body("deny"),
                                     headers=_AUTH)
            assert resp.status == 200
            data = await resp.json()
        assert data["hermes_command"]["status"] == "ok"
        assert ev.is_set()

    @pytest.mark.asyncio
    async def test_deny_slash_carries_reason(self, zelda_env, monkeypatch):
        """/deny <reason> dispatches the runner's REAL /deny handler through the gate's
        busy lane — same path Telegram takes mid-turn. The handler resolves the wait
        and relays the reason to the agent."""
        from tools import approval as appr

        store, adapter, runner = zelda_env
        sid = _seed_sinda_session(store)
        ev = _seed_approval(sid, request_id="req-reason")
        seen = {}

        async def _stub_deny(event):
            seen["args"] = event.get_command_args()
            return "DENIED"

        monkeypatch.setattr(runner, "_handle_deny_command", _stub_deny)

        async def _fail_run_agent(**kwargs):
            return ({"final_response": "SHOULD-NOT-RUN"}, {})

        monkeypatch.setattr(adapter, "_run_agent", _fail_run_agent)
        async with TestClient(TestServer(_make_app(adapter, runner))) as client:
            resp = await client.post("/v1/chat/completions",
                                     json=_chat_body("/deny too risky today"),
                                     headers=_AUTH)
            assert resp.status == 200
            data = await resp.json()
        assert data["hermes_command"]["status"] == "ok"
        # The gate dispatched the runner's /deny handler with the args intact.
        assert seen.get("args") == "too risky today"
        assert data["hermes_command"]["command"] == "/deny too risky today"

    @pytest.mark.asyncio
    async def test_ordinary_prose_falls_through(self, zelda_env, monkeypatch):
        """While blocked, ordinary prose is NOT consumed — it reaches the normal turn
        (the blocked agent already owns the socket; the follow-up queues naturally)."""
        from tools import approval as appr

        store, adapter, runner = zelda_env
        sid = _seed_sinda_session(store)
        _seed_approval(sid, request_id="req-prose")
        spawned = []

        async def _fake_run_agent(**kwargs):
            spawned.append(kwargs)
            return ({"final_response": "LLM-REACHED"}, {})

        monkeypatch.setattr(adapter, "_run_agent", _fake_run_agent)
        async with TestClient(TestServer(_make_app(adapter, runner))) as client:
            resp = await client.post("/v1/chat/completions",
                                     json=_chat_body("what is the capital of France"),
                                     headers=_AUTH)
            assert resp.status == 200
            data = await resp.json()
        assert data["choices"][0]["message"]["content"] == "LLM-REACHED"
        assert len(spawned) == 1
        # The approval stays pending.
        assert appr.has_blocking_approval(sid)

    @pytest.mark.asyncio
    async def test_no_approval_pending_words_are_normal_turn(self, zelda_env, monkeypatch):
        """The word map is gated on a REAL pending approval — ``approve`` in normal
        conversation reaches the LLM like any other word."""
        store, adapter, runner = zelda_env
        sid = _seed_sinda_session(store)
        spawned = []

        async def _fake_run_agent(**kwargs):
            spawned.append(kwargs)
            return ({"final_response": "LLM-REACHED"}, {})

        monkeypatch.setattr(adapter, "_run_agent", _fake_run_agent)
        async with TestClient(TestServer(_make_app(adapter, runner))) as client:
            resp = await client.post("/v1/chat/completions", json=_chat_body("approve"),
                                     headers=_AUTH)
            assert resp.status == 200
            data = await resp.json()
        assert data["choices"][0]["message"]["content"] == "LLM-REACHED"
        assert len(spawned) == 1


class TestBoundaryClear:
    def test_boundary_clear_cancels_pending_approval(self, zelda_env):
        """_clear_session_boundary_security_state wakes a blocked approval waiter —
        a /new must unwind the blocked run instead of leaving it wedged."""
        store, adapter, runner = zelda_env
        sid = _seed_sinda_session(store)
        ev = _seed_approval(sid, request_id="req-clear")
        assert ev.is_set() is False

        from gateway.run_agent_cache import GatewayAgentCacheMixin

        cache = object.__new__(GatewayAgentCacheMixin)
        cache._peek_session_state = lambda key: None  # type: ignore[method-assign]
        cache._clear_session_boundary_security_state(sid)
        from tools import approval as appr
        assert appr.has_blocking_approval(sid) is False
        assert ev.is_set() is True
