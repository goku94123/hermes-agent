"""Zelda fork: interactive components — the clarify surface (P2.2).

A Sinda chat turn (``X-Zelda-Client: 1``, streaming) whose agent calls the ``clarify``
tool must reach the phone: the question streams as an SSE ``hermes.clarify`` event and
the user's next normal chat turn resolves the pending prompt instead of spawning a
second agent on top of the blocked one.

Contracts (behavior, never snapshots):
- Wiring is header-gated: only ``X-Zelda-Client: 1`` turns get ``agent.clarify_callback``.
- The callback registers in ``tools.clarify_gateway`` under the caller's session id,
  notifies (the SSE frame payload), then blocks on the shared primitive; a resolved
  entry returns the user's answer; unanswered returns the sentinel shape.
- An answer turn resolves the pending entry and returns a ZERO-content ack (the real
  continuation streams on the original POST) — and does NOT reach ``_run_agent``.
- An invalid selection keeps the prompt armed and returns a teaching hint
  (4018-style), still without spawning an agent.
- Free prose on a native-choice prompt resolves as "Other" (never dead-end the words).
- A slash command falls through to the gate; the clarify stays pending.
- A session-boundary clear cancels the pending entry so a blocked worker wakes.

Module-level ``allow_real_home_io``: importing ``gateway.run`` pulls hermes_bootstrap →
pm's dependency activation, which stats ``<repo-parent>/manifest.json``. On a default
install (repo inside the home) that is a file in the REAL home, so the guard refuses
every test that imports the runner. The tests below only touch ``tmp_path`` stores;
the real-home touch is pm's one metadata stat (false positive, no state access).
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
SINDA_ID = "20260927_130000_clar01"
ROOT_PRED = "zelda:20260927_120000_clarroot"


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


def _post_prompt(session_id: str, question: str = "Pick a lane",
                 choices=None, multi_select=False) -> str:
    """Register a pending clarify exactly the way the wired callback does."""
    import uuid as _uuid

    from tools import clarify_gateway as cg

    cid = _uuid.uuid4().hex[:10]
    cg.register(clarify_id=cid, session_key=session_id, question=question,
                choices=list(choices) if choices else None, multi_select=multi_select)
    return cid


@pytest.fixture(autouse=True)
def _clean_clarify_registry():
    from tools import clarify_gateway as cg

    with cg._lock:
        cg._entries.clear()
        cg._session_index.clear()
    yield
    with cg._lock:
        cg._entries.clear()
        cg._session_index.clear()


class TestClarifyCallbackWiring:
    @pytest.mark.asyncio
    async def test_zelda_streaming_turn_gets_clarify_kwargs(self, zelda_env, monkeypatch):
        """A Sinda streaming turn passes the clarify kwargs into _run_agent; the
        non-zelda shape gets none of them."""
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
        assert seen.get("clarify_session_key") == sid
        assert callable(seen.get("clarify_notify_callback"))

    @pytest.mark.asyncio
    async def test_non_zelda_client_gets_no_clarify_kwargs(self, zelda_env, monkeypatch):
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
        assert "clarify_session_key" not in seen
        assert "clarify_notify_callback" not in seen


class TestZeldaClarifyCallback:
    """The REAL callback built by APIServerAdapter._build_zelda_clarify_callback."""

    def _cb(self, adapter, payloads):
        return adapter._build_zelda_clarify_callback(
            lambda payload, _loop=None: payloads.append(payload), SINDA_ID)

    def test_registers_notifies_and_returns_answer(self, zelda_env):
        from tools import clarify_gateway as cg

        store, adapter, runner = zelda_env
        _seed_sinda_session(store)
        payloads = []
        cb = self._cb(adapter, payloads)
        gate = threading.Event()

        def _resolver():
            while not payloads:
                if gate.wait(timeout=0.05):
                    break
            cg.resolve_gateway_clarify(payloads[0]["clarifyId"], "Option B")

        threading.Thread(target=_resolver, daemon=True).start()
        answer = cb("Pick a lane", ["Option A", "Option B"])
        assert answer == "Option B"
        assert payloads and payloads[0]["question"] == "Pick a lane"
        assert payloads[0]["choices"] == ["Option A", "Option B"]
        assert payloads[0]["multiSelect"] is False
        assert payloads[0]["sessionId"] == SINDA_ID
        # Answer consumed: nothing left pending for the session.
        assert cg.get_pending_for_session(SINDA_ID, include_choice_prompts=True) is None

    def test_sentinel_on_empty_wake(self, zelda_env):
        """A boundary clear ("" wake) yields the did-not-respond sentinel, not ""."""
        from tools import clarify_gateway as cg

        store, adapter, runner = zelda_env
        payloads = []
        cb = self._cb(adapter, payloads)
        gate = threading.Event()

        def _clearer():
            while not payloads:
                if gate.wait(timeout=0.05):
                    break
            cg.clear_session(SINDA_ID)

        threading.Thread(target=_clearer, daemon=True).start()
        answer = cb("Still there?", None)
        assert answer.startswith("[user did not respond within")
        assert cg.get_pending_for_session(SINDA_ID, include_choice_prompts=True) is None


class TestClarifyAnswerIntercept:
    @pytest.mark.asyncio
    async def test_answer_consumes_prompt_without_agent(self, zelda_env, monkeypatch):
        """A typed selection resolves the pending prompt, ACKs with an empty
        completion, and never spawns an agent."""
        from tools import clarify_gateway as cg

        store, adapter, runner = zelda_env
        sid = _seed_sinda_session(store)
        cid = _post_prompt(sid, choices=["Option A", "Option B"])
        spawned = []

        async def _fail_run_agent(**kwargs):
            spawned.append(kwargs)
            return ({"final_response": "SHOULD-NOT-RUN"}, {})

        monkeypatch.setattr(adapter, "_run_agent", _fail_run_agent)
        async with TestClient(TestServer(_make_app(adapter, runner))) as client:
            resp = await client.post("/v1/chat/completions", json=_chat_body("2"),
                                     headers=_AUTH)
            assert resp.status == 200
            data = await resp.json()
        assert spawned == []
        assert data["choices"][0]["message"]["content"] == ""
        assert data["hermes_command"]["status"] == "ok"
        assert data["hermes_command"]["command"] == "clarify"
        # Delta: the entry actually resolved with the mapped choice ("2" -> "Option B").
        # (Index cleanup is the blocked waiter's job in production — wait_for_response
        # pops it on wake; this test has no waiter, so assert resolution, not removal.)
        with cg._lock:
            entry = cg._entries.get(cid)
            assert entry is not None and entry.response == "Option B" and entry.event.is_set()

    @pytest.mark.asyncio
    async def test_invalid_selection_keeps_prompt_and_teaches(self, zelda_env, monkeypatch):
        from tools import clarify_gateway as cg

        store, adapter, runner = zelda_env
        sid = _seed_sinda_session(store)
        cid = _post_prompt(sid, choices=["Option A", "Option B"])

        async def _fail_run_agent(**kwargs):
            return ({"final_response": "SHOULD-NOT-RUN"}, {})

        monkeypatch.setattr(adapter, "_run_agent", _fail_run_agent)
        async with TestClient(TestServer(_make_app(adapter, runner))) as client:
            resp = await client.post("/v1/chat/completions", json=_chat_body("9"),
                                     headers=_AUTH)
            assert resp.status == 200
            data = await resp.json()
        content = data["choices"][0]["message"]["content"]
        assert content and "number" in content.lower()
        assert data["hermes_command"]["status"] == "rejected"
        # The prompt is STILL armed for a retry.
        assert cg.get_pending_for_session(sid, include_choice_prompts=True) is not None
        assert cg.get_pending_for_session(sid, include_choice_prompts=True).clarify_id == cid

    @pytest.mark.asyncio
    async def test_prose_resolves_as_other(self, zelda_env, monkeypatch):
        """Free prose on a native-choice prompt resolves with the user's words."""
        from tools import clarify_gateway as cg

        store, adapter, runner = zelda_env
        sid = _seed_sinda_session(store)
        cid = _post_prompt(sid, choices=["Option A", "Option B"])

        async def _fail_run_agent(**kwargs):
            return ({"final_response": "SHOULD-NOT-RUN"}, {})

        monkeypatch.setattr(adapter, "_run_agent", _fail_run_agent)
        async with TestClient(TestServer(_make_app(adapter, runner))) as client:
            resp = await client.post("/v1/chat/completions",
                                     json=_chat_body("neither — do it offline"),
                                     headers=_AUTH)
            assert resp.status == 200
            await resp.json()
        # Prose resolved the choice prompt as "Other" with the user's own words
        # (index cleanup rides the blocked waiter's wake in production).
        with cg._lock:
            entry = cg._entries.get(cid)
            assert entry is not None and entry.response == "neither — do it offline" and entry.event.is_set()

    @pytest.mark.asyncio
    async def test_slash_command_falls_through_clarify_stays(self, zelda_env, monkeypatch):
        """A slash-prefixed turn goes to the gate path, not the intercept; the pending
        clarify survives it."""
        from tools import clarify_gateway as cg

        store, adapter, runner = zelda_env
        sid = _seed_sinda_session(store)
        cid = _post_prompt(sid, choices=["Option A", "Option B"])
        spawned = []

        async def _fake_run_agent(**kwargs):
            spawned.append(kwargs)
            return ({"final_response": "LLM-REACHED"}, {})

        monkeypatch.setattr(adapter, "_run_agent", _fake_run_agent)
        async with TestClient(TestServer(_make_app(adapter, runner))) as client:
            resp = await client.post("/v1/chat/completions",
                                     json=_chat_body("/definitely-not-a-command"),
                                     headers=_AUTH)
            assert resp.status == 200
            data = await resp.json()
        # Reached the LLM (gate passed it through), clarify untouched.
        assert data["choices"][0]["message"]["content"] == "LLM-REACHED"
        assert cg.get_pending_for_session(sid, include_choice_prompts=True) is not None
        assert cg.get_pending_for_session(sid, include_choice_prompts=True).clarify_id == cid

    @pytest.mark.asyncio
    async def test_streaming_ack_frame_shape(self, zelda_env, monkeypatch):
        """stream=true ack: a single stop frame with hermes_command, NO content deltas."""
        store, adapter, runner = zelda_env
        sid = _seed_sinda_session(store)
        _post_prompt(sid, choices=["Option A", "Option B"])

        async def _fail_run_agent(**kwargs):
            return ({"final_response": "SHOULD-NOT-RUN"}, {})

        monkeypatch.setattr(adapter, "_run_agent", _fail_run_agent)
        async with TestClient(TestServer(_make_app(adapter, runner))) as client:
            resp = await client.post("/v1/chat/completions", json=_chat_body("1", stream=True),
                                     headers=_AUTH)
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
        assert len(payloads) == 1
        assert payloads[0]["choices"][0]["finish_reason"] == "stop"
        assert payloads[0]["choices"][0]["delta"] == {}
        assert payloads[0]["hermes_command"]["status"] == "ok"


class TestBoundaryClear:
    def test_boundary_clear_cancels_pending_clarify(self, zelda_env):
        """_clear_session_boundary_security_state drops the pending entry — a /new
        must wake a blocked clarify worker instead of leaving it armed."""
        store, adapter, runner = zelda_env
        sid = _seed_sinda_session(store)
        _post_prompt(sid, choices=["A", "B"])

        # Drive the REAL funnel helper on a minimal cache-like object; it imports
        # tools.clarify_gateway fresh and clears by session key.
        from gateway.run_agent_cache import GatewayAgentCacheMixin

        cache = object.__new__(GatewayAgentCacheMixin)
        # _peek_session_state: None is the "no SessionState" path the funnel tolerates.
        cache._peek_session_state = lambda key: None  # type: ignore[method-assign]
        cache._clear_session_boundary_security_state(sid)
        from tools import clarify_gateway as cg
        assert cg.get_pending_for_session(sid, include_choice_prompts=True) is None
