"""Zelda fork: POST /api/commands must dispatch against the caller's REAL id-addressed session.

Sinda (the ``X-Zelda-Client`` fork app) addresses sessions by ``X-Hermes-Session-Id``; the
Sep-2026 ``POST /api/commands`` dispatch built a phantom source (``chat_id="zelda-fork"``)
whose generated session key no Sinda row ever carries, so an idle-path command recovered a
FRESH empty session instead of the caller's conversation.

Contracts (mirroring the /new parity handler's id-addressed resolution):
- The provided ``X-Hermes-Session-Id`` resolves to its live continuation tip.
- The dispatch source is built from the row's stored identity, so an idle-path command
  recovers the caller's row via the peer finder and operates on it (observable delta).
- No phantom session row is created by the dispatch.
- Unknown session ids are refused cleanly (404; the seeded row is untouched).
- The response carries ``session_id`` and ``status`` alongside the legacy fields.
- Auth stays required (401 without a Bearer key when one is configured).

Drives the REAL ``_handle_command_dispatch`` through aiohttp's TestClient against a REAL
SessionStore + SessionDB (SQLite in tmp_path); only the runner is a partial fake — the same
shape as tests/gateway/test_branch_thread_command.py.
"""

from __future__ import annotations

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import GatewayConfig, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware
from gateway.session import SessionStore

API_KEY = "sk-zelda-test"
SINDA_ID = "20260925_120000_aaaa01"
ROOT_PRED = "zelda:20260925_110000_root0"


# ---------------------------------------------------------------------------
# Harness
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


def _seed_sinda_session(store: SessionStore, session_id: str = SINDA_ID, title: str = "orig title"):
    """Seed a Sinda-shaped session row: source=api_server, NULL session_key, peer tuple
    (user_id/chat_id = zelda:<pred>), with messages (peer recovery requires messages)."""
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
    """Partial GatewayRunner exposing the real idle command handlers over the temp store."""
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
    app.router.add_post("/api/commands", adapter._handle_command_dispatch)
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


_AUTH = {"Authorization": f"Bearer {API_KEY}"}


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


class TestDispatchAddressesRealSession:
    @pytest.mark.asyncio
    async def test_dispatch_hits_callers_session(self, zelda_env):
        """The observable delta: /title <new> mutates the CALLER's row, not a phantom."""
        store, adapter, runner = zelda_env
        sid = _seed_sinda_session(store)
        async with TestClient(TestServer(_make_app(adapter, runner))) as client:
            resp = await client.post(
                "/api/commands", json={"command": "/title renamed-title"},
                headers={**{"Authorization": f"Bearer {API_KEY}"}, "X-Hermes-Session-Id": sid})
            assert resp.status == 200
            data = await resp.json()
            assert data["object"] == "hermes.command.result"
            assert data["handled"] is True
            assert data["status"] == "ok"
            assert data["sessionId"] == sid
        # The delta: the CALLER's row carries the new title, and it is the same row.
        assert store._db.get_session_title(sid) == "renamed-title"

    @pytest.mark.asyncio
    async def test_dispatch_creates_no_phantom_session(self, zelda_env):
        """Dispatch must not mint a fresh session row as a side effect."""
        store, adapter, runner = zelda_env
        sid = _seed_sinda_session(store)
        before = {r["id"] for r in store._db.list_sessions_rich(limit=100)}
        async with TestClient(TestServer(_make_app(adapter, runner))) as client:
            resp = await client.post(
                "/api/commands", json={"command": "/title phantom-check"},
                headers={**{"Authorization": f"Bearer {API_KEY}"}, "X-Hermes-Session-Id": sid})
            assert resp.status == 200
        after = {r["id"] for r in store._db.list_sessions_rich(limit=100)}
        assert after == before

    @pytest.mark.asyncio
    async def test_dispatch_unknown_session_id_refuses_cleanly(self, zelda_env):
        """An id nobody seeded must 404 and touch nothing."""
        store, adapter, runner = zelda_env
        sid = _seed_sinda_session(store)
        bogus = "20260925_999999_bogus9"
        async with TestClient(TestServer(_make_app(adapter, runner))) as client:
            resp = await client.post(
                "/api/commands", json={"command": "/title nope"},
                headers={**{"Authorization": f"Bearer {API_KEY}"}, "X-Hermes-Session-Id": bogus})
            assert resp.status == 404
        assert store._db.get_session_title(sid) == "orig title"
        assert len(store._db.list_sessions_rich(limit=100)) == 1

    @pytest.mark.asyncio
    async def test_dispatch_without_session_id_keeps_legacy_shape(self, zelda_env):
        """Header-less callers keep the legacy behavior: same response object, no crash."""
        store, adapter, runner = zelda_env
        _seed_sinda_session(store)
        async with TestClient(TestServer(_make_app(adapter, runner))) as client:
            resp = await client.post(
                "/api/commands", json={"command": "/status"},
                headers={"Authorization": f"Bearer {API_KEY}"})
            assert resp.status == 200
            data = await resp.json()
            assert data["object"] == "hermes.command.result"
            assert "handled" in data and "result" in data

    @pytest.mark.asyncio
    async def test_dispatch_requires_auth(self, zelda_env):
        """The endpoint stays authenticated when a key is configured."""
        store, adapter, runner = zelda_env
        sid = _seed_sinda_session(store)
        async with TestClient(TestServer(_make_app(adapter, runner))) as client:
            resp = await client.post(
                "/api/commands", json={"command": "/title sneaky"},
                headers={"X-Hermes-Session-Id": sid})
            assert resp.status == 401
        assert store._db.get_session_title(sid) == "orig title"


class TestPlainTableDispatch:
    @pytest.mark.asyncio
    async def test_plain_table_command_reaches_caller(self, zelda_env, monkeypatch):
        """PLAIN-table commands (shared idle+busy handlers, e.g. /status) dispatch through
        the fork machinery too. The real idle path consults
        ``_gateway_plain_command_handlers`` FIRST (``_hm_dispatch_canonical_command``);
        skipping it answered 'not available' for commands Telegram answers."""
        store, adapter, runner = zelda_env
        sid = _seed_sinda_session(store)

        seen = []

        async def _stub_status(event):
            seen.append(event.get_command())
            return "STATUS-STUB-OK"

        monkeypatch.setattr(runner, "_handle_status_command", _stub_status)
        async with TestClient(TestServer(_make_app(adapter, runner))) as client:
            resp = await client.post(
                "/api/commands", json={"command": "/status"},
                headers={**{"Authorization": f"Bearer {API_KEY}"}, "X-Hermes-Session-Id": sid})
            assert resp.status == 200
            data = await resp.json()
        assert data["handled"] is True
        assert data["status"] == "ok"
        assert data["result"] == "STATUS-STUB-OK"
        assert seen == ["status"]

