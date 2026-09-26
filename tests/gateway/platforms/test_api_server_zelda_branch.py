"""Zelda fork: /branch on Sinda (Task 3) — DB clone + header rebind, no thread needed.

Sinda is not thread-capable (in-app topics ARE the threads), so ``/branch`` clones the
caller's id-addressed session at the DB level (mirror of the upstream branch handler minus
thread creation) and the fork creates a new topic bound to the clone via the response
``X-Hermes-Session-Id`` header — the same rebind flow as the shipped /new parity.

Contracts:
- plain ``/branch <name>``: caller's row UNTOUCHED (still routable, still titled); a NEW row
  exists with parent_session_id == caller id, copied history, title lineage, and the response
  header names the clone (sessionId + header).
- ``/branch --here <name>``: caller's topic REBINDS onto the clone (old row ended as
  session_switch boundary, new row is the continuation tip; response header = new id).
- Unknown/empty-history session: refused cleanly, nothing minted.
- /branch is reachable through the Task 2 gate (excluded there, handled here).
"""

from __future__ import annotations

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import GatewayConfig, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.session import SessionStore

API_KEY = "sk-zelda-test"
SINDA_ID = "20260925_120000_cccc03"
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


def _seed_sinda_session(store: SessionStore, session_id: str = SINDA_ID, title: str = "parent title"):
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
    app = web.Application()
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


def _chat_body(text: str) -> dict:
    return {"model": "test-model", "messages": [{"role": "user", "content": text}], "stream": False}


class TestBranch:
    @pytest.mark.asyncio
    async def test_branch_clones_and_rebinds_header(self, zelda_env):
        """Plain /branch: caller untouched; new row is the parent's child with copied history."""
        store, adapter, runner = zelda_env
        sid = _seed_sinda_session(store)
        async with TestClient(TestServer(_make_app(adapter, runner))) as client:
            resp = await client.post("/v1/chat/completions", json=_chat_body("/branch alt-plan"),
                                     headers=_AUTH)
            assert resp.status == 200
            data = await resp.json()
            branch_id = resp.headers.get("X-Hermes-Session-Id")
        meta = data["hermes_command"]
        assert meta["status"] == "ok"
        assert branch_id and branch_id != sid
        # Caller's row untouched: same title, still live.
        assert store._db.get_session_title(sid) == "parent title"
        # The clone: parent link + copied history + identity columns.
        row = store._db.get_session(branch_id)
        assert row["parent_session_id"] == sid
        msgs = store._db.get_messages(branch_id)
        assert [m["content"] for m in msgs] == ["hello from sinda", "hi"]

    @pytest.mark.asyncio
    async def test_branch_here_rebinds_in_place(self, zelda_env):
        """/branch --here: the topic moves onto the clone; old row ends as session_switch."""
        store, adapter, runner = zelda_env
        sid = _seed_sinda_session(store)
        async with TestClient(TestServer(_make_app(adapter, runner))) as client:
            resp = await client.post("/v1/chat/completions",
                                     json=_chat_body("/branch --here here-plan"), headers=_AUTH)
            assert resp.status == 200
            data = await resp.json()
            new_id = resp.headers.get("X-Hermes-Session-Id")
        meta = data["hermes_command"]
        assert meta["status"] == "ok"
        assert new_id and new_id != sid
        row = store._db.get_session(new_id)
        assert row["parent_session_id"] == sid
        assert store._db.get_session_title(new_id) == "here-plan"
        old_row = store._db.get_session(sid)
        assert old_row["ended_at"] is not None

    @pytest.mark.asyncio
    async def test_branch_unknown_session_404(self, zelda_env):
        store, adapter, runner = zelda_env
        _seed_sinda_session(store)
        headers = {**_AUTH, "X-Hermes-Session-Id": "20260925_999999_bogus9"}
        async with TestClient(TestServer(_make_app(adapter, runner))) as client:
            resp = await client.post("/v1/chat/completions", json=_chat_body("/branch x"),
                                     headers=headers)
            assert resp.status == 404
        assert len(store._db.list_sessions_rich(limit=100)) == 1
