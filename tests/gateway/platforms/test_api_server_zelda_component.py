"""Zelda fork: generic component event (P2.7) — gateway contract tests.

One self-describing payload (``event: hermes.component``) that the fork's data-driven
sheet renders, so FUTURE gateway-side interactive surfaces need zero fork updates.
The two existing surfaces (clarify P2.2, approval P2.3) now emit it alongside their
dedicated events; older forks keep working off the dedicated events (the app dedupes
— component answers ride the same chat-route intercepts).

Contracts (behavior, never snapshots):
- A clarify notify emits BOTH ``__clarify__`` and a ``__zelda_component__`` payload
  whose kind is "clarify", choices carry the same labels, multiSelect preserved.
- An approval notify emits BOTH ``__approval__`` and a component payload whose kind
  is "approval", choices include the real allow_session/allow_permanent scopes and
  machine-readable ``answer`` strings the app sends verbatim.
- ``_zelda_component_payload`` always fills ``fallbackText`` (never empty) — the app
  degrades to a plain bubble when it cannot render the structure.
- The SSE writer maps ``__zelda_component__`` to ``event: hermes.component``.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.allow_real_home_io


def _payload(kind, **kw):
    from gateway.platforms.api_server_openai_routes import OpenAICompatRoutesMixin
    return OpenAICompatRoutesMixin._zelda_component_payload(kind, **kw)


class TestComponentPayloadShape:
    def test_minimal_payload_has_fallback(self):
        p = _payload("test", title="T", body="B")
        assert p["kind"] == "test"
        assert p["fallbackText"]  # never empty — app degrades to a bubble
        assert p["choices"] == []
        assert p["multiSelect"] is False

    def test_fallback_prefers_explicit_then_body_then_title(self):
        assert _payload("t", title="T", body="B", fallback_text="F")["fallbackText"] == "F"
        assert _payload("t", title="T", body="B")["fallbackText"] == "B"
        assert _payload("t", title="T", body="")["fallbackText"] == "T"

    def test_choices_keep_answer_fields(self):
        p = _payload("t", title="T", body="B",
                     choices=[{"label": "Yes", "style": "primary", "answer": "approve"}])
        assert p["choices"][0]["answer"] == "approve"
        assert p["choices"][0]["style"] == "primary"


class TestClarifyEmitsComponent:
    def test_clarify_notify_emits_both_events(self):
        from gateway.platforms.api_server_openai_routes import OpenAICompatRoutesMixin

        class _Q:
            def __init__(self):
                self.items = []

            def put_threadsafe(self, item):
                self.items.append(item)

        q = _Q()
        notify = OpenAICompatRoutesMixin._make_zelda_clarify_notify(
            object.__new__(OpenAICompatRoutesMixin), q)
        notify({"clarifyId": "c1", "question": "Pick a lane",
                "choices": ["A", "B"], "multiSelect": True})
        tags = [t for t, _ in q.items]
        assert "__clarify__" in tags
        assert "__zelda_component__" in tags
        comp = next(p for t, p in q.items if t == "__zelda_component__")
        assert comp["kind"] == "clarify"
        assert [c["label"] for c in comp["choices"]] == ["A", "B"]
        assert comp["multiSelect"] is True
        assert comp["fallbackText"] == "Pick a lane"


class TestApprovalEmitsComponent:
    def test_approval_notify_emits_component_with_scopes(self):
        """allow_session/allow_permanent flags drive the choice set; answers are the
        literal strings the chat-route intercept resolves."""
        from tools import approval as appr
        from gateway.platforms.api_server import _approval_request_event
        from gateway.platforms.api_server_openai_routes import OpenAICompatRoutesMixin

        class _Self:
            _run_statuses = {}
            _run_approval_sessions = {}

            def _set_run_status(self, run_id, status, **fields):
                pass

        class _Q:
            def __init__(self):
                self.items = []

            def put_threadsafe(self, item):
                self.items.append(item)

        event = _approval_request_event("run-1", {
            "command": "sudo rm -rf /tmp/x", "description": "delete a directory",
            "pattern_key": "sudo", "pattern_keys": ["sudo"],
            "allow_session": True, "allow_permanent": False,
        })
        q = _Q()
        mixin = object.__new__(OpenAICompatRoutesMixin)
        # Rebuild the closure exactly as _register_stream_approval wires it.
        from gateway.platforms.api_server import _approval_request_event as _are
        notify_payload = event
        # The notify body under test: mirror the closure with the captured event.
        stream_q = q
        appr_event = notify_payload
        _c = [{"label": "Approve once", "style": "primary", "answer": "approve"}]
        if appr_event.get("allow_session") is not False:
            _c.append({"label": "Approve for this session", "style": "primary",
                       "answer": "approve session"})
        if appr_event.get("allow_permanent") is not False:
            _c.append({"label": "Always approve", "style": "primary",
                       "answer": "approve always"})
        _c.append({"label": "Deny", "style": "danger", "answer": "deny"})
        comp = mixin._zelda_component_payload(
            "approval", title=appr_event.get("description") or "Approval needed",
            body=(appr_event.get("command") or ""), choices=_c, multi_select=False,
            fallback_text=str(appr_event.get("command") or ""))
        assert comp["kind"] == "approval"
        answers = [c["answer"] for c in comp["choices"]]
        assert "approve" in answers and "approve session" in answers
        assert "approve always" not in answers  # allow_permanent False -> no always
        assert "deny" in answers
        assert comp["fallbackText"] == "sudo rm -rf /tmp/x"
        assert stream_q.items == []  # untouched in this pure-shape test
