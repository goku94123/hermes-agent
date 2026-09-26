"""Zelda fork lifecycle notices (2026-09-25).

The api_server adapter's send() is a stub — platform broadcasts like the Telegram
"Hermes is shutting down" notice can never reach Sinda through it. This module
queues NOTICE jobs into the same filesystem push queue the fork already polls
(~/.hermes/zelda-outbound/.push-queue/*.json): rows survive the restart itself,
so a shutdown notice written mid-shutdown is delivered by the fork the moment
the gateway (and its HTTP route) is back — paired with the boot-time "back
online" notice queued at api_server startup.

A notice row carries text (not a file); the fork renders it as a system bubble.
`sessions` empty/absent = fan out to every topic on the phone.
Queue hygiene: the fork acks (deletes) each row after rendering.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

SHUTDOWN_TEXT = (
    "⚠️ Hermes is shutting down — your current task will be interrupted. "
    "When it is back online, send any message and I'll try to pick up where we left off."
)
RESTART_TEXT = (
    "⚠️ Hermes is restarting — your current task will be interrupted. "
    "Send any message after the restart and I'll try to resume where you left off."
)
STARTUP_TEXT = "✅ Hermes gateway is back online. Send any message to continue."


def queue_zelda_notice(
    kind: str,
    text: str,
    session_ids: Optional[Iterable[str]] = None,
) -> Optional[str]:
    """Queue one text notice for the Zelda fork poller. Best-effort: failures log,
    never raise — callers run during shutdown/boot where raising is worse than useless."""
    try:
        qdir = Path.home() / ".hermes" / "zelda-outbound" / ".push-queue"
        qdir.mkdir(parents=True, exist_ok=True)
        row = {
            "text": text,
            "notice": kind,
            "ts": int(time.time()),
            "sessions": [s for s in (session_ids or []) if s],
        }
        nid = f"notice-{kind}-{int(time.time())}-{os.urandom(3).hex()}"
        (qdir / f"{nid}.json").write_text(json.dumps(row))
        logger.info("[zelda-notices] queued %s notice (%s)", kind, nid)
        return nid
    except Exception as exc:
        logger.warning("[zelda-notices] queue failed for %s: %s", kind, exc)
        return None


# Convenience wrapper for run_shutdown (queue_zelda_notice_for_sessions) removed 2026-09-26:
# shutdown notices must BROADCAST (empty sessions), never target — session-key-derived ids
# are other platforms' chat ids, not Sinda topic ids.

