"""In-process session registry for the dashboard (plan §11): each submitted
pair gets a session holding its ingested documents, computed delta, a live
ChatIndex, and the on-disk report/markup paths.

Deliberately in-memory, not a database: this is a single-process demo
dashboard (`make dashboard`), not a deployed multi-worker service.
Sessions don't survive a process restart -- an accepted, documented scope
boundary (see README), not an oversight. The report/markup *files* it
writes do survive on disk (dashboard_runs/{session_id}/), so the durable
artifacts aren't lost even though the in-memory index/chat-history is.
"""

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from src.canonical.model import CanonicalDocument
from src.chat.answer import ChatAnswer
from src.chat.index import ChatIndex
from src.delta.engine import DeltaReport


@dataclass
class ChatTurn:
    question: str
    answer: ChatAnswer


@dataclass
class PairSession:
    session_id: str
    pair_label: str
    doc_a: CanonicalDocument
    doc_b: CanonicalDocument
    report: DeltaReport
    report_paths: dict[str, Path]
    chat_index: ChatIndex
    created_at: datetime
    chat_history: list[ChatTurn] = field(default_factory=list)


_SESSIONS: dict[str, PairSession] = {}


def create_session(
    pair_label: str,
    doc_a: CanonicalDocument,
    doc_b: CanonicalDocument,
    report: DeltaReport,
    report_paths: dict[str, Path],
    chat_index: ChatIndex,
) -> PairSession:
    session_id = uuid.uuid4().hex[:12]
    session = PairSession(
        session_id=session_id,
        pair_label=pair_label,
        doc_a=doc_a,
        doc_b=doc_b,
        report=report,
        report_paths=report_paths,
        chat_index=chat_index,
        created_at=datetime.now(UTC),
    )
    _SESSIONS[session_id] = session
    return session


def get_session(session_id: str) -> PairSession | None:
    return _SESSIONS.get(session_id)


def list_sessions() -> list[PairSession]:
    return sorted(_SESSIONS.values(), key=lambda s: s.created_at, reverse=True)


def reset_sessions() -> None:
    """Test-only escape hatch -- the registry is module-level state, and
    tests need a clean slate between runs without restarting the process."""
    _SESSIONS.clear()
