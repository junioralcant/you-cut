from datetime import datetime, timezone
from pathlib import Path

import pytest

from youcut.models import ClipRecord, SessionData
from youcut.session_store import list_sessions, load_session, save_session


def _make_session(
    session_id: str = "sess-001",
    created_at: datetime | None = None,
    tmp_path: Path | None = None,
) -> SessionData:
    base = tmp_path or Path("/tmp")
    return SessionData(
        session_id=session_id,
        source_url="https://youtube.com/watch?v=abc123",
        cut_mode="youtube",
        transcription_cache_path=base / "transcription.json",
        clips=[
            ClipRecord(
                title="Clip 1",
                start_time=0.0,
                end_time=300.0,
                clip_path=base / "clip1.mp4",
                thumbnail_path=base / "thumb1.png",
                approved=True,
            )
        ],
        created_at=created_at or datetime(2026, 4, 25, 10, 0, 0, tzinfo=timezone.utc),
        output_dir=base / "output",
    )


@pytest.fixture(autouse=True)
def patch_sessions_dir(tmp_path, monkeypatch):
    sessions_dir = tmp_path / "sessions"
    monkeypatch.setattr("youcut.session_store._SESSIONS_DIR", sessions_dir)
    return sessions_dir


def test_list_sessions_skips_corrupted_file(tmp_path):
    session = _make_session(tmp_path=tmp_path)
    save_session(session)
    corrupt = (tmp_path / "sessions" / "corrupt.json")
    corrupt.parent.mkdir(parents=True, exist_ok=True)
    corrupt.write_text("not valid json", encoding="utf-8")
    sessions = list_sessions()
    assert len(sessions) == 1
    assert sessions[0].session_id == session.session_id


def test_save_creates_file(tmp_path):
    session = _make_session(tmp_path=tmp_path)
    path = save_session(session)
    assert path.exists()
    assert path.name == f"{session.session_id}.json"


def test_round_trip(tmp_path):
    session = _make_session(tmp_path=tmp_path)
    save_session(session)
    loaded = load_session(session.session_id)

    assert loaded.session_id == session.session_id
    assert loaded.source_url == session.source_url
    assert loaded.cut_mode == session.cut_mode
    assert loaded.transcription_cache_path == session.transcription_cache_path
    assert loaded.output_dir == session.output_dir
    assert loaded.created_at == session.created_at
    assert len(loaded.clips) == len(session.clips)
    assert loaded.clips[0].clip_path == session.clips[0].clip_path
    assert loaded.clips[0].thumbnail_path == session.clips[0].thumbnail_path


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="nonexistent"):
        load_session("nonexistent")


def test_list_sessions_ordered_desc(tmp_path):
    dates = [
        datetime(2026, 4, 23, tzinfo=timezone.utc),
        datetime(2026, 4, 25, tzinfo=timezone.utc),
        datetime(2026, 4, 24, tzinfo=timezone.utc),
    ]
    for i, dt in enumerate(dates):
        save_session(_make_session(session_id=f"sess-{i:03d}", created_at=dt, tmp_path=tmp_path))

    sessions = list_sessions()
    assert len(sessions) == 3
    assert sessions[0].created_at > sessions[1].created_at > sessions[2].created_at


def test_sessions_dir_created_automatically(tmp_path, monkeypatch):
    new_dir = tmp_path / "brand_new" / "sessions"
    monkeypatch.setattr("youcut.session_store._SESSIONS_DIR", new_dir)
    assert not new_dir.exists()
    session = _make_session(tmp_path=tmp_path)
    save_session(session)
    assert new_dir.exists()


def test_list_empty_sessions():
    assert list_sessions() == []
