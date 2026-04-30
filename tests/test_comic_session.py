from datetime import datetime
from pathlib import Path

import pytest

from youcut.comic.session import (
    list_motion_comic_sessions,
    load_motion_comic_session,
    save_motion_comic_session,
)
from youcut.models import (
    CastMember,
    ClipRecord,
    MotionComicSession,
    Panel,
    PanelRenderResult,
    SessionData,
)
from youcut.session_store import save_session


@pytest.fixture(autouse=True)
def isolated_sessions_dir(tmp_path, monkeypatch):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    monkeypatch.setattr("youcut.session_store._SESSIONS_DIR", sessions_dir)
    return sessions_dir


def _sample_motion_comic(session_id: str = "mc_001") -> MotionComicSession:
    return MotionComicSession(
        session_id=session_id,
        video_path=Path("input.mp4"),
        created_at=datetime(2025, 1, 1, 12, 0, 0),
        cast=[CastMember(character_id="p1", text_card="ficha")],
        panels=[
            Panel(
                index=0,
                start_time=0.0,
                end_time=3.0,
                participants=["p1"],
                framing="close",
                scene="rua",
                pose_description="parado",
                panel_seconds_target=3.0,
            )
        ],
        panel_results=[
            PanelRenderResult(
                panel_index=0,
                base_image_path=Path("base.png"),
                clip_path=Path("clip.mp4"),
                clip_seconds=3.0,
                cost_usd=0.10,
            )
        ],
        total_cost_usd=0.10,
    )


def test_save_and_load_round_trip(isolated_sessions_dir):
    original = _sample_motion_comic()
    saved_path = save_motion_comic_session(original)
    assert saved_path.exists()
    assert saved_path.parent == isolated_sessions_dir

    loaded = load_motion_comic_session(original.session_id)
    assert loaded.session_id == original.session_id
    assert loaded.video_path == original.video_path
    assert len(loaded.cast) == 1
    assert loaded.cast[0].character_id == "p1"
    assert len(loaded.panels) == 1
    assert loaded.panels[0].framing == "close"
    assert len(loaded.panel_results) == 1
    assert loaded.total_cost_usd == 0.10


def test_load_unknown_session_raises():
    with pytest.raises(FileNotFoundError, match=r"não encontrada"):
        load_motion_comic_session("does-not-exist")


def test_list_filters_legacy_sessions(isolated_sessions_dir):
    legacy = SessionData(
        session_id="legacy_1",
        source_url="https://youtube.com/x",
        cut_mode="social",
        transcription_cache_path=Path("/tmp/x.json"),
        clips=[],
        created_at=datetime(2025, 1, 1),
        output_dir=Path("/tmp/output"),
    )
    save_session(legacy)

    motion_a = _sample_motion_comic(session_id="mc_a")
    motion_b = _sample_motion_comic(session_id="mc_b")
    motion_b.created_at = datetime(2025, 1, 2, 12, 0, 0)
    save_motion_comic_session(motion_a)
    save_motion_comic_session(motion_b)

    result = list_motion_comic_sessions()
    ids = [s.session_id for s in result]
    assert "legacy_1" not in ids
    assert ids == ["mc_b", "mc_a"]


def test_list_returns_empty_when_no_motion_comic_sessions(isolated_sessions_dir):
    legacy = SessionData(
        session_id="legacy_only",
        source_url="https://youtube.com/x",
        cut_mode="social",
        transcription_cache_path=Path("/tmp/x.json"),
        clips=[ClipRecord(
            title="t",
            start_time=0.0,
            end_time=1.0,
            clip_path=Path("c.mp4"),
            thumbnail_path=None,
        )],
        created_at=datetime(2025, 1, 1),
        output_dir=Path("/tmp/output"),
    )
    save_session(legacy)

    result = list_motion_comic_sessions()
    assert result == []


def test_list_ignores_corrupt_files(isolated_sessions_dir):
    motion = _sample_motion_comic()
    save_motion_comic_session(motion)
    (isolated_sessions_dir / "corrupt.json").write_text("not-json", encoding="utf-8")

    result = list_motion_comic_sessions()
    assert len(result) == 1
    assert result[0].session_id == motion.session_id


def test_save_overwrites_existing_session(isolated_sessions_dir):
    a = _sample_motion_comic()
    save_motion_comic_session(a)
    a.total_cost_usd = 9.99
    save_motion_comic_session(a)

    loaded = load_motion_comic_session(a.session_id)
    assert loaded.total_cost_usd == 9.99


def test_legacy_session_is_not_overwritten_by_motion_comic(isolated_sessions_dir):
    """Sessões legadas em arquivos com nome diferente coexistem."""
    legacy = SessionData(
        session_id="legacy_x",
        source_url="https://youtube.com/x",
        cut_mode="social",
        transcription_cache_path=Path("/tmp/x.json"),
        clips=[],
        created_at=datetime(2025, 1, 1),
        output_dir=Path("/tmp/output"),
    )
    save_session(legacy)

    motion = _sample_motion_comic(session_id="mc_y")
    save_motion_comic_session(motion)

    legacy_files = list(isolated_sessions_dir.glob("legacy_x.json"))
    motion_files = list(isolated_sessions_dir.glob("mc_y.json"))
    assert len(legacy_files) == 1
    assert len(motion_files) == 1
