"""Testes de youcut.presenters.detector.

A maioria mockeia o cliente Claude e a extração de frames, já que
o caminho real depende de ffmpeg + áudio/imagem reais.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from youcut.presenters.catalog import load_catalog
from youcut.presenters.detector import detect_presenters


class _FakeClaude:
    def __init__(self, slugs: list[str] | None):
        self._slugs = slugs
        self.messages = SimpleNamespace(create=self._create)
        self.call_count = 0
        self.last_content_len = 0

    def _create(self, model, max_tokens, messages):
        self.call_count += 1
        msg = messages[0]
        self.last_content_len = len(msg.get("content", []))
        if self._slugs is None:
            payload = "{\"detected_slugs\": []}"
        else:
            slugs_json = ", ".join(f'"{s}"' for s in self._slugs)
            payload = f'{{"detected_slugs": [{slugs_json}]}}'
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=payload)])


def _populate(tmp_path: Path, slugs: list[str]) -> Path:
    for slug in slugs:
        (tmp_path / f"{slug}.png").write_bytes(b"fake")
    return tmp_path


def test_detect_empty_catalog_returns_empty(tmp_path: Path):
    catalog = load_catalog(tmp_path)  # vazio
    fake_claude = _FakeClaude(["tiago_leifert"])
    result = detect_presenters(tmp_path / "video.mp4", catalog, fake_claude, "claude")
    assert result.profiles == []
    # Catálogo vazio NÃO deve chamar o LLM.
    assert fake_claude.call_count == 0


def test_detect_no_client_returns_empty(tmp_path: Path):
    _populate(tmp_path, ["tiago_leifert"])
    catalog = load_catalog(tmp_path)
    result = detect_presenters(tmp_path / "fake.mp4", catalog, None, "claude")
    assert result.profiles == []
    assert result.source_method == "vision"


def test_detect_picks_matching_slug(tmp_path: Path):
    _populate(tmp_path, ["tiago_leifert", "outro_apresentador"])
    catalog = load_catalog(tmp_path)
    fake_claude = _FakeClaude(["tiago_leifert"])
    # Mock o sampling de frames pra retornar 1 frame fake.
    with patch(
        "youcut.presenters.detector._sample_video_frames",
        return_value=[b"\x89PNG\r\n\x1a\n" + b"\x00" * 100],
    ):
        result = detect_presenters(
            tmp_path / "fake.mp4", catalog, fake_claude, "claude-test",
        )
    assert len(result.profiles) == 1
    assert result.profiles[0].slug == "tiago_leifert"
    assert fake_claude.call_count == 1


def test_detect_ignores_unknown_slug(tmp_path: Path):
    _populate(tmp_path, ["tiago_leifert"])
    catalog = load_catalog(tmp_path)
    fake_claude = _FakeClaude(["messi"])  # não está no catálogo
    with patch(
        "youcut.presenters.detector._sample_video_frames",
        return_value=[b"frame"],
    ):
        result = detect_presenters(tmp_path / "v.mp4", catalog, fake_claude, "claude")
    assert result.profiles == []


def test_detect_no_frames_returns_empty(tmp_path: Path):
    _populate(tmp_path, ["tiago_leifert"])
    catalog = load_catalog(tmp_path)
    fake_claude = _FakeClaude(["tiago_leifert"])
    with patch(
        "youcut.presenters.detector._sample_video_frames",
        return_value=[],  # nenhum frame extraído
    ):
        result = detect_presenters(tmp_path / "v.mp4", catalog, fake_claude, "claude")
    assert result.profiles == []
    # Sem frames, o LLM nem deve ser chamado.
    assert fake_claude.call_count == 0


def test_detect_claude_failure_is_graceful(tmp_path: Path):
    _populate(tmp_path, ["tiago_leifert"])
    catalog = load_catalog(tmp_path)

    class _Boom:
        messages = SimpleNamespace(
            create=lambda **_: (_ for _ in ()).throw(RuntimeError("credits low"))
        )

    with patch(
        "youcut.presenters.detector._sample_video_frames",
        return_value=[b"frame"],
    ):
        result = detect_presenters(tmp_path / "v.mp4", catalog, _Boom(), "claude")
    assert result.profiles == []


def test_detect_invalid_json_returns_empty(tmp_path: Path):
    _populate(tmp_path, ["tiago_leifert"])
    catalog = load_catalog(tmp_path)

    class _BadJson:
        def __init__(self):
            self.messages = SimpleNamespace(create=self._create)

        def _create(self, **_):
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="lorem ipsum, no json")]
            )

    with patch(
        "youcut.presenters.detector._sample_video_frames",
        return_value=[b"frame"],
    ):
        result = detect_presenters(tmp_path / "v.mp4", catalog, _BadJson(), "claude")
    assert result.profiles == []


def test_detect_dedupes_slugs(tmp_path: Path):
    _populate(tmp_path, ["tiago_leifert"])
    catalog = load_catalog(tmp_path)
    # LLM retorna o mesmo slug 3 vezes — deduplicar.
    fake_claude = _FakeClaude(["tiago_leifert", "tiago_leifert", "tiago_leifert"])
    with patch(
        "youcut.presenters.detector._sample_video_frames",
        return_value=[b"frame"],
    ):
        result = detect_presenters(tmp_path / "v.mp4", catalog, fake_claude, "claude")
    assert len(result.profiles) == 1
