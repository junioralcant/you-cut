"""Testes unitários para TrackMoodClassifier em youcut/music/classifier.py."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import anthropic as real_anthropic


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig
    return PipelineConfig()


def _mock_client_returning_mood(mood: str) -> MagicMock:
    """Cliente mockado que retorna um único bloco tool_use com `mood`."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = "classify_track_mood"
    block.input = {"mood": mood}

    response = MagicMock()
    response.content = [block]

    client = MagicMock()
    client.with_options.return_value = client
    client.messages.create.return_value = response
    return client


def _mock_client_returning_blocks(blocks: list) -> MagicMock:
    response = MagicMock()
    response.content = blocks

    client = MagicMock()
    client.with_options.return_value = client
    client.messages.create.return_value = response
    return client


def _mock_client_raising(exc: Exception) -> MagicMock:
    client = MagicMock()
    client.with_options.return_value = client
    client.messages.create.side_effect = exc
    return client


class TestClassify:
    def test_returns_canonical_mood(self, config):
        from youcut.music.classifier import TrackMoodClassifier

        with patch(
            "youcut.music.classifier.anthropic.Anthropic",
            return_value=_mock_client_returning_mood("energico"),
        ):
            classifier = TrackMoodClassifier(config)
            result = classifier.classify(
                title="Treino pesado",
                description="EDM intensa para academia",
                tags=["workout", "edm"],
            )
        assert result == "energico"

    def test_invalid_mood_returns_none(self, config):
        """RF-09: mood fora do enum vira None (faixa marcada como 'indefinido' pelo syncer)."""
        from youcut.music.classifier import TrackMoodClassifier

        with patch(
            "youcut.music.classifier.anthropic.Anthropic",
            return_value=_mock_client_returning_mood("psychedelic"),
        ):
            classifier = TrackMoodClassifier(config)
            result = classifier.classify(title="X", description="Y", tags=[])
        assert result is None

    def test_no_tool_use_block_returns_none(self, config):
        from youcut.music.classifier import TrackMoodClassifier

        text_block = MagicMock()
        text_block.type = "text"

        with patch(
            "youcut.music.classifier.anthropic.Anthropic",
            return_value=_mock_client_returning_blocks([text_block]),
        ):
            classifier = TrackMoodClassifier(config)
            result = classifier.classify(title="X", description="Y", tags=[])
        assert result is None

    def test_api_error_returns_none(self, config):
        from youcut.music.classifier import TrackMoodClassifier

        api_error = real_anthropic.APIError(
            "boom",
            request=MagicMock(),
            body=None,
        )
        with patch(
            "youcut.music.classifier.anthropic.Anthropic",
            return_value=_mock_client_raising(api_error),
        ):
            classifier = TrackMoodClassifier(config)
            result = classifier.classify(title="X", description="Y", tags=[])
        assert result is None

    def test_unexpected_exception_returns_none(self, config):
        from youcut.music.classifier import TrackMoodClassifier

        with patch(
            "youcut.music.classifier.anthropic.Anthropic",
            return_value=_mock_client_raising(RuntimeError("network down")),
        ):
            classifier = TrackMoodClassifier(config)
            result = classifier.classify(title="X", description="Y", tags=[])
        assert result is None

    def test_prompt_includes_title_description_and_tags(self, config):
        """Asserts sobre o prompt enviado ao Claude (techspec)."""
        from youcut.music.classifier import TrackMoodClassifier

        client = _mock_client_returning_mood("feliz")
        with patch("youcut.music.classifier.anthropic.Anthropic", return_value=client):
            classifier = TrackMoodClassifier(config)
            classifier.classify(
                title="Praia ao Sol",
                description="Reggae alegre",
                tags=["reggae", "summer"],
            )
        call = client.messages.create.call_args
        kwargs = call.kwargs
        # tool_choice forçado para nossa tool
        assert kwargs["tool_choice"]["name"] == "classify_track_mood"
        # schema fechado em mood
        tool = kwargs["tools"][0]
        assert tool["name"] == "classify_track_mood"
        assert set(tool["input_schema"]["properties"]["mood"]["enum"]) == {
            "motivacional", "reflexivo", "energico", "emocional", "feliz", "dramatico",
        }
        # user prompt contém título, descrição e tags
        user_msg = kwargs["messages"][0]["content"]
        assert "Praia ao Sol" in user_msg
        assert "Reggae alegre" in user_msg
        assert "reggae" in user_msg and "summer" in user_msg

    def test_empty_metadata_still_calls_classifier(self, config):
        from youcut.music.classifier import TrackMoodClassifier

        client = _mock_client_returning_mood("reflexivo")
        with patch("youcut.music.classifier.anthropic.Anthropic", return_value=client):
            classifier = TrackMoodClassifier(config)
            result = classifier.classify(title="", description="", tags=[])
        assert result == "reflexivo"
