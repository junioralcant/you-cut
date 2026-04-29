from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import anthropic as real_anthropic

from youcut.analyzer import (
    MAX_CLIP_DURATION,
    MIN_CLIP_DURATION,
    _SYSTEM_PROMPT,
    _normalize_thumbnail_text,
    analyze,
)
from youcut.models import TranscriptionResult, TranscriptionSegment, ViralClip, WordTimestamp


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from youcut.config import PipelineConfig
    return PipelineConfig()


@pytest.fixture
def short_transcription():
    return TranscriptionResult(
        segments=[
            TranscriptionSegment(
                start=0.0,
                end=30.0,
                text="Este é um conteúdo incrível sobre Python",
                words=[WordTimestamp(word="Este", start=0.0, end=0.5)],
            ),
            TranscriptionSegment(
                start=30.0,
                end=60.0,
                text="Veja como fazer em 30 segundos",
                words=[WordTimestamp(word="Veja", start=30.0, end=30.5)],
            ),
            TranscriptionSegment(
                start=60.0,
                end=90.0,
                text="Você não vai acreditar no resultado",
                words=[WordTimestamp(word="Você", start=60.0, end=60.5)],
            ),
        ],
        language="pt",
        source_path=Path("test_video.mp4"),
    )


def _make_mock_client(clips_data: list[dict]) -> MagicMock:
    """Build a mock Anthropic client that returns a tool_use response."""
    mock_block = MagicMock()
    mock_block.type = "tool_use"
    mock_block.name = "identify_viral_clips"
    mock_block.input = {"clips": clips_data}

    mock_response = MagicMock()
    mock_response.content = [mock_block]

    mock_client = MagicMock()
    mock_client.with_options.return_value = mock_client
    mock_client.messages.create.return_value = mock_response
    return mock_client


class TestAnalyzeMapping:
    def test_returns_viral_clip_instances(self, config, short_transcription):
        clips_data = [
            {
                "title": "Conteúdo Incrível!",
                "reason": "Gancho forte no início",
                "viral_score": 9.0,
                "start_time": 0.0,
                "end_time": 30.0,
                "description": "Veja este conteúdo incrível",
                "hashtags": ["#python", "#viral"],
                "thumbnail_idea": "Frame com expressão de surpresa",
                "thumbnail_text": "MOMENTO IMPACTANTE",
            }
        ]
        mock_client = _make_mock_client(clips_data)

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            result = analyze(short_transcription, config)

        assert len(result) == 1
        assert isinstance(result[0], ViralClip)
        assert result[0].title == "Conteúdo Incrível!"
        assert result[0].viral_score == 9.0
        assert result[0].hashtags == ["#python", "#viral"]
        assert result[0].thumbnail_idea == "Frame com expressão de surpresa"

    def test_all_fields_mapped_correctly(self, config, short_transcription):
        raw = {
            "title": "Título Teste",
            "reason": "Razão Teste",
            "viral_score": 7.5,
            "start_time": 10.0,
            "end_time": 40.0,
            "description": "Descrição Teste",
            "hashtags": ["#a", "#b"],
            "thumbnail_idea": "Ideia Thumbnail",
            "thumbnail_text": "MOMENTO IMPACTANTE",
            "social_hook_title": "ALERTA TOTAL",
            "social_image_prompt": "Cena editorial de tensão política sem texto",
            "social_visual_style": "claro e vivo",
        }
        mock_client = _make_mock_client([raw])

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            result = analyze(short_transcription, config)

        clip = result[0]
        assert clip.title == raw["title"]
        assert clip.reason == raw["reason"]
        assert clip.viral_score == raw["viral_score"]
        assert clip.start_time == raw["start_time"]
        assert clip.end_time == raw["end_time"]
        assert clip.description == raw["description"]
        assert clip.hashtags == raw["hashtags"]
        assert clip.thumbnail_idea == raw["thumbnail_idea"]
        assert clip.thumbnail_text == raw["thumbnail_text"]
        assert clip.social_hook_title == raw["social_hook_title"]
        assert clip.social_image_prompt == raw["social_image_prompt"]
        assert clip.social_visual_style == raw["social_visual_style"]

    def test_social_fields_are_derived_when_missing(self, config, short_transcription):
        raw = {
            "title": "Título Social Teste",
            "reason": "Momento forte sobre crise política",
            "viral_score": 7.5,
            "start_time": 10.0,
            "end_time": 40.0,
            "description": "Descrição Teste",
            "hashtags": ["#a", "#b"],
            "thumbnail_idea": "Debate acalorado em estúdio",
            "thumbnail_text": "MOMENTO IMPACTANTE",
        }
        mock_client = _make_mock_client([raw])

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            result = analyze(short_transcription, config)

        clip = result[0]
        assert clip.social_hook_title == "TÍTULO SOCIAL TESTE"
        assert "Debate acalorado em estúdio" in clip.social_image_prompt
        assert clip.social_visual_style

    def test_thumbnail_text_is_normalized_to_prd_style(self, config, short_transcription):
        raw = {
            "title": "Título Teste",
            "reason": "Razão Teste",
            "viral_score": 7.5,
            "start_time": 10.0,
            "end_time": 40.0,
            "description": "Descrição Teste",
            "hashtags": ["#a", "#b"],
            "thumbnail_idea": "Ideia Thumbnail",
            "thumbnail_text": '  eleiÇÃO   pega fogo!!!  ',
        }
        mock_client = _make_mock_client([raw])

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            result = analyze(short_transcription, config)

        assert result[0].thumbnail_text == "ELEIÇÃO PEGA FOGO"

    def test_thumbnail_text_empty_falls_back_to_derived_text(self, config, short_transcription):
        raw = {
            "title": "Renan Santos fala sobre eleições e segurança",
            "reason": "Discussão forte sobre segurança pública e campanha",
            "viral_score": 7.5,
            "start_time": 10.0,
            "end_time": 40.0,
            "description": "Descrição Teste",
            "hashtags": ["#a", "#b"],
            "thumbnail_idea": "Ideia Thumbnail",
            "thumbnail_text": "",
        }
        mock_client = _make_mock_client([raw])

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            result = analyze(short_transcription, config)

        assert result[0].thumbnail_text == "DISCUSSÃO FORTE SEGURANÇA PÚBLICA CAMPANHA"

    def test_thumbnail_text_equal_to_title_falls_back_to_derived_text(self, config, short_transcription):
        raw = {
            "title": "Renan Santos fala sobre eleições e segurança",
            "reason": "Discussão forte sobre segurança pública e campanha",
            "viral_score": 7.5,
            "start_time": 10.0,
            "end_time": 40.0,
            "description": "Descrição Teste",
            "hashtags": ["#a", "#b"],
            "thumbnail_idea": "Ideia Thumbnail",
            "thumbnail_text": "Renan Santos fala sobre eleições e segurança",
        }
        mock_client = _make_mock_client([raw])

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            result = analyze(short_transcription, config)

        assert result[0].thumbnail_text == "DISCUSSÃO FORTE SEGURANÇA PÚBLICA CAMPANHA"

    def test_thumbnail_text_too_similar_to_title_falls_back_to_derived_text(self, config, short_transcription):
        raw = {
            "title": "Renan Santos fala sobre eleições e segurança",
            "reason": "Discussão forte sobre segurança pública e campanha",
            "viral_score": 7.5,
            "start_time": 10.0,
            "end_time": 40.0,
            "description": "Descrição Teste",
            "hashtags": ["#a", "#b"],
            "thumbnail_idea": "Ideia Thumbnail",
            "thumbnail_text": "Renan Santos nas eleições",
        }
        mock_client = _make_mock_client([raw])

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            result = analyze(short_transcription, config)

        assert result[0].thumbnail_text == "DISCUSSÃO FORTE SEGURANÇA PÚBLICA CAMPANHA"

    def test_normalize_thumbnail_text_truncates_to_six_words(self):
        normalized = _normalize_thumbnail_text(
            "um texto muito longo com palavras demais para thumbnail",
            title="Titulo qualquer",
            reason="Razão qualquer",
        )

        assert normalized == "UM TEXTO MUITO LONGO COM PALAVRAS"


class TestAnalyzeSorting:
    def test_sorted_by_viral_score_descending(self, config, short_transcription):
        clips_data = [
            {
                "title": "Baixo", "reason": "R", "viral_score": 5.0,
                "start_time": 0.0, "end_time": 30.0, "description": "D",
                "hashtags": [], "thumbnail_idea": "T", "thumbnail_text": "MOMENTO IMPACTANTE",
            },
            {
                "title": "Alto", "reason": "R", "viral_score": 9.5,
                "start_time": 30.0, "end_time": 60.0, "description": "D",
                "hashtags": [], "thumbnail_idea": "T", "thumbnail_text": "MOMENTO IMPACTANTE",
            },
            {
                "title": "Médio", "reason": "R", "viral_score": 7.0,
                "start_time": 60.0, "end_time": 90.0, "description": "D",
                "hashtags": [], "thumbnail_idea": "T", "thumbnail_text": "MOMENTO IMPACTANTE",
            },
        ]
        mock_client = _make_mock_client(clips_data)

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            result = analyze(short_transcription, config)

        scores = [c.viral_score for c in result]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == 9.5
        assert scores[-1] == 5.0


class TestAnalyzeFiltering:
    def test_clips_below_min_duration_filtered(self, config, short_transcription):
        clips_data = [
            {
                "title": "Muito Curto", "reason": "R", "viral_score": 9.0,
                "start_time": 0.0, "end_time": MIN_CLIP_DURATION - 1,  # too short
                "description": "D", "hashtags": [], "thumbnail_idea": "T",
                "thumbnail_text": "MOMENTO IMPACTANTE",
            },
            {
                "title": "Válido", "reason": "R", "viral_score": 8.0,
                "start_time": 0.0, "end_time": 30.0,  # valid
                "description": "D", "hashtags": [], "thumbnail_idea": "T",
                "thumbnail_text": "MOMENTO IMPACTANTE",
            },
        ]
        mock_client = _make_mock_client(clips_data)

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            result = analyze(short_transcription, config)

        assert len(result) == 1
        assert result[0].title == "Válido"

    def test_clips_above_max_duration_filtered(self, config, short_transcription):
        clips_data = [
            {
                "title": "Muito Longo", "reason": "R", "viral_score": 9.0,
                "start_time": 0.0, "end_time": MAX_CLIP_DURATION + 1,  # too long
                "description": "D", "hashtags": [], "thumbnail_idea": "T",
                "thumbnail_text": "MOMENTO IMPACTANTE",
            },
            {
                "title": "Válido", "reason": "R", "viral_score": 8.0,
                "start_time": 0.0, "end_time": 45.0,  # valid
                "description": "D", "hashtags": [], "thumbnail_idea": "T",
                "thumbnail_text": "MOMENTO IMPACTANTE",
            },
        ]
        mock_client = _make_mock_client(clips_data)

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            result = analyze(short_transcription, config)

        assert len(result) == 1
        assert result[0].title == "Válido"

    def test_boundary_min_duration_accepted(self, config, short_transcription):
        clips_data = [
            {
                "title": "Exatamente 15s", "reason": "R", "viral_score": 8.0,
                "start_time": 0.0, "end_time": float(MIN_CLIP_DURATION),
                "description": "D", "hashtags": [], "thumbnail_idea": "T",
                "thumbnail_text": "MOMENTO IMPACTANTE",
            }
        ]
        mock_client = _make_mock_client(clips_data)

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            result = analyze(short_transcription, config)

        assert len(result) == 1

    def test_boundary_max_duration_accepted(self, config, short_transcription):
        clips_data = [
            {
                "title": "Exatamente 60s", "reason": "R", "viral_score": 8.0,
                "start_time": 0.0, "end_time": float(MAX_CLIP_DURATION),
                "description": "D", "hashtags": [], "thumbnail_idea": "T",
                "thumbnail_text": "MOMENTO IMPACTANTE",
            }
        ]
        mock_client = _make_mock_client(clips_data)

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            result = analyze(short_transcription, config)

        assert len(result) == 1


class TestAnalyzeErrorHandling:
    def test_api_error_reraised_as_runtime_error(self, config, short_transcription):
        mock_request = MagicMock()
        api_error = real_anthropic.APIConnectionError(request=mock_request)

        mock_client = MagicMock()
        mock_client.with_options.return_value = mock_client
        mock_client.messages.create.side_effect = api_error

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            with pytest.raises(RuntimeError) as exc_info:
                analyze(short_transcription, config)

        assert "Claude" in str(exc_info.value) or "API" in str(exc_info.value)

    def test_runtime_error_message_is_friendly(self, config, short_transcription):
        mock_request = MagicMock()
        api_error = real_anthropic.APIConnectionError(request=mock_request)

        mock_client = MagicMock()
        mock_client.with_options.return_value = mock_client
        mock_client.messages.create.side_effect = api_error

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            with pytest.raises(RuntimeError) as exc_info:
                analyze(short_transcription, config)

        error_msg = str(exc_info.value)
        assert "transcrição" in error_msg or "API" in error_msg

    def test_original_error_chained(self, config, short_transcription):
        mock_request = MagicMock()
        api_error = real_anthropic.APIConnectionError(request=mock_request)

        mock_client = MagicMock()
        mock_client.with_options.return_value = mock_client
        mock_client.messages.create.side_effect = api_error

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            with pytest.raises(RuntimeError) as exc_info:
                analyze(short_transcription, config)

        assert exc_info.value.__cause__ is api_error


class TestAnalyzePromptContent:
    def test_prompt_contains_virality_criteria(self, config, short_transcription):
        """Verify SYSTEM_PROMPT contains all 5 PRD virality criteria."""
        assert "Gancho" in _SYSTEM_PROMPT or "gancho" in _SYSTEM_PROMPT
        assert "impactante" in _SYSTEM_PROMPT or "curiosidade" in _SYSTEM_PROMPT
        assert "prática" in _SYSTEM_PROMPT or "humor" in _SYSTEM_PROMPT or "emoção" in _SYSTEM_PROMPT
        assert "intensidade" in _SYSTEM_PROMPT or "alta intensidade" in _SYSTEM_PROMPT.lower()
        assert "isoladamente" in _SYSTEM_PROMPT or "contexto externo" in _SYSTEM_PROMPT

    def test_prompt_mentions_duration_rules(self, config, short_transcription):
        """Verify social-mode SYSTEM_PROMPT mentions 15-180s duration constraint."""
        assert "15" in _SYSTEM_PROMPT and "180" in _SYSTEM_PROMPT

    def test_transcription_passed_to_api(self, config, short_transcription):
        mock_client = _make_mock_client([])

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            analyze(short_transcription, config)

        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args.kwargs
        messages = call_kwargs.get("messages", [])
        user_content = messages[0]["content"]
        text_block = next(b for b in user_content if b["type"] == "text")
        # Transcription segments should appear in the prompt text
        assert "Este é um conteúdo incrível" in text_block["text"]

    def test_cache_control_applied_to_transcription(self, config, short_transcription):
        mock_client = _make_mock_client([])

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            analyze(short_transcription, config)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        messages = call_kwargs.get("messages", [])
        user_content = messages[0]["content"]
        text_block = next(b for b in user_content if b["type"] == "text")
        assert text_block.get("cache_control") == {"type": "ephemeral"}


class TestAnalyzeEdgeCases:
    def test_empty_transcription_returns_empty_list(self, config):
        empty = TranscriptionResult(
            segments=[], language="pt", source_path=Path("test.mp4")
        )
        with patch("youcut.analyzer.anthropic.Anthropic"):
            result = analyze(empty, config)

        assert result == []

    def test_empty_api_response_returns_empty_list(self, config, short_transcription):
        mock_client = _make_mock_client([])

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            result = analyze(short_transcription, config)

        assert result == []

    def test_invalid_clip_data_skipped_gracefully(self, config, short_transcription):
        clips_data = [
            {"title": "Válido", "reason": "R", "viral_score": 8.0,
             "start_time": 0.0, "end_time": 30.0, "description": "D",
             "hashtags": [], "thumbnail_idea": "T", "thumbnail_text": "MOMENTO IMPACTANTE"},
            {"title": "Inválido - falta campos"},  # missing required fields
        ]
        mock_client = _make_mock_client(clips_data)

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            result = analyze(short_transcription, config)

        assert len(result) == 1
        assert result[0].title == "Válido"


class TestAnalyzeChunking:
    def _make_long_transcription(self, minutes: int) -> TranscriptionResult:
        segments = []
        for i in range(minutes):
            start = float(i * 60)
            end = float((i + 1) * 60)
            segments.append(
                TranscriptionSegment(
                    start=start,
                    end=end,
                    text=f"Conteúdo do minuto {i + 1}",
                    words=[WordTimestamp(word="Conteúdo", start=start, end=start + 0.5)],
                )
            )
        return TranscriptionResult(
            segments=segments,
            language="pt",
            source_path=Path("long_video.mp4"),
        )

    def test_short_transcription_single_api_call(self, config):
        """Transcription ≤30 min should make exactly 1 API call."""
        short = self._make_long_transcription(20)
        mock_client = _make_mock_client([])

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            analyze(short, config)

        assert mock_client.messages.create.call_count == 1

    def test_long_transcription_multiple_api_calls(self, config):
        """Transcription >30 min should be chunked into multiple calls."""
        long = self._make_long_transcription(35)  # 35 minutes
        clip_data = {
            "title": "Clipe", "reason": "R", "viral_score": 8.0,
            "start_time": 0.0, "end_time": 30.0, "description": "D",
            "hashtags": [], "thumbnail_idea": "T", "thumbnail_text": "MOMENTO IMPACTANTE",
        }
        mock_client = _make_mock_client([clip_data])

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            result = analyze(long, config)

        assert mock_client.messages.create.call_count >= 2
        assert len(result) > 0

    def test_chunked_results_merged_and_sorted(self, config):
        """Results from multiple chunks are merged and sorted by viral_score."""
        long = self._make_long_transcription(35)

        call_count = 0
        chunk_responses = [
            [{"title": "C1", "reason": "R", "viral_score": 6.0,
              "start_time": 0.0, "end_time": 30.0, "description": "D",
              "hashtags": [], "thumbnail_idea": "T", "thumbnail_text": "MOMENTO IMPACTANTE"}],
            [{"title": "C2", "reason": "R", "viral_score": 9.0,
              "start_time": 1800.0, "end_time": 1830.0, "description": "D",
              "hashtags": [], "thumbnail_idea": "T", "thumbnail_text": "MOMENTO IMPACTANTE"}],
        ]

        def side_effect(**kwargs):
            nonlocal call_count
            data = chunk_responses[call_count] if call_count < len(chunk_responses) else []
            call_count += 1
            mock_block = MagicMock()
            mock_block.type = "tool_use"
            mock_block.name = "identify_viral_clips"
            mock_block.input = {"clips": data}
            resp = MagicMock()
            resp.content = [mock_block]
            return resp

        mock_client = MagicMock()
        mock_client.with_options.return_value = mock_client
        mock_client.messages.create.side_effect = side_effect

        with patch("youcut.analyzer.anthropic.Anthropic", return_value=mock_client):
            result = analyze(long, config)

        assert len(result) == 2
        assert result[0].viral_score == 9.0
        assert result[1].viral_score == 6.0
