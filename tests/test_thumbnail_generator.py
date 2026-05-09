import base64
import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from youcut.models import ThumbnailFrameResult, ViralClip
from youcut.thumbnail_generator import (
    _build_thumbnail_clip_context,
    _build_thumbnail_generation_prompt,
    _build_transcript_visual_context,
    _compose_text_overlay,
    _extract_transcript_visual_elements_fallback,
    _extract_frames_candidates,
    _generate_thumbnail_via_ai,
    _generate_thumbnail_via_ai_result,
    _load_transcript_excerpt,
    _resize_to_youtube_format,
    _resolve_image_cost_profile,
    _resolve_image_request_params,
    _run_thumbnail_skill_script,
    _select_supporting_reference_frames_via_ai_result,
    _select_generation_reference_frames,
    _select_best_face_frame,
    _select_best_frame_via_ai,
    _select_best_frame_via_ai_result,
    _build_thumbnail_result,
    generate_thumbnail,
)


def _make_clip(thumbnail_text: str = "MOMENTO IMPACTANTE") -> ViralClip:
    return ViralClip(
        title="Momento Viral Incrível",
        reason="High energy",
        viral_score=9.0,
        start_time=60.0,
        end_time=660.0,
        description="The host explains the main topic.",
        hashtags=["#youtube"],
        thumbnail_idea="Host explaining excitedly with charts in background",
        thumbnail_text=thumbnail_text,
        cut_mode="youtube",
    )


def _make_png(width: int = 640, height: int = 480, color: tuple[int, int, int] = (100, 150, 200)) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_completed_process(stdout: bytes | str, returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["ffmpeg"], returncode=returncode, stdout=stdout, stderr=b"")


def _make_mediapipe_detection(score: float, xmin: float, ymin: float, width: float, height: float):
    det = MagicMock()
    det.score = [score]
    bb = MagicMock()
    bb.xmin = xmin
    bb.ymin = ymin
    bb.width = width
    bb.height = height
    det.location_data.relative_bounding_box = bb
    return det


def _make_cv2_mock():
    cv2 = MagicMock()
    cv2.COLOR_BGR2RGB = 4
    cv2.IMREAD_COLOR = 1
    fake_frame = MagicMock()
    fake_frame.shape = (720, 1280, 3)
    cv2.imdecode.return_value = fake_frame
    cv2.cvtColor.return_value = fake_frame
    return cv2


def _make_numpy_mock():
    np = MagicMock()
    np.frombuffer.return_value = MagicMock()
    np.uint8 = MagicMock()
    return np


def _import_raising_for_mediapipe(name, *args, **kwargs):
    if name == "mediapipe":
        raise ImportError("No module named 'mediapipe'")
    return __import__(name, *args, **kwargs)


def test_generate_thumbnail_returns_correct_path(tmp_path):
    clip = _make_clip()
    fake_mp4 = tmp_path / "clip.mp4"
    fake_mp4.write_bytes(b"fake")

    result_model = ThumbnailFrameResult(
        frame_timestamp=1.0,
        frame_score=0.9,
        segmentation_applied=False,
        output_path=tmp_path / "thumbnails" / "clip_03.png",
    )

    with patch("youcut.thumbnail_generator._build_thumbnail_result", return_value=result_model):
        result = generate_thumbnail(clip, tmp_path, clip_index=3, clip_path=fake_mp4)

    assert result == tmp_path / "thumbnails" / "clip_03.png"


def test_generate_thumbnail_creates_thumbnails_dir(tmp_path):
    clip = _make_clip()
    fake_mp4 = tmp_path / "clip.mp4"
    fake_mp4.write_bytes(b"fake")

    result_model = ThumbnailFrameResult(
        frame_timestamp=1.0,
        frame_score=0.9,
        segmentation_applied=False,
        output_path=tmp_path / "thumbnails" / "clip_01.png",
    )

    with patch("youcut.thumbnail_generator._build_thumbnail_result", return_value=result_model):
        generate_thumbnail(clip, tmp_path, clip_index=1, clip_path=fake_mp4)

    assert (tmp_path / "thumbnails").exists()


def test_generate_thumbnail_raises_when_clip_path_missing(tmp_path):
    clip = _make_clip()
    with pytest.raises(ValueError):
        generate_thumbnail(clip, tmp_path, clip_index=0, clip_path=None)


def test_build_thumbnail_result_without_config_uses_local_path(tmp_path):
    clip = _make_clip("")
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"fake")
    output_path = tmp_path / "thumb.png"
    frame_bytes = _make_png()
    thumbnail_bytes = _make_png(1280, 720)

    with (
        patch("youcut.thumbnail_generator._extract_frames_candidates", return_value=[(1.5, frame_bytes)]),
        patch("youcut.thumbnail_generator._build_ai_clients", return_value=(None, None)),
        patch("youcut.thumbnail_generator._select_best_local_candidate", return_value=(1.5, frame_bytes, 0.42)),
        patch("youcut.thumbnail_generator._generate_thumbnail_via_ai_result", return_value=(thumbnail_bytes, "local")),
    ):
        result = _build_thumbnail_result(clip, clip_path, output_path, None)

    assert result.selection_method == "local"
    assert result.generation_method == "local"
    assert output_path.exists()


def test_build_thumbnail_result_with_config_calls_ai_stages(tmp_path):
    clip = _make_clip("")
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"fake")
    output_path = tmp_path / "thumb.png"
    frame_bytes = _make_png()
    thumbnail_bytes = _make_png(1280, 720)

    ai_selection = patch("youcut.thumbnail_generator._select_best_frame_via_ai_result", return_value=(2.5, frame_bytes, "ai", 1.0))
    ai_generation = patch("youcut.thumbnail_generator._generate_thumbnail_via_ai_result", return_value=(thumbnail_bytes, "ai"))
    with (
        patch("youcut.thumbnail_generator._extract_frames_candidates", return_value=[(2.5, frame_bytes)]),
        patch("youcut.thumbnail_generator._build_ai_clients", return_value=(object(), SimpleNamespace(api_key="test-key"))),
        ai_selection as mock_selection,
        ai_generation as mock_generation,
    ):
        result = _build_thumbnail_result(clip, clip_path, output_path, SimpleNamespace())

    mock_selection.assert_called_once()
    mock_generation.assert_called_once()
    assert result.selection_method == "ai"
    assert result.generation_method == "ai"


def test_build_thumbnail_result_skips_local_text_overlay_when_ai_generation_succeeds(tmp_path):
    clip = _make_clip("LEGADO MBL")
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"fake")
    output_path = tmp_path / "thumb.png"
    frame_bytes = _make_png()
    thumbnail_bytes = _make_png(1280, 720)

    with (
        patch("youcut.thumbnail_generator._extract_frames_candidates", return_value=[(2.5, frame_bytes)]),
        patch("youcut.thumbnail_generator._build_ai_clients", return_value=(object(), SimpleNamespace(api_key="test-key"))),
        patch("youcut.thumbnail_generator._select_best_frame_via_ai_result", return_value=(2.5, frame_bytes, "ai", 1.0)),
        patch("youcut.thumbnail_generator._generate_thumbnail_via_ai_result", return_value=(thumbnail_bytes, "ai")),
        patch("youcut.thumbnail_generator._compose_text_overlay") as mock_overlay,
    ):
        result = _build_thumbnail_result(clip, clip_path, output_path, SimpleNamespace())

    mock_overlay.assert_not_called()
    assert result.generation_method == "ai"
    assert output_path.exists()


def test_build_thumbnail_result_methods_populated(tmp_path):
    clip = _make_clip("")
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"fake")
    output_path = tmp_path / "thumb.png"
    frame_bytes = _make_png()
    thumbnail_bytes = _make_png(1280, 720)

    with (
        patch("youcut.thumbnail_generator._extract_frames_candidates", return_value=[(3.0, frame_bytes)]),
        patch("youcut.thumbnail_generator._build_ai_clients", return_value=(None, None)),
        patch("youcut.thumbnail_generator._select_best_local_candidate", return_value=(3.0, frame_bytes, 0.33)),
        patch("youcut.thumbnail_generator._generate_thumbnail_via_ai_result", return_value=(thumbnail_bytes, "local")),
    ):
        result = _build_thumbnail_result(clip, clip_path, output_path, None)

    assert result.selection_method in {"ai", "local"}
    assert result.generation_method in {"ai", "local"}


def test_compose_text_overlay_empty_returns_original(caplog):
    img = Image.new("RGB", (1280, 720), color=(50, 100, 150))
    original_bytes = img.tobytes()

    with caplog.at_level("DEBUG"):
        result = _compose_text_overlay(img, "")

    assert result.tobytes() == original_bytes
    assert "thumbnail_text vazio" in caplog.text


def test_compose_text_overlay_short_text_within_7_percent():
    img = Image.new("RGB", (1280, 720), color=(50, 100, 150))
    result = _compose_text_overlay(img, "CHOQUE")
    assert result.size == img.size
    assert result.tobytes() != img.tobytes()


def test_compose_text_overlay_long_text_reduces_font(caplog):
    img = Image.new("RGB", (1280, 720), color=(50, 100, 150))
    long_title = "UMA FRASE MUITO LONGA COM MUITAS PALAVRAS PARA FORCAR AJUSTE DE FONTE NA THUMBNAIL"

    with caplog.at_level("WARNING"):
        result = _compose_text_overlay(img, long_title)

    assert result.size == (1280, 720)
    assert "original_size" in caplog.text
    assert "adjusted_size" in caplog.text


def test_apply_frame_processing_returns_pil_image():
    frame_bytes = _make_png(640, 480)
    from youcut.thumbnail_generator import _apply_frame_processing

    result = _apply_frame_processing(frame_bytes)
    assert isinstance(result, Image.Image)


def test_resize_to_youtube_format_produces_correct_dimensions(tmp_path):
    img = Image.new("RGB", (1792, 1024), color=(255, 0, 0))
    img_path = tmp_path / "thumb.png"
    img.save(img_path, format="PNG")

    _resize_to_youtube_format(img_path)

    with Image.open(img_path) as result:
        assert result.size == (1280, 720)


def test_extract_frames_candidates_returns_correct_count(tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake")
    frame_png = _make_png(1600, 900)

    def run_side_effect(cmd, **kwargs):
        if cmd[0] == "ffprobe":
            return _make_completed_process(json.dumps({"format": {"duration": "60.0"}}))
        return _make_completed_process(frame_png)

    with patch("youcut.thumbnail_generator.subprocess.run", side_effect=run_side_effect):
        result = _extract_frames_candidates(video_path, n_frames=10)

    assert len(result) == 10


def test_extract_frames_candidates_timestamps_within_range(tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake")
    frame_png = _make_png()

    def run_side_effect(cmd, **kwargs):
        if cmd[0] == "ffprobe":
            return _make_completed_process(json.dumps({"format": {"duration": "100.0"}}))
        return _make_completed_process(frame_png)

    with patch("youcut.thumbnail_generator.subprocess.run", side_effect=run_side_effect):
        result = _extract_frames_candidates(video_path, n_frames=5)

    assert all(5.0 <= ts <= 95.0 for ts, _ in result)


@pytest.mark.parametrize(("requested", "expected"), [(0, 5), (20, 15)])
def test_extract_frames_candidates_clamps_n_frames(tmp_path, requested, expected):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake")
    frame_png = _make_png()

    def run_side_effect(cmd, **kwargs):
        if cmd[0] == "ffprobe":
            return _make_completed_process(json.dumps({"format": {"duration": "40.0"}}))
        return _make_completed_process(frame_png)

    with patch("youcut.thumbnail_generator.subprocess.run", side_effect=run_side_effect):
        result = _extract_frames_candidates(video_path, n_frames=requested)

    assert len(result) == expected


def test_extract_frames_candidates_skips_failed_frame(tmp_path, caplog):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake")
    frame_png = _make_png()
    counter = {"frames": 0}

    def run_side_effect(cmd, **kwargs):
        if cmd[0] == "ffprobe":
            return _make_completed_process(json.dumps({"format": {"duration": "40.0"}}))
        counter["frames"] += 1
        if counter["frames"] == 3:
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd)
        return _make_completed_process(frame_png)

    with patch("youcut.thumbnail_generator.subprocess.run", side_effect=run_side_effect), caplog.at_level("WARNING"):
        result = _extract_frames_candidates(video_path, n_frames=5)

    assert len(result) == 4
    assert "Falha ao extrair frame candidato" in caplog.text


def test_extract_frames_candidates_raises_if_all_fail(tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake")

    def run_side_effect(cmd, **kwargs):
        if cmd[0] == "ffprobe":
            return _make_completed_process(json.dumps({"format": {"duration": "40.0"}}))
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd)

    with patch("youcut.thumbnail_generator.subprocess.run", side_effect=run_side_effect):
        with pytest.raises(RuntimeError):
            _extract_frames_candidates(video_path, n_frames=5)


def test_select_best_frame_via_ai_returns_correct_frame():
    frames = [(1.0, b"a"), (2.0, b"b"), (3.0, b"c")]
    response = SimpleNamespace(content=[SimpleNamespace(text='{"selected_index": 2, "reason": "bright face"}')])
    client = MagicMock()
    client.with_options.return_value = client
    client.messages.create.return_value = response

    result = _select_best_frame_via_ai(frames, "Teste", client)

    assert result == (3.0, b"c")


def test_select_best_frame_via_ai_fallback_on_invalid_json(caplog):
    frames = [(1.0, b"a"), (2.0, b"b")]
    response = SimpleNamespace(content=[SimpleNamespace(text="sem json")])
    client = MagicMock()
    client.with_options.return_value = client
    client.messages.create.return_value = response

    with patch("youcut.thumbnail_generator._select_best_local_candidate", return_value=(1.0, b"a", 0.2)) as mock_local, caplog.at_level("WARNING"):
        result = _select_best_frame_via_ai_result(frames, "Teste", client)

    mock_local.assert_called_once()
    assert result[2] == "local"
    assert "Falha na seleção por IA" in caplog.text


def test_select_best_frame_via_ai_fallback_on_timeout(caplog):
    frames = [(1.0, b"a"), (2.0, b"b")]
    client = MagicMock()
    client.with_options.return_value = client
    client.messages.create.side_effect = TimeoutError("timeout")

    with patch("youcut.thumbnail_generator._select_best_local_candidate", return_value=(1.0, b"a", 0.2)), caplog.at_level("WARNING"):
        result = _select_best_frame_via_ai_result(frames, "Teste", client)

    assert result[2] == "local"
    assert "timeout" in caplog.text


def test_select_best_frame_via_ai_fallback_on_api_error(caplog):
    frames = [(1.0, b"a"), (2.0, b"b")]
    client = MagicMock()
    client.with_options.return_value = client
    client.messages.create.side_effect = RuntimeError("api error")

    with patch("youcut.thumbnail_generator._select_best_local_candidate", return_value=(1.0, b"a", 0.2)), caplog.at_level("WARNING"):
        result = _select_best_frame_via_ai_result(frames, "Teste", client)

    assert result[2] == "local"
    assert "api error" in caplog.text


def test_select_best_frame_via_ai_logs_reason_on_success(caplog):
    frames = [(1.0, b"a"), (2.0, b"b")]
    response = SimpleNamespace(content=[SimpleNamespace(text='{"selected_index": 0, "reason": "bright face"}')])
    client = MagicMock()
    client.with_options.return_value = client
    client.messages.create.return_value = response

    with caplog.at_level("INFO"):
        result = _select_best_frame_via_ai_result(frames, "Teste", client)

    assert result[2] == "ai"
    assert "bright face" in caplog.text


def test_generate_thumbnail_via_ai_returns_correct_size():
    input_png = _make_png(640, 360)
    client = SimpleNamespace(api_key="test-key")

    with patch("youcut.thumbnail_generator._run_thumbnail_skill_script", return_value=_make_png(1536, 864)):
        result = _generate_thumbnail_via_ai(input_png, "Teste", client)

    with Image.open(io.BytesIO(result)) as img:
        assert img.size == (1280, 720)


def test_build_thumbnail_generation_prompt_uses_doc_context():
    with patch(
        "youcut.thumbnail_generator._load_thumbnail_prompt_context",
        return_value="Skill guidance: safe-zone. PRD guidance: no embedded text by default.",
    ):
        prompt = _build_thumbnail_generation_prompt("Teste")

    assert "safe-zone" in prompt
    assert "no embedded text by default" in prompt
    assert "Video title context: Teste." in prompt


def test_build_thumbnail_generation_prompt_embeds_text_rules_when_requested():
    with patch(
        "youcut.thumbnail_generator._load_thumbnail_prompt_context",
        return_value="PRD guidance: text area max 7%.",
    ):
        prompt = _build_thumbnail_generation_prompt(
            "Teste",
            thumbnail_text="LEGADO DO MBL",
            clip_context="Debate sobre estrategia digital, MBL e publico jovem.",
            transcript_visual_context="referência sutil a eleição e campanha | referência sutil a segurança pública",
        )

    assert 'Embed exactly this thumbnail text inside the generated image: "LEGADO DO MBL".' in prompt
    assert 'Render the secondary text in bright yellow (#FFD54A): "LEGADO DO".' in prompt
    assert 'Render the hero text larger in vivid orange (#FF8A00): "MBL".' in prompt
    assert "Prefer bright yellow and vivid orange text accents" in prompt
    assert "at most 7% of the total image area" in prompt
    assert "Do not add any text, captions, words" not in prompt
    assert "If the reference frames show more than one relevant person" in prompt
    assert "Use additional reference frames to bring in complementary people" in prompt
    assert "The people and their expressions must remain the main focus of the thumbnail." in prompt
    assert "Clip thematic context: Debate sobre estrategia digital, MBL e publico jovem." in prompt
    assert "Transcript-derived subtle visual motifs: referência sutil a eleição e campanha | referência sutil a segurança pública." in prompt


def test_build_thumbnail_clip_context_uses_clip_metadata():
    clip = _make_clip("LEGADO DO MBL")

    context = _build_thumbnail_clip_context(clip)

    assert "Host explaining excitedly" in context
    assert "High energy" in context
    assert "The host explains the main topic." in context
    assert "#youtube" in context


def test_load_transcript_excerpt_reads_sidecar_cache(tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake")
    transcript_path = tmp_path / "clip_transcript.json"
    transcript_path.write_text(
        json.dumps({
            "result": {
                "segments": [
                    {"text": "Primeiro trecho do debate."},
                    {"text": "Segundo trecho com seguranca publica."},
                ]
            }
        }),
        encoding="utf-8",
    )

    excerpt = _load_transcript_excerpt(video_path)

    assert "Primeiro trecho do debate." in excerpt
    assert "Segundo trecho com seguranca publica." in excerpt


def test_extract_transcript_visual_elements_fallback_maps_themes():
    excerpt = "Tema de eleicoes, campanha, seguranca publica e redes sociais entre candidatos."

    elements = _extract_transcript_visual_elements_fallback(excerpt)

    assert "referência sutil a eleição e campanha" in elements
    assert "referência sutil a segurança pública" in elements
    assert "referência sutil a mídia e ambiente digital" in elements


def test_build_transcript_visual_context_uses_fallback_without_ai(tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake")
    transcript_path = tmp_path / "clip_transcript.json"
    transcript_path.write_text(
        json.dumps({
            "result": {
                "segments": [
                    {"text": "Discussao sobre eleicoes e seguranca publica."},
                ]
            }
        }),
        encoding="utf-8",
    )

    context = _build_transcript_visual_context(video_path, "Teste", None)

    assert "referência sutil a eleição e campanha" in context


def test_generate_thumbnail_via_ai_fallback_when_no_client(caplog):
    input_png = _make_png(640, 360)

    with patch("youcut.thumbnail_generator._render_local_thumbnail_bytes", return_value=_make_png(1280, 720)) as mock_local, caplog.at_level("WARNING"):
        result, method = _generate_thumbnail_via_ai_result(input_png, "Teste", None)

    mock_local.assert_called_once()
    assert method == "local"
    assert result
    assert "Cliente OpenAI ausente" in caplog.text


def test_generate_thumbnail_via_ai_fallback_on_timeout(caplog):
    input_png = _make_png(640, 360)
    client = SimpleNamespace(api_key="test-key")

    with (
        patch("youcut.thumbnail_generator._run_thumbnail_skill_script", side_effect=TimeoutError("timeout")),
        patch("youcut.thumbnail_generator._render_local_thumbnail_bytes", return_value=_make_png(1280, 720)),
        caplog.at_level("WARNING"),
    ):
        _, method = _generate_thumbnail_via_ai_result(input_png, "Teste", client)

    assert method == "local"
    assert "timeout" in caplog.text


def test_generate_thumbnail_via_ai_fallback_on_http_error(caplog):
    input_png = _make_png(640, 360)
    client = SimpleNamespace(api_key="test-key")

    with (
        patch("youcut.thumbnail_generator._run_thumbnail_skill_script", side_effect=RuntimeError("429")),
        patch("youcut.thumbnail_generator._render_local_thumbnail_bytes", return_value=_make_png(1280, 720)),
        caplog.at_level("WARNING"),
    ):
        _, method = _generate_thumbnail_via_ai_result(input_png, "Teste", client)

    assert method == "local"
    assert "429" in caplog.text


def test_generate_thumbnail_via_ai_logs_method_ai_on_success(caplog):
    input_png = _make_png(640, 360)
    client = SimpleNamespace(api_key="test-key")

    with patch("youcut.thumbnail_generator._run_thumbnail_skill_script", return_value=_make_png(1536, 864)), caplog.at_level("INFO"):
        _, method = _generate_thumbnail_via_ai_result(input_png, "Teste", client)

    assert method == "ai"
    assert "method=ai" in caplog.text


def test_generate_thumbnail_via_ai_uses_built_prompt():
    input_png = _make_png(640, 360)
    client = SimpleNamespace(api_key="test-key")

    with patch(
        "youcut.thumbnail_generator._build_thumbnail_generation_prompt",
        return_value="PROMPT_FROM_DOCS",
    ), patch(
        "youcut.thumbnail_generator._run_thumbnail_skill_script",
        return_value=_make_png(1536, 864),
    ) as mock_skill:
        _generate_thumbnail_via_ai_result(input_png, "Teste", client)

    assert mock_skill.call_args.kwargs["prompt"] == "PROMPT_FROM_DOCS"


def test_generate_thumbnail_via_ai_passes_thumbnail_text_to_prompt_builder():
    input_png = _make_png(640, 360)
    client = SimpleNamespace(api_key="test-key")

    with patch(
        "youcut.thumbnail_generator._build_thumbnail_generation_prompt",
        return_value="PROMPT_FROM_DOCS",
    ) as mock_prompt, patch(
        "youcut.thumbnail_generator._run_thumbnail_skill_script",
        return_value=_make_png(1536, 864),
    ):
        _generate_thumbnail_via_ai_result(input_png, "Teste", client, thumbnail_text="LEGADO MBL")

    assert mock_prompt.call_args.kwargs["thumbnail_text"] == "LEGADO MBL"


def test_select_supporting_reference_frames_via_ai_returns_principal_plus_supporting():
    frames = [(0.0, b"a"), (1.0, b"b"), (2.0, b"c"), (3.0, b"d")]
    response = SimpleNamespace(content=[SimpleNamespace(text='{"supporting_indices": [3, 0], "reason": "second guest and wider shot"}')])
    client = MagicMock()
    client.with_options.return_value = client
    client.messages.create.return_value = response

    result = _select_supporting_reference_frames_via_ai_result(
        frames,
        "Teste",
        selected_timestamp=1.0,
        anthropic_client=client,
        clip_context="Debate entre duas pessoas",
        max_frames=4,
    )

    assert result == [b"b", b"d", b"a"]


def test_select_supporting_reference_frames_via_ai_falls_back_to_neighbors():
    frames = [(float(index), bytes([index])) for index in range(5)]
    client = MagicMock()
    client.with_options.return_value = client
    client.messages.create.side_effect = RuntimeError("api error")

    result = _select_supporting_reference_frames_via_ai_result(
        frames,
        "Teste",
        selected_timestamp=2.0,
        anthropic_client=client,
        max_frames=4,
    )

    assert result == [b"\x02", b"\x01", b"\x03", b"\x00"]


def test_select_generation_reference_frames_prefers_neighbors():
    frames = [(float(index), bytes([index])) for index in range(10)]

    result = _select_generation_reference_frames(frames, selected_timestamp=4.0, max_frames=5)

    assert result == [b"\x04", b"\x03", b"\x05", b"\x02", b"\x06"]


@patch("youcut.thumbnail_generator._extract_frame")
def test_select_best_face_frame_falls_back_when_mediapipe_not_installed(mock_extract_fallback, tmp_path):
    mock_extract_fallback.return_value = b"fallback_no_mp"
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake")

    with patch("builtins.__import__", side_effect=_import_raising_for_mediapipe):
        result = _select_best_face_frame(video_path)

    assert result == b"fallback_no_mp"


@patch("youcut.thumbnail_generator._extract_frames_candidates")
def test_select_best_face_frame_returns_highest_score_frame(mock_candidates, tmp_path):
    import sys

    fake_mp = MagicMock()
    detector_instance = MagicMock()

    def make_result(detections):
        result = MagicMock()
        result.detections = detections
        return result

    good_det = _make_mediapipe_detection(score=0.95, xmin=0.35, ymin=0.25, width=0.30, height=0.50)
    detector_instance.process.side_effect = [
        make_result([]),
        make_result([good_det]),
    ]
    fake_mp.solutions.face_detection.FaceDetection.return_value = detector_instance
    mock_candidates.return_value = [(1.0, b"a"), (2.0, b"b")]

    with patch.dict(sys.modules, {"mediapipe": fake_mp, "cv2": _make_cv2_mock(), "numpy": _make_numpy_mock()}):
        result = _select_best_face_frame(tmp_path / "clip.mp4", n_samples=2)

    assert result == b"b"


# ────────────────────────────────────────────────────────────────────────────────
# Task 1.0 — perfil de custo + wrapper subprocess parametrizado
# ────────────────────────────────────────────────────────────────────────────────


def _make_config(cut_mode: str = "social"):
    return SimpleNamespace(cut_mode=cut_mode)


def test_resolve_image_cost_profile_social():
    assert _resolve_image_cost_profile(_make_config("social")) == "social"


def test_resolve_image_cost_profile_youtube():
    assert _resolve_image_cost_profile(_make_config("youtube")) == "standard"


def test_resolve_image_cost_profile_none_config():
    assert _resolve_image_cost_profile(None) == "standard"


def test_resolve_image_request_params_standard():
    assert _resolve_image_request_params("standard") == ("gpt-image-1.5", "1536x1024", "low")


def test_resolve_image_request_params_social():
    assert _resolve_image_request_params("social") == ("gpt-image-1.5", "1024x1024", "low")


def _capture_run_args(captured: dict, output_bytes: bytes):
    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = kwargs.get("cwd")
        # Simula a saída do script: cria thumbnails/<output_name>.png na cwd
        output_name = cmd[3]
        thumbnails_dir = Path(captured["cwd"]) / "thumbnails"
        thumbnails_dir.mkdir(parents=True, exist_ok=True)
        (thumbnails_dir / output_name).write_bytes(output_bytes)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
    return fake_run


def test_run_skill_script_passes_size_arg_for_social(tmp_path):
    captured: dict = {}
    fake_run = _capture_run_args(captured, _make_png(8, 8))
    with patch("youcut.thumbnail_generator.subprocess.run", side_effect=fake_run):
        result = _run_thumbnail_skill_script(
            prompt="hello",
            reference_frames=[_make_png(8, 8)],
            openai_api_key="sk-test",
            timeout=5.0,
            config=_make_config("social"),
        )
    assert isinstance(result, bytes) and len(result) > 0
    cmd = captured["cmd"]
    assert "--size" in cmd
    assert cmd[cmd.index("--size") + 1] == "1024x1024"
    assert cmd[cmd.index("--quality") + 1] == "low"
    assert cmd[cmd.index("--model") + 1] == "gpt-image-1.5"


def test_run_skill_script_passes_size_arg_for_youtube(tmp_path):
    captured: dict = {}
    fake_run = _capture_run_args(captured, _make_png(8, 8))
    with patch("youcut.thumbnail_generator.subprocess.run", side_effect=fake_run):
        _run_thumbnail_skill_script(
            prompt="hello",
            reference_frames=[_make_png(8, 8)],
            openai_api_key="sk-test",
            timeout=5.0,
            config=_make_config("youtube"),
        )
    cmd = captured["cmd"]
    assert cmd[cmd.index("--size") + 1] == "1536x1024"


def test_run_skill_script_no_config_keeps_legacy_args(tmp_path):
    captured: dict = {}
    fake_run = _capture_run_args(captured, _make_png(8, 8))
    with patch("youcut.thumbnail_generator.subprocess.run", side_effect=fake_run):
        _run_thumbnail_skill_script(
            prompt="hello",
            reference_frames=[_make_png(8, 8)],
            openai_api_key="sk-test",
            timeout=5.0,
        )
    cmd = captured["cmd"]
    assert cmd[cmd.index("--size") + 1] == "1536x1024"
    assert cmd[cmd.index("--model") + 1] == "gpt-image-1.5"
    assert cmd[cmd.index("--quality") + 1] == "low"


def test_run_skill_script_emits_info_log(caplog):
    captured: dict = {}
    fake_run = _capture_run_args(captured, _make_png(8, 8))
    with patch("youcut.thumbnail_generator.subprocess.run", side_effect=fake_run), caplog.at_level("INFO"):
        _run_thumbnail_skill_script(
            prompt="hi",
            reference_frames=[_make_png(8, 8)],
            openai_api_key="sk-test",
            timeout=5.0,
            config=_make_config("social"),
        )
    assert "profile=social" in caplog.text
    assert "size=1024x1024" in caplog.text
    assert "model=gpt-image-1.5" in caplog.text


def test_run_skill_script_raises_when_no_api_key():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        _run_thumbnail_skill_script(
            prompt="x",
            reference_frames=[_make_png(8, 8)],
            openai_api_key=None,
            timeout=5.0,
        )


def test_run_skill_script_raises_when_no_frames():
    with pytest.raises(ValueError, match="frame de referência"):
        _run_thumbnail_skill_script(
            prompt="x",
            reference_frames=[],
            openai_api_key="sk-test",
            timeout=5.0,
        )


# ────────────────────────────────────────────────────────────────────────────────
# Task 2.0 — propagação de config nos callers + cut_mode opcional + regressão
# ────────────────────────────────────────────────────────────────────────────────


def _social_clip() -> ViralClip:
    return ViralClip(
        title="Hook viral",
        reason="energy",
        viral_score=8.0,
        start_time=0.0,
        end_time=30.0,
        description="d",
        hashtags=["#x"],
        thumbnail_idea="t",
        thumbnail_text="HOOK",
        cut_mode="social",
    )


def test_generate_thumbnail_via_ai_result_propagates_config_social():
    captured: dict = {}

    def fake_skill(*, prompt, reference_frames, openai_api_key, timeout, config=None):
        captured["config"] = config
        return _make_png(1536, 864)

    with patch("youcut.thumbnail_generator._run_thumbnail_skill_script", side_effect=fake_skill):
        cfg = _make_config("social")
        _generate_thumbnail_via_ai_result(
            frame_bytes=_make_png(8, 8),
            title="t",
            openai_client=MagicMock(),
            reference_frames=[_make_png(8, 8)],
            openai_api_key="sk-test",
            config=cfg,
        )
    assert captured["config"] is cfg


def test_generate_thumbnail_via_ai_result_no_config_keeps_legacy():
    captured: dict = {}

    def fake_skill(*, prompt, reference_frames, openai_api_key, timeout, config=None):
        captured["config"] = config
        return _make_png(1536, 864)

    with patch("youcut.thumbnail_generator._run_thumbnail_skill_script", side_effect=fake_skill):
        _generate_thumbnail_via_ai_result(
            frame_bytes=_make_png(8, 8),
            title="t",
            openai_client=MagicMock(),
            reference_frames=[_make_png(8, 8)],
            openai_api_key="sk-test",
        )
    assert captured["config"] is None


def test_apply_cut_mode_override_no_override_returns_same_config():
    from youcut.thumbnail_generator import _apply_cut_mode_override
    cfg = _make_config("youtube")
    assert _apply_cut_mode_override(cfg, None) is cfg


def test_apply_cut_mode_override_creates_namespace_when_no_config():
    from youcut.thumbnail_generator import _apply_cut_mode_override
    result = _apply_cut_mode_override(None, "social")
    assert getattr(result, "cut_mode", None) == "social"


def test_apply_cut_mode_override_uses_model_copy_when_available():
    from youcut.thumbnail_generator import _apply_cut_mode_override

    class FakePipelineConfig:
        def __init__(self, cut_mode, other="kept"):
            self.cut_mode = cut_mode
            self.other = other

        def model_copy(self, *, update):
            new_cm = update.get("cut_mode", self.cut_mode)
            return FakePipelineConfig(cut_mode=new_cm, other=self.other)

    cfg = FakePipelineConfig(cut_mode="youtube")
    result = _apply_cut_mode_override(cfg, "social")
    assert result is not cfg
    assert result.cut_mode == "social"
    assert result.other == "kept"


def test_run_skill_script_arch_invariant_not_imported_in_comic():
    """RF4: pipeline `comic` continua independente do skill script.

    Garante que nenhum módulo em `youcut/comic/**` importa
    `_run_thumbnail_skill_script` por engano (Tech Spec §Riscos item 5).
    """
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    comic_dir = repo_root / "youcut" / "comic"
    forbidden = "_run_thumbnail_skill_script"
    offenders: list[str] = []
    for py in comic_dir.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        if forbidden in text:
            offenders.append(str(py.relative_to(repo_root)))
    assert offenders == [], (
        f"Pipeline `comic` não pode depender de {forbidden} (RF4 do PRD). "
        f"Encontrado em: {offenders}"
    )
