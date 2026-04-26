"""Face tracking module — detects active speaker's face and crops video accordingly."""

import logging
from pathlib import Path
from typing import NamedTuple

from youcut.config import PipelineConfig
from youcut.models import CropRegion, SpeakerSegment

logger = logging.getLogger(__name__)

# Output canvas for split-screen mode (9:16 portrait)
_SPLIT_OUTPUT_W = 1080
_SPLIT_OUTPUT_H = 1920
_SPLIT_HALF_H = _SPLIT_OUTPUT_H // 2  # 960

# Vertical padding applied around detected face bounding box (fraction of bbox height)
_FACE_PADDING_VERTICAL = 0.40

# Aspect ratio for single-speaker crop: 9 wide, 16 tall
_ASPECT_W = 9
_ASPECT_H = 16


class _BBox(NamedTuple):
    x: int
    y: int
    w: int
    h: int


# ---------------------------------------------------------------------------
# ROI computation helpers
# ---------------------------------------------------------------------------

def _compute_single_speaker_roi(bbox: _BBox, frame_w: int, frame_h: int) -> CropRegion:
    """Compute a 9:16 crop region centred on *bbox* within a *frame_w*×*frame_h* frame.

    Applies 40% vertical padding around the face, then expands/shrinks to fit
    the 9:16 aspect ratio. Never upscales beyond the source resolution.
    """
    pad_v = int(bbox.h * _FACE_PADDING_VERTICAL)
    face_cx = bbox.x + bbox.w // 2
    face_cy = bbox.y + bbox.h // 2

    # Start from a padded height and derive width from 9:16 ratio
    crop_h = bbox.h + 2 * pad_v
    crop_w = crop_h * _ASPECT_W // _ASPECT_H

    # If derived width exceeds frame, constrain by width instead
    if crop_w > frame_w:
        crop_w = frame_w
        crop_h = crop_w * _ASPECT_H // _ASPECT_W

    # Cap height at frame height too
    if crop_h > frame_h:
        crop_h = frame_h
        crop_w = crop_h * _ASPECT_W // _ASPECT_H

    # Centre the crop on the face
    x = face_cx - crop_w // 2
    y = face_cy - crop_h // 2

    # Clamp to frame boundaries
    x = max(0, min(x, frame_w - crop_w))
    y = max(0, min(y, frame_h - crop_h))
    crop_w = min(crop_w, frame_w)
    crop_h = min(crop_h, frame_h)

    return CropRegion(x=x, y=y, w=crop_w, h=crop_h)


def _build_split_screen_regions(
    box1: _BBox,
    box2: _BBox,
    frame_w: int,
    frame_h: int,
) -> tuple[CropRegion, CropRegion]:
    """Build two crop regions for a 50/50 vertical split-screen layout.

    The returned regions are independent crops in source-frame coordinates,
    each covering the respective speaker's face. The first region is
    displayed in the top half (y=0), the second in the bottom half.
    """
    def _half_crop(bbox: _BBox) -> CropRegion:
        # Each half is 1080×960; derive a 16:9 source crop that fills it
        target_w = _SPLIT_OUTPUT_W
        target_h = _SPLIT_HALF_H  # 960
        # Desired source aspect ratio to fill the half: 16:9 horizontally
        # (target_w / target_h = 1080/960 ≈ 9:8; we crop source to match that)
        crop_h = bbox.h + int(bbox.h * _FACE_PADDING_VERTICAL * 2)
        crop_w = crop_h * target_w // target_h
        if crop_w > frame_w:
            crop_w = frame_w
            crop_h = crop_w * target_h // target_w
        if crop_h > frame_h:
            crop_h = frame_h
            crop_w = crop_h * target_w // target_h
        cx = bbox.x + bbox.w // 2
        cy = bbox.y + bbox.h // 2
        x = max(0, min(cx - crop_w // 2, frame_w - crop_w))
        y = max(0, min(cy - crop_h // 2, frame_h - crop_h))
        return CropRegion(x=x, y=y, w=max(1, crop_w), h=max(1, crop_h))

    return _half_crop(box1), _half_crop(box2)


# ---------------------------------------------------------------------------
# Speaker ↔ face association
# ---------------------------------------------------------------------------

def _active_speakers_at(segments: list[SpeakerSegment], timestamp: float) -> list[str]:
    """Return IDs of speakers active at *timestamp* (sorted for determinism)."""
    active = [s.speaker_id for s in segments if s.start <= timestamp <= s.end]
    return sorted(set(active))


def _assign_speakers_to_faces(
    faces: list[_BBox],
    speaker_segments: list[SpeakerSegment],
    timestamp: float,
    speaker_order: list[str] | None = None,
) -> dict[str, _BBox]:
    """Associate detected faces with active speakers at *timestamp*.

    Returns a mapping of {speaker_id: bbox}. With two faces and two speakers,
    the left-most face is assigned to the speaker whose ID comes first in
    *speaker_order* (determined by the first frame of simultaneous speech and
    passed in from the caller so order remains consistent across frames).

    When *speaker_order* is None, order is determined by speaker_id
    lexicographic sort.
    """
    active = _active_speakers_at(speaker_segments, timestamp)

    if not faces or not active:
        return {}

    if len(faces) == 1:
        return {active[0]: faces[0]}

    if len(faces) >= 2 and len(active) >= 2:
        order = speaker_order if speaker_order else active[:2]
        left, right = sorted(faces[:2], key=lambda b: b.x)
        return {order[0]: left, order[1]: right}

    # More faces than active speakers — use the face closest to centre
    if len(active) == 1:
        cx = sum(b.x + b.w // 2 for b in faces) / len(faces)
        closest = min(faces, key=lambda b: abs((b.x + b.w // 2) - cx))
        return {active[0]: closest}

    return {}


# ---------------------------------------------------------------------------
# Frame-level detection (MediaPipe wrapper)
# ---------------------------------------------------------------------------

def _detect_faces_in_frame(frame, detector) -> list[_BBox]:
    """Run MediaPipe face detection on a single BGR *frame* and return bboxes."""
    import cv2  # type: ignore[import]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.process(rgb)
    if not result.detections:
        return []

    h, w = frame.shape[:2]
    boxes: list[_BBox] = []
    for det in result.detections:
        bb = det.location_data.relative_bounding_box
        x = max(0, int(bb.xmin * w))
        y = max(0, int(bb.ymin * h))
        bw = min(int(bb.width * w), w - x)
        bh = min(int(bb.height * h), h - y)
        if bw > 0 and bh > 0:
            boxes.append(_BBox(x=x, y=y, w=bw, h=bh))
    return boxes


def _make_face_detector(min_confidence: float):
    """Instantiate a MediaPipe FaceDetection detector."""
    import mediapipe as mp  # type: ignore[import]
    return mp.solutions.face_detection.FaceDetection(
        min_detection_confidence=min_confidence,
    )


# ---------------------------------------------------------------------------
# Public entry point (stub — rendering added in Task 4.0)
# ---------------------------------------------------------------------------

def apply_face_tracking(clip_path: Path, config: PipelineConfig) -> Path:
    """Apply face tracking to *clip_path* and return the processed clip path.

    Returns *clip_path* unchanged (fallback) if face tracking fails or no
    faces are detected. Full rendering is implemented in Task 4.0; this stub
    validates the detection and ROI pipeline.
    """
    if not config.face_tracking:
        return clip_path

    try:
        import cv2  # type: ignore[import]
        import mediapipe  # noqa: F401  # type: ignore[import]
    except ImportError as exc:
        logger.warning("Face tracking: dependências ausentes (%s) — exportando clipe original.", exc)
        return clip_path

    try:
        return _run_face_tracking(clip_path, config)
    except Exception as exc:
        logger.warning("Face tracking falhou com erro: %s — exportando clipe original.", exc)
        return clip_path


def _run_face_tracking(clip_path: Path, config: PipelineConfig) -> Path:
    import cv2  # type: ignore[import]
    import time

    from youcut.diarizer import diarize

    speaker_segments = diarize(clip_path, config)

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {clip_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    t0 = time.time()
    detector = _make_face_detector(config.face_detection_confidence)
    cap = cv2.VideoCapture(str(clip_path))

    from youcut.models import FaceTrackingResult

    frame_regions: list[CropRegion | None] = []
    secondary_regions: list[CropRegion | None] = []
    is_split_screen: list[bool] = []
    had_faces = False
    speaker_order: list[str] | None = None
    last_region: CropRegion | None = None

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp = frame_idx / fps

        if frame_idx % 30 == 0:
            faces = _detect_faces_in_frame(frame, detector)
            logger.info("Face tracking: %d rostos detectados no frame %d", len(faces), frame_idx)
        else:
            faces = _detect_faces_in_frame(frame, detector)

        assignment = _assign_speakers_to_faces(faces, speaker_segments, timestamp, speaker_order)

        if len(assignment) == 2 and speaker_order is None:
            speaker_order = list(assignment.keys())

        if len(assignment) == 2:
            speakers = list(assignment.keys())
            box1 = assignment[speakers[0]]
            box2 = assignment[speakers[1]]
            roi1, roi2 = _build_split_screen_regions(box1, box2, frame_w, frame_h)
            frame_regions.append(roi1)
            secondary_regions.append(roi2)
            is_split_screen.append(True)
            last_region = roi1
            had_faces = True
        elif len(assignment) == 1:
            speaker_id = next(iter(assignment))
            bbox = assignment[speaker_id]
            roi = _compute_single_speaker_roi(bbox, frame_w, frame_h)
            frame_regions.append(roi)
            secondary_regions.append(None)
            is_split_screen.append(False)
            last_region = roi
            had_faces = True
        else:
            frame_regions.append(last_region)
            secondary_regions.append(None)
            is_split_screen.append(False)

        frame_idx += 1

    cap.release()
    detector.close()

    elapsed = time.time() - t0
    logger.info("Face tracking concluído em %.1fs para %s", elapsed, clip_path.name)

    if not had_faces:
        logger.warning("Face tracking: nenhum rosto detectado — usando fallback para clipe %s", clip_path.name)
        return clip_path

    result = FaceTrackingResult(
        frame_regions=frame_regions,
        had_faces=had_faces,
        is_split_screen=is_split_screen,
        secondary_regions=secondary_regions,
    )

    return _render_clip(clip_path, result, fps, frame_w, frame_h, config)


# ---------------------------------------------------------------------------
# EMA smoothing
# ---------------------------------------------------------------------------

def _smooth_regions(
    regions: list[CropRegion | None],
    alpha: float = 0.15,
) -> list[CropRegion]:
    """Apply Exponential Moving Average to crop region coordinates.

    None entries are replaced by the last valid region. If the list starts
    with None entries, they are filled forward from the first valid region.
    """
    if not regions:
        return []

    # Find first valid region to initialise EMA
    first_valid: CropRegion | None = None
    for r in regions:
        if r is not None:
            first_valid = r
            break

    if first_valid is None:
        # No valid regions at all — return zeros
        zero = CropRegion(x=0, y=0, w=1, h=1)
        return [zero] * len(regions)

    smoothed: list[CropRegion] = []
    ex = float(first_valid.x)
    ey = float(first_valid.y)
    ew = float(first_valid.w)
    eh = float(first_valid.h)

    for r in regions:
        if r is not None:
            ex = alpha * r.x + (1 - alpha) * ex
            ey = alpha * r.y + (1 - alpha) * ey
            ew = alpha * r.w + (1 - alpha) * ew
            eh = alpha * r.h + (1 - alpha) * eh
        smoothed.append(CropRegion(x=int(ex), y=int(ey), w=max(1, int(ew)), h=max(1, int(eh))))

    return smoothed


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_OUTPUT_W = 1080
_OUTPUT_H = 1920


def _crop_frame(frame, roi: CropRegion):
    """Crop *frame* to *roi* and resize to the standard output resolution."""
    import cv2  # type: ignore[import]

    h, w = frame.shape[:2]
    x = max(0, min(roi.x, w - 1))
    y = max(0, min(roi.y, h - 1))
    x2 = max(x + 1, min(roi.x + roi.w, w))
    y2 = max(y + 1, min(roi.y + roi.h, h))
    cropped = frame[y:y2, x:x2]
    return cv2.resize(cropped, (_OUTPUT_W, _OUTPUT_H), interpolation=cv2.INTER_LINEAR)


def _build_split_frame(frame, roi1: CropRegion, roi2: CropRegion):
    """Compose a split-screen frame from two source crop regions."""
    import cv2  # type: ignore[import]
    import numpy as np

    top = _crop_frame(frame, roi1)
    top = cv2.resize(top, (_OUTPUT_W, _SPLIT_HALF_H), interpolation=cv2.INTER_LINEAR)
    bottom = _crop_frame(frame, roi2)
    bottom = cv2.resize(bottom, (_OUTPUT_W, _SPLIT_HALF_H), interpolation=cv2.INTER_LINEAR)
    return np.vstack([top, bottom])


def _mux_audio(video_path: Path, audio_source: Path, output_path: Path) -> None:
    """Mux audio from *audio_source* into *video_path*, writing to *output_path*."""
    import subprocess

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_source),
            "-c:v", "copy",
            "-c:a", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )


def _render_clip(
    clip_path: Path,
    result,
    fps: float,
    frame_w: int,
    frame_h: int,
    config: PipelineConfig,
) -> Path:
    """Render a new video applying the smoothed crop regions and muxing the original audio."""
    import cv2  # type: ignore[import]
    import tempfile

    smoothed_primary = _smooth_regions(result.frame_regions)
    smoothed_secondary = _smooth_regions(
        [r for r in result.secondary_regions]
    ) if any(r is not None for r in result.secondary_regions) else []

    output_path = clip_path.with_name(clip_path.stem + "_tracked.mp4")
    tmp_video = Path(tempfile.mktemp(suffix=".mp4"))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_video), fourcc, fps, (_OUTPUT_W, _OUTPUT_H))

    cap = cv2.VideoCapture(str(clip_path))
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        is_split = result.is_split_screen[frame_idx] if frame_idx < len(result.is_split_screen) else False

        if is_split and smoothed_secondary and frame_idx < len(smoothed_secondary):
            roi1 = smoothed_primary[frame_idx]
            roi2 = smoothed_secondary[frame_idx]
            out_frame = _build_split_frame(frame, roi1, roi2)
        elif frame_idx < len(smoothed_primary):
            out_frame = _crop_frame(frame, smoothed_primary[frame_idx])
        else:
            out_frame = cv2.resize(frame, (_OUTPUT_W, _OUTPUT_H))

        writer.write(out_frame)
        frame_idx += 1

    cap.release()
    writer.release()

    try:
        _mux_audio(tmp_video, clip_path, output_path)
        tmp_video.unlink(missing_ok=True)
    except Exception as exc:
        logger.warning("Face tracking: falha no mux de áudio (%s) — usando vídeo sem áudio", exc)
        tmp_video.rename(output_path)

    logger.info("Face tracking: arquivo gerado em %s", output_path)
    return output_path
