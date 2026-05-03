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


def _detect_faces_with_mediapipe(frame, min_confidence: float) -> list[_BBox]:
    """Try MediaPipe face detection. Returns [] when the legacy ``solutions``
    API is unavailable (e.g. mediapipe>=0.10 on Python 3.13)."""
    try:
        import mediapipe as mp  # type: ignore[import]
        if not hasattr(mp, "solutions"):
            return []
        with mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=min_confidence,
        ) as detector:
            return _detect_faces_in_frame(frame, detector)
    except Exception:
        return []


def _detect_faces_with_opencv(frame) -> list[_BBox]:
    """Fallback face detection using OpenCV's bundled Haar cascade."""
    try:
        import cv2  # type: ignore[import]
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(str(cascade_path))
        if cascade.empty():
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))
        return [_BBox(x=int(x), y=int(y), w=int(w), h=int(h)) for (x, y, w, h) in rects]
    except Exception:
        return []


def detect_dominant_face_y_norm(
    clip_path: Path,
    *,
    sample_count: int = 8,
    min_confidence: float = 0.5,
) -> float | None:
    """Sample frames from *clip_path* and return the average normalized vertical
    center (0..1) of the dominant face across detections.

    "Dominant" = the largest-area face per frame, which on talking-head sources
    is reliably the speaker. Returns ``None`` when neither MediaPipe nor OpenCV
    can be used, the video can't be opened, or no face is detected in any sample.

    Used by the social composer to anchor the bottom-panel crop on the speaker's
    face instead of doing a naive centre crop.
    """
    try:
        import cv2  # type: ignore[import]
    except ImportError as exc:
        logger.info("Face anchor: OpenCV ausente (%s)", exc)
        return None

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        logger.info("Face anchor: não foi possível abrir %s", clip_path)
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None

    sample_count = max(1, sample_count)
    positions = [int(total * (i + 0.5) / sample_count) for i in range(sample_count)]

    samples: list[tuple[float, int]] = []  # (y_norm, area)
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok:
            continue
        faces = _detect_faces_with_mediapipe(frame, min_confidence)
        if not faces:
            faces = _detect_faces_with_opencv(frame)
        if not faces:
            continue
        h_frame = frame.shape[0]
        if h_frame <= 0:
            continue
        dominant = max(faces, key=lambda b: b.w * b.h)
        y_center_norm = (dominant.y + dominant.h / 2.0) / h_frame
        samples.append((max(0.0, min(1.0, y_center_norm)), dominant.w * dominant.h))

    cap.release()

    if not samples:
        logger.info("Face anchor: nenhum rosto detectado em %s", clip_path.name)
        return None

    # Drop detections with area below 50% of the largest sample to filter out
    # Haar false positives on buttons / chair fabric while keeping real faces.
    max_area = max(area for _, area in samples)
    filtered = [y for y, area in samples if area >= max_area * 0.5]
    if not filtered:
        filtered = [y for y, _ in samples]

    # Use the median: robust to remaining outliers, no averaging artifacts.
    filtered.sort()
    mid = len(filtered) // 2
    median = filtered[mid] if len(filtered) % 2 else (filtered[mid - 1] + filtered[mid]) / 2.0

    logger.info(
        "Face anchor: rosto dominante em y_norm=%.3f (n=%d/%d) para %s",
        median, len(filtered), len(samples), clip_path.name,
    )
    return median


# ---------------------------------------------------------------------------
# Public entry point (stub — rendering added in Task 4.0)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Face-aware framing for editorial social layout (bottom panel)
# ---------------------------------------------------------------------------

# Faces with bbox dimension below this many pixels are dropped as likely
# false positives (background heads, watermarks, posters).
_MIN_FACE_DIM_PX = 40

# Padding applied around the aggregate bounding box of all valid faces.
# Vertical is larger than horizontal so the framing keeps headroom above hair.
_FRAMING_HORIZONTAL_PADDING = 0.10
_FRAMING_VERTICAL_PADDING = 0.20

# Cluster face centres into horizontal bins of this width (pixels) when
# aggregating across the clip. Bins capture distinct speaker positions.
_CLUSTER_BIN_PX = 50

# A cluster (bin) is kept only if it holds at least this fraction of the total
# detections. Drops transient false positives (posters, someone walking by)
# without dropping a real but less-detected speaker — unlike a global
# percentile clip, this is robust to detection imbalance between speakers.
_CLUSTER_MIN_FRACTION = 0.03


def _compute_panel_crop_for_faces(
    faces: list[_BBox],
    *,
    frame_w: int,
    frame_h: int,
    target_w: int,
    target_h: int,
) -> CropRegion:
    """Compute a crop with target_w:target_h aspect ratio that contains all
    *faces* (with padding) and is centred on their union bounding box.

    Falls back to a centred crop when *faces* is empty.
    """
    target_ratio = target_w / target_h

    if not faces:
        if frame_w / frame_h >= target_ratio:
            crop_h = frame_h
            crop_w = int(round(crop_h * target_ratio))
        else:
            crop_w = frame_w
            crop_h = int(round(crop_w / target_ratio))
        x = max(0, (frame_w - crop_w) // 2)
        y = max(0, (frame_h - crop_h) // 2)
        return CropRegion(x=x, y=y, w=max(1, crop_w), h=max(1, crop_h))

    x_min = min(f.x for f in faces)
    y_min = min(f.y for f in faces)
    x_max = max(f.x + f.w for f in faces)
    y_max = max(f.y + f.h for f in faces)
    union_w = max(1, x_max - x_min)
    union_h = max(1, y_max - y_min)

    pad_x = int(union_w * _FRAMING_HORIZONTAL_PADDING)
    pad_y = int(union_h * _FRAMING_VERTICAL_PADDING)
    desired_w = union_w + 2 * pad_x
    desired_h = union_h + 2 * pad_y

    desired_ratio = desired_w / desired_h
    if desired_ratio < target_ratio:
        desired_w = int(round(desired_h * target_ratio))
    else:
        desired_h = int(round(desired_w / target_ratio))

    if desired_w > frame_w:
        desired_w = frame_w
        desired_h = int(round(desired_w / target_ratio))
    if desired_h > frame_h:
        desired_h = frame_h
        desired_w = int(round(desired_h * target_ratio))

    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    x = max(0, min(cx - desired_w // 2, frame_w - desired_w))
    y = max(0, min(cy - desired_h // 2, frame_h - desired_h))

    return CropRegion(x=x, y=y, w=max(1, desired_w), h=max(1, desired_h))


def _ffmpeg_center_crop(
    src: Path, dst: Path, *, target_w: int, target_h: int,
) -> Path:
    """Re-encode *src* into *dst* using a static centred crop at target dims."""
    import subprocess

    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-vf", f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h}",
        "-c:v", "libx264",
        "-c:a", "aac",
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return dst


def frame_for_panel(
    clip_path: Path,
    *,
    target_w: int,
    target_h: int,
    config: PipelineConfig,
) -> Path:
    """Re-frame *clip_path* into a target_w×target_h video using face-aware crop.

    Tries speaker-aware scene cuts first (lip-movement-based, in
    ``youcut.speaker_framing``); falls back to a single static aggregate
    crop, then to a centred re-encode if no faces are found or any
    dependency is missing.
    """
    output_path = clip_path.with_name(clip_path.stem + "_framed.mp4")
    try:
        import cv2  # type: ignore[import]  # noqa: F401
    except ImportError as exc:
        logger.info("Face framing: OpenCV ausente (%s) — center crop", exc)
        return _ffmpeg_center_crop(clip_path, output_path, target_w=target_w, target_h=target_h)

    try:
        from youcut.speaker_framing import frame_with_speaker_scenes
        scene_result = frame_with_speaker_scenes(
            clip_path, output_path,
            target_w=target_w, target_h=target_h, config=config,
        )
        if scene_result is not None:
            return scene_result
    except Exception as exc:
        logger.warning("Speaker framing falhou (%s) — usando crop estático", exc)

    try:
        region = _analyze_clip_for_static_crop(
            clip_path,
            target_w=target_w,
            target_h=target_h,
            config=config,
        )
    except Exception as exc:
        logger.warning("Face framing: análise falhou (%s) — center crop", exc)
        return _ffmpeg_center_crop(clip_path, output_path, target_w=target_w, target_h=target_h)

    if region is None:
        logger.info(
            "Face framing: nenhum rosto consistente em %s — center crop",
            clip_path.name,
        )
        return _ffmpeg_center_crop(
            clip_path, output_path, target_w=target_w, target_h=target_h,
        )

    try:
        return _ffmpeg_static_crop(
            clip_path, output_path,
            region=region, target_w=target_w, target_h=target_h,
        )
    except Exception as exc:
        logger.warning("Face framing: render estático falhou (%s) — center crop", exc)
        return _ffmpeg_center_crop(clip_path, output_path, target_w=target_w, target_h=target_h)


def _analyze_clip_for_static_crop(
    clip_path: Path,
    *,
    target_w: int,
    target_h: int,
    config: PipelineConfig,
) -> CropRegion | None:
    """Scan *clip_path* and return a single CropRegion that contains the faces
    seen across the whole clip, or ``None`` when no faces are detected.

    Uses 5th/95th percentile of detection coordinates to be robust to a small
    number of spurious detections.
    """
    import cv2  # type: ignore[import]
    import time

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {clip_path}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_w <= 0 or frame_h <= 0:
        cap.release()
        raise RuntimeError(f"Dimensões inválidas para {clip_path}: {frame_w}x{frame_h}")

    detector = None
    try:
        detector = _make_face_detector(config.face_detection_confidence)
        use_mediapipe = True
    except Exception as exc:
        logger.info("Face framing: MediaPipe indisponível (%s) — usando OpenCV", exc)
        use_mediapipe = False

    t0 = time.time()
    all_faces: list[_BBox] = []
    frames_with_faces = 0
    total_frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if use_mediapipe and detector is not None:
            try:
                faces = _detect_faces_in_frame(frame, detector)
            except Exception:
                faces = _detect_faces_with_opencv(frame)
        else:
            faces = _detect_faces_with_opencv(frame)

        valid = [f for f in faces if f.w >= _MIN_FACE_DIM_PX and f.h >= _MIN_FACE_DIM_PX]
        if valid:
            all_faces.extend(valid)
            frames_with_faces += 1
        total_frames += 1

    cap.release()
    if detector is not None:
        try:
            detector.close()
        except Exception:
            pass

    if not all_faces:
        return None

    aggregate = _aggregate_face_bbox(all_faces)
    if aggregate is None:
        return None

    region = _compute_panel_crop_for_faces(
        [aggregate],
        frame_w=frame_w, frame_h=frame_h,
        target_w=target_w, target_h=target_h,
    )

    elapsed = time.time() - t0
    logger.info(
        "Face framing: análise em %.1fs — %d detecções em %d/%d frames; "
        "bbox agregada %dx%d em (%d,%d) → crop %dx%d em (%d,%d)",
        elapsed, len(all_faces), frames_with_faces, total_frames,
        aggregate.w, aggregate.h, aggregate.x, aggregate.y,
        region.w, region.h, region.x, region.y,
    )
    return region


def _aggregate_face_bbox(faces: list[_BBox]) -> _BBox | None:
    """Return a bbox covering all *consistent* face clusters.

    Bins faces by horizontal centre and keeps only bins that hold at least
    ``_CLUSTER_MIN_FRACTION`` of detections. The bbox is then the min/max over
    every face in every accepted cluster — so a speaker who looks down or
    away half the time still stays in the frame, while a poster on the wall
    detected for 10 frames does not.
    """
    if not faces:
        return None

    bins: dict[int, list[_BBox]] = {}
    for f in faces:
        cx = f.x + f.w // 2
        bin_idx = cx // _CLUSTER_BIN_PX
        bins.setdefault(bin_idx, []).append(f)

    threshold = max(1, int(len(faces) * _CLUSTER_MIN_FRACTION))
    accepted: list[_BBox] = []
    for bucket in bins.values():
        if len(bucket) >= threshold:
            accepted.extend(bucket)

    if not accepted:
        accepted = faces

    x_min = min(f.x for f in accepted)
    y_min = min(f.y for f in accepted)
    x_max = max(f.x + f.w for f in accepted)
    y_max = max(f.y + f.h for f in accepted)

    if x_max <= x_min or y_max <= y_min:
        return None

    return _BBox(x=x_min, y=y_min, w=x_max - x_min, h=y_max - y_min)


def _ffmpeg_static_crop(
    src: Path,
    dst: Path,
    *,
    region: CropRegion,
    target_w: int,
    target_h: int,
) -> Path:
    """Render *src* into *dst* applying a single static crop+scale via ffmpeg."""
    import subprocess

    vf = (
        f"crop={region.w}:{region.h}:{region.x}:{region.y},"
        f"scale={target_w}:{target_h}"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-vf", vf,
        "-c:v", "libx264",
        "-c:a", "aac",
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return dst


# ---------------------------------------------------------------------------
# Public entry point (legacy split-screen pipeline)
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
