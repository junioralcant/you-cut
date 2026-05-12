"""Speaker-aware framing for the editorial bottom panel.

Uses MediaPipe FaceLandmarker (478 landmarks) to track each speaker's lip
aperture across the clip, then classifies who is speaking per frame, builds
"scenes" of consecutive frames sharing the same active set, and renders each
scene with its own static crop. Output is concatenated into a single clip with
editor-style cuts between speakers — no moving zoom, no per-frame jitter.

When detection fails or only one speaker is identified the public entry returns
``None`` so the caller can fall back to the static aggregate crop.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from youcut.config import PipelineConfig
from youcut.face_tracker import (
    _BBox,
    _CLUSTER_BIN_PX,
    _CLUSTER_MIN_FRACTION,
    _MIN_FACE_DIM_PX,
    _compute_panel_crop_for_faces,
)
from youcut.models import CropRegion


@dataclass(frozen=True)
class SpeakerScene:
    """Cena de fala: intervalo [start, end) em segundos no clip source, com os
    bboxes (pixel-space original) dos clusters ATIVOS nesse intervalo."""
    start_s: float
    end_s: float
    active_bboxes: tuple[_BBox, ...]


@dataclass(frozen=True)
class SpeakerScenesAnalysis:
    """Saída da análise de fala. ``frame_w``/``frame_h`` são as dimensões do
    source clip; ``scenes`` cobrem a duração total sem buracos."""
    frame_w: int
    frame_h: int
    fps: float
    cluster_bboxes: tuple[_BBox, ...]
    scenes: tuple[SpeakerScene, ...]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Cached FaceLandmarker model file (~3.7 MB, downloaded once).
_MODEL_DIR = Path.home() / ".youcut" / "models"
_FACE_LANDMARKER_MODEL_PATH = _MODEL_DIR / "face_landmarker.task"
_FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# MediaPipe lip landmark indices (478-point face mesh).
_LIP_UPPER_INNER = 13
_LIP_LOWER_INNER = 14
_LIP_LEFT_CORNER = 78
_LIP_RIGHT_CORNER = 308

# Minimum scene duration. Shorter activity bursts are merged into the
# previous scene so the cut never feels neurotic.
_MIN_SCENE_DURATION_S = 1.2

# How far (px) a detected face centre may be from a cluster centre and still
# be assigned to that cluster.
_CLUSTER_ASSIGN_TOLERANCE_PX = 200

# Lip-delta smoothing window (seconds).
_LIP_SMOOTHING_WINDOW_S = 0.5


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class _PerFrameFace(NamedTuple):
    bbox: _BBox
    lip_aperture: float


class _Cluster(NamedTuple):
    cluster_id: int
    bbox: _BBox
    cx: int
    cy: int


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def frame_with_speaker_scenes(
    clip_path: Path,
    output_path: Path,
    *,
    target_w: int,
    target_h: int,
    config: PipelineConfig,
) -> Path | None:
    """Re-frame *clip_path* with speaker-aware scene cuts.

    Returns the rendered file path on success, or ``None`` when speaker
    detection cannot run (no model, no MediaPipe Tasks API, single speaker
    only, no faces detected).
    """
    model_path = _ensure_face_landmarker_model()
    if model_path is None:
        return None

    try:
        face_data = _collect_face_mesh_data(clip_path, model_path, config)
    except Exception as exc:
        logger.info("Speaker framing: análise FaceLandmarker falhou (%s)", exc)
        return None

    if face_data is None:
        return None

    frame_w, frame_h, fps, frames = face_data
    if frame_w <= 0 or frame_h <= 0 or not frames:
        return None

    all_bboxes = [face.bbox for per_frame in frames for face in per_frame]
    clusters = _identify_clusters_from_bboxes(all_bboxes)

    if len(clusters) < 2:
        logger.info(
            "Speaker framing: %d cluster(s) — fallback estático",
            len(clusters),
        )
        return None

    series = _per_cluster_aperture_series(frames, clusters)
    smoothed = _smoothed_lip_deltas_per_cluster(series, fps)
    active_per_frame = _classify_active_per_frame(smoothed)

    min_scene_frames = max(1, int(round(fps * _MIN_SCENE_DURATION_S)))
    raw_scenes = _build_scenes_from_activity(active_per_frame, min_scene_frames)
    if not raw_scenes:
        return None

    scenes_with_crop: list[tuple[int, int, CropRegion]] = []
    for start_frame, end_frame, active_ids in raw_scenes:
        if active_ids:
            chosen = [c for c in clusters if c.cluster_id in active_ids]
        else:
            chosen = clusters
        bboxes = [c.bbox for c in chosen]
        crop = _compute_panel_crop_for_faces(
            bboxes,
            frame_w=frame_w, frame_h=frame_h,
            target_w=target_w, target_h=target_h,
        )
        scenes_with_crop.append((start_frame, end_frame, crop))

    logger.info(
        "Speaker framing: %d cluster(s), %d cena(s) construídas",
        len(clusters), len(scenes_with_crop),
    )

    return _render_scenes(
        clip_path, output_path, scenes_with_crop, fps,
        target_w=target_w, target_h=target_h,
    )


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

def analyze_speaker_scenes(
    clip_path: Path,
    config: PipelineConfig,
) -> SpeakerScenesAnalysis | None:
    """Roda só a etapa de **análise** do speaker framing — sem renderizar.

    Retorna metadata (cluster bboxes em pixel-space do source + lista de cenas
    com clusters ativos por intervalo). ``None`` quando o pipeline não pode
    rodar (model ausente, MediaPipe quebrado, < 2 clusters, sem rostos).
    """
    model_path = _ensure_face_landmarker_model()
    if model_path is None:
        return None

    try:
        face_data = _collect_face_mesh_data(clip_path, model_path, config)
    except Exception as exc:
        logger.info("analyze_speaker_scenes: análise FaceLandmarker falhou (%s)", exc)
        return None
    if face_data is None:
        return None

    frame_w, frame_h, fps, frames = face_data
    if frame_w <= 0 or frame_h <= 0 or not frames or fps <= 0:
        return None

    all_bboxes = [face.bbox for per_frame in frames for face in per_frame]
    clusters = _identify_clusters_from_bboxes(all_bboxes)
    if len(clusters) < 2:
        return None

    series = _per_cluster_aperture_series(frames, clusters)
    smoothed = _smoothed_lip_deltas_per_cluster(series, fps)
    active_per_frame = _classify_active_per_frame(smoothed)

    min_scene_frames = max(1, int(round(fps * _MIN_SCENE_DURATION_S)))
    raw_scenes = _build_scenes_from_activity(active_per_frame, min_scene_frames)
    if not raw_scenes:
        return None

    cluster_by_id = {c.cluster_id: c for c in clusters}
    scenes: list[SpeakerScene] = []
    for start_frame, end_frame, active_ids in raw_scenes:
        bboxes = tuple(
            cluster_by_id[cid].bbox
            for cid in active_ids
            if cid in cluster_by_id
        )
        scenes.append(
            SpeakerScene(
                start_s=start_frame / fps,
                end_s=end_frame / fps,
                active_bboxes=bboxes,
            )
        )

    return SpeakerScenesAnalysis(
        frame_w=frame_w,
        frame_h=frame_h,
        fps=fps,
        cluster_bboxes=tuple(c.bbox for c in clusters),
        scenes=tuple(scenes),
    )


def _ensure_face_landmarker_model() -> Path | None:
    if _FACE_LANDMARKER_MODEL_PATH.exists():
        return _FACE_LANDMARKER_MODEL_PATH

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Baixando modelo FaceLandmarker (uma única vez)...")
        urllib.request.urlretrieve(
            _FACE_LANDMARKER_MODEL_URL, _FACE_LANDMARKER_MODEL_PATH,
        )
        logger.info(
            "Modelo FaceLandmarker em cache: %s",
            _FACE_LANDMARKER_MODEL_PATH,
        )
        return _FACE_LANDMARKER_MODEL_PATH
    except Exception as exc:
        logger.warning("Falha ao baixar modelo FaceLandmarker: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Frame-level data collection (impure: opens video + runs MediaPipe)
# ---------------------------------------------------------------------------

def _collect_face_mesh_data(
    clip_path: Path,
    model_path: Path,
    config: PipelineConfig,
) -> tuple[int, int, float, list[list[_PerFrameFace]]] | None:
    import cv2  # type: ignore[import]

    try:
        from mediapipe import Image, ImageFormat  # type: ignore[import]
        from mediapipe.tasks import python as mp_python  # type: ignore[import]
        from mediapipe.tasks.python import vision  # type: ignore[import]
    except ImportError as exc:
        logger.info("Speaker framing: MediaPipe Tasks indisponível (%s)", exc)
        return None

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_w <= 0 or frame_h <= 0:
        cap.release()
        return None

    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=4,
        min_face_detection_confidence=config.face_detection_confidence,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        landmarker = vision.FaceLandmarker.create_from_options(options)
    except Exception as exc:
        cap.release()
        logger.info("Speaker framing: falha ao criar FaceLandmarker (%s)", exc)
        return None

    frames: list[list[_PerFrameFace]] = []
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
            ts_ms = int(round(frame_idx * 1000 / fps))

            try:
                result = landmarker.detect_for_video(mp_image, ts_ms)
            except Exception:
                frames.append([])
                frame_idx += 1
                continue

            per_frame: list[_PerFrameFace] = []
            if result.face_landmarks:
                for face_landmarks in result.face_landmarks:
                    face = _build_face_from_landmarks(face_landmarks, frame_w, frame_h)
                    if face is not None:
                        per_frame.append(face)

            frames.append(per_frame)
            frame_idx += 1
    finally:
        cap.release()
        try:
            landmarker.close()
        except Exception:
            pass

    return frame_w, frame_h, fps, frames


def _build_face_from_landmarks(
    landmarks, frame_w: int, frame_h: int,
) -> _PerFrameFace | None:
    if len(landmarks) < 470:
        return None

    xs = [int(lm.x * frame_w) for lm in landmarks]
    ys = [int(lm.y * frame_h) for lm in landmarks]
    x_min = max(0, min(xs))
    x_max = min(frame_w, max(xs))
    y_min = max(0, min(ys))
    y_max = min(frame_h, max(ys))
    bw = x_max - x_min
    bh = y_max - y_min
    if bw < _MIN_FACE_DIM_PX or bh < _MIN_FACE_DIM_PX:
        return None
    bbox = _BBox(x=x_min, y=y_min, w=bw, h=bh)

    upper = landmarks[_LIP_UPPER_INNER]
    lower = landmarks[_LIP_LOWER_INNER]
    left = landmarks[_LIP_LEFT_CORNER]
    right = landmarks[_LIP_RIGHT_CORNER]

    lip_dy = abs(upper.y - lower.y) * frame_h
    mouth_w = abs(right.x - left.x) * frame_w
    aperture = lip_dy / mouth_w if mouth_w > 1.0 else 0.0

    return _PerFrameFace(bbox=bbox, lip_aperture=aperture)


# ---------------------------------------------------------------------------
# Pure analysis helpers (testable)
# ---------------------------------------------------------------------------

def _identify_clusters_from_bboxes(all_bboxes: list[_BBox]) -> list[_Cluster]:
    """Bin face centres horizontally and merge contiguous bins with enough
    mass (≥ ``_CLUSTER_MIN_FRACTION`` of total) into clusters.

    Returns clusters sorted left-to-right with re-numbered IDs.
    """
    if not all_bboxes:
        return []

    bins: dict[int, list[_BBox]] = {}
    for b in all_bboxes:
        cx = b.x + b.w // 2
        bin_idx = cx // _CLUSTER_BIN_PX
        bins.setdefault(bin_idx, []).append(b)

    threshold = max(1, int(len(all_bboxes) * _CLUSTER_MIN_FRACTION))
    accepted = sorted(b for b, bboxes in bins.items() if len(bboxes) >= threshold)
    if not accepted:
        return []

    groups: list[list[int]] = [[accepted[0]]]
    for b in accepted[1:]:
        if b - groups[-1][-1] <= 2:
            groups[-1].append(b)
        else:
            groups.append([b])

    raw_clusters: list[_Cluster] = []
    for group in groups:
        cluster_bboxes: list[_BBox] = []
        for bin_idx in group:
            cluster_bboxes.extend(bins[bin_idx])
        x_min = min(bb.x for bb in cluster_bboxes)
        y_min = min(bb.y for bb in cluster_bboxes)
        x_max = max(bb.x + bb.w for bb in cluster_bboxes)
        y_max = max(bb.y + bb.h for bb in cluster_bboxes)
        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2
        raw_clusters.append(_Cluster(
            cluster_id=-1,
            bbox=_BBox(x=x_min, y=y_min, w=x_max - x_min, h=y_max - y_min),
            cx=cx, cy=cy,
        ))

    raw_clusters.sort(key=lambda c: c.cx)
    return [c._replace(cluster_id=i) for i, c in enumerate(raw_clusters)]


def _per_cluster_aperture_series(
    frames: list[list[_PerFrameFace]],
    clusters: list[_Cluster],
) -> dict[int, list[float | None]]:
    """For each cluster, return a per-frame list of lip apertures.

    A frame entry is ``None`` when no face was detected near that cluster.
    If multiple faces map to the same cluster on a frame the larger aperture
    wins (which conservatively reflects activity).
    """
    n_frames = len(frames)
    series: dict[int, list[float | None]] = {
        c.cluster_id: [None] * n_frames for c in clusters
    }
    if not clusters:
        return series

    for frame_idx, per_frame in enumerate(frames):
        for face in per_frame:
            cx = face.bbox.x + face.bbox.w // 2
            best = min(clusters, key=lambda c: abs(c.cx - cx))
            if abs(best.cx - cx) > _CLUSTER_ASSIGN_TOLERANCE_PX:
                continue
            existing = series[best.cluster_id][frame_idx]
            if existing is None or face.lip_aperture > existing:
                series[best.cluster_id][frame_idx] = face.lip_aperture

    return series


def _smoothed_lip_deltas_per_cluster(
    series: dict[int, list[float | None]],
    fps: float,
) -> dict[int, list[float]]:
    """Convert each cluster's aperture series into a smoothed |Δaperture|
    series. Frames without a detection contribute zero delta.
    """
    window = max(1, int(round(fps * _LIP_SMOOTHING_WINDOW_S)))
    out: dict[int, list[float]] = {}
    for cid, apertures in series.items():
        deltas: list[float] = []
        prev: float | None = None
        for v in apertures:
            if v is None or prev is None:
                deltas.append(0.0)
            else:
                deltas.append(abs(v - prev))
            prev = v
        out[cid] = _moving_average(deltas, window)
    return out


def _moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or not values:
        return list(values)
    n = len(values)
    cumsum = [0.0] * (n + 1)
    for i, v in enumerate(values):
        cumsum[i + 1] = cumsum[i] + v
    half = window // 2
    out = [0.0] * n
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        out[i] = (cumsum[b] - cumsum[a]) / (b - a)
    return out


def _classify_active_per_frame(
    smoothed: dict[int, list[float]],
) -> list[list[int]]:
    """Per cluster, mark a frame as 'speaking' when its smoothed delta is
    above the cluster's own median of non-zero deltas. Returns the list of
    active cluster IDs per frame.
    """
    if not smoothed:
        return []

    n_frames = len(next(iter(smoothed.values())))

    thresholds: dict[int, float] = {}
    for cid, vals in smoothed.items():
        non_zero = sorted(v for v in vals if v > 1e-5)
        if non_zero:
            thresholds[cid] = _median(non_zero)
        else:
            thresholds[cid] = float("inf")

    active_per_frame: list[list[int]] = []
    for i in range(n_frames):
        active = sorted(
            cid for cid, vals in smoothed.items()
            if vals[i] > thresholds[cid]
        )
        active_per_frame.append(active)

    return active_per_frame


def _median(sorted_values: list[float]) -> float:
    n = len(sorted_values)
    if n == 0:
        raise ValueError("median of empty list")
    if n % 2 == 1:
        return sorted_values[n // 2]
    return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2.0


def _build_scenes_from_activity(
    active_per_frame: list[list[int]],
    min_scene_frames: int,
) -> list[tuple[int, int, list[int]]]:
    """Group consecutive frames sharing an active set into scenes; merge runs
    shorter than ``min_scene_frames`` into the previous scene to keep cuts
    deliberate.
    """
    if not active_per_frame:
        return []

    runs: list[tuple[int, int, tuple[int, ...]]] = []
    current = tuple(active_per_frame[0])
    start = 0
    for i in range(1, len(active_per_frame)):
        active = tuple(active_per_frame[i])
        if active != current:
            runs.append((start, i, current))
            current = active
            start = i
    runs.append((start, len(active_per_frame), current))

    merged: list[tuple[int, int, tuple[int, ...]]] = []
    for run in runs:
        s, e, active = run
        if merged and (e - s) < min_scene_frames:
            # Short transient → swallow into the previous scene without
            # changing its active set.
            prev_s, _, prev_active = merged[-1]
            merged[-1] = (prev_s, e, prev_active)
        elif merged and merged[-1][2] == active:
            # Same active set as the previous scene (e.g. after a short run
            # got absorbed) → coalesce.
            prev_s, _, prev_active = merged[-1]
            merged[-1] = (prev_s, e, prev_active)
        else:
            merged.append(run)

    return [(s, e, list(active)) for s, e, active in merged]


# ---------------------------------------------------------------------------
# Rendering (impure: ffmpeg)
# ---------------------------------------------------------------------------

def _render_scenes(
    clip_path: Path,
    output_path: Path,
    scenes: list[tuple[int, int, CropRegion]],
    fps: float,
    *,
    target_w: int,
    target_h: int,
) -> Path:
    if not scenes:
        raise RuntimeError("Speaker framing: lista de cenas vazia")

    if len(scenes) == 1:
        start_frame, end_frame, crop = scenes[0]
        return _render_single_scene(
            clip_path, output_path, start_frame, end_frame, crop, fps,
            target_w=target_w, target_h=target_h,
        )

    temp_dir = Path(tempfile.mkdtemp(prefix="youcut_scenes_"))
    try:
        scene_paths: list[Path] = []
        for idx, (start_frame, end_frame, crop) in enumerate(scenes):
            scene_path = temp_dir / f"scene_{idx:04d}.mp4"
            _render_single_scene(
                clip_path, scene_path, start_frame, end_frame, crop, fps,
                target_w=target_w, target_h=target_h,
            )
            scene_paths.append(scene_path)

        concat_list = temp_dir / "concat.txt"
        with concat_list.open("w") as f:
            for sp in scene_paths:
                f.write(f"file '{sp.resolve()}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _render_single_scene(
    clip_path: Path,
    output_path: Path,
    start_frame: int,
    end_frame: int,
    crop: CropRegion,
    fps: float,
    *,
    target_w: int,
    target_h: int,
) -> Path:
    start_time = start_frame / fps
    duration = (end_frame - start_frame) / fps
    if duration <= 0:
        raise RuntimeError(
            f"Speaker framing: duração inválida (start={start_frame}, end={end_frame})",
        )

    vf = (
        f"crop={crop.w}:{crop.h}:{crop.x}:{crop.y},"
        f"scale={target_w}:{target_h}"
    )
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-i", str(clip_path),
        "-t", str(duration),
        "-vf", vf,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-c:a", "aac",
        "-ar", "48000",
        "-ac", "2",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path
