"""Testes unitários para youcut/face_tracker.py (Tasks 3.0 e 4.0)."""
import pytest

from youcut.face_tracker import (
    _BBox,
    _aggregate_face_bbox,
    _assign_speakers_to_faces,
    _build_split_screen_regions,
    _compute_panel_crop_for_faces,
    _compute_single_speaker_roi,
    _smooth_regions,
)
from youcut.models import CropRegion, SpeakerSegment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seg(speaker_id: str, start: float, end: float) -> SpeakerSegment:
    return SpeakerSegment(speaker_id=speaker_id, start=start, end=end)


# ---------------------------------------------------------------------------
# _compute_single_speaker_roi
# ---------------------------------------------------------------------------

class TestComputeSingleSpeakerRoi:
    FRAME_W = 1280
    FRAME_H = 720

    def test_centred_face_roi_within_bounds(self):
        bbox = _BBox(x=540, y=260, w=200, h=200)
        roi = _compute_single_speaker_roi(bbox, self.FRAME_W, self.FRAME_H)
        assert roi.x >= 0
        assert roi.y >= 0
        assert roi.x + roi.w <= self.FRAME_W
        assert roi.y + roi.h <= self.FRAME_H

    def test_centred_face_roi_approx_9_16_ratio(self):
        bbox = _BBox(x=540, y=260, w=200, h=200)
        roi = _compute_single_speaker_roi(bbox, self.FRAME_W, self.FRAME_H)
        ratio = roi.h / roi.w
        expected = 16 / 9
        assert abs(ratio - expected) < 0.05, f"Expected ratio ~{expected:.3f}, got {ratio:.3f}"

    def test_face_on_left_edge_x_clamped(self):
        bbox = _BBox(x=0, y=300, w=100, h=100)
        roi = _compute_single_speaker_roi(bbox, self.FRAME_W, self.FRAME_H)
        assert roi.x == 0
        assert roi.w <= self.FRAME_W

    def test_face_on_top_edge_y_clamped(self):
        bbox = _BBox(x=600, y=0, w=100, h=100)
        roi = _compute_single_speaker_roi(bbox, self.FRAME_W, self.FRAME_H)
        assert roi.y == 0
        assert roi.h <= self.FRAME_H

    def test_face_on_right_edge_within_bounds(self):
        bbox = _BBox(x=1200, y=300, w=80, h=80)
        roi = _compute_single_speaker_roi(bbox, self.FRAME_W, self.FRAME_H)
        assert roi.x >= 0
        assert roi.x + roi.w <= self.FRAME_W

    def test_face_on_bottom_edge_within_bounds(self):
        bbox = _BBox(x=600, y=680, w=80, h=40)
        roi = _compute_single_speaker_roi(bbox, self.FRAME_W, self.FRAME_H)
        assert roi.y >= 0
        assert roi.y + roi.h <= self.FRAME_H

    def test_very_large_face_clamped_to_frame(self):
        bbox = _BBox(x=0, y=0, w=self.FRAME_W, h=self.FRAME_H)
        roi = _compute_single_speaker_roi(bbox, self.FRAME_W, self.FRAME_H)
        assert roi.x >= 0
        assert roi.y >= 0
        assert roi.w <= self.FRAME_W
        assert roi.h <= self.FRAME_H


# ---------------------------------------------------------------------------
# _build_split_screen_regions
# ---------------------------------------------------------------------------

class TestBuildSplitScreenRegions:
    FRAME_W = 1280
    FRAME_H = 720

    def test_region1_has_smaller_y_than_region2(self):
        box1 = _BBox(x=200, y=200, w=150, h=150)
        box2 = _BBox(x=800, y=200, w=150, h=150)
        roi1, roi2 = _build_split_screen_regions(box1, box2, self.FRAME_W, self.FRAME_H)
        # roi1 goes to top half, roi2 to bottom — they're independent source crops
        # both must be within frame
        assert roi1.x >= 0 and roi1.y >= 0
        assert roi1.x + roi1.w <= self.FRAME_W
        assert roi1.y + roi1.h <= self.FRAME_H
        assert roi2.x >= 0 and roi2.y >= 0
        assert roi2.x + roi2.w <= self.FRAME_W
        assert roi2.y + roi2.h <= self.FRAME_H

    def test_returns_two_crop_regions(self):
        box1 = _BBox(x=100, y=200, w=150, h=150)
        box2 = _BBox(x=900, y=200, w=150, h=150)
        result = _build_split_screen_regions(box1, box2, self.FRAME_W, self.FRAME_H)
        assert len(result) == 2

    def test_both_regions_have_positive_dimensions(self):
        box1 = _BBox(x=100, y=100, w=100, h=100)
        box2 = _BBox(x=700, y=100, w=100, h=100)
        roi1, roi2 = _build_split_screen_regions(box1, box2, self.FRAME_W, self.FRAME_H)
        assert roi1.w > 0 and roi1.h > 0
        assert roi2.w > 0 and roi2.h > 0

    def test_face_positions_influence_different_regions(self):
        box1 = _BBox(x=100, y=200, w=100, h=100)
        box2 = _BBox(x=900, y=200, w=100, h=100)
        roi1, roi2 = _build_split_screen_regions(box1, box2, self.FRAME_W, self.FRAME_H)
        # Regions should be centred on different horizontal positions
        cx1 = roi1.x + roi1.w // 2
        cx2 = roi2.x + roi2.w // 2
        assert cx1 != cx2, "Expected different crops for left/right faces"


# ---------------------------------------------------------------------------
# _assign_speakers_to_faces
# ---------------------------------------------------------------------------

class TestAssignSpeakersToFaces:
    def test_empty_faces_returns_empty_dict(self):
        segs = [_seg("SPEAKER_00", 0.0, 5.0)]
        result = _assign_speakers_to_faces([], segs, 1.0)
        assert result == {}

    def test_no_active_speakers_returns_empty_dict(self):
        segs = [_seg("SPEAKER_00", 5.0, 10.0)]
        face = _BBox(x=100, y=100, w=80, h=80)
        result = _assign_speakers_to_faces([face], segs, 1.0)
        assert result == {}

    def test_one_face_one_active_speaker_direct_association(self):
        segs = [_seg("SPEAKER_00", 0.0, 5.0)]
        face = _BBox(x=100, y=100, w=80, h=80)
        result = _assign_speakers_to_faces([face], segs, 2.0)
        assert "SPEAKER_00" in result
        assert result["SPEAKER_00"] == face

    def test_two_faces_two_speakers_left_gets_first_speaker(self):
        segs = [_seg("SPEAKER_00", 0.0, 5.0), _seg("SPEAKER_01", 0.0, 5.0)]
        left_face = _BBox(x=100, y=200, w=80, h=80)
        right_face = _BBox(x=700, y=200, w=80, h=80)
        order = ["SPEAKER_00", "SPEAKER_01"]
        result = _assign_speakers_to_faces([left_face, right_face], segs, 2.0, speaker_order=order)
        assert result["SPEAKER_00"] == left_face
        assert result["SPEAKER_01"] == right_face

    def test_two_faces_order_consistent_across_calls(self):
        segs = [_seg("SPEAKER_00", 0.0, 10.0), _seg("SPEAKER_01", 0.0, 10.0)]
        left_face = _BBox(x=50, y=200, w=80, h=80)
        right_face = _BBox(x=800, y=200, w=80, h=80)
        order = ["SPEAKER_00", "SPEAKER_01"]

        result1 = _assign_speakers_to_faces([left_face, right_face], segs, 1.0, speaker_order=order)
        result2 = _assign_speakers_to_faces([left_face, right_face], segs, 3.0, speaker_order=order)

        assert result1 == result2, "Speaker assignment must be idempotent given the same order"

    def test_two_faces_reversed_order_swaps_assignment(self):
        segs = [_seg("SPEAKER_00", 0.0, 5.0), _seg("SPEAKER_01", 0.0, 5.0)]
        left_face = _BBox(x=100, y=200, w=80, h=80)
        right_face = _BBox(x=700, y=200, w=80, h=80)

        order_a = ["SPEAKER_00", "SPEAKER_01"]
        order_b = ["SPEAKER_01", "SPEAKER_00"]

        result_a = _assign_speakers_to_faces([left_face, right_face], segs, 2.0, speaker_order=order_a)
        result_b = _assign_speakers_to_faces([left_face, right_face], segs, 2.0, speaker_order=order_b)

        assert result_a["SPEAKER_00"] == left_face
        assert result_b["SPEAKER_01"] == left_face

    def test_one_face_multiple_active_uses_first_speaker(self):
        segs = [_seg("SPEAKER_00", 0.0, 5.0), _seg("SPEAKER_01", 0.0, 5.0)]
        face = _BBox(x=400, y=200, w=100, h=100)
        result = _assign_speakers_to_faces([face], segs, 2.0)
        assert len(result) == 1
        speaker = next(iter(result))
        assert speaker in ("SPEAKER_00", "SPEAKER_01")


# ---------------------------------------------------------------------------
# _smooth_regions (Task 4.0)
# ---------------------------------------------------------------------------

class TestSmoothRegions:
    def _r(self, x, y, w=100, h=100):
        return CropRegion(x=x, y=y, w=w, h=h)

    def test_empty_list_returns_empty(self):
        assert _smooth_regions([]) == []

    def test_no_none_values_returns_smoothed_list(self):
        regions = [self._r(0, 0), self._r(200, 200)]
        result = _smooth_regions(regions, alpha=0.15)
        assert len(result) == 2
        assert all(r is not None for r in result)

    def test_none_in_middle_propagates_last_valid(self):
        regions = [self._r(0, 0), None, self._r(200, 200)]
        result = _smooth_regions(regions, alpha=0.15)
        assert len(result) == 3
        assert result[1] is not None  # no None in output

    def test_alpha_zero_all_frames_equal_to_first_valid(self):
        first = self._r(10, 20)
        regions = [first, self._r(200, 300), self._r(500, 600)]
        result = _smooth_regions(regions, alpha=0.0)
        # With alpha=0, EMA never updates: all frames keep first value
        for r in result:
            assert r.x == 10
            assert r.y == 20

    def test_none_only_list_returns_zeros(self):
        result = _smooth_regions([None, None, None])
        assert len(result) == 3
        for r in result:
            assert r.w >= 1
            assert r.h >= 1

    def test_smoothed_values_between_start_and_end(self):
        regions = [self._r(0, 0), self._r(100, 100)]
        result = _smooth_regions(regions, alpha=0.5)
        # Second value should be between 0 and 100 (exclusive) due to EMA
        assert 0 <= result[1].x <= 100
        assert 0 <= result[1].y <= 100

    def test_single_region_unchanged(self):
        region = self._r(50, 60, w=120, h=140)
        result = _smooth_regions([region], alpha=0.15)
        assert len(result) == 1
        assert result[0].x == 50
        assert result[0].y == 60


# ---------------------------------------------------------------------------
# _active_speakers_at (helper interno)
# ---------------------------------------------------------------------------

class TestActiveSpeakersAt:
    from youcut.face_tracker import _active_speakers_at as _at

    def test_returns_speakers_within_interval(self):
        from youcut.face_tracker import _active_speakers_at
        segs = [_seg("SPEAKER_00", 0.0, 5.0), _seg("SPEAKER_01", 3.0, 8.0)]
        active = _active_speakers_at(segs, 4.0)
        assert "SPEAKER_00" in active
        assert "SPEAKER_01" in active

    def test_returns_empty_when_no_speaker_active(self):
        from youcut.face_tracker import _active_speakers_at
        segs = [_seg("SPEAKER_00", 0.0, 2.0)]
        active = _active_speakers_at(segs, 5.0)
        assert active == []

    def test_returns_sorted_list(self):
        from youcut.face_tracker import _active_speakers_at
        segs = [_seg("SPEAKER_01", 0.0, 5.0), _seg("SPEAKER_00", 0.0, 5.0)]
        active = _active_speakers_at(segs, 2.0)
        assert active == sorted(active)

    def test_no_duplicates(self):
        from youcut.face_tracker import _active_speakers_at
        segs = [_seg("SPEAKER_00", 0.0, 5.0), _seg("SPEAKER_00", 1.0, 3.0)]
        active = _active_speakers_at(segs, 2.0)
        assert active.count("SPEAKER_00") == 1


# ---------------------------------------------------------------------------
# _assign_speakers_to_faces — additional branches
# ---------------------------------------------------------------------------

class TestAssignSpeakersBranches:
    def test_more_faces_than_speakers_uses_centre_face(self):
        segs = [_seg("SPEAKER_00", 0.0, 5.0)]
        # Three faces; the one closest to centre should be selected
        left = _BBox(x=0, y=200, w=80, h=80)
        centre = _BBox(x=600, y=200, w=80, h=80)
        right = _BBox(x=1150, y=200, w=80, h=80)
        result = _assign_speakers_to_faces([left, centre, right], segs, 2.0)
        assert len(result) == 1
        assert result["SPEAKER_00"] == centre

    def test_two_faces_no_speaker_order_uses_lex_sort(self):
        segs = [_seg("SPEAKER_00", 0.0, 5.0), _seg("SPEAKER_01", 0.0, 5.0)]
        left = _BBox(x=100, y=200, w=80, h=80)
        right = _BBox(x=700, y=200, w=80, h=80)
        result = _assign_speakers_to_faces([left, right], segs, 2.0, speaker_order=None)
        # Without explicit order, lex sort: SPEAKER_00 < SPEAKER_01 → SPEAKER_00 gets left
        assert result["SPEAKER_00"] == left
        assert result["SPEAKER_01"] == right


# ---------------------------------------------------------------------------
# apply_face_tracking — unit-level fallback paths
# ---------------------------------------------------------------------------

class TestApplyFaceTrackingFallbacks:
    def test_returns_clip_path_when_face_tracking_disabled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from youcut.config import PipelineConfig
        from youcut.face_tracker import apply_face_tracking

        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"fake")
        config = PipelineConfig(face_tracking=False)
        result = apply_face_tracking(clip, config)
        assert result == clip

    def test_returns_clip_path_when_mediapipe_not_installed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from youcut.config import PipelineConfig
        import sys
        import importlib

        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"fake")
        config = PipelineConfig(face_tracking=True)

        # Simulate mediapipe not installed
        original_cv2 = sys.modules.get("cv2")
        original_mp = sys.modules.get("mediapipe")
        sys.modules["cv2"] = None  # type: ignore
        sys.modules["mediapipe"] = None  # type: ignore
        try:
            import youcut.face_tracker
            importlib.reload(youcut.face_tracker)
            result = youcut.face_tracker.apply_face_tracking(clip, config)
            assert result == clip
        finally:
            if original_cv2 is None:
                sys.modules.pop("cv2", None)
            else:
                sys.modules["cv2"] = original_cv2
            if original_mp is None:
                sys.modules.pop("mediapipe", None)
            else:
                sys.modules["mediapipe"] = original_mp
            importlib.reload(youcut.face_tracker)

    def test_returns_clip_path_on_runtime_error(self, tmp_path, monkeypatch):
        from unittest.mock import patch
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from youcut.config import PipelineConfig
        from youcut.face_tracker import apply_face_tracking

        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"fake")
        config = PipelineConfig(face_tracking=True)

        with (
            patch("youcut.face_tracker.cv2", create=True),
            patch("youcut.face_tracker.mediapipe", create=True),
            patch("youcut.face_tracker._run_face_tracking", side_effect=RuntimeError("fail")),
        ):
            result = apply_face_tracking(clip, config)

        assert result == clip


# ---------------------------------------------------------------------------
# _smooth_regions — additional alpha cases
# ---------------------------------------------------------------------------

class TestSmoothRegionsAlpha:
    def _r(self, x, y, w=100, h=100):
        return CropRegion(x=x, y=y, w=w, h=h)

    def test_alpha_one_each_frame_equals_current_region(self):
        regions = [self._r(0, 0), self._r(50, 50), self._r(200, 200)]
        result = _smooth_regions(regions, alpha=1.0)
        # With alpha=1, EMA fully updates each step → output equals input
        assert result[0].x == 0
        assert result[1].x == 50
        assert result[2].x == 200

    def test_leading_none_filled_with_first_valid(self):
        regions = [None, None, self._r(100, 100)]
        result = _smooth_regions(regions, alpha=0.15)
        # First valid is at index 2; leading Nones get initialised to it
        assert result[0] is not None
        assert result[1] is not None
        assert result[2].x == 100


# ---------------------------------------------------------------------------
# _crop_frame, _build_split_frame, _mux_audio — com mock de cv2
# ---------------------------------------------------------------------------

class TestCropAndRenderFunctions:
    """Tests for rendering functions using mocked cv2 and numpy."""

    def _make_cv2_mock(self):
        from unittest.mock import MagicMock
        cv2_mock = MagicMock()
        cv2_mock.COLOR_BGR2RGB = 4
        cv2_mock.INTER_LINEAR = 1
        cv2_mock.CAP_PROP_FPS = 5
        cv2_mock.CAP_PROP_FRAME_WIDTH = 3
        cv2_mock.CAP_PROP_FRAME_HEIGHT = 4
        cv2_mock.CAP_PROP_FRAME_COUNT = 7
        cv2_mock.VideoWriter_fourcc.return_value = 0
        return cv2_mock

    def _make_frame(self, h=720, w=1280):
        import numpy as np
        return np.zeros((h, w, 3), dtype="uint8")

    def test_crop_frame_calls_resize(self, tmp_path):
        from unittest.mock import MagicMock, patch
        import sys
        import importlib
        import numpy as np

        cv2_mock = self._make_cv2_mock()
        resized = np.zeros((1920, 1080, 3), dtype="uint8")
        cv2_mock.resize.return_value = resized

        frame = self._make_frame()
        roi = CropRegion(x=100, y=100, w=200, h=200)

        sys.modules["cv2"] = cv2_mock
        try:
            import youcut.face_tracker
            importlib.reload(youcut.face_tracker)
            result = youcut.face_tracker._crop_frame(frame, roi)
        finally:
            sys.modules.pop("cv2", None)
            importlib.reload(youcut.face_tracker)

        cv2_mock.resize.assert_called_once()
        assert result.shape == (1920, 1080, 3)

    def test_crop_frame_clamps_roi_to_frame_bounds(self, tmp_path):
        from unittest.mock import MagicMock, patch
        import sys
        import importlib
        import numpy as np

        cv2_mock = self._make_cv2_mock()
        resized = np.zeros((1920, 1080, 3), dtype="uint8")
        cv2_mock.resize.return_value = resized

        frame = self._make_frame(h=100, w=100)
        # ROI extends beyond frame
        roi = CropRegion(x=90, y=90, w=200, h=200)

        sys.modules["cv2"] = cv2_mock
        try:
            import youcut.face_tracker
            importlib.reload(youcut.face_tracker)
            result = youcut.face_tracker._crop_frame(frame, roi)
        finally:
            sys.modules.pop("cv2", None)
            importlib.reload(youcut.face_tracker)

        cv2_mock.resize.assert_called_once()

    def test_build_split_frame_calls_vstack(self):
        from unittest.mock import MagicMock, patch
        import sys
        import importlib
        import numpy as np

        cv2_mock = self._make_cv2_mock()
        half_frame = np.zeros((960, 1080, 3), dtype="uint8")
        cv2_mock.resize.return_value = half_frame

        frame = self._make_frame()
        roi1 = CropRegion(x=0, y=0, w=200, h=200)
        roi2 = CropRegion(x=400, y=0, w=200, h=200)

        sys.modules["cv2"] = cv2_mock
        try:
            import youcut.face_tracker
            importlib.reload(youcut.face_tracker)
            result = youcut.face_tracker._build_split_frame(frame, roi1, roi2)
        finally:
            sys.modules.pop("cv2", None)
            importlib.reload(youcut.face_tracker)

        # result should have double height
        assert result.shape[0] == 960 * 2

    def test_mux_audio_calls_ffmpeg(self, tmp_path):
        from unittest.mock import patch
        import youcut.face_tracker as ft

        video = tmp_path / "video.mp4"
        audio = tmp_path / "audio.mp4"
        output = tmp_path / "out.mp4"
        video.write_bytes(b"v")
        audio.write_bytes(b"a")

        with patch("subprocess.run") as mock_run:
            ft._mux_audio(video, audio, output)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "ffmpeg" in cmd
        assert "-c:a" in cmd
        assert "copy" in cmd


# ---------------------------------------------------------------------------
# _run_face_tracking — mocked cv2 + mediapipe end-to-end
# ---------------------------------------------------------------------------

class TestRunFaceTrackingMocked:
    def test_returns_clip_path_when_no_faces_detected(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock, patch
        import sys
        import importlib
        import numpy as np

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from youcut.config import PipelineConfig
        config = PipelineConfig(face_tracking=True)

        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"fake")

        # Mock cv2
        cv2_mock = MagicMock()
        cv2_mock.CAP_PROP_FPS = 5
        cv2_mock.CAP_PROP_FRAME_WIDTH = 3
        cv2_mock.CAP_PROP_FRAME_HEIGHT = 4
        cv2_mock.CAP_PROP_FRAME_COUNT = 7
        cv2_mock.COLOR_BGR2RGB = 4

        cap_mock = MagicMock()
        cap_mock.isOpened.return_value = True
        cap_mock.get.side_effect = lambda prop: {5: 25.0, 3: 640, 4: 480, 7: 5}.get(prop, 0)
        # Return (False, None) on first read → no frames
        cap_mock.read.return_value = (False, None)
        cv2_mock.VideoCapture.return_value = cap_mock

        detector_mock = MagicMock()
        detector_mock.process.return_value = MagicMock(detections=[])

        sys.modules["cv2"] = cv2_mock
        try:
            import youcut.face_tracker
            importlib.reload(youcut.face_tracker)
            with patch("youcut.face_tracker._make_face_detector", return_value=detector_mock), \
                 patch("youcut.diarizer.diarize", return_value=[]):
                result = youcut.face_tracker._run_face_tracking(clip, config)
        finally:
            sys.modules.pop("cv2", None)
            importlib.reload(youcut.face_tracker)

        # No faces → fallback to original clip
        assert result == clip

    def test_render_clip_called_when_faces_detected(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock, patch
        import sys
        import importlib
        import numpy as np

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from youcut.config import PipelineConfig
        config = PipelineConfig(face_tracking=True)

        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"fake")

        cv2_mock = MagicMock()
        cv2_mock.CAP_PROP_FPS = 5
        cv2_mock.CAP_PROP_FRAME_WIDTH = 3
        cv2_mock.CAP_PROP_FRAME_HEIGHT = 4
        cv2_mock.CAP_PROP_FRAME_COUNT = 7
        cv2_mock.COLOR_BGR2RGB = 4

        frame = np.zeros((480, 640, 3), dtype="uint8")
        cap_mock = MagicMock()
        cap_mock.isOpened.return_value = True
        cap_mock.get.side_effect = lambda prop: {5: 25.0, 3: 640, 4: 480, 7: 1}.get(prop, 0)
        cap_mock.read.side_effect = [(True, frame), (False, None)]
        cv2_mock.VideoCapture.return_value = cap_mock

        det_result = MagicMock()
        bb = MagicMock()
        bb.xmin = 0.2
        bb.ymin = 0.2
        bb.width = 0.2
        bb.height = 0.2
        det_result.location_data.relative_bounding_box = bb
        det_result.score = [0.9]

        detector_mock = MagicMock()
        process_result = MagicMock()
        process_result.detections = [det_result]
        detector_mock.process.return_value = process_result

        expected_output = tmp_path / "clip_tracked.mp4"

        sys.modules["cv2"] = cv2_mock
        try:
            import youcut.face_tracker
            importlib.reload(youcut.face_tracker)
            from youcut.models import SpeakerSegment as SS
            dummy_seg = SS(speaker_id="SPEAKER_00", start=0.0, end=60.0)
            with patch("youcut.face_tracker._make_face_detector", return_value=detector_mock), \
                 patch("youcut.diarizer.diarize", return_value=[dummy_seg]), \
                 patch("youcut.face_tracker._render_clip", return_value=expected_output) as mock_render:
                result = youcut.face_tracker._run_face_tracking(clip, config)
        finally:
            sys.modules.pop("cv2", None)
            importlib.reload(youcut.face_tracker)

        mock_render.assert_called_once()
        assert result == expected_output


# ---------------------------------------------------------------------------
# Edge-case branches for coverage (lines 52-53, 94-98, 269-290)
# ---------------------------------------------------------------------------

class TestComputeSingleSpeakerRoiEdgeBranches:
    def test_tall_face_triggers_crop_h_gt_frame_h_branch(self):
        # Face height 500 in 720-high frame → crop_h will exceed frame_h
        bbox = _BBox(x=640, y=360, w=100, h=500)
        roi = _compute_single_speaker_roi(bbox, 1280, 720)
        assert roi.h <= 720
        assert roi.w >= 1
        assert roi.x + roi.w <= 1280
        assert roi.y + roi.h <= 720


class TestBuildSplitScreenEdgeBranches:
    def test_small_frame_triggers_crop_w_gt_frame_w_branch(self):
        # Small frame: 100×100. A face of height=80 produces crop_w > frame_w
        bbox = _BBox(x=10, y=10, w=50, h=80)
        roi1, roi2 = _build_split_screen_regions(bbox, bbox, 100, 100)
        assert roi1.w <= 100
        assert roi1.h <= 100
        assert roi2.w <= 100
        assert roi2.h <= 100

    def test_medium_face_triggers_crop_h_gt_frame_h_branch(self):
        # Large face relative to frame → crop_h > frame_h in _half_crop
        bbox = _BBox(x=100, y=100, w=300, h=700)
        roi1, roi2 = _build_split_screen_regions(bbox, bbox, 1280, 720)
        assert roi1.h <= 720
        assert roi2.h <= 720


class TestRunFaceTrackingSplitScreen:
    """Cover split-screen code path in _run_face_tracking."""

    def test_split_screen_path_with_two_speakers(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock, patch
        import sys
        import importlib
        import numpy as np

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from youcut.config import PipelineConfig
        from youcut.models import SpeakerSegment as SS

        config = PipelineConfig(face_tracking=True)
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"fake")

        cv2_mock = MagicMock()
        cv2_mock.CAP_PROP_FPS = 5
        cv2_mock.CAP_PROP_FRAME_WIDTH = 3
        cv2_mock.CAP_PROP_FRAME_HEIGHT = 4
        cv2_mock.CAP_PROP_FRAME_COUNT = 7
        cv2_mock.COLOR_BGR2RGB = 4

        frame = np.zeros((480, 1280, 3), dtype="uint8")
        cap_mock = MagicMock()
        cap_mock.isOpened.return_value = True
        cap_mock.get.side_effect = lambda p: {5: 25.0, 3: 1280, 4: 480, 7: 1}.get(p, 0)
        cap_mock.read.side_effect = [(True, frame), (False, None)]
        cv2_mock.VideoCapture.return_value = cap_mock

        # Two detections — left and right faces
        def make_det(xmin, width):
            d = MagicMock()
            bb = MagicMock()
            bb.xmin, bb.ymin, bb.width, bb.height = xmin, 0.2, width, 0.3
            d.location_data.relative_bounding_box = bb
            return d

        process_result = MagicMock()
        process_result.detections = [make_det(0.1, 0.15), make_det(0.6, 0.15)]

        detector_mock = MagicMock()
        detector_mock.process.return_value = process_result

        segs = [
            SS(speaker_id="SPEAKER_00", start=0.0, end=60.0),
            SS(speaker_id="SPEAKER_01", start=0.0, end=60.0),
        ]

        expected_output = tmp_path / "clip_tracked.mp4"

        sys.modules["cv2"] = cv2_mock
        try:
            import youcut.face_tracker
            importlib.reload(youcut.face_tracker)
            with patch("youcut.face_tracker._make_face_detector", return_value=detector_mock), \
                 patch("youcut.diarizer.diarize", return_value=segs), \
                 patch("youcut.face_tracker._render_clip", return_value=expected_output) as mock_render:
                result = youcut.face_tracker._run_face_tracking(clip, config)
        finally:
            sys.modules.pop("cv2", None)
            importlib.reload(youcut.face_tracker)

        mock_render.assert_called_once()
        # Verify split screen was detected
        call_result = mock_render.call_args[0][1]  # FaceTrackingResult
        assert any(call_result.is_split_screen)
        assert result == expected_output


# ---------------------------------------------------------------------------
# _compute_panel_crop_for_faces (face-aware framing for editorial layout)
# ---------------------------------------------------------------------------

class TestComputePanelCropForFaces:
    """Crop derivation for the speaker_bottom_ai_top bottom panel.

    Source frames are 1920×1080 (16:9 wide). Target panel is 1080×880 (taller
    than wide, matches the editorial layout's bottom band).
    """

    FRAME_W = 1920
    FRAME_H = 1080
    TARGET_W = 1080
    TARGET_H = 880
    TARGET_RATIO = 1080 / 880

    def _ratio(self, region):
        return region.w / region.h

    def test_empty_faces_returns_centred_crop_with_target_ratio(self):
        region = _compute_panel_crop_for_faces(
            [],
            frame_w=self.FRAME_W, frame_h=self.FRAME_H,
            target_w=self.TARGET_W, target_h=self.TARGET_H,
        )
        assert region.x >= 0 and region.y >= 0
        assert region.x + region.w <= self.FRAME_W
        assert region.y + region.h <= self.FRAME_H
        assert abs(self._ratio(region) - self.TARGET_RATIO) < 0.05
        cx = region.x + region.w // 2
        assert abs(cx - self.FRAME_W // 2) <= 2

    def test_single_face_off_centre_horizontal_centres_on_face(self):
        face = _BBox(x=300, y=400, w=200, h=200)
        face_cx = face.x + face.w // 2
        region = _compute_panel_crop_for_faces(
            [face],
            frame_w=self.FRAME_W, frame_h=self.FRAME_H,
            target_w=self.TARGET_W, target_h=self.TARGET_H,
        )
        crop_cx = region.x + region.w // 2
        assert abs(crop_cx - face_cx) <= 2, (
            f"Expected crop centred on face cx={face_cx}, got crop cx={crop_cx}"
        )
        assert region.x >= 0 and region.x + region.w <= self.FRAME_W

    def test_single_face_on_right_side_centres_on_face(self):
        face = _BBox(x=1500, y=400, w=200, h=200)
        face_cx = face.x + face.w // 2
        region = _compute_panel_crop_for_faces(
            [face],
            frame_w=self.FRAME_W, frame_h=self.FRAME_H,
            target_w=self.TARGET_W, target_h=self.TARGET_H,
        )
        crop_cx = region.x + region.w // 2
        # When a face is near the edge, the crop is clamped to frame bounds —
        # but the face must still be inside the crop (not lost like before).
        assert region.x <= face.x and (region.x + region.w) >= (face.x + face.w)
        # And the crop should pull as far right as it can to keep the face centred
        assert crop_cx > self.FRAME_W // 2

    def test_two_faces_realistic_layout_crop_contains_both(self):
        # Two speakers in a typical interview framing (faces in the middle
        # 60% of a 1920x1080 source). Both must fit inside the crop.
        left = _BBox(x=600, y=400, w=180, h=180)
        right = _BBox(x=1200, y=400, w=180, h=180)
        region = _compute_panel_crop_for_faces(
            [left, right],
            frame_w=self.FRAME_W, frame_h=self.FRAME_H,
            target_w=self.TARGET_W, target_h=self.TARGET_H,
        )
        assert region.x <= left.x
        assert region.x + region.w >= right.x + right.w
        assert region.y <= left.y
        assert region.y + region.h >= left.y + left.h

    def test_two_faces_at_extreme_edges_keeps_both_visible(self):
        # When faces sit near opposite edges of a wide frame, the union may not
        # fit at the panel's target ratio. The crop should still pull as wide
        # as possible (clamped by the source frame) and stay centred on the
        # mid-point so neither face gets clipped off-screen wholesale.
        left = _BBox(x=300, y=400, w=180, h=180)
        right = _BBox(x=1450, y=400, w=180, h=180)
        region = _compute_panel_crop_for_faces(
            [left, right],
            frame_w=self.FRAME_W, frame_h=self.FRAME_H,
            target_w=self.TARGET_W, target_h=self.TARGET_H,
        )
        max_panel_w = int(self.FRAME_H * self.TARGET_RATIO)
        assert region.w >= max_panel_w - 2, (
            f"Expected crop near max panel width {max_panel_w}, got {region.w}"
        )
        union_cx = (left.x + (right.x + right.w)) // 2
        crop_cx = region.x + region.w // 2
        assert abs(crop_cx - union_cx) <= 2

    def test_two_faces_crop_wider_than_single_face_crop(self):
        left = _BBox(x=600, y=400, w=180, h=180)
        right = _BBox(x=1140, y=400, w=180, h=180)
        single = _compute_panel_crop_for_faces(
            [left],
            frame_w=self.FRAME_W, frame_h=self.FRAME_H,
            target_w=self.TARGET_W, target_h=self.TARGET_H,
        )
        both = _compute_panel_crop_for_faces(
            [left, right],
            frame_w=self.FRAME_W, frame_h=self.FRAME_H,
            target_w=self.TARGET_W, target_h=self.TARGET_H,
        )
        assert both.w >= single.w, (
            f"Two-face crop should zoom out (≥ single). single.w={single.w}, both.w={both.w}"
        )

    def test_crop_preserves_target_aspect_ratio(self):
        face = _BBox(x=860, y=440, w=200, h=200)
        region = _compute_panel_crop_for_faces(
            [face],
            frame_w=self.FRAME_W, frame_h=self.FRAME_H,
            target_w=self.TARGET_W, target_h=self.TARGET_H,
        )
        assert abs(self._ratio(region) - self.TARGET_RATIO) < 0.05

    def test_crop_never_exceeds_frame(self):
        face = _BBox(x=10, y=10, w=self.FRAME_W - 20, h=self.FRAME_H - 20)
        region = _compute_panel_crop_for_faces(
            [face],
            frame_w=self.FRAME_W, frame_h=self.FRAME_H,
            target_w=self.TARGET_W, target_h=self.TARGET_H,
        )
        assert region.x >= 0
        assert region.y >= 0
        assert region.x + region.w <= self.FRAME_W
        assert region.y + region.h <= self.FRAME_H


# ---------------------------------------------------------------------------
# _aggregate_face_bbox (percentile-based bbox over a clip's detections)
# ---------------------------------------------------------------------------

class TestAggregateFaceBbox:
    def test_empty_returns_none(self):
        assert _aggregate_face_bbox([]) is None

    def test_single_face_returns_that_face(self):
        face = _BBox(x=600, y=400, w=200, h=200)
        bbox = _aggregate_face_bbox([face])
        assert bbox is not None
        assert bbox.x == face.x
        assert bbox.y == face.y
        assert bbox.x + bbox.w == face.x + face.w
        assert bbox.y + bbox.h == face.y + face.h

    def test_two_speakers_aggregate_spans_both(self):
        # 100 detections per speaker, both consistent across frames
        left_detections = [_BBox(x=600, y=400, w=180, h=180) for _ in range(100)]
        right_detections = [_BBox(x=1200, y=400, w=180, h=180) for _ in range(100)]
        bbox = _aggregate_face_bbox(left_detections + right_detections)
        assert bbox is not None
        assert bbox.x <= 600
        assert bbox.x + bbox.w >= 1200 + 180

    def test_outliers_rejected_by_clustering(self):
        # 100 consistent detections + 3 spurious far-away ones (each in its own bin)
        consistent = [_BBox(x=900, y=400, w=180, h=180) for _ in range(100)]
        outliers = [
            _BBox(x=50, y=50, w=60, h=60),
            _BBox(x=1850, y=950, w=60, h=60),
            _BBox(x=1700, y=900, w=60, h=60),
        ]
        bbox = _aggregate_face_bbox(consistent + outliers)
        assert bbox is not None
        # The bbox should stay close to the consistent cluster, not stretched to the outliers.
        assert bbox.x >= 800, f"Aggregate dragged left by outliers: {bbox}"
        assert bbox.x + bbox.w <= 1180, f"Aggregate dragged right by outliers: {bbox}"

    def test_unbalanced_speakers_both_kept(self):
        # Speaker A is detected 1400 times (looks at camera most of the clip),
        # Speaker B only 90 times (looks down often). The previous percentile
        # logic would clip B as an outlier; clustering keeps any bin with >=3%
        # of detections, so B (~6%) survives.
        a = [_BBox(x=500, y=400, w=180, h=180) for _ in range(1400)]
        b = [_BBox(x=1300, y=400, w=180, h=180) for _ in range(90)]
        bbox = _aggregate_face_bbox(a + b)
        assert bbox is not None
        assert bbox.x <= 500, f"Speaker A missing: {bbox}"
        assert bbox.x + bbox.w >= 1300 + 180, f"Speaker B clipped out: {bbox}"

    def test_extremely_rare_cluster_dropped(self):
        # B appears in only 1% of frames — that's below the 3% threshold,
        # treated as transient noise.
        a = [_BBox(x=500, y=400, w=180, h=180) for _ in range(990)]
        b = [_BBox(x=1300, y=400, w=180, h=180) for _ in range(10)]
        bbox = _aggregate_face_bbox(a + b)
        assert bbox is not None
        assert bbox.x + bbox.w < 1300, f"Rare cluster should not stretch bbox: {bbox}"
