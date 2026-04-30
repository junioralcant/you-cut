"""Unit tests for the pure analysis helpers in youcut.speaker_framing."""
from __future__ import annotations

import math

import pytest

from youcut.face_tracker import _BBox
from youcut.speaker_framing import (
    _PerFrameFace,
    _build_scenes_from_activity,
    _classify_active_per_frame,
    _identify_clusters_from_bboxes,
    _moving_average,
    _per_cluster_aperture_series,
    _smoothed_lip_deltas_per_cluster,
)


# ---------------------------------------------------------------------------
# _identify_clusters_from_bboxes
# ---------------------------------------------------------------------------

class TestIdentifyClusters:
    def test_empty_returns_empty(self):
        assert _identify_clusters_from_bboxes([]) == []

    def test_single_cluster_one_id(self):
        bboxes = [_BBox(x=600, y=400, w=180, h=180) for _ in range(50)]
        clusters = _identify_clusters_from_bboxes(bboxes)
        assert len(clusters) == 1
        assert clusters[0].cluster_id == 0

    def test_two_distinct_clusters_left_to_right(self):
        left = [_BBox(x=400, y=400, w=180, h=180) for _ in range(50)]
        right = [_BBox(x=1300, y=400, w=180, h=180) for _ in range(50)]
        clusters = _identify_clusters_from_bboxes(left + right)
        assert len(clusters) == 2
        assert clusters[0].cx < clusters[1].cx
        assert clusters[0].cluster_id == 0
        assert clusters[1].cluster_id == 1

    def test_outliers_dropped_below_min_fraction(self):
        consistent = [_BBox(x=600, y=400, w=180, h=180) for _ in range(100)]
        # Three single-frame faces in different bins, none reach the 3% threshold.
        outliers = [
            _BBox(x=50, y=50, w=60, h=60),
            _BBox(x=1850, y=950, w=60, h=60),
            _BBox(x=1700, y=900, w=60, h=60),
        ]
        clusters = _identify_clusters_from_bboxes(consistent + outliers)
        assert len(clusters) == 1
        assert 500 <= clusters[0].cx <= 700

    def test_unbalanced_speakers_both_cluster(self):
        a = [_BBox(x=500, y=400, w=180, h=180) for _ in range(800)]
        b = [_BBox(x=1300, y=400, w=180, h=180) for _ in range(40)]
        clusters = _identify_clusters_from_bboxes(a + b)
        assert len(clusters) == 2


# ---------------------------------------------------------------------------
# _per_cluster_aperture_series
# ---------------------------------------------------------------------------

class TestPerClusterApertureSeries:
    def test_no_clusters_returns_empty_dict(self):
        frames = [[_PerFrameFace(bbox=_BBox(x=0, y=0, w=100, h=100), lip_aperture=0.1)]]
        out = _per_cluster_aperture_series(frames, [])
        assert out == {}

    def test_single_cluster_assigns_apertures(self):
        bboxes = [_BBox(x=600, y=400, w=180, h=180) for _ in range(10)]
        clusters = _identify_clusters_from_bboxes(bboxes)
        frames = [
            [_PerFrameFace(bbox=_BBox(x=600, y=400, w=180, h=180), lip_aperture=0.1 * (i + 1))]
            for i in range(5)
        ]
        out = _per_cluster_aperture_series(frames, clusters)
        assert clusters[0].cluster_id in out
        actual = out[clusters[0].cluster_id]
        expected = [0.1, 0.2, 0.3, 0.4, 0.5]
        assert actual == pytest.approx(expected)

    def test_missing_face_yields_none(self):
        bboxes = [_BBox(x=600, y=400, w=180, h=180) for _ in range(10)]
        clusters = _identify_clusters_from_bboxes(bboxes)
        frames = [
            [_PerFrameFace(bbox=_BBox(x=600, y=400, w=180, h=180), lip_aperture=0.2)],
            [],
            [_PerFrameFace(bbox=_BBox(x=600, y=400, w=180, h=180), lip_aperture=0.4)],
        ]
        out = _per_cluster_aperture_series(frames, clusters)
        assert out[clusters[0].cluster_id] == [0.2, None, 0.4]

    def test_two_clusters_route_correctly(self):
        bboxes = (
            [_BBox(x=400, y=400, w=180, h=180) for _ in range(40)]
            + [_BBox(x=1300, y=400, w=180, h=180) for _ in range(40)]
        )
        clusters = _identify_clusters_from_bboxes(bboxes)
        assert len(clusters) == 2

        frames = [
            [
                _PerFrameFace(bbox=_BBox(x=400, y=400, w=180, h=180), lip_aperture=0.1),
                _PerFrameFace(bbox=_BBox(x=1300, y=400, w=180, h=180), lip_aperture=0.5),
            ],
        ]
        out = _per_cluster_aperture_series(frames, clusters)
        assert out[clusters[0].cluster_id] == [0.1]  # left
        assert out[clusters[1].cluster_id] == [0.5]  # right


# ---------------------------------------------------------------------------
# _moving_average / _smoothed_lip_deltas_per_cluster
# ---------------------------------------------------------------------------

class TestMovingAverage:
    def test_empty_returns_empty(self):
        assert _moving_average([], 5) == []

    def test_window_one_returns_input(self):
        assert _moving_average([1.0, 2.0, 3.0], 1) == [1.0, 2.0, 3.0]

    def test_constant_signal_stays_constant(self):
        out = _moving_average([5.0, 5.0, 5.0, 5.0], 3)
        assert all(abs(v - 5.0) < 1e-9 for v in out)

    def test_centred_window(self):
        out = _moving_average([1.0, 2.0, 3.0, 4.0, 5.0], 3)
        # Window size 3 is centred; for the middle element (index 2), avg of
        # 2,3,4 = 3. For edges, the window shrinks to what's available.
        assert out[2] == pytest.approx(3.0)


class TestSmoothedLipDeltas:
    def test_zero_when_apertures_constant(self):
        series = {0: [0.2, 0.2, 0.2, 0.2, 0.2]}
        out = _smoothed_lip_deltas_per_cluster(series, fps=25.0)
        assert all(v == 0.0 for v in out[0])

    def test_nonzero_when_apertures_vary(self):
        series = {0: [0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5]}
        out = _smoothed_lip_deltas_per_cluster(series, fps=25.0)
        assert max(out[0]) > 0.1

    def test_none_entries_treated_as_silence(self):
        series = {0: [0.1, None, None, None, 0.5]}
        out = _smoothed_lip_deltas_per_cluster(series, fps=25.0)
        # All deltas are 0 because consecutive None breaks the chain.
        assert all(v == 0.0 for v in out[0])


# ---------------------------------------------------------------------------
# _classify_active_per_frame
# ---------------------------------------------------------------------------

class TestClassifyActivePerFrame:
    def test_empty_returns_empty(self):
        assert _classify_active_per_frame({}) == []

    def test_silent_cluster_never_active(self):
        # All zero deltas → threshold is +inf → never active.
        out = _classify_active_per_frame({0: [0.0, 0.0, 0.0, 0.0]})
        assert out == [[], [], [], []]

    def test_only_speaking_cluster_above_median_marked(self):
        # Cluster 0 has high deltas in second half only.
        deltas = [0.001, 0.001, 0.001, 0.001, 0.5, 0.5, 0.5, 0.5]
        out = _classify_active_per_frame({0: deltas})
        # Median of non-zero deltas: between 0.001 and 0.5 → ~0.001.
        # Frames 4-7 sit above median (= 0.001), frame 0-3 are at median.
        assert out[5] == [0]
        assert out[6] == [0]

    def test_two_clusters_independent_thresholds(self):
        # Cluster 0 talks first, cluster 1 talks last.
        c0 = [0.5, 0.5, 0.5, 0.001, 0.001, 0.001]
        c1 = [0.001, 0.001, 0.001, 0.5, 0.5, 0.5]
        out = _classify_active_per_frame({0: c0, 1: c1})
        assert 0 in out[0] and 1 not in out[0]
        assert 0 not in out[5] and 1 in out[5]


# ---------------------------------------------------------------------------
# _build_scenes_from_activity
# ---------------------------------------------------------------------------

class TestBuildScenesFromActivity:
    def test_empty_returns_empty(self):
        assert _build_scenes_from_activity([], min_scene_frames=10) == []

    def test_single_run_yields_one_scene(self):
        active = [[0]] * 50
        scenes = _build_scenes_from_activity(active, min_scene_frames=10)
        assert scenes == [(0, 50, [0])]

    def test_two_runs_produce_two_scenes(self):
        active = [[0]] * 30 + [[1]] * 30
        scenes = _build_scenes_from_activity(active, min_scene_frames=10)
        assert len(scenes) == 2
        assert scenes[0] == (0, 30, [0])
        assert scenes[1] == (30, 60, [1])

    def test_short_run_merged_into_previous(self):
        # 30 frames of [0], then 5 frames of [1] (below 10 threshold), then
        # 30 frames of [0]. The [1] run is too short → merged into the
        # preceding [0] scene (which then absorbs the next [0] too).
        active = [[0]] * 30 + [[1]] * 5 + [[0]] * 30
        scenes = _build_scenes_from_activity(active, min_scene_frames=10)
        # Result should be a single long scene of [0]
        assert len(scenes) == 1
        assert scenes[0] == (0, 65, [0])

    def test_alternating_long_runs_preserved(self):
        active = [[0]] * 30 + [[1]] * 30 + [[0]] * 30
        scenes = _build_scenes_from_activity(active, min_scene_frames=10)
        assert [(s[0], s[1], s[2]) for s in scenes] == [
            (0, 30, [0]),
            (30, 60, [1]),
            (60, 90, [0]),
        ]

    def test_silence_modeled_as_empty_active(self):
        active = [[]] * 20 + [[0]] * 30
        scenes = _build_scenes_from_activity(active, min_scene_frames=10)
        assert len(scenes) == 2
        assert scenes[0] == (0, 20, [])
        assert scenes[1] == (20, 50, [0])
