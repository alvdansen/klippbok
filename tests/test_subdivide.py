"""Tests for _subdivide_segments and max_frames integration.

Tests the scene subdivision logic that splits long scenes into
shorter clips when max_frames is set. Pure logic tests (no ffmpeg).
"""

from klippbok.video.split import _subdivide_segments


class TestSubdivideSegments:
    """Tests for _subdivide_segments()."""

    def test_no_subdivision_needed(self) -> None:
        """Segments shorter than max_duration pass through unchanged."""
        segments = [(0.0, 3.0), (3.0, 5.0)]
        result = _subdivide_segments(segments, max_duration=5.0)

        assert len(result) == 2
        assert result[0] == (0.0, 3.0, 0, -1)  # sub_index -1 = no split
        assert result[1] == (3.0, 5.0, 1, -1)

    def test_exact_fit(self) -> None:
        """Segment exactly at max_duration should not be split."""
        segments = [(0.0, 5.0)]
        result = _subdivide_segments(segments, max_duration=5.0)

        assert len(result) == 1
        assert result[0] == (0.0, 5.0, 0, -1)

    def test_splits_long_segment(self) -> None:
        """A 10s segment with 5s max should produce 2 chunks."""
        segments = [(0.0, 10.0)]
        result = _subdivide_segments(segments, max_duration=5.0)

        assert len(result) == 2
        assert result[0] == (0.0, 5.0, 0, 0)
        assert result[1] == (5.0, 10.0, 0, 1)

    def test_splits_into_three(self) -> None:
        """A 15s segment with 5.0625s max (81 frames at 16fps)."""
        max_dur = 81 / 16  # 5.0625s
        segments = [(0.0, 15.0)]
        result = _subdivide_segments(segments, max_duration=max_dur)

        assert len(result) == 3
        # First two are max_dur, last is the remainder
        assert result[0][0] == 0.0
        assert abs(result[0][1] - max_dur) < 0.001
        assert result[0][2] == 0  # scene index
        assert result[0][3] == 0  # sub_index

        assert abs(result[1][0] - max_dur) < 0.001
        assert abs(result[1][1] - 2 * max_dur) < 0.001
        assert result[1][3] == 1

        assert abs(result[2][0] - 2 * max_dur) < 0.001
        assert result[2][1] == 15.0
        assert result[2][3] == 2

    def test_preserves_scene_indices(self) -> None:
        """Scene indices are preserved across segments."""
        segments = [(0.0, 2.0), (2.0, 12.0), (12.0, 14.0)]
        result = _subdivide_segments(segments, max_duration=5.0)

        # Scene 0: 2s, no split
        assert result[0] == (0.0, 2.0, 0, -1)
        # Scene 1: 10s, split into 2
        assert result[1][2] == 1  # scene_index
        assert result[2][2] == 1
        # Scene 2: 2s, no split
        assert result[3] == (12.0, 14.0, 2, -1)

    def test_tiny_tail_skipped(self) -> None:
        """Tail segments < 50ms are absorbed by the previous chunk."""
        # 5.04s with 5.0s max — the 0.04s tail should be absorbed
        segments = [(0.0, 5.04)]
        result = _subdivide_segments(segments, max_duration=5.0)

        # Should be just 1 chunk (tail is < 50ms tolerance)
        assert len(result) == 1

    def test_empty_segments(self) -> None:
        """Empty segment list returns empty result."""
        result = _subdivide_segments([], max_duration=5.0)
        assert result == []

    def test_mixed_short_and_long(self) -> None:
        """Mix of short and long segments."""
        segments = [
            (0.0, 1.0),    # short, no split
            (1.0, 12.0),   # 11s, splits into 3 at 5s max
            (12.0, 13.0),  # short, no split
        ]
        result = _subdivide_segments(segments, max_duration=5.0)

        # Scene 0: pass-through
        assert result[0] == (0.0, 1.0, 0, -1)
        # Scene 1: 3 chunks (5 + 5 + 1)
        assert result[1][3] == 0  # sub_index 0
        assert result[2][3] == 1  # sub_index 1
        assert result[3][3] == 2  # sub_index 2
        # Scene 2: pass-through
        assert result[4] == (12.0, 13.0, 2, -1)

    def test_default_81_frames_at_16fps(self) -> None:
        """Real-world case: 81 frames at 16fps = 5.0625s max."""
        max_dur = 81 / 16  # 5.0625s

        # A 30-second scene (like a long movie shot)
        segments = [(0.0, 30.0)]
        result = _subdivide_segments(segments, max_duration=max_dur)

        # 30 / 5.0625 = 5.926... -> 6 chunks
        assert len(result) == 6

        # Verify continuity — each chunk starts where previous ends
        for i in range(1, len(result)):
            assert abs(result[i][0] - result[i - 1][1]) < 0.001

        # All should be scene 0
        for chunk in result:
            assert chunk[2] == 0


class TestMaxFramesInVideoConfig:
    """Tests for max_frames field in VideoConfig."""

    def test_default_81(self) -> None:
        """Default max_frames is 81."""
        from klippbok.config.data_schema import VideoConfig
        config = VideoConfig()
        assert config.max_frames == 81

    def test_none_means_no_limit(self) -> None:
        """max_frames=None means no frame count limit."""
        from klippbok.config.data_schema import VideoConfig
        config = VideoConfig(max_frames=None)
        assert config.max_frames is None

    def test_custom_value(self) -> None:
        """max_frames can be set to any positive integer."""
        from klippbok.config.data_schema import VideoConfig
        config = VideoConfig(max_frames=49)
        assert config.max_frames == 49

    def test_model_copy_override(self) -> None:
        """max_frames can be overridden via model_copy."""
        from klippbok.config.data_schema import VideoConfig
        config = VideoConfig(max_frames=81)
        updated = config.model_copy(update={"max_frames": 49})
        assert updated.max_frames == 49
        assert config.max_frames == 81  # original unchanged
