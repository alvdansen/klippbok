"""Tests for klippbok.video.scene — requires PySceneDetect + ffmpeg.

Tests are skipped if dependencies are not available.
"""

from pathlib import Path

import pytest

from tests.conftest import requires_ffmpeg, requires_scenedetect

from klippbok.video.errors import SceneDetectNotFoundError


# ---------------------------------------------------------------------------
# Import guard test (always runs)
# ---------------------------------------------------------------------------

class TestSceneDetectImport:
    """Test that missing scenedetect raises helpful error."""

    def test_check_scenedetect_not_installed(self, monkeypatch) -> None:
        """SceneDetectNotFoundError when scenedetect not importable."""
        import klippbok.video.scene as scene_module

        # Simulate missing import
        original_check = scene_module._check_scenedetect

        def mock_check():
            raise SceneDetectNotFoundError()

        monkeypatch.setattr(scene_module, "_check_scenedetect", mock_check)

        with pytest.raises(SceneDetectNotFoundError, match="pip install"):
            scene_module.detect_scenes("/dummy/path")

        # Restore
        monkeypatch.setattr(scene_module, "_check_scenedetect", original_check)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

@requires_ffmpeg
@requires_scenedetect
class TestDetectScenes:
    """Scene detection integration tests."""

    def test_no_cuts_in_single_scene(self, tiny_video: Path) -> None:
        """A single-scene test video has no cuts."""
        from klippbok.video.scene import detect_scenes
        boundaries = detect_scenes(tiny_video)
        assert len(boundaries) == 0

    def test_verify_no_cuts(self, tiny_video: Path) -> None:
        """verify_no_cuts returns True for clean clip."""
        from klippbok.video.scene import verify_no_cuts
        assert verify_no_cuts(tiny_video) is True

    def test_detect_cut_in_two_scenes(self, two_scene_video: Path) -> None:
        """Two concatenated scenes should produce at least one boundary."""
        from klippbok.video.scene import detect_scenes
        boundaries = detect_scenes(two_scene_video, threshold=20.0)
        # The abrupt color change should be detected
        assert len(boundaries) >= 1
        assert boundaries[0].frame_number > 0

    def test_verify_no_cuts_fails_on_two_scenes(self, two_scene_video: Path) -> None:
        """verify_no_cuts returns False when cuts are present."""
        from klippbok.video.scene import verify_no_cuts
        assert verify_no_cuts(two_scene_video) is False
