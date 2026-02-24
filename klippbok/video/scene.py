"""Scene detection via PySceneDetect.

Detects scene cuts in video files — both for splitting long videos into
training clips AND for verifying that pre-cut clips have no cuts
(temporal coherence is essential for video training).

PySceneDetect is an optional dependency. If not installed, functions
raise SceneDetectNotFoundError with install instructions.
"""

from __future__ import annotations

from pathlib import Path

from klippbok.video.errors import FFmpegNotFoundError, SceneDetectNotFoundError
from klippbok.video.models import SceneBoundary


def _check_scenedetect() -> None:
    """Verify PySceneDetect is available.

    Raises SceneDetectNotFoundError if not installed.
    """
    try:
        import scenedetect  # noqa: F401
    except ImportError:
        raise SceneDetectNotFoundError()


def detect_scenes(
    path: str | Path,
    threshold: float = 27.0,
    min_scene_length: float = 0.5,
) -> list[SceneBoundary]:
    """Detect scene cuts in a video file.

    Uses PySceneDetect's ContentDetector to find visual discontinuities
    (cuts, dissolves, fades). Returns a list of scene boundaries.

    For splitting long videos into training clips, these boundaries tell
    you where to cut. For pre-cut clips, an empty list means no cuts
    (good — the clip is temporally coherent).

    Args:
        path: Path to the video file.
        threshold: Content detection threshold (default 27.0).
            Higher = less sensitive (misses subtle cuts).
            Lower = more sensitive (may false-positive on fast motion).
            27.0 is PySceneDetect's default and works well for most content.
        min_scene_length: Minimum scene length in seconds.
            Scenes shorter than this are merged with neighbors.

    Returns:
        List of SceneBoundary objects, sorted by frame number.

    Raises:
        SceneDetectNotFoundError: if PySceneDetect is not installed.
        FFmpegNotFoundError: if ffmpeg/ffprobe is not in PATH.
    """
    _check_scenedetect()
    path = Path(path).resolve()

    from scenedetect import ContentDetector, open_video, SceneManager

    try:
        video = open_video(str(path))
    except Exception as e:
        error_msg = str(e).lower()
        if "ffmpeg" in error_msg or "ffprobe" in error_msg:
            raise FFmpegNotFoundError("ffmpeg")
        raise

    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(
            threshold=threshold,
            min_scene_len=int(min_scene_length * video.frame_rate),
        )
    )

    # Process the entire video
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    # Convert to our model. Scene list is pairs of (start, end) FrameTimecodes.
    # We want the CUT points — where one scene ends and the next begins.
    # The first scene starts at frame 0, so we skip it and take the
    # start of each subsequent scene as a boundary.
    boundaries: list[SceneBoundary] = []
    for i, (start, _end) in enumerate(scene_list):
        if i == 0:
            # First scene starts at the beginning — not a cut
            continue
        boundaries.append(SceneBoundary(
            frame_number=start.get_frames(),
            timecode=start.get_seconds(),
            confidence=1.0,  # ContentDetector doesn't provide per-cut confidence
        ))

    return boundaries


def verify_no_cuts(
    path: str | Path,
    threshold: float = 27.0,
) -> bool:
    """Quick check: does this clip contain any scene cuts?

    Pre-cut training clips should be temporally coherent — no cuts.
    This function returns True if the clip is clean (no cuts detected),
    False if cuts were found.

    Uses a slightly higher threshold than default scene detection to
    avoid false positives on fast camera movement.

    Args:
        path: Path to the video clip.
        threshold: Detection threshold (default 27.0).

    Returns:
        True if no cuts detected (clip is clean).
        False if one or more cuts were found.

    Raises:
        SceneDetectNotFoundError: if PySceneDetect is not installed.
    """
    boundaries = detect_scenes(path, threshold=threshold)
    return len(boundaries) == 0
