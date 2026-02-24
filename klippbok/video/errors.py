"""Klippbok video pipeline errors.

Custom exceptions for video processing operations. Every error includes
a clear message about what went wrong AND how to fix it.
"""

from __future__ import annotations


class KlippbokVideoError(Exception):
    """Base exception for all video pipeline errors.

    Subclasses provide specific context for different failure modes.
    All error messages follow the pattern: what happened + how to fix it.
    """
    pass


class FFmpegNotFoundError(KlippbokVideoError):
    """Raised when ffmpeg or ffprobe is not found in PATH.

    ffmpeg is a system dependency — it must be installed separately
    from Python packages. On Windows: winget install ffmpeg
    """

    def __init__(self, tool: str = "ffmpeg") -> None:
        self.tool = tool
        super().__init__(
            f"{tool} not found in PATH. "
            f"Install it with: winget install ffmpeg (Windows) "
            f"or visit https://ffmpeg.org/download.html"
        )


class SceneDetectNotFoundError(KlippbokVideoError):
    """Raised when PySceneDetect is not installed.

    PySceneDetect is an optional dependency — only needed for scene
    detection in long videos, not for scanning pre-cut clips.
    """

    def __init__(self) -> None:
        super().__init__(
            "PySceneDetect not installed. "
            "Install it with: pip install 'klippbok[video]' "
            "or: pip install scenedetect[opencv]"
        )


class ProbeError(KlippbokVideoError):
    """Raised when ffprobe fails to read a video file.

    Common causes: corrupted file, unsupported codec, zero-length file.
    """

    def __init__(self, path: str, detail: str) -> None:
        self.path = path
        self.detail = detail
        super().__init__(
            f"Failed to probe '{path}': {detail}"
        )


class SplitError(KlippbokVideoError):
    """Raised when ffmpeg fails to split or normalize a video.

    Common causes: insufficient disk space, codec issues, permission errors.
    """

    def __init__(self, path: str, detail: str) -> None:
        self.path = path
        self.detail = detail
        super().__init__(
            f"Failed to process '{path}': {detail}"
        )


class ExtractionError(KlippbokVideoError):
    """Raised when frame extraction from a video fails.

    Common causes: corrupted video, codec not supported by ffmpeg,
    requesting a frame number beyond the video's length.
    """

    def __init__(self, path: str, detail: str) -> None:
        self.path = path
        self.detail = detail
        super().__init__(
            f"Failed to extract frame from '{path}': {detail}"
        )
