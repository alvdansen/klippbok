"""ffmpeg/ffprobe discovery and PATH management.

WinGet installs ffmpeg to a deep package directory that requires a shell
restart to appear in PATH. This module finds it automatically so users
don't need to restart their terminal after installing ffmpeg.

The discovery runs once at import time and caches the result.
"""

from __future__ import annotations

import glob
import os
import subprocess


def _discover_ffmpeg_path() -> None:
    """Add WinGet-installed ffmpeg to PATH if not already findable.

    Checks common install locations on Windows:
    1. Already in PATH (no action needed)
    2. WinGet package directory (deep nested path)
    3. Chocolatey bin directory
    4. Scoop shims directory

    This runs once at import time. If ffmpeg is already in PATH,
    it returns immediately with no overhead.
    """
    # Quick check: is ffmpeg already in PATH?
    try:
        result = subprocess.run(
            ["ffprobe", "-version"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0:
            return  # Already available, nothing to do
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Search WinGet install location
    local_appdata = os.environ.get("LOCALAPPDATA", "")
    if local_appdata:
        pattern = os.path.join(
            local_appdata,
            "Microsoft", "WinGet", "Packages", "*ffmpeg*", "**", "bin",
        )
        for bin_dir in glob.glob(pattern, recursive=True):
            ffprobe_path = os.path.join(bin_dir, "ffprobe.exe")
            if os.path.isfile(ffprobe_path):
                os.environ["PATH"] = bin_dir + ";" + os.environ["PATH"]
                return

    # Search Chocolatey
    choco_bin = r"C:\ProgramData\chocolatey\bin"
    if os.path.isfile(os.path.join(choco_bin, "ffprobe.exe")):
        if choco_bin not in os.environ["PATH"]:
            os.environ["PATH"] = choco_bin + ";" + os.environ["PATH"]
        return

    # Search Scoop
    userprofile = os.environ.get("USERPROFILE", "")
    if userprofile:
        scoop_shims = os.path.join(userprofile, "scoop", "shims")
        if os.path.isfile(os.path.join(scoop_shims, "ffprobe.exe")):
            if scoop_shims not in os.environ["PATH"]:
                os.environ["PATH"] = scoop_shims + ";" + os.environ["PATH"]
            return


# Run discovery once when the video package is first imported
_discover_ffmpeg_path()
