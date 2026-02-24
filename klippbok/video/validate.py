"""Video clip validation against Klippbok data config.

Compares probed video metadata against the VideoConfig to find structural
issues: wrong fps, invalid frame count, resolution problems, non-square SAR.

This is pure logic -no ffmpeg/ffprobe needed. All inputs are Python objects,
making it fast and easy to test.

Quality checks (blur, exposure, motion) are deferred to Phase 4.
"""

from __future__ import annotations

from pathlib import Path

from klippbok.config.data_schema import VideoConfig
from klippbok.config.defaults import VALID_FRAME_COUNTS
from klippbok.video.models import (
    ClipValidation,
    IssueCode,
    ScanReport,
    Severity,
    ValidationIssue,
    VideoMetadata,
)


def nearest_valid_frame_count(count: int, direction: str = "down") -> int:
    """Find the nearest valid 4n+1 frame count.

    Valid frame counts for Wan's 3D causal VAE: 1, 5, 9, 13, 17, 21, ...
    We always trim DOWN -never pad or add frames, because fabricated
    frames would introduce artifacts.

    Args:
        count: The actual frame count to adjust.
        direction: 'down' (default) trims to nearest valid count below.
            'up' would go to the next valid count above (not used in practice).

    Returns:
        The nearest valid 4n+1 frame count.

    Examples:
        >>> nearest_valid_frame_count(83)  # trim from 83 to 81
        81
        >>> nearest_valid_frame_count(81)  # already valid
        81
        >>> nearest_valid_frame_count(4)   # trim from 4 to 1
        1
    """
    if count <= 0:
        return 1

    if direction == "down":
        # 4n+1 formula: subtract 1, floor-divide by 4, multiply by 4, add 1
        n = (count - 1) // 4
        return 4 * n + 1
    else:
        # Round up to next valid count
        n = (count - 1 + 3) // 4  # ceiling division
        return 4 * n + 1


def validate_clip(
    metadata: VideoMetadata,
    video_config: VideoConfig,
) -> ClipValidation:
    """Validate a video clip against the data config's video settings.

    Checks (in order):
    1. Resolution: is it at or above the target? Below minimum?
    2. SAR: are pixels square?
    3. FPS: does it match the target?
    4. Frame count: is it a valid 4n+1 count?

    Each check produces a ValidationIssue with:
    - Machine-readable code
    - Severity (error = can't use, warning = will re-encode)
    - Human message: what's wrong + how to fix it

    Args:
        metadata: Probed video metadata.
        video_config: Target video specs from data config.

    Returns:
        ClipValidation with all findings.
    """
    issues: list[ValidationIssue] = []
    needs_reencode = False
    recommended_fc: int | None = None

    # ── Resolution checks ──
    # Use height as the resolution metric (480p, 720p)
    clip_height = metadata.height

    # Check against minimum quality threshold
    min_res = video_config.resolution  # target is also effective minimum
    if clip_height < min_res:
        # Below target = would need upscaling
        if video_config.upscale_policy == "never":
            issues.append(ValidationIssue(
                code=IssueCode.RESOLUTION_BELOW_TARGET,
                severity=Severity.ERROR,
                message=(
                    f"Clip is {metadata.display_resolution} ({clip_height}p) -"
                    f"below target {min_res}p. Upscaling is disabled (upscale_policy: never). "
                    f"Either use higher-resolution source material or set upscale_policy: warn."
                ),
                field="resolution",
                actual=clip_height,
                expected=min_res,
            ))
        else:
            issues.append(ValidationIssue(
                code=IssueCode.RESOLUTION_BELOW_TARGET,
                severity=Severity.WARNING,
                message=(
                    f"Clip is {metadata.display_resolution} ({clip_height}p) -"
                    f"below target {min_res}p. Will need upscaling, which Wan was not "
                    f"trained to handle. Quality may suffer."
                ),
                field="resolution",
                actual=clip_height,
                expected=min_res,
            ))
            needs_reencode = True

    elif clip_height > min_res:
        # Above target -will downscale (this is fine, just informational)
        issues.append(ValidationIssue(
            code=IssueCode.RESOLUTION_ABOVE_TARGET,
            severity=Severity.INFO,
            message=(
                f"Clip is {metadata.display_resolution} ({clip_height}p) -"
                f"above target {min_res}p. Will be downscaled during normalization."
            ),
            field="resolution",
            actual=clip_height,
            expected=min_res,
        ))
        needs_reencode = True

    # ── SAR check ──
    if not metadata.is_square_sar:
        if video_config.sar_policy == "reject":
            issues.append(ValidationIssue(
                code=IssueCode.NON_SQUARE_SAR,
                severity=Severity.ERROR,
                message=(
                    f"Non-square pixel aspect ratio (SAR: {metadata.sar}). "
                    f"sar_policy is 'reject'. Either fix the source file "
                    f"or set sar_policy: auto_correct to resample automatically."
                ),
                field="sar",
                actual=metadata.sar,
                expected="1:1",
            ))
        else:
            issues.append(ValidationIssue(
                code=IssueCode.NON_SQUARE_SAR,
                severity=Severity.WARNING,
                message=(
                    f"Non-square pixel aspect ratio (SAR: {metadata.sar}). "
                    f"Will be corrected to square pixels during normalization."
                ),
                field="sar",
                actual=metadata.sar,
                expected="1:1",
            ))
            needs_reencode = True

    # ── FPS check ──
    target_fps = video_config.fps
    # Allow small floating-point tolerance (e.g. 23.976 vs 24.0)
    if abs(metadata.fps - target_fps) > 0.5:
        issues.append(ValidationIssue(
            code=IssueCode.FPS_MISMATCH,
            severity=Severity.WARNING,
            message=(
                f"FPS is {metadata.fps} -target is {target_fps}. "
                f"Will be re-encoded at {target_fps} fps during normalization."
            ),
            field="fps",
            actual=metadata.fps,
            expected=target_fps,
        ))
        needs_reencode = True

    # ── Frame count checks ──
    actual_fc = metadata.frame_count

    # Determine target frame count
    if video_config.frame_count == "auto":
        # Auto: use nearest valid 4n+1 count at or below actual
        target_fc = nearest_valid_frame_count(actual_fc, "down")
    else:
        target_fc = int(video_config.frame_count)

    # Check if count is a valid 4n+1
    if actual_fc not in VALID_FRAME_COUNTS:
        valid_down = nearest_valid_frame_count(actual_fc, "down")
        trim_amount = actual_fc - valid_down

        # Is the trim reasonable? (≤3 frames is fine, >3 is concerning)
        if trim_amount > 3:
            issues.append(ValidationIssue(
                code=IssueCode.FRAME_COUNT_LARGE_TRIM,
                severity=Severity.WARNING,
                message=(
                    f"Frame count {actual_fc} is not a valid 4n+1 count. "
                    f"Nearest valid count is {valid_down} (trimming {trim_amount} frames "
                    f"from end). This is a significant trim -consider re-cutting the clip."
                ),
                field="frame_count",
                actual=actual_fc,
                expected=valid_down,
            ))
        else:
            issues.append(ValidationIssue(
                code=IssueCode.INVALID_FRAME_COUNT,
                severity=Severity.WARNING,
                message=(
                    f"Frame count {actual_fc} is not a valid 4n+1 count. "
                    f"Will be trimmed to {valid_down} ({trim_amount} frames from end)."
                ),
                field="frame_count",
                actual=actual_fc,
                expected=valid_down,
            ))
        recommended_fc = valid_down
        needs_reencode = True

    # Check minimum viable frame count (at least 5 frames = 1 seed + 4)
    valid_fc = recommended_fc if recommended_fc is not None else actual_fc
    if valid_fc < 5:
        issues.append(ValidationIssue(
            code=IssueCode.FRAME_COUNT_TOO_SHORT,
            severity=Severity.ERROR,
            message=(
                f"Clip has only {actual_fc} frames (valid: {valid_fc}). "
                f"Minimum useful clip length is 5 frames (1 seed + 4 for VAE). "
                f"This clip is too short for training."
            ),
            field="frame_count",
            actual=actual_fc,
            expected="≥5",
        ))

    return ClipValidation(
        metadata=metadata,
        issues=issues,
        needs_reencode=needs_reencode,
        recommended_frame_count=recommended_fc,
    )


def validate_directory(
    directory: str | Path,
    video_config: VideoConfig,
    metadata_list: list[VideoMetadata] | None = None,
) -> ScanReport:
    """Validate all video clips in a directory.

    If metadata_list is provided, uses that instead of probing
    (useful when you've already probed the directory).

    Args:
        directory: Path to the directory.
        video_config: Target video specs.
        metadata_list: Pre-probed metadata (optional).

    Returns:
        ScanReport with per-clip results and summary.
    """
    directory = Path(directory).resolve()

    if metadata_list is None:
        # Import here to avoid circular dependency and allow
        # validate to work without ffprobe for pure-Python testing
        from klippbok.video.probe import probe_directory
        metadata_list = probe_directory(directory)

    clips = [validate_clip(meta, video_config) for meta in metadata_list]

    return ScanReport(directory=directory, clips=clips)


def _format_config_hint(video_config: "VideoConfig | None") -> list[str]:
    """Build the 'current settings + how to change them' lines.

    Leads with what the tool will do automatically, then offers
    optional overrides. Aimed at curators, not coders.
    """
    lines: list[str] = []
    if video_config is not None:
        lines.append(
            f"  Normalize will convert to: {video_config.fps}fps, "
            f"{video_config.resolution}p"
        )
    lines.append("  To use a different fps:        add --fps 24 to this command")
    lines.append("  To use a different resolution:  create a klippbok_data.yaml with a")
    lines.append("                                  video: section (see examples/ folder)")
    return lines


_SEVERITY_ICON = {
    Severity.ERROR: "ERROR",
    Severity.WARNING: "WARN",
    Severity.INFO: "INFO",
}


def format_scan_report(
    report: ScanReport,
    video_config: "VideoConfig | None" = None,
) -> str:
    """Format a ScanReport as a compact, grouped console report.

    When many clips share the same issues (common with batch recordings),
    groups them together instead of repeating identical info per clip.
    Unique problems are listed individually so nothing gets hidden.

    Shows the current target settings and how to override them so users
    always know where to change fps, resolution, etc.

    Use ``format_scan_report_verbose`` for the full per-clip listing.

    Args:
        report: The scan report to format.
        video_config: Optional — the VideoConfig used for the scan.
            Included in the output so users can see what targets they're
            being validated against.

    Returns:
        Formatted string for console output.
    """
    lines: list[str] = []

    # Header
    lines.append(f"Scan Report: {report.directory}")
    lines.append("=" * 60)
    lines.append("")

    # Summary box — three states a curator cares about:
    #   ready    = no issues at all, already matches target specs
    #   fixable  = needs re-encoding but no errors (normalize will handle it)
    #   unusable = has errors that can't be fixed automatically
    ready = sum(1 for c in report.clips if not c.issues)
    fixable = report.needs_reencode - report.invalid
    unusable = report.invalid

    lines.append(f"  Clips scanned:    {report.total}")
    if ready:
        lines.append(f"  Ready to use:     {ready}")
    if fixable:
        lines.append(f"  Normalize recommended: {fixable}")
    if unusable:
        lines.append(f"  Unusable:         {unusable}")
    if ready == report.total:
        lines.append(f"  All clips match target specs!")
    lines.append("")

    # Config hint — always visible
    lines.extend(_format_config_hint(video_config))
    lines.append("")

    # Issue summary (quick at-a-glance breakdown by type)
    summary = report.issue_summary
    if summary:
        lines.append("Issue Summary:")
        for code, count in sorted(summary.items(), key=lambda x: x[0].value):
            lines.append(f"  {code.value}: {count}")
        lines.append("")

    # No issues? Done.
    clips_with_issues = [c for c in report.clips if c.issues]
    if not clips_with_issues:
        lines.append("All clips are valid! No issues found.")
        lines.append("")
        return "\n".join(lines)

    # --- Group clips by their issue pattern ---
    from collections import defaultdict

    groups: dict[tuple, list[ClipValidation]] = defaultdict(list)
    for clip in clips_with_issues:
        pattern = tuple((i.code, i.severity) for i in clip.issues)
        groups[pattern].append(clip)

    lines.append("Issues Found:")
    lines.append("-" * 60)

    for pattern, group_clips in groups.items():
        count = len(group_clips)

        # Compute spec ranges across the group
        heights = sorted({c.metadata.height for c in group_clips})
        fps_vals = sorted({c.metadata.fps for c in group_clips})
        fc_vals = sorted({c.metadata.frame_count for c in group_clips})

        def _range_str(vals: list) -> str:
            if len(vals) == 1:
                return str(vals[0])
            return f"{vals[0]}-{vals[-1]}"

        spec_parts = []
        if heights:
            spec_parts.append(f"{_range_str(heights)}p")
        if fps_vals:
            spec_parts.append(f"{_range_str(fps_vals)}fps")
        if fc_vals:
            spec_parts.append(f"{_range_str(fc_vals)} frames")
        spec_str = ", ".join(spec_parts)

        lines.append("")
        if count == 1:
            clip = group_clips[0]
            lines.append(f"  {clip.metadata.path.name}  ({spec_str})")
        else:
            lines.append(f"  {count} clips  ({spec_str})")

        # Show issue messages from the representative clip
        representative = group_clips[0]
        for issue in representative.issues:
            icon = _SEVERITY_ICON[issue.severity]
            lines.append(f"    [{icon}] {issue.message}")

        # Small groups: list filenames for reference
        if 2 <= count <= 6:
            names = [c.metadata.path.name for c in group_clips]
            lines.append(f"    Files: {', '.join(names)}")

    # Footer hint
    lines.append("")
    lines.append("Run with -v for full per-clip details.")
    lines.append("")
    return "\n".join(lines)


def format_scan_report_verbose(
    report: ScanReport,
    video_config: "VideoConfig | None" = None,
) -> str:
    """Format a ScanReport with full per-clip details.

    Lists every clip individually with its specs and all issues.
    Use this when you need to see exactly what's going on with each file.

    Args:
        report: The scan report to format.
        video_config: Optional — the VideoConfig used for the scan.

    Returns:
        Formatted string for console output.
    """
    lines: list[str] = []

    # Header
    lines.append(f"Scan Report (verbose): {report.directory}")
    lines.append("=" * 60)
    lines.append("")

    # Summary — same three-state breakdown as compact report
    ready = sum(1 for c in report.clips if not c.issues)
    fixable = report.needs_reencode - report.invalid
    unusable = report.invalid

    lines.append(f"  Clips scanned:    {report.total}")
    if ready:
        lines.append(f"  Ready to use:     {ready}")
    if fixable:
        lines.append(f"  Normalize recommended: {fixable}")
    if unusable:
        lines.append(f"  Unusable:         {unusable}")
    if ready == report.total:
        lines.append(f"  All clips match target specs!")
    lines.append("")

    # Config hint
    lines.extend(_format_config_hint(video_config))
    lines.append("")

    # Every clip, whether it has issues or not
    lines.append("Per-Clip Details:")
    lines.append("-" * 60)

    for clip in report.clips:
        name = clip.metadata.path.name
        res = clip.metadata.display_resolution
        fps = clip.metadata.fps
        fc = clip.metadata.frame_count
        dur = clip.metadata.duration

        status = "OK" if not clip.issues else ("ERROR" if not clip.is_valid else "WARN")
        lines.append(f"\n  {name}  [{status}]")
        lines.append(f"    {res}, {fps}fps, {fc} frames, {dur:.1f}s")

        if clip.needs_reencode:
            lines.append(f"    Needs re-encode: yes")
        if clip.recommended_frame_count is not None:
            lines.append(f"    Recommended frame count: {clip.recommended_frame_count}")

        if clip.issues:
            for issue in clip.issues:
                icon = _SEVERITY_ICON[issue.severity]
                lines.append(f"    [{icon}] {issue.message}")
        else:
            lines.append(f"    No issues.")

    lines.append("")
    return "\n".join(lines)
