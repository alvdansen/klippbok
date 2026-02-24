"""Dataset validation reporting — rich and plaintext output.

Two output modes:
- Rich: colorful tables, panels, and progress when the `rich` library
  is installed. This is the default for terminal users.
- Plaintext: ASCII-safe fallback when rich is not installed or when
  --json mode is used. Always works, even on minimal Python installs.

All output uses ASCII-safe characters for Windows cp1252 compatibility.
"""

from __future__ import annotations

from klippbok.dataset.bucketing import BucketingResult
from klippbok.dataset.models import DatasetReport, OrganizeResult
from klippbok.video.models import Severity


# ---------------------------------------------------------------------------
# Plaintext report (always available)
# ---------------------------------------------------------------------------

def format_report_plaintext(report: DatasetReport) -> str:
    """Format a DatasetReport as ASCII-safe plaintext.

    This is the fallback when rich is not installed. It produces
    a scannable, copy-pasteable report.

    Args:
        report: The validation report.

    Returns:
        Formatted string for console output.
    """
    lines: list[str] = []

    # Header
    lines.append("Dataset Validation Report")
    lines.append("=" * 60)

    # Summary
    lines.append(f"Sources:         {report.total_sources}")
    lines.append(f"Total samples:   {report.total_samples}")
    lines.append(f"Valid:           {report.valid_samples}")
    lines.append(f"Invalid:         {report.invalid_samples}")
    lines.append(f"Errors:          {report.error_count}")
    lines.append(f"Warnings:        {report.warning_count}")
    status = "PASS" if report.is_valid else "FAIL"
    lines.append(f"Status:          {status}")
    lines.append("")

    # Issue summary
    summary = report.issue_summary
    if summary:
        lines.append("Issue Summary:")
        lines.append("-" * 40)
        for code, count in sorted(summary.items(), key=lambda x: x[0].value):
            lines.append(f"  {code.value}: {count}")
        lines.append("")

    # Per-dataset details
    for ds in report.datasets:
        lines.append(f"Source: {ds.source_path}")
        lines.append(f"  Structure: {ds.structure.value}")
        lines.append(f"  Samples: {ds.total_samples} ({ds.valid_samples} valid)")
        lines.append("")

        # Dataset-level issues
        for issue in ds.dataset_issues:
            icon = _severity_icon(issue.severity)
            lines.append(f"  [{icon}] {issue.message}")

        # Per-sample issues (only samples with issues)
        for sample in ds.samples:
            if not sample.issues:
                continue
            lines.append(f"  {sample.stem}:")
            for issue in sample.issues:
                icon = _severity_icon(issue.severity)
                lines.append(f"    [{icon}] {issue.message}")
        lines.append("")

    # Cross-dataset issues
    if report.cross_dataset_issues:
        lines.append("Cross-Dataset Issues:")
        lines.append("-" * 40)
        for issue in report.cross_dataset_issues:
            icon = _severity_icon(issue.severity)
            lines.append(f"  [{icon}] {issue.message}")
        lines.append("")

    # All-clear
    if report.is_valid and report.total_samples > 0:
        lines.append("All samples passed validation!")
    elif report.total_samples == 0:
        lines.append("No samples found to validate.")

    lines.append("")
    return "\n".join(lines)


def format_bucketing_plaintext(result: BucketingResult) -> str:
    """Format a BucketingResult as ASCII-safe plaintext.

    Args:
        result: The bucketing preview result.

    Returns:
        Formatted string for console output.
    """
    lines: list[str] = []

    lines.append("Bucketing Preview")
    lines.append("=" * 60)
    lines.append(f"Step size: {result.step_size}px")
    lines.append(f"Total assigned: {result.total_assigned}")
    lines.append(f"Total buckets: {result.total_buckets}")
    lines.append("")

    if result.buckets:
        # Table header
        lines.append(f"  {'Bucket Key':<20} {'Count':>6}  Samples")
        lines.append(f"  {'-'*20} {'-'*6}  {'-'*30}")
        for bucket in result.buckets:
            stems = ", ".join(bucket.samples[:5])
            if len(bucket.samples) > 5:
                stems += f" (+{len(bucket.samples) - 5} more)"
            lines.append(f"  {bucket.bucket_key:<20} {bucket.count:>6}  {stems}")
        lines.append("")

    if result.issues:
        lines.append("Bucketing Issues:")
        for issue in result.issues:
            icon = _severity_icon(issue.severity)
            lines.append(f"  [{icon}] {issue.message}")
        lines.append("")

    return "\n".join(lines)


def _severity_icon(severity: Severity) -> str:
    """ASCII-safe severity indicator."""
    return {
        Severity.ERROR: "ERROR",
        Severity.WARNING: "WARN ",
        Severity.INFO: "INFO ",
    }[severity]


# ---------------------------------------------------------------------------
# Rich report (optional dependency)
# ---------------------------------------------------------------------------

def print_validation_report(report: DatasetReport, console: object | None = None) -> None:
    """Print a validation report using rich formatting.

    Falls back to plaintext if rich is not installed.

    Args:
        report: The validation report.
        console: Optional rich Console instance. Created if not provided.
    """
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
    except ImportError:
        print(format_report_plaintext(report))
        return

    if console is None:
        console = Console()

    # Summary panel
    status_color = "green" if report.is_valid else "red"
    status_text = "PASS" if report.is_valid else "FAIL"

    summary_lines = [
        f"Sources: {report.total_sources}",
        f"Total samples: {report.total_samples}",
        f"Valid: {report.valid_samples}",
        f"Invalid: {report.invalid_samples}",
        f"Errors: {report.error_count}",
        f"Warnings: {report.warning_count}",
        f"[bold {status_color}]Status: {status_text}[/]",
    ]
    console.print(Panel(
        "\n".join(summary_lines),
        title="Dataset Validation",
        border_style=status_color,
    ))

    # Issue summary table
    summary = report.issue_summary
    if summary:
        table = Table(title="Issue Summary")
        table.add_column("Code", style="cyan")
        table.add_column("Count", justify="right")
        for code, count in sorted(summary.items(), key=lambda x: x[0].value):
            table.add_row(code.value, str(count))
        console.print(table)

    # Per-dataset details
    for ds in report.datasets:
        ds_color = "green" if ds.is_valid else "red"
        console.print(f"\n[bold]Source:[/] {ds.source_path}")
        console.print(f"  Structure: {ds.structure.value}, "
                       f"Samples: {ds.total_samples} ({ds.valid_samples} valid)")

        for issue in ds.dataset_issues:
            _print_rich_issue(console, issue)

        for sample in ds.samples:
            if not sample.issues:
                continue
            console.print(f"  [bold]{sample.stem}[/]:")
            for issue in sample.issues:
                _print_rich_issue(console, issue, indent=4)

    # Cross-dataset issues
    if report.cross_dataset_issues:
        console.print("\n[bold]Cross-Dataset Issues:[/]")
        for issue in report.cross_dataset_issues:
            _print_rich_issue(console, issue)

    # All-clear
    if report.is_valid and report.total_samples > 0:
        console.print("\n[bold green]All samples passed validation![/]")


def print_bucketing_report(result: BucketingResult, console: object | None = None) -> None:
    """Print bucketing preview using rich formatting.

    Falls back to plaintext if rich is not installed.

    Args:
        result: The bucketing preview result.
        console: Optional rich Console instance.
    """
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        print(format_bucketing_plaintext(result))
        return

    if console is None:
        console = Console()

    console.print(f"\n[bold]Bucketing Preview[/] (step: {result.step_size}px)")
    console.print(f"Assigned: {result.total_assigned}, Buckets: {result.total_buckets}")

    if result.buckets:
        table = Table()
        table.add_column("Bucket Key", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Samples")
        for bucket in result.buckets:
            stems = ", ".join(bucket.samples[:5])
            if len(bucket.samples) > 5:
                stems += f" (+{len(bucket.samples) - 5} more)"
            table.add_row(bucket.bucket_key, str(bucket.count), stems)
        console.print(table)

    for issue in result.issues:
        _print_rich_issue(console, issue)


def _print_rich_issue(console: object, issue: object, indent: int = 2) -> None:
    """Print a single issue with rich formatting."""
    from klippbok.video.models import Severity as Sev
    prefix = " " * indent
    color = {
        Sev.ERROR: "red",
        Sev.WARNING: "yellow",
        Sev.INFO: "blue",
    }.get(issue.severity, "white")  # type: ignore[union-attr]
    label = _severity_icon(issue.severity)  # type: ignore[union-attr]
    console.print(f"{prefix}[{color}][{label}][/{color}] {issue.message}")  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Organize report (plaintext)
# ---------------------------------------------------------------------------

def format_organize_plaintext(result: OrganizeResult) -> str:
    """Format an OrganizeResult as ASCII-safe plaintext.

    Args:
        result: The organize result.

    Returns:
        Formatted string for console output.
    """
    lines: list[str] = []

    # Header
    mode = "dry-run" if result.dry_run else ("move" if not result.organized else "copy")
    # Detect move vs copy from the result — if dry_run we don't know,
    # but the header still shows the layout and counts.
    lines.append("Organizing dataset")
    lines.append(f"  Output: {result.output_dir}")
    lines.append(f"  Layout: {result.layout.value}")
    if result.dry_run:
        lines.append("  Mode:   dry-run (no files touched)")
    lines.append("")

    # Organized samples
    for i, sample in enumerate(result.organized, 1):
        parts = [sample.stem]
        if sample.target_dest:
            parts.append(sample.target_dest.name)
        if sample.caption_dest:
            parts.append(sample.caption_dest.suffix)
        if sample.reference_dest:
            parts.append(sample.reference_dest.suffix)
        detail = " + ".join(parts[1:]) if len(parts) > 1 else ""
        lines.append(f"  [{i}/{result.organized_count}] {sample.stem} -> {detail}")

    # Skipped samples
    for sample in result.skipped:
        lines.append(f"  [SKIP] {sample.stem}: {sample.skip_reason}")

    lines.append("")

    # Summary
    lines.append(f"Done: {result.organized_count} organized, {result.skipped_count} skipped")

    # Trainer configs
    for config_path in result.trainer_configs:
        lines.append(f"  Trainer config: {config_path}")

    lines.append("")
    return "\n".join(lines)


def print_organize_report(result: OrganizeResult, console: object | None = None) -> None:
    """Print an organize report using rich formatting.

    Falls back to plaintext if rich is not installed.

    Args:
        result: The organize result.
        console: Optional rich Console instance.
    """
    try:
        from rich.console import Console
    except ImportError:
        print(format_organize_plaintext(result))
        return

    if console is None:
        console = Console()

    # Header
    console.print("\n[bold]Organizing dataset[/]")
    console.print(f"  Output: {result.output_dir}")
    console.print(f"  Layout: {result.layout.value}")
    if result.dry_run:
        console.print("  Mode:   [yellow]dry-run (no files touched)[/]")
    console.print()

    # Organized samples
    for i, sample in enumerate(result.organized, 1):
        parts = []
        if sample.target_dest:
            parts.append(sample.target_dest.name)
        if sample.caption_dest:
            parts.append(sample.caption_dest.suffix)
        if sample.reference_dest:
            parts.append(sample.reference_dest.suffix)
        detail = " + ".join(parts)
        console.print(f"  [{i}/{result.organized_count}] {sample.stem} -> {detail}")

    # Skipped
    for sample in result.skipped:
        console.print(f"  [red][SKIP][/red] {sample.stem}: {sample.skip_reason}")

    console.print()

    # Summary
    color = "green" if result.skipped_count == 0 else "yellow"
    console.print(f"[bold {color}]Done: {result.organized_count} organized, {result.skipped_count} skipped[/]")

    for config_path in result.trainer_configs:
        console.print(f"  Trainer config: {config_path}")
