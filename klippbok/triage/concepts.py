"""Concept reference discovery from folder structure.

Scans a concepts/ directory for reference images organized by type:

    concepts/
        character/
            holly.jpg         -> ConceptReference("holly", CHARACTER, ...)
            paul.png          -> ConceptReference("paul", CHARACTER, ...)
        setting/
            tiffanys.webp     -> ConceptReference("tiffanys", SETTING, ...)
        object/
            cat.jpg           -> ConceptReference("cat", OBJECT, ...)

Folder names are matched to concept types via TYPE_ALIASES — the user
can name folders naturally ("humans", "People", "actors") and Klippbok
maps them to the right type.

Unknown folder names still work — the references are loaded and used
for matching, but won't auto-select a captioning prompt.
"""

from __future__ import annotations

from pathlib import Path

from klippbok.triage.models import (
    IMAGE_EXTENSIONS,
    ConceptReference,
    ConceptType,
    resolve_concept_type,
)


def discover_concepts(concepts_dir: str | Path) -> list[ConceptReference]:
    """Scan a concepts directory for reference images.

    Expects a structure where each subfolder represents a concept type
    and contains one or more reference images. Each image becomes a
    ConceptReference with its name derived from the filename.

    Args:
        concepts_dir: Path to the concepts directory.

    Returns:
        List of discovered ConceptReference objects, sorted by type
        then name.

    Raises:
        FileNotFoundError: if concepts_dir doesn't exist.
    """
    concepts_dir = Path(concepts_dir).resolve()

    if not concepts_dir.exists():
        raise FileNotFoundError(f"Concepts directory not found: {concepts_dir}")

    if not concepts_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {concepts_dir}")

    references: list[ConceptReference] = []

    # Scan each subfolder
    for folder in sorted(concepts_dir.iterdir()):
        if not folder.is_dir():
            # Skip files at the root level — we need subfolders
            continue

        folder_name = folder.name
        concept_type = resolve_concept_type(folder_name)

        if concept_type is None:
            print(f"  Note: folder '{folder_name}' doesn't match a known type. "
                  f"Images will still be used for matching but won't auto-select "
                  f"a captioning prompt.")
            print(f"  Known types: character, style, motion, object, setting")

        # Find all image files in this folder
        images = sorted(
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )

        for image_path in images:
            references.append(ConceptReference(
                name=image_path.stem,
                concept_type=concept_type,
                image_path=image_path,
                folder_name=folder_name,
            ))

    return references


def print_concept_summary(references: list[ConceptReference]) -> None:
    """Print a summary of discovered concepts.

    Groups concepts by type and lists them with their folder names.

    Args:
        references: List of ConceptReference objects to summarize.
    """
    if not references:
        print("  No concept references found.")
        return

    # Group by type
    by_type: dict[ConceptType | None, list[ConceptReference]] = {}
    for ref in references:
        by_type.setdefault(ref.concept_type, []).append(ref)

    for concept_type, refs in sorted(by_type.items(), key=lambda x: (x[0] is None, str(x[0]))):
        type_label = concept_type.value if concept_type else "unknown"
        names = ", ".join(r.name for r in refs)
        folder = refs[0].folder_name
        print(f"  {type_label} ({folder}/): {names}")
