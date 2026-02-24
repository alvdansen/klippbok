"""Tests for klippbok.triage.concepts — concept discovery from folders.

Uses tmp_path for filesystem tests. No external dependencies required.
"""

import pytest
from pathlib import Path

from klippbok.triage.models import ConceptReference, ConceptType
from klippbok.triage.concepts import discover_concepts, print_concept_summary


def _create_image(path: Path) -> None:
    """Create a minimal file that looks like an image (by extension)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xd8\xff\xe0")  # JPEG magic bytes


class TestDiscoverConcepts:
    """Tests for discover_concepts()."""

    def test_canonical_structure(self, tmp_path: Path) -> None:
        """Standard concepts/ structure with canonical folder names."""
        _create_image(tmp_path / "character" / "holly.jpg")
        _create_image(tmp_path / "setting" / "tiffanys.webp")
        _create_image(tmp_path / "object" / "cat.png")

        refs = discover_concepts(tmp_path)

        assert len(refs) == 3
        names = {r.name for r in refs}
        assert names == {"holly", "tiffanys", "cat"}

        # Check types were resolved
        by_name = {r.name: r for r in refs}
        assert by_name["holly"].concept_type == ConceptType.CHARACTER
        assert by_name["tiffanys"].concept_type == ConceptType.SETTING
        assert by_name["cat"].concept_type == ConceptType.OBJECT

    def test_squishy_folder_names(self, tmp_path: Path) -> None:
        """Squishy aliases like 'humans' resolve to the right type."""
        _create_image(tmp_path / "humans" / "person1.jpg")
        _create_image(tmp_path / "places" / "park.jpg")

        refs = discover_concepts(tmp_path)

        by_name = {r.name: r for r in refs}
        assert by_name["person1"].concept_type == ConceptType.CHARACTER
        assert by_name["park"].concept_type == ConceptType.SETTING

    def test_unknown_folder(self, tmp_path: Path) -> None:
        """Unknown folder name results in None concept_type."""
        _create_image(tmp_path / "mystery" / "thing.jpg")

        refs = discover_concepts(tmp_path)

        assert len(refs) == 1
        assert refs[0].name == "thing"
        assert refs[0].concept_type is None
        assert refs[0].folder_name == "mystery"

    def test_multiple_images_per_folder(self, tmp_path: Path) -> None:
        """Multiple reference images in one folder all get discovered."""
        _create_image(tmp_path / "character" / "holly.jpg")
        _create_image(tmp_path / "character" / "paul.png")
        _create_image(tmp_path / "character" / "fred.webp")

        refs = discover_concepts(tmp_path)

        assert len(refs) == 3
        names = {r.name for r in refs}
        assert names == {"holly", "paul", "fred"}
        # All should be CHARACTER
        assert all(r.concept_type == ConceptType.CHARACTER for r in refs)

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Empty concepts directory returns empty list."""
        refs = discover_concepts(tmp_path)
        assert refs == []

    def test_nonexistent_directory(self) -> None:
        """Nonexistent directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            discover_concepts(Path("/nonexistent/path"))

    def test_not_a_directory(self, tmp_path: Path) -> None:
        """File path raises NotADirectoryError."""
        f = tmp_path / "file.txt"
        f.write_text("not a dir")
        with pytest.raises(NotADirectoryError):
            discover_concepts(f)

    def test_skips_root_level_files(self, tmp_path: Path) -> None:
        """Files at the root of concepts/ are ignored (need subfolders)."""
        (tmp_path / "stray_image.jpg").write_bytes(b"\xff\xd8")
        _create_image(tmp_path / "character" / "holly.jpg")

        refs = discover_concepts(tmp_path)

        assert len(refs) == 1
        assert refs[0].name == "holly"

    def test_skips_non_image_files(self, tmp_path: Path) -> None:
        """Non-image files in type folders are ignored."""
        _create_image(tmp_path / "character" / "holly.jpg")
        (tmp_path / "character" / "notes.txt").write_text("notes")
        (tmp_path / "character" / "data.json").write_text("{}")

        refs = discover_concepts(tmp_path)

        assert len(refs) == 1
        assert refs[0].name == "holly"

    def test_preserves_absolute_paths(self, tmp_path: Path) -> None:
        """Image paths are absolute."""
        _create_image(tmp_path / "character" / "holly.jpg")

        refs = discover_concepts(tmp_path)
        assert refs[0].image_path.is_absolute()

    def test_sorted_output(self, tmp_path: Path) -> None:
        """Results are sorted by folder then name."""
        _create_image(tmp_path / "setting" / "b_place.jpg")
        _create_image(tmp_path / "character" / "a_person.jpg")
        _create_image(tmp_path / "character" / "z_person.jpg")

        refs = discover_concepts(tmp_path)

        # character/ comes before setting/ alphabetically
        assert refs[0].folder_name == "character"
        assert refs[0].name == "a_person"
        assert refs[1].folder_name == "character"
        assert refs[1].name == "z_person"
        assert refs[2].folder_name == "setting"

    def test_image_extensions(self, tmp_path: Path) -> None:
        """All supported image extensions are discovered."""
        for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"]:
            _create_image(tmp_path / "object" / f"file{ext}")

        refs = discover_concepts(tmp_path)
        assert len(refs) == 7

    def test_case_insensitive_extensions(self, tmp_path: Path) -> None:
        """Image extensions are case-insensitive."""
        _create_image(tmp_path / "object" / "file.JPG")
        _create_image(tmp_path / "object" / "file2.Png")

        refs = discover_concepts(tmp_path)
        assert len(refs) == 2


class TestPrintConceptSummary:
    """Tests for print_concept_summary()."""

    def test_empty_list(self, capsys) -> None:
        print_concept_summary([])
        captured = capsys.readouterr()
        assert "No concept references found" in captured.out

    def test_grouped_by_type(self, capsys) -> None:
        refs = [
            ConceptReference("holly", ConceptType.CHARACTER, Path("x.jpg"), "character"),
            ConceptReference("paul", ConceptType.CHARACTER, Path("y.jpg"), "character"),
            ConceptReference("tiffanys", ConceptType.SETTING, Path("z.jpg"), "setting"),
        ]
        print_concept_summary(refs)
        captured = capsys.readouterr()
        assert "character" in captured.out
        assert "holly" in captured.out
        assert "paul" in captured.out
        assert "setting" in captured.out
        assert "tiffanys" in captured.out
