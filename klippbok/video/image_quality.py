"""Image quality scoring for extracted reference images.

Uses OpenCV to compute sharpness (Laplacian variance) and detect
blank/uniform frames. These are the quality gates that prevent garbage
reference images from corrupting the VAE latent representation.

WHY Laplacian variance: the Laplacian operator detects edges and
texture. A sharp image has high Laplacian variance (lots of detail).
A blurry or blank image has low variance (uniform regions). This is
the standard computational sharpness metric — simple, fast, robust.

All functions work on file paths (not pre-loaded arrays) so they
can be used standalone from the CLI without loading OpenCV upfront.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from klippbok.video.extract_models import ImageValidation


def compute_sharpness(image_path: str | Path) -> float:
    """Compute sharpness of an image using Laplacian variance.

    Loads the image as grayscale, applies the Laplacian operator
    (second-order derivative — detects edges), and returns the
    variance of the result. Higher = sharper = more detail.

    Typical ranges:
    - Blank/solid color: < 1.0
    - Very blurry: 1-50
    - Normal video frame: 50-500
    - Sharp photograph: 500-5000+

    Args:
        image_path: Path to the image file (PNG, JPG, etc.).

    Returns:
        Laplacian variance as a float. Higher = sharper.

    Raises:
        FileNotFoundError: if the image file doesn't exist.
        ValueError: if the file can't be read as an image.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # cv2.imread returns None for unreadable files (not an exception)
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(
            f"Cannot read image '{image_path}'. "
            f"File may be corrupted or not a supported image format."
        )

    # Laplacian detects edges; variance measures how much edge content there is
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def is_blank(image_path: str | Path, threshold: float = 5.0) -> bool:
    """Detect if an image is effectively blank (uniform color).

    A blank frame has almost no texture or edge information — it's
    a solid color, a black frame, or a white frame. These are useless
    as reference images because they carry no visual information for
    the VAE to encode.

    Uses Laplacian variance: if the variance is below the threshold,
    the image is considered blank.

    Args:
        image_path: Path to the image file.
        threshold: Laplacian variance below this = blank.
            Default 5.0 catches solid colors and near-uniform frames.
            Increase to catch slightly textured but still useless frames.

    Returns:
        True if the image is blank/uniform, False if it has content.

    Raises:
        FileNotFoundError: if the image file doesn't exist.
        ValueError: if the file can't be read as an image.
    """
    return compute_sharpness(image_path) < threshold


def validate_extracted_image(
    image_path: str | Path,
    expected_width: int | None = None,
    expected_height: int | None = None,
) -> ImageValidation:
    """Validate an extracted reference image for quality and resolution.

    Checks three things:
    1. Sharpness — is there enough detail for the VAE?
    2. Blank detection — is this a solid color / black / white frame?
    3. Resolution — does it match the expected dimensions from the source video?

    Args:
        image_path: Path to the image to validate.
        expected_width: Expected width in pixels (from source video metadata).
            None to skip resolution check.
        expected_height: Expected height in pixels.
            None to skip resolution check.

    Returns:
        ImageValidation with all quality metrics.

    Raises:
        FileNotFoundError: if the image file doesn't exist.
        ValueError: if the file can't be read as an image.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load to get dimensions (need color image for width/height)
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(
            f"Cannot read image '{image_path}'. "
            f"File may be corrupted or not a supported image format."
        )

    height, width = img.shape[:2]

    # Compute sharpness on grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(laplacian.var())

    # Resolution check
    resolution_ok = True
    if expected_width is not None and expected_height is not None:
        resolution_ok = (width == expected_width and height == expected_height)

    return ImageValidation(
        path=image_path,
        width=width,
        height=height,
        sharpness=sharpness,
        is_blank=sharpness < 5.0,
        resolution_ok=resolution_ok,
        expected_width=expected_width,
        expected_height=expected_height,
    )
