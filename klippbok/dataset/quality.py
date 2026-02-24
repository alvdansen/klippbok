"""Dataset quality metrics: exposure, motion, perceptual hashing.

Pure measurement functions — path in, number out. Follows the same
pattern as klippbok.video.image_quality (compute_sharpness). Each metric
is independent and can be used standalone.

Why these three:
- Exposure: catches under/overexposed frames that wash out VAE encoding
- Motion: catches static shots (useless for video) and chaotic motion (unlearnable)
- dHash: catches near-duplicate images that waste training compute

dHash is implemented manually in numpy (~15 lines) to avoid adding
imagehash as a dependency. It's the simplest perceptual hash that still
works well for near-duplicate detection.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Exposure measurement
# ---------------------------------------------------------------------------

def compute_exposure(image_path: str | Path) -> tuple[float, float]:
    """Measure brightness of an image as mean and standard deviation.

    Loads the image as grayscale and computes mean pixel intensity
    (0.0 = pure black, 1.0 = pure white) and standard deviation
    (low = uniform brightness, high = high contrast).

    Why normalized: raw pixel values (0-255) are hardware-dependent.
    Normalizing to 0.0-1.0 makes thresholds portable across color depths.

    Args:
        image_path: Path to the image file.

    Returns:
        Tuple of (mean_brightness, std_brightness), both in 0.0-1.0 range.

    Raises:
        FileNotFoundError: if the image file doesn't exist.
        ValueError: if the file can't be read as an image.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(
            f"Cannot read image '{image_path}'. "
            f"File may be corrupted or not a supported image format."
        )

    mean, std = cv2.meanStdDev(gray)
    # cv2.meanStdDev returns nested arrays: [[mean]], [[std]]
    return float(mean[0][0]) / 255.0, float(std[0][0]) / 255.0


# ---------------------------------------------------------------------------
# Motion intensity measurement
# ---------------------------------------------------------------------------

def compute_motion_intensity(
    video_path: str | Path,
    sample_count: int = 5,
) -> float:
    """Estimate motion intensity of a video clip via frame differencing.

    Samples evenly-spaced consecutive frame pairs and computes the mean
    absolute pixel difference. Higher = more motion between frames.

    Returns a percentage (0.0-100.0) where:
    - 0.0 = completely static (all frames identical)
    - ~1-5 = subtle motion (talking head, gentle sway)
    - ~5-20 = moderate motion (walking, panning)
    - ~20+ = intense motion (action, fast cuts)

    Why frame differencing: it's simple, fast, and correlates well with
    how much the model needs to learn about temporal dynamics. More
    sophisticated optical flow isn't worth the cost for a screening metric.

    Args:
        video_path: Path to the video file.
        sample_count: Number of frame pairs to sample. More = more accurate
            but slower. Default 5 is plenty for screening.

    Returns:
        Mean pixel change as a percentage (0.0-100.0).

    Raises:
        FileNotFoundError: if the video file doesn't exist.
        ValueError: if the video can't be opened or has too few frames.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(
            f"Cannot open video '{video_path}'. "
            f"File may be corrupted or codec not supported."
        )

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 2:
            raise ValueError(
                f"Video '{video_path.name}' has {total_frames} frame(s). "
                f"Need at least 2 frames to measure motion."
            )

        # Pick evenly-spaced frame indices to sample
        # We need pairs of consecutive frames, so pick start indices
        max_start = total_frames - 2  # last valid start for a pair
        if sample_count >= max_start + 1:
            # Sample all available pairs
            indices = list(range(max_start + 1))
        else:
            step = max_start / sample_count
            indices = [int(i * step) for i in range(sample_count)]

        diffs: list[float] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok1, frame1 = cap.read()
            ok2, frame2 = cap.read()
            if not ok1 or not ok2:
                continue

            # Convert to grayscale for consistent comparison
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            # Mean absolute difference as percentage of max pixel value
            diff = cv2.absdiff(gray1, gray2)
            mean_diff = float(np.mean(diff)) / 255.0 * 100.0
            diffs.append(mean_diff)

        if not diffs:
            raise ValueError(
                f"Could not read any frames from '{video_path.name}'."
            )

        return float(np.mean(diffs))

    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Perceptual hashing (dHash)
# ---------------------------------------------------------------------------

def compute_dhash(image_path: str | Path, hash_size: int = 8) -> int:
    """Compute a difference hash (dHash) for perceptual duplicate detection.

    dHash encodes the relative brightness gradient between adjacent pixels.
    Two images that look similar to humans will have similar dHash values,
    even if they differ in resolution, compression, or minor cropping.

    Algorithm:
    1. Resize to (hash_size+1) x hash_size (one extra column for gradients)
    2. Convert to grayscale
    3. For each pixel, is the right neighbor brighter? → 1 bit
    4. Pack bits into an integer

    The result is a (hash_size * hash_size)-bit integer. Default hash_size=8
    produces a 64-bit hash.

    Args:
        image_path: Path to the image file.
        hash_size: Size of the hash grid. Default 8 = 64-bit hash.
            Larger = more discriminating but less tolerant of edits.

    Returns:
        Integer hash value.

    Raises:
        FileNotFoundError: if the image file doesn't exist.
        ValueError: if the file can't be read as an image.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load and resize to small grid
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(
            f"Cannot read image '{image_path}'. "
            f"File may be corrupted or not a supported image format."
        )

    # Resize to (hash_size+1) columns x hash_size rows
    resized = cv2.resize(img, (hash_size + 1, hash_size))

    # Compute horizontal gradient: is the right pixel brighter?
    # This produces a hash_size x hash_size boolean matrix
    diff = resized[:, 1:] > resized[:, :-1]

    # Pack bits into an integer
    # Flatten row-by-row and convert booleans to a single integer
    bits = diff.flatten()
    hash_value = 0
    for bit in bits:
        hash_value = (hash_value << 1) | int(bit)

    return hash_value


def hamming_distance(hash1: int, hash2: int) -> int:
    """Count the number of different bits between two hash values.

    Lower distance = more similar images. For 64-bit dHash:
    - 0 = identical (or extremely similar)
    - 1-5 = very similar (likely same image, different compression)
    - 6-10 = somewhat similar (same scene, different angle/crop)
    - 10+ = different images

    Args:
        hash1: First hash value.
        hash2: Second hash value.

    Returns:
        Number of differing bits.
    """
    xor = hash1 ^ hash2
    return bin(xor).count("1")


def find_duplicates(
    image_paths: list[Path],
    threshold: int = 6,
    hash_size: int = 8,
) -> list[list[Path]]:
    """Find groups of perceptually similar images.

    Computes dHash for each image, then groups images whose hamming
    distance is at or below the threshold. Uses simple pairwise comparison
    — efficient enough for training dataset sizes (hundreds, not millions).

    Args:
        image_paths: List of image file paths to compare.
        threshold: Maximum hamming distance to consider as duplicate.
            Default 6 catches most near-duplicates.
        hash_size: dHash grid size. Default 8 = 64-bit hash.

    Returns:
        List of duplicate groups, where each group is a list of 2+ paths
        that are perceptually similar. Images that have no duplicates are
        not included.
    """
    if len(image_paths) < 2:
        return []

    # Compute hashes for all images
    hashes: list[tuple[Path, int]] = []
    for path in image_paths:
        try:
            h = compute_dhash(path, hash_size)
            hashes.append((path, h))
        except (FileNotFoundError, ValueError):
            # Skip unreadable images — they'll be caught by other validators
            continue

    if len(hashes) < 2:
        return []

    # Pairwise comparison using union-find for grouping
    n = len(hashes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]  # path compression
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if hamming_distance(hashes[i][1], hashes[j][1]) <= threshold:
                union(i, j)

    # Collect groups
    groups: dict[int, list[Path]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(hashes[i][0])

    # Only return groups with 2+ members (actual duplicates)
    return [group for group in groups.values() if len(group) >= 2]
