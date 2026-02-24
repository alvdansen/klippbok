"""CLIP embedding computation for triage matching.

Uses OpenAI's CLIP model (via HuggingFace Transformers) to compute
image embeddings for both reference images and sampled clip frames.
Cosine similarity between embeddings determines how closely a clip
frame matches a concept reference.

CLIP is the right tool here because it understands semantic content —
it can match a reference portrait of Holly Golightly against movie
frames showing her from different angles, lighting, and distances.

Requires: pip install torch transformers
These are optional dependencies — a clear install message is shown
if they're missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Lazy imports — torch and transformers are optional heavy dependencies.
# We check at runtime and give a clear install message if missing.
_CLIP_AVAILABLE = False
_IMPORT_ERROR: str | None = None

try:
    import torch
    from transformers import CLIPModel, CLIPProcessor
    _CLIP_AVAILABLE = True
except ImportError as e:
    _IMPORT_ERROR = str(e)


def check_clip_available() -> None:
    """Raise a helpful error if CLIP dependencies aren't installed.

    Raises:
        ImportError: with install instructions if torch or transformers
            are missing.
    """
    if not _CLIP_AVAILABLE:
        raise ImportError(
            "Triage requires PyTorch and Transformers for CLIP embeddings.\n"
            "Install with:\n"
            "  pip install torch transformers\n"
            "\n"
            "On Windows with CUDA:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
            "  pip install transformers\n"
            f"\n(Original error: {_IMPORT_ERROR})"
        )


class CLIPEmbedder:
    """Computes CLIP image embeddings for triage matching.

    Loads a CLIP model once and reuses it for all images. Embeddings
    are L2-normalized so cosine similarity = dot product.

    The model runs on CPU by default — for 25 clips x 5 frames = 125
    images + a few references, CPU is fast enough (< 1 min total).

    Usage:
        embedder = CLIPEmbedder()
        emb1 = embedder.encode_image(Path("reference.jpg"))
        emb2 = embedder.encode_image(Path("frame.png"))
        similarity = embedder.similarity(emb1, emb2)
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """Load the CLIP model and processor.

        Args:
            model_name: HuggingFace model ID for the CLIP model.
                Default is the standard ViT-B/32 (~600MB download on
                first use, cached afterwards).

        Raises:
            ImportError: if torch or transformers aren't installed.
        """
        check_clip_available()

        self.model_name = model_name
        self._processor = CLIPProcessor.from_pretrained(model_name)
        self._model = CLIPModel.from_pretrained(model_name)
        self._model.eval()

    def _get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run pixel values through the vision encoder + projection.

        Uses the model's components directly rather than get_image_features()
        because newer transformers versions wrap the output in a ModelOutput
        object, which breaks batch indexing. This approach is stable across
        all versions.

        Args:
            pixel_values: Preprocessed image tensor from CLIPProcessor.

        Returns:
            Tensor of shape (batch_size, 512) — projected image embeddings.
        """
        vision_outputs = self._model.vision_model(pixel_values=pixel_values)
        # pooler_output is the CLS token after layer norm: (batch_size, 768)
        pooled = vision_outputs.pooler_output
        # visual_projection maps 768 -> 512 (shared embedding space)
        return self._model.visual_projection(pooled)

    def encode_image(self, image_path: str | Path) -> np.ndarray:
        """Compute a CLIP embedding for a single image.

        The embedding is L2-normalized, so cosine similarity between
        two embeddings is simply their dot product.

        Args:
            image_path: Path to the image file (PNG, JPG, WebP, etc.)

        Returns:
            1D numpy array of shape (512,) — the normalized embedding.

        Raises:
            FileNotFoundError: if the image doesn't exist.
            ValueError: if the image can't be loaded.
        """
        from PIL import Image

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        with torch.no_grad():
            inputs = self._processor(images=image, return_tensors="pt")
            features = self._get_image_features(inputs["pixel_values"])

        # features shape: (1, 512) — one image, 512-dim embedding
        embedding = features[0].numpy()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def encode_images(self, image_paths: list[Path]) -> list[np.ndarray]:
        """Compute CLIP embeddings for multiple images (batched).

        More efficient than calling encode_image() in a loop because
        images are processed in a single forward pass.

        Args:
            image_paths: List of image file paths.

        Returns:
            List of normalized embedding arrays, same order as input.
        """
        from PIL import Image

        if not image_paths:
            return []

        images = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            images.append(img)

        with torch.no_grad():
            inputs = self._processor(images=images, return_tensors="pt")
            features = self._get_image_features(inputs["pixel_values"])

        # features shape: (batch_size, 512)
        embeddings = []
        for i in range(len(image_paths)):
            emb = features[i].numpy()
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            embeddings.append(emb)

        return embeddings

    def encode_text(self, text: str) -> np.ndarray:
        """Compute a CLIP embedding for a text prompt.

        CLIP maps images and text into the same embedding space, so
        text embeddings can be compared directly against image embeddings.
        This enables zero-shot classification — e.g. checking if a frame
        looks like "a movie title card with text overlay".

        Args:
            text: The text prompt to encode.

        Returns:
            1D numpy array of shape (512,) — the normalized embedding.
        """
        with torch.no_grad():
            inputs = self._processor(text=[text], return_tensors="pt", padding=True)
            text_outputs = self._model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            # pooler_output: CLS token after layer norm (batch_size, 512 or 768)
            pooled = text_outputs.pooler_output
            # text_projection maps to shared 512-dim space
            features = self._model.text_projection(pooled)

        embedding = features[0].numpy()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def encode_texts(self, texts: list[str]) -> list[np.ndarray]:
        """Compute CLIP embeddings for multiple text prompts (batched).

        Args:
            texts: List of text prompts.

        Returns:
            List of normalized embedding arrays, same order as input.
        """
        if not texts:
            return []

        with torch.no_grad():
            inputs = self._processor(text=texts, return_tensors="pt", padding=True)
            text_outputs = self._model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            pooled = text_outputs.pooler_output
            features = self._model.text_projection(pooled)

        embeddings = []
        for i in range(len(texts)):
            emb = features[i].numpy()
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            embeddings.append(emb)
        return embeddings

    @staticmethod
    def similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two normalized embeddings.

        Since embeddings are L2-normalized, this is just a dot product.

        Args:
            embedding1: First embedding (1D array).
            embedding2: Second embedding (1D array).

        Returns:
            Similarity score from -1.0 to 1.0 (practically 0.0 to 1.0
            for image embeddings). Higher = more similar.
        """
        return float(np.dot(embedding1, embedding2))

    @staticmethod
    def best_match_score(
        frame_embeddings: list[np.ndarray],
        reference_embedding: np.ndarray,
    ) -> tuple[float, int]:
        """Find which frame best matches a reference, and how well.

        Compares each frame embedding against the reference and returns
        the highest similarity score and the index of the best frame.

        Args:
            frame_embeddings: List of embeddings from sampled clip frames.
            reference_embedding: Embedding of the concept reference image.

        Returns:
            Tuple of (best_similarity, best_frame_index).
            Returns (0.0, 0) if frame_embeddings is empty.
        """
        if not frame_embeddings:
            return 0.0, 0

        best_score = -1.0
        best_index = 0

        for i, frame_emb in enumerate(frame_embeddings):
            score = float(np.dot(frame_emb, reference_embedding))
            if score > best_score:
                best_score = score
                best_index = i

        return max(best_score, 0.0), best_index
