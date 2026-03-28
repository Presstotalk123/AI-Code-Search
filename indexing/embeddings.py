"""Sentence-transformers embedding model for Solr vector indexing"""
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Wrapper for sentence-transformers that produces raw float arrays for Solr vector indexing"""

    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', device: str = 'cpu'):
        """
        Initialize embedding model

        Args:
            model_name: HuggingFace model name (default: all-MiniLM-L6-v2, 384 dims)
            device: 'cpu' or 'cuda'
        """
        self.model_name = model_name
        self.device = device

        logger.info(f"Initializing embedding model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        logger.info("Embedding model initialized successfully")

    def encode(self, texts: list, batch_size: int = 32) -> np.ndarray:
        """
        Encode a list of texts into embedding vectors.

        Args:
            texts: List of strings to embed
            batch_size: Number of texts per encoding batch

        Returns:
            numpy array of shape (len(texts), 384)
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2-normalise for cosine similarity
        )

    def __call__(self, texts: list) -> np.ndarray:
        """Make the instance callable - returns raw numpy array"""
        return self.encode(texts)
