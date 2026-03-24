"""Sentence-transformers embedding wrapper for ChromaDB"""
import logging
import chromadb.utils.embedding_functions as embedding_functions

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Wrapper for sentence-transformers models compatible with ChromaDB"""

    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', device: str = 'cpu'):
        """
        Initialize embedding model

        Args:
            model_name: HuggingFace model name (default: all-MiniLM-L6-v2)
            device: 'cpu' or 'cuda'
        """
        self.model_name = model_name
        self.device = device

        logger.info(f"Initializing embedding model: {model_name} on {device}")

        # Create ChromaDB-compatible embedding function
        self.chroma_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name,
            device=device
        )

        logger.info("Embedding model initialized successfully")

    def __call__(self, texts):
        """Make the instance callable for ChromaDB"""
        return self.chroma_ef(texts)
