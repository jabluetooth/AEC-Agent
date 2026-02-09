"""
Embedding service using sentence-transformers.

Generates vector embeddings for semantic search using pgvector.
"""


import structlog

from aec_agent.config.settings import get_settings

logger = structlog.get_logger(__name__)

# Optional import - gracefully handle missing sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None


class EmbeddingServiceError(Exception):
    """Embedding service error."""
    pass


class EmbeddingService:
    """
    Service for generating text embeddings.

    Uses sentence-transformers models for high-quality embeddings
    suitable for semantic search.
    """

    def __init__(self, model_name: str | None = None):
        """
        Initialize embedding service.

        Args:
            model_name: Sentence transformer model name
                        (default: from settings)
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise EmbeddingServiceError(
                "sentence-transformers is not installed. "
                "Install with: pip install sentence-transformers"
            )

        settings = get_settings()
        self._model_name = model_name or settings.embedding_model
        self._expected_dimension = settings.embedding_dimension
        self._model: SentenceTransformer | None = None

    def _ensure_model(self) -> SentenceTransformer:
        """Lazy load the model."""
        if self._model is None:
            logger.info("Loading embedding model", model=self._model_name)
            self._model = SentenceTransformer(self._model_name)

            # Verify dimension matches expected
            actual_dim = self._model.get_sentence_embedding_dimension()
            if actual_dim != self._expected_dimension:
                logger.warning(
                    "Embedding dimension mismatch",
                    expected=self._expected_dimension,
                    actual=actual_dim,
                    model=self._model_name
                )

            logger.info(
                "Embedding model loaded",
                model=self._model_name,
                dimension=actual_dim
            )

        return self._model

    def generate_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        if not text or not text.strip():
            return []

        model = self._ensure_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        More efficient than calling generate_embedding in a loop.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Filter empty texts but keep track of positions
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)

        if not valid_texts:
            return [[] for _ in texts]

        model = self._ensure_model()
        embeddings = model.encode(valid_texts, convert_to_numpy=True)

        # Reconstruct full list with empty vectors for empty texts
        result = [[] for _ in texts]
        for i, idx in enumerate(valid_indices):
            result[idx] = embeddings[i].tolist()

        return result

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._model is not None:
            return self._model.get_sentence_embedding_dimension()
        return self._expected_dimension

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None


# Global instance (lazy initialized)
_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService | None:
    """Get the global embedding service instance."""
    return _embedding_service


def initialize_embedding_service(model_name: str | None = None) -> EmbeddingService:
    """
    Initialize the global embedding service.

    Args:
        model_name: Optional model name override

    Returns:
        Initialized EmbeddingService
    """
    global _embedding_service

    if _embedding_service is None:
        _embedding_service = EmbeddingService(model_name)

    return _embedding_service


def close_embedding_service() -> None:
    """Release embedding service resources."""
    global _embedding_service

    if _embedding_service is not None:
        # Clear model from memory
        _embedding_service._model = None
        _embedding_service = None
        logger.info("Embedding service closed")
