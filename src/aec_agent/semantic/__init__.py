"""
Semantic search layer for AEC Agent.

Provides embeddings generation and semantic search using sentence-transformers and pgvector.
"""

from aec_agent.semantic.description_generator import (
    generate_autocad_description,
    generate_description,
    generate_revit_description,
)
from aec_agent.semantic.embeddings import EmbeddingService, get_embedding_service
from aec_agent.semantic.search import SemanticSearch, resolve_element

__all__ = [
    "EmbeddingService",
    "get_embedding_service",
    "generate_description",
    "generate_autocad_description",
    "generate_revit_description",
    "SemanticSearch",
    "resolve_element",
]
