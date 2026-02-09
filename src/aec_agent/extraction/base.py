"""
Base extractor interface and configuration.

Defines the common interface for AutoCAD and Revit extractors.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from uuid import UUID

from aec_agent.db.models import ExtractionResult


@dataclass
class ExtractionConfig:
    """Configuration for extraction operations."""

    batch_size: int = 1000
    include_geometry: bool = True
    include_properties: bool = True
    include_xdata: bool = True
    layer_filter: str | None = None
    category_filter: list[str] | None = None
    generate_descriptions: bool = True
    compute_embeddings: bool = True


class BaseExtractor(ABC):
    """
    Abstract base class for CAD extractors.

    Provides common interface for AutoCAD and Revit extraction.
    """

    def __init__(self, config: ExtractionConfig | None = None):
        """
        Initialize extractor.

        Args:
            config: Extraction configuration
        """
        self.config = config or ExtractionConfig()

    @property
    @abstractmethod
    def source(self) -> str:
        """Return source identifier ('autocad' or 'revit')."""
        pass

    @abstractmethod
    async def extract_full(self, file_path: str) -> ExtractionResult:
        """
        Perform full extraction of all entities.

        Args:
            file_path: Path to the drawing/model

        Returns:
            Extraction result with statistics
        """
        pass

    @abstractmethod
    async def extract_incremental(
        self,
        project_id: UUID,
        changed_ids: list[str]
    ) -> ExtractionResult:
        """
        Extract only changed entities.

        Args:
            project_id: Project to update
            changed_ids: List of changed entity IDs

        Returns:
            Extraction result with statistics
        """
        pass

    @abstractmethod
    async def stream_entities(
        self,
        batch_size: int | None = None
    ) -> AsyncIterator[list[dict]]:
        """
        Stream entities in batches.

        Args:
            batch_size: Override default batch size

        Yields:
            Batches of entity dicts
        """
        pass

    @abstractmethod
    async def compute_file_hash(self, file_path: str) -> str:
        """
        Compute hash of file for change detection.

        Args:
            file_path: Path to file

        Returns:
            File hash string
        """
        pass

    async def check_file_changed(
        self,
        file_path: str,
        stored_hash: str | None
    ) -> bool:
        """
        Check if file has changed since last extraction.

        Args:
            file_path: Path to file
            stored_hash: Previously stored hash

        Returns:
            True if file changed
        """
        if stored_hash is None:
            return True

        current_hash = await self.compute_file_hash(file_path)
        return current_hash != stored_hash
