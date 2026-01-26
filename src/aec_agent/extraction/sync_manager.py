"""
Sync manager for metadata extraction.

Orchestrates extraction, embedding generation, and relationship computation.
Handles both full syncs and incremental updates with debouncing.
"""

import asyncio
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Set, List
from uuid import UUID
import time

import structlog

from aec_agent.config.settings import get_settings
from aec_agent.db.connection import DatabasePool
from aec_agent.db.models import ExtractionResult, SyncStatus
from aec_agent.db.repository import ElementRepository
from aec_agent.db.queries.relationships import compute_all_relationships
from aec_agent.extraction.autocad_extractor import AutoCADExtractor
from aec_agent.extraction.revit_extractor import RevitExtractor
from aec_agent.extraction.base import ExtractionConfig
from aec_agent.semantic.embeddings import EmbeddingService
from aec_agent.semantic.description_generator import generate_description

logger = structlog.get_logger(__name__)


class SyncManager:
    """
    Orchestrates metadata synchronization.

    Manages:
    - Full extraction on document open
    - Incremental updates on modifications
    - Debounced sync to avoid excessive updates
    - Background embedding generation
    - Relationship computation
    """

    def __init__(
        self,
        pool: DatabasePool,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        """
        Initialize sync manager.

        Args:
            pool: Database connection pool
            embedding_service: Optional embedding service (can be lazy loaded)
        """
        self._pool = pool
        self._embeddings = embedding_service
        self._repository = ElementRepository(pool)

        settings = get_settings()
        self._debounce_ms = settings.sync_debounce_ms
        self._distance_threshold = settings.relationship_distance_threshold

        # Track pending syncs per project
        self._pending_syncs: Dict[UUID, Set[str]] = {}
        self._debounce_tasks: Dict[UUID, asyncio.Task] = {}
        self._sync_status: Dict[UUID, SyncStatus] = {}

    async def trigger_full_sync(
        self,
        source: str,
        file_path: str,
        force: bool = False,
    ) -> ExtractionResult:
        """
        Trigger full extraction and sync.

        Args:
            source: 'autocad' or 'revit'
            file_path: Path to drawing/model
            force: Force re-extraction even if file unchanged

        Returns:
            Extraction result
        """
        logger.info("Starting full sync", source=source, file_path=file_path, force=force)

        # Check for unchanged file (hash-based skip detection)
        if not force and file_path:
            existing_project = await self._repository.get_project_by_file_path(file_path)
            if existing_project:
                current_hash = await self._compute_file_hash(file_path)
                if current_hash and existing_project.file_hash == current_hash:
                    logger.info(
                        "File unchanged, skipping extraction",
                        project_id=str(existing_project.id),
                        file_hash=current_hash
                    )
                    return ExtractionResult(
                        project_id=existing_project.id,
                        elements_extracted=0,
                        elements_updated=0,
                        elements_deleted=0,
                        embeddings_generated=0,
                        relationships_computed=0,
                        duration_ms=0,
                        success=True,
                    )

        config = ExtractionConfig()

        if source == "autocad":
            extractor = AutoCADExtractor(self._pool, config)
        else:
            extractor = RevitExtractor(self._pool, config)

        # Extract elements
        result = await extractor.extract_full(file_path)

        if result.success:
            # Generate descriptions and embeddings
            result.embeddings_generated = await self._generate_embeddings_for_project(
                result.project_id
            )

            # Compute relationships
            rel_counts = await compute_all_relationships(
                self._pool,
                result.project_id,
                self._distance_threshold,
            )
            result.relationships_computed = sum(rel_counts.values())

            # Update sync status
            self._sync_status[result.project_id] = SyncStatus(
                project_id=result.project_id,
                last_sync=datetime.utcnow(),
                is_syncing=False,
            )

        return result

    async def trigger_incremental_sync(
        self,
        project_id: UUID,
        changed_ids: List[str],
        source: str = "autocad",
    ) -> ExtractionResult:
        """
        Trigger incremental extraction for changed elements.

        Args:
            project_id: Project to update
            changed_ids: List of changed element IDs
            source: 'autocad' or 'revit'

        Returns:
            Extraction result
        """
        logger.info(
            "Starting incremental sync",
            project_id=str(project_id),
            count=len(changed_ids)
        )

        config = ExtractionConfig()

        if source == "autocad":
            extractor = AutoCADExtractor(self._pool, config)
        else:
            extractor = RevitExtractor(self._pool, config)

        # Extract changed elements
        result = await extractor.extract_incremental(project_id, changed_ids)

        if result.success:
            # Generate embeddings for updated elements
            await self._generate_embeddings_for_elements(
                project_id, changed_ids
            )

            # Recompute relationships for affected elements
            from aec_agent.db.queries.relationships import recompute_relationships_for_element
            for source_id in changed_ids:
                element = await self._repository.get_element_by_source_id(
                    project_id, source_id
                )
                if element:
                    await recompute_relationships_for_element(
                        self._pool, element.id, self._distance_threshold
                    )

            # Update sync status
            if project_id in self._sync_status:
                self._sync_status[project_id].last_sync = datetime.utcnow()
                self._sync_status[project_id].is_syncing = False

        return result

    async def schedule_sync(
        self,
        project_id: UUID,
        changed_ids: List[str],
        source: str = "autocad",
    ) -> None:
        """
        Schedule a debounced sync for changed elements.

        Multiple changes within the debounce window are batched.

        Args:
            project_id: Project to update
            changed_ids: List of changed element IDs
            source: 'autocad' or 'revit'
        """
        # Add to pending changes
        if project_id not in self._pending_syncs:
            self._pending_syncs[project_id] = set()
        self._pending_syncs[project_id].update(changed_ids)

        # Cancel existing debounce task
        if project_id in self._debounce_tasks:
            self._debounce_tasks[project_id].cancel()

        # Schedule new debounced sync
        async def debounced_sync():
            await asyncio.sleep(self._debounce_ms / 1000)
            ids = list(self._pending_syncs.pop(project_id, set()))
            if ids:
                await self.trigger_incremental_sync(project_id, ids, source)

        self._debounce_tasks[project_id] = asyncio.create_task(debounced_sync())

    async def _generate_embeddings_for_project(
        self,
        project_id: UUID,
    ) -> int:
        """
        Generate embeddings for all elements in a project.

        Args:
            project_id: Project to process

        Returns:
            Number of embeddings generated
        """
        if not self._embeddings:
            logger.warning("Embedding service not available")
            return 0

        count = 0
        batch_size = 100

        # Query elements without embeddings
        query = """
            SELECT
                id, source_id, source, entity_type, layer, category,
                family, type_name, properties, description
            FROM elements
            WHERE project_id = $1
              AND deleted_at IS NULL
              AND embedding IS NULL
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, project_id)

        # Process in batches
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]

            # Generate descriptions
            descriptions = []
            element_ids = []
            for row in batch:
                element_dict = dict(row)
                source = element_dict["source"]
                desc = generate_description(element_dict, source)
                if desc:
                    descriptions.append(desc)
                    element_ids.append(row["id"])

            if not descriptions:
                continue

            # Generate embeddings
            embeddings = self._embeddings.generate_embeddings_batch(descriptions)

            # Update elements
            async with self._pool.transaction() as conn:
                for elem_id, desc, embedding in zip(element_ids, descriptions, embeddings):
                    if embedding:
                        await conn.execute(
                            """
                            UPDATE elements
                            SET description = $1, embedding = $2, updated_at = NOW()
                            WHERE id = $3
                            """,
                            desc,
                            embedding,
                            elem_id,
                        )
                        count += 1

        logger.info(
            "Generated embeddings",
            project_id=str(project_id),
            count=count
        )
        return count

    async def _generate_embeddings_for_elements(
        self,
        project_id: UUID,
        source_ids: List[str],
    ) -> int:
        """
        Generate embeddings for specific elements.

        Args:
            project_id: Project ID
            source_ids: Element source IDs

        Returns:
            Number of embeddings generated
        """
        if not self._embeddings:
            return 0

        count = 0
        for source_id in source_ids:
            element = await self._repository.get_element_by_source_id(
                project_id, source_id
            )
            if not element:
                continue

            # Generate description
            element_dict = {
                "entity_type": element.entity_type,
                "layer": element.layer,
                "category": element.category,
                "family": element.family,
                "type_name": element.type_name,
                "properties": element.properties,
                "centroid": element.centroid.model_dump() if element.centroid else None,
            }
            desc = generate_description(element_dict, element.source)
            if not desc:
                continue

            # Generate embedding
            embedding = self._embeddings.generate_embedding(desc)
            if not embedding:
                continue

            # Update element
            await self._pool.execute(
                """
                UPDATE elements
                SET description = $1, embedding = $2, updated_at = NOW()
                WHERE id = $3
                """,
                desc,
                embedding,
                element.id,
            )
            count += 1

        return count

    def get_sync_status(self, project_id: UUID) -> Optional[SyncStatus]:
        """Get sync status for a project."""
        return self._sync_status.get(project_id)

    def set_embedding_service(self, service: EmbeddingService) -> None:
        """Set or update the embedding service."""
        self._embeddings = service

    async def _compute_file_hash(self, file_path: str) -> str:
        """
        Compute file hash for change detection.

        Uses file size and modification time for fast hashing
        (actual file content hashing would be slow for large CAD files).

        Args:
            file_path: Path to the file

        Returns:
            Hash string or empty string if file not accessible
        """
        try:
            path = Path(file_path)
            if path.exists():
                stats = path.stat()
                # Combine size and mtime for a fast "fingerprint"
                content = f"{stats.st_size}:{stats.st_mtime}"
                return hashlib.sha256(content.encode()).hexdigest()[:16]
        except Exception as e:
            logger.warning("Could not compute file hash", file_path=file_path, error=str(e))

        return ""
