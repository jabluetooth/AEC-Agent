"""
AutoCAD extraction pipeline.

Extracts entity metadata from AutoCAD via the .NET sidecar
and streams to PostgreSQL.
"""

import hashlib
import time
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

import structlog

from aec_agent.db.connection import DatabasePool
from aec_agent.db.models import Element, ExtractionResult, Project
from aec_agent.db.repository import ElementRepository
from aec_agent.extraction.base import BaseExtractor, ExtractionConfig
from aec_agent.extraction.geometry import (
    autocad_geometry_to_wkt,
    compute_bounds,
    compute_centroid,
)
from aec_agent.mcp.sidecar_client import SidecarError, call_autocad_command

logger = structlog.get_logger(__name__)


class AutoCADExtractor(BaseExtractor):
    """
    Extractor for AutoCAD drawings.

    Uses the .NET sidecar to extract entity data including:
    - Handle (stable ID)
    - ObjectId (session ID)
    - Entity type
    - Layer, color, linetype
    - Geometry (type-specific)
    - Bounding box
    - Extended data (XData)
    - Block attributes
    """

    def __init__(
        self,
        pool: DatabasePool,
        config: ExtractionConfig | None = None
    ):
        """
        Initialize AutoCAD extractor.

        Args:
            pool: Database connection pool
            config: Extraction configuration
        """
        super().__init__(config)
        self._pool = pool
        self._repository = ElementRepository(pool)

    @property
    def source(self) -> str:
        return "autocad"

    async def extract_full(self, file_path: str) -> ExtractionResult:
        """
        Perform full extraction of all entities.

        Args:
            file_path: Path to the drawing (for tracking)

        Returns:
            Extraction result with statistics
        """
        start_time = time.time()
        result = ExtractionResult(project_id=uuid4())

        try:
            # Get or create project
            project = await self._repository.get_project_by_file_path(file_path)
            if not project:
                # Compute file hash for change detection
                file_hash = await self.compute_file_hash(file_path)

                project = Project(
                    name=Path(file_path).stem,
                    source="autocad",
                    file_path=file_path,
                    file_hash=file_hash,
                    extracted_at=datetime.utcnow(),
                )
                await self._repository.create_project(project)
            else:
                # Clear existing elements for re-extraction
                await self._repository.delete_project_elements(project.id)

            result.project_id = project.id

            # Stream entities and ingest
            batch_count = 0
            async for batch in self.stream_entities():
                elements = []
                for entity_data in batch:
                    element = self._entity_to_element(entity_data, project.id)
                    if element:
                        elements.append(element)

                if elements:
                    count = await self._repository.upsert_elements_batch(elements)
                    result.elements_extracted += count
                    batch_count += 1
                    logger.debug(
                        "Extracted batch",
                        batch=batch_count,
                        count=count
                    )

            # Update extraction time
            await self._repository.update_project_extraction_time(project.id)

            logger.info(
                "AutoCAD extraction complete",
                project_id=str(project.id),
                elements=result.elements_extracted,
                duration_ms=int((time.time() - start_time) * 1000)
            )

        except SidecarError as e:
            result.errors.append(f"Sidecar error: {e.message}")
            logger.error("Extraction failed", error=str(e))

        except Exception as e:
            result.errors.append(str(e))
            logger.error("Extraction failed", error=str(e))

        result.duration_ms = int((time.time() - start_time) * 1000)
        return result

    async def extract_incremental(
        self,
        project_id: UUID,
        changed_ids: list[str]
    ) -> ExtractionResult:
        """
        Extract only changed entities.

        Args:
            project_id: Project to update
            changed_ids: List of changed entity handles

        Returns:
            Extraction result with statistics
        """
        start_time = time.time()
        result = ExtractionResult(project_id=project_id)

        try:
            for handle in changed_ids:
                # Get entity by handle
                entity_response = await call_autocad_command(
                    "get_entity",
                    {
                        "handle": handle,
                        "include_geometry": self.config.include_geometry,
                        "include_xdata": self.config.include_xdata,
                    }
                )

                if entity_response.get("success"):
                    entity_data = entity_response.get("data", {})
                    element = self._entity_to_element(entity_data, project_id)
                    if element:
                        await self._repository.upsert_element(element)
                        result.elements_updated += 1
                else:
                    # Entity may have been deleted
                    existing = await self._repository.get_element_by_source_id(
                        project_id, handle, "autocad"
                    )
                    if existing:
                        await self._repository.soft_delete_element(existing.id)
                        result.elements_deleted += 1

            logger.info(
                "Incremental extraction complete",
                project_id=str(project_id),
                updated=result.elements_updated,
                deleted=result.elements_deleted
            )

        except Exception as e:
            result.errors.append(str(e))
            logger.error("Incremental extraction failed", error=str(e))

        result.duration_ms = int((time.time() - start_time) * 1000)
        return result

    async def stream_entities(
        self,
        batch_size: int | None = None
    ) -> AsyncIterator[list[dict]]:
        """
        Stream entities in batches from the sidecar.

        Uses the extract_all_entities command which returns paginated
        results with full geometry for PostgreSQL storage.

        Args:
            batch_size: Override default batch size

        Yields:
            Batches of entity dicts
        """
        batch_size = batch_size or self.config.batch_size
        offset = 0

        try:
            while True:
                params = {
                    "offset": offset,
                    "limit": batch_size,
                    "include_geometry": self.config.include_geometry,
                    "include_xdata": self.config.include_xdata,
                }

                if self.config.layer_filter:
                    params["layer_filter"] = self.config.layer_filter

                try:
                    response = await call_autocad_command("extract_all_entities", params)

                    if not response.get("success"):
                        error = response.get("error", {})
                        logger.error(
                            "Extraction batch failed",
                            error=error.get("message", "Unknown error")
                        )
                        break

                    data = response.get("data", {})
                    entities = data.get("entities", [])

                    if not entities:
                        break

                    yield entities

                    # Check if there are more
                    has_more = data.get("has_more", False)
                    if not has_more:
                        break

                    offset += len(entities)

                except SidecarError as e:
                    logger.error("Stream entities failed", error=str(e))
                    break
        except GeneratorExit:
            logger.debug("Entity stream closed")
        finally:
            logger.debug("Entity stream finished", offset=offset)

    async def compute_file_hash(self, file_path: str) -> str:
        """
        Compute hash of drawing file.

        Args:
            file_path: Path to DWG file

        Returns:
            SHA256 hash string
        """
        try:
            path = Path(file_path)
            if path.exists():
                # Use file stats for quick hash (size + mtime)
                stats = path.stat()
                content = f"{stats.st_size}:{stats.st_mtime}"
                return hashlib.sha256(content.encode()).hexdigest()[:16]
        except Exception as e:
            logger.warning("Could not compute file hash", error=str(e))

        # Fallback: use current timestamp
        return hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]

    def _entity_to_element(
        self,
        entity_data: dict,
        project_id: UUID
    ) -> Element | None:
        """
        Convert entity data from sidecar to Element model.

        Args:
            entity_data: Raw entity data from sidecar
            project_id: Parent project ID

        Returns:
            Element model or None if invalid
        """
        handle = entity_data.get("handle") or entity_data.get("Handle")
        if not handle:
            return None

        entity_type = entity_data.get("entity_type") or entity_data.get("EntityType", "Unknown")
        layer = entity_data.get("layer") or entity_data.get("Layer")

        # Process geometry
        geometry_info = entity_data.get("geometry") or entity_data.get("Geometry", {})
        if not geometry_info and "bounds" in entity_data:
            # Create minimal geometry from bounds
            geometry_info = {"type": entity_type, "bounds": entity_data["bounds"]}

        geom_wkt = None
        centroid = None
        bounds = None

        if geometry_info:
            geom_wkt = autocad_geometry_to_wkt(geometry_info)
            centroid = compute_centroid(geometry_info, "autocad")
            bounds = compute_bounds(geometry_info, "autocad")

        # Extract properties
        properties = entity_data.get("properties") or entity_data.get("Properties", {})
        xdata = entity_data.get("xdata") or entity_data.get("XData", {})
        if xdata:
            properties["xdata"] = xdata

        # Add color and linetype to properties
        if "color" in entity_data or "Color" in entity_data:
            properties["color"] = entity_data.get("color") or entity_data.get("Color")
        if "linetype" in entity_data or "Linetype" in entity_data:
            properties["linetype"] = entity_data.get("linetype") or entity_data.get("Linetype")

        return Element(
            project_id=project_id,
            source_id=handle,
            source="autocad",
            entity_type=entity_type,
            layer=layer,
            geom_wkt=geom_wkt,
            centroid=centroid,
            bounds=bounds,
            properties=properties,
        )
