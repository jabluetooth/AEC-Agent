"""
Revit extraction pipeline.

Extracts element metadata from Revit via the pyRevit sidecar
and streams to PostgreSQL.
"""

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, List, AsyncIterator
from uuid import UUID, uuid4
import time

import structlog

from aec_agent.db.connection import DatabasePool
from aec_agent.db.models import Element, Project, ElementRelationship, ExtractionResult
from aec_agent.db.repository import ElementRepository
from aec_agent.extraction.base import BaseExtractor, ExtractionConfig
from aec_agent.extraction.geometry import (
    revit_location_to_wkt,
    compute_centroid,
    compute_bounds,
)
from aec_agent.mcp.sidecar_client import call_sidecar, SidecarError

logger = structlog.get_logger(__name__)


# Conversion factor: Revit internal units are feet
FEET_TO_METERS = 0.3048


class RevitExtractor(BaseExtractor):
    """
    Extractor for Revit models.

    Uses the pyRevit sidecar to extract element data including:
    - ElementId (stable within document)
    - UniqueId (GUID, stable across sessions)
    - Category, Family, Type
    - Level, Phase
    - Location (point or curve)
    - Bounding box
    - Parameters (instance + type)
    - Host relationships (doors in walls)
    - Structural connections
    """

    def __init__(
        self,
        pool: DatabasePool,
        config: Optional[ExtractionConfig] = None
    ):
        """
        Initialize Revit extractor.

        Args:
            pool: Database connection pool
            config: Extraction configuration
        """
        super().__init__(config)
        self._pool = pool
        self._repository = ElementRepository(pool)

    @property
    def source(self) -> str:
        return "revit"

    async def extract_full(self, file_path: str) -> ExtractionResult:
        """
        Perform full extraction of all elements.

        Args:
            file_path: Path to the Revit model (for tracking)

        Returns:
            Extraction result with statistics
        """
        start_time = time.time()
        result = ExtractionResult(project_id=uuid4())

        try:
            # Get or create project
            project = await self._repository.get_project_by_file_path(file_path)
            if not project:
                file_hash = await self.compute_file_hash(file_path)

                project = Project(
                    name=Path(file_path).stem,
                    source="revit",
                    file_path=file_path,
                    file_hash=file_hash,
                    extracted_at=datetime.utcnow(),
                )
                await self._repository.create_project(project)
            else:
                # Clear existing elements for re-extraction
                await self._repository.delete_project_elements(project.id)

            result.project_id = project.id

            # Stream elements and ingest
            batch_count = 0
            async for batch in self.stream_entities():
                elements = []
                for element_data in batch:
                    element = self._element_data_to_element(element_data, project.id)
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

            # Extract relationships from Revit API
            relationships = await self._extract_relationships(project.id)
            for rel in relationships:
                await self._repository.create_relationship(rel)
            result.relationships_computed = len(relationships)

            # Update extraction time
            await self._repository.update_project_extraction_time(project.id)

            logger.info(
                "Revit extraction complete",
                project_id=str(project.id),
                elements=result.elements_extracted,
                relationships=result.relationships_computed,
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
        changed_ids: List[str]
    ) -> ExtractionResult:
        """
        Extract only changed elements.

        Args:
            project_id: Project to update
            changed_ids: List of changed element IDs

        Returns:
            Extraction result with statistics
        """
        start_time = time.time()
        result = ExtractionResult(project_id=project_id)

        try:
            for element_id in changed_ids:
                response = await call_sidecar(
                    endpoint="/mcp/elements/get",
                    method="POST",
                    payload={"element_id": int(element_id)},
                    sidecar_type="revit"
                )

                if response.get("success"):
                    element_data = response.get("data", {})
                    element = self._element_data_to_element(element_data, project_id)
                    if element:
                        await self._repository.upsert_element(element)
                        result.elements_updated += 1
                else:
                    # Element may have been deleted
                    existing = await self._repository.get_element_by_source_id(
                        project_id, element_id, "revit"
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
        batch_size: Optional[int] = None
    ) -> AsyncIterator[List[dict]]:
        """
        Stream elements in batches from the sidecar.

        Args:
            batch_size: Override default batch size

        Yields:
            Batches of element dicts
        """
        batch_size = batch_size or self.config.batch_size
        offset = 0

        while True:
            payload = {
                "offset": offset,
                "limit": batch_size,
                "include_parameters": self.config.include_properties,
            }

            if self.config.category_filter:
                payload["categories"] = self.config.category_filter

            try:
                response = await call_sidecar(
                    endpoint="/mcp/extract/batch",
                    method="POST",
                    payload=payload,
                    sidecar_type="revit"
                )

                if not response.get("success"):
                    error = response.get("error", {})
                    logger.error(
                        "Extraction batch failed",
                        error=error.get("message", "Unknown error")
                    )
                    break

                data = response.get("data", {})
                elements = data.get("elements", [])

                if not elements:
                    break

                yield elements

                has_more = data.get("has_more", False)
                if not has_more:
                    break

                offset += len(elements)

            except SidecarError as e:
                logger.error("Stream elements failed", error=str(e))
                break

    async def compute_file_hash(self, file_path: str) -> str:
        """
        Compute hash of Revit model file.

        Args:
            file_path: Path to RVT file

        Returns:
            SHA256 hash string
        """
        try:
            path = Path(file_path)
            if path.exists():
                stats = path.stat()
                content = f"{stats.st_size}:{stats.st_mtime}"
                return hashlib.sha256(content.encode()).hexdigest()[:16]
        except Exception as e:
            logger.warning("Could not compute file hash", error=str(e))

        return hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]

    async def _extract_relationships(self, project_id: UUID) -> List[ElementRelationship]:
        """
        Extract relationships from Revit API.

        Gets host relationships (doors in walls) and structural connections.

        Args:
            project_id: Project ID

        Returns:
            List of relationships
        """
        relationships = []

        try:
            response = await call_sidecar(
                endpoint="/mcp/extract/relationships",
                method="POST",
                payload={},
                sidecar_type="revit"
            )

            if response.get("success"):
                rel_data = response.get("data", {})

                # Host relationships (doors/windows in walls)
                for host_rel in rel_data.get("hosted", []):
                    host_id = str(host_rel.get("host_id"))
                    hosted_id = str(host_rel.get("hosted_id"))

                    # Get element UUIDs
                    host_elem = await self._repository.get_element_by_source_id(
                        project_id, host_id, "revit"
                    )
                    hosted_elem = await self._repository.get_element_by_source_id(
                        project_id, hosted_id, "revit"
                    )

                    if host_elem and hosted_elem:
                        relationships.append(ElementRelationship(
                            project_id=project_id,
                            from_element_id=host_elem.id,
                            to_element_id=hosted_elem.id,
                            relation_type="hosts",
                            source="revit_api",
                            confidence=1.0,
                        ))

                # Structural connections
                for conn in rel_data.get("connections", []):
                    from_id = str(conn.get("from_id"))
                    to_id = str(conn.get("to_id"))

                    from_elem = await self._repository.get_element_by_source_id(
                        project_id, from_id, "revit"
                    )
                    to_elem = await self._repository.get_element_by_source_id(
                        project_id, to_id, "revit"
                    )

                    if from_elem and to_elem:
                        relationships.append(ElementRelationship(
                            project_id=project_id,
                            from_element_id=from_elem.id,
                            to_element_id=to_elem.id,
                            relation_type="connected_to",
                            source="revit_api",
                            confidence=1.0,
                        ))

        except SidecarError as e:
            logger.warning("Could not extract relationships", error=str(e))

        return relationships

    def _element_data_to_element(
        self,
        element_data: dict,
        project_id: UUID
    ) -> Optional[Element]:
        """
        Convert element data from sidecar to Element model.

        Args:
            element_data: Raw element data from sidecar
            project_id: Parent project ID

        Returns:
            Element model or None if invalid
        """
        element_id = element_data.get("element_id") or element_data.get("ElementId")
        if not element_id:
            return None

        category = element_data.get("category") or element_data.get("Category")
        family = element_data.get("family") or element_data.get("Family")
        type_name = element_data.get("type_name") or element_data.get("TypeName")

        # Process location (already in meters from sidecar)
        location_info = element_data.get("location") or element_data.get("Location", {})
        geom_wkt = None
        centroid = None
        bounds = None

        if location_info:
            geom_wkt = revit_location_to_wkt(location_info)
            centroid = compute_centroid(location_info, "revit")
            bounds = compute_bounds(location_info, "revit")

        # Extract parameters as properties
        properties = {}
        parameters = element_data.get("parameters") or element_data.get("Parameters", {})
        if parameters:
            properties.update(parameters)

        # Add level if present
        level = element_data.get("level") or element_data.get("Level")
        if level:
            properties["level"] = level

        # Map category to entity_type for consistency
        entity_type = self._category_to_entity_type(category)

        return Element(
            project_id=project_id,
            source_id=str(element_id),
            source="revit",
            entity_type=entity_type,
            category=category,
            family=family,
            type_name=type_name,
            geom_wkt=geom_wkt,
            centroid=centroid,
            bounds=bounds,
            properties=properties,
        )

    def _category_to_entity_type(self, category: str) -> str:
        """
        Map Revit category to normalized entity type.

        Args:
            category: Revit category name

        Returns:
            Normalized entity type
        """
        if not category:
            return "Unknown"

        # Normalize common categories
        category_map = {
            "Walls": "WALL",
            "Doors": "DOOR",
            "Windows": "WINDOW",
            "Floors": "FLOOR",
            "Ceilings": "CEILING",
            "Roofs": "ROOF",
            "Stairs": "STAIR",
            "Railings": "RAILING",
            "Columns": "COLUMN",
            "Structural Columns": "COLUMN",
            "Structural Framing": "BEAM",
            "Furniture": "FURNITURE",
            "Mechanical Equipment": "EQUIPMENT",
            "Electrical Equipment": "EQUIPMENT",
            "Plumbing Fixtures": "FIXTURE",
            "Rooms": "ROOM",
            "Areas": "AREA",
            "Levels": "LEVEL",
            "Grids": "GRID",
        }

        return category_map.get(category, category.upper())
