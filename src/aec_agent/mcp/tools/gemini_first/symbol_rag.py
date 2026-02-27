"""
RAG-based Symbol Recognition using CLIP embeddings.

Uses visual similarity search to identify CAD symbols from image patches,
mapping them to block names for AutoCAD insertion.

Architecture:
1. Extract symbol region from drawing image
2. Generate CLIP embedding (512-dim)
3. Query pgvector for nearest neighbors
4. Return matched block with confidence

References:
- CLIP: https://openai.com/research/clip
- pgvector: https://github.com/pgvector/pgvector
"""

from __future__ import annotations

import base64
import hashlib
import io
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from uuid import UUID

import numpy as np
import structlog

if TYPE_CHECKING:
    from PIL import Image as PILImage

logger = structlog.get_logger(__name__)

# Check for optional dependencies
try:
    import torch
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    Image = None

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    clip = None

try:
    from transformers import CLIPProcessor, CLIPModel
    TRANSFORMERS_CLIP_AVAILABLE = True
except ImportError:
    TRANSFORMERS_CLIP_AVAILABLE = False
    CLIPProcessor = None
    CLIPModel = None


# =============================================================================
# Data Models
# =============================================================================

class SymbolDomain(str, Enum):
    """Symbol domain categories matching NCS layer prefixes."""
    ELECTRICAL = "electrical"
    MECHANICAL = "mechanical"
    PLUMBING = "plumbing"
    FIRE = "fire"
    ARCHITECTURAL = "architectural"
    STRUCTURAL = "structural"
    CIVIL = "civil"


@dataclass
class SymbolMatch:
    """Result of a symbol recognition query."""
    symbol_id: UUID
    block_name: str
    display_name: str
    domain: SymbolDomain
    category: str
    subcategory: Optional[str]
    layer: str
    confidence: float  # 0.0 - 1.0 (1 - cosine_distance)
    distance: float  # Raw cosine distance
    attributes: dict = field(default_factory=dict)
    default_scale: float = 1.0
    default_rotation: float = 0.0


@dataclass
class SymbolLibraryEntry:
    """A symbol in the library."""
    id: UUID
    block_name: str
    display_name: str
    description: Optional[str]
    domain: str
    category: str
    subcategory: Optional[str]
    layer: str
    embedding: Optional[list[float]]
    preview_image: Optional[bytes]
    default_scale: float = 1.0
    default_rotation: float = 0.0
    attributes: dict = field(default_factory=dict)
    standards: list[str] = field(default_factory=list)
    jurisdiction: Optional[str] = None
    code_references: list[str] = field(default_factory=list)
    is_active: bool = True


@dataclass
class SymbolRecognitionConfig:
    """Configuration for symbol recognition."""
    # Model settings
    model_name: str = "openai/clip-vit-base-patch32"  # HuggingFace model ID
    embedding_dim: int = 512
    device: Optional[str] = None  # auto-detect if None

    # Search settings
    top_k: int = 5  # Number of candidates to return
    min_confidence: float = 0.5  # Minimum confidence threshold
    domain_filter: Optional[SymbolDomain] = None  # Filter by domain

    # Image preprocessing
    image_size: int = 224  # CLIP input size
    normalize: bool = True

    # Caching
    cache_embeddings: bool = True


# =============================================================================
# CLIP Encoder
# =============================================================================

class CLIPEncoder:
    """
    CLIP-based image encoder for symbol embeddings.

    Uses OpenAI's CLIP ViT-B/32 model (512-dim embeddings).
    Supports both OpenAI's clip package and HuggingFace transformers.
    """

    _instance: Optional["CLIPEncoder"] = None

    def __init__(self, config: Optional[SymbolRecognitionConfig] = None):
        """
        Initialize CLIP encoder.

        Args:
            config: Recognition configuration
        """
        self.config = config or SymbolRecognitionConfig()
        self._model = None
        self._preprocess = None
        self._processor = None
        self._device = None
        self._initialized = False
        self._use_transformers = False

    @classmethod
    def get_instance(cls, config: Optional[SymbolRecognitionConfig] = None) -> "CLIPEncoder":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    def _detect_device(self) -> str:
        """Detect best available device."""
        if self.config.device:
            return self.config.device

        if TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda"
        elif TORCH_AVAILABLE and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    async def initialize(self) -> None:
        """Initialize CLIP model (lazy loading)."""
        if self._initialized:
            return

        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is required for CLIP encoding. "
                "Install with: pip install torch torchvision"
            )

        self._device = self._detect_device()
        logger.info("Initializing CLIP encoder", device=self._device)

        # Try OpenAI CLIP first (faster, simpler)
        if CLIP_AVAILABLE:
            try:
                self._model, self._preprocess = clip.load("ViT-B/32", device=self._device)
                self._model.eval()
                self._use_transformers = False
                self._initialized = True
                logger.info("CLIP encoder initialized (OpenAI clip)")
                return
            except Exception as e:
                logger.warning(f"OpenAI CLIP failed, trying HuggingFace: {e}")

        # Fall back to HuggingFace transformers
        if TRANSFORMERS_CLIP_AVAILABLE:
            try:
                self._processor = CLIPProcessor.from_pretrained(self.config.model_name)
                self._model = CLIPModel.from_pretrained(self.config.model_name)
                self._model.to(self._device)
                self._model.eval()
                self._use_transformers = True
                self._initialized = True
                logger.info("CLIP encoder initialized (HuggingFace)")
                return
            except Exception as e:
                logger.error(f"HuggingFace CLIP failed: {e}")
                raise

        raise RuntimeError(
            "No CLIP implementation available. Install one of:\n"
            "  pip install git+https://github.com/openai/CLIP.git\n"
            "  pip install transformers"
        )

    def encode_image(self, image: "PILImage.Image") -> np.ndarray:
        """
        Encode an image to CLIP embedding.

        Args:
            image: PIL Image (any size, will be resized)

        Returns:
            512-dimensional normalized embedding
        """
        if not self._initialized:
            raise RuntimeError("CLIP encoder not initialized. Call initialize() first.")

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        with torch.no_grad():
            if self._use_transformers:
                # HuggingFace transformers
                inputs = self._processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                features = self._model.get_image_features(**inputs)
            else:
                # OpenAI CLIP
                image_tensor = self._preprocess(image).unsqueeze(0).to(self._device)
                features = self._model.encode_image(image_tensor)

            # Normalize to unit vector
            features = features / features.norm(dim=-1, keepdim=True)
            embedding = features.cpu().numpy().flatten()

        return embedding

    def encode_images_batch(self, images: list["PILImage.Image"]) -> np.ndarray:
        """
        Encode multiple images in a batch.

        Args:
            images: List of PIL Images

        Returns:
            Array of shape (N, 512) with normalized embeddings
        """
        if not self._initialized:
            raise RuntimeError("CLIP encoder not initialized. Call initialize() first.")

        # Convert to RGB
        rgb_images = [
            img.convert("RGB") if img.mode != "RGB" else img
            for img in images
        ]

        with torch.no_grad():
            if self._use_transformers:
                inputs = self._processor(images=rgb_images, return_tensors="pt", padding=True)
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                features = self._model.get_image_features(**inputs)
            else:
                image_tensors = torch.stack([
                    self._preprocess(img) for img in rgb_images
                ]).to(self._device)
                features = self._model.encode_image(image_tensors)

            # Normalize
            features = features / features.norm(dim=-1, keepdim=True)
            embeddings = features.cpu().numpy()

        return embeddings

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to CLIP embedding (for text-based symbol search).

        Args:
            text: Text description (e.g., "electrical outlet duplex")

        Returns:
            512-dimensional normalized embedding
        """
        if not self._initialized:
            raise RuntimeError("CLIP encoder not initialized. Call initialize() first.")

        with torch.no_grad():
            if self._use_transformers:
                inputs = self._processor(text=[text], return_tensors="pt", padding=True)
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                features = self._model.get_text_features(**inputs)
            else:
                tokens = clip.tokenize([text]).to(self._device)
                features = self._model.encode_text(tokens)

            features = features / features.norm(dim=-1, keepdim=True)
            embedding = features.cpu().numpy().flatten()

        return embedding


# =============================================================================
# Symbol RAG Repository
# =============================================================================

class SymbolRAGRepository:
    """
    Repository for symbol library operations with vector search.

    Integrates with PostgreSQL/pgvector for similarity search.
    """

    def __init__(self, pool):
        """
        Initialize repository.

        Args:
            pool: DatabasePool instance
        """
        from aec_agent.db.connection import DatabasePool
        self._pool: DatabasePool = pool

    async def add_symbol(self, symbol: SymbolLibraryEntry) -> UUID:
        """
        Add a symbol to the library.

        Args:
            symbol: Symbol entry to add

        Returns:
            Symbol ID
        """
        import json

        query = """
            INSERT INTO symbol_library (
                id, block_name, display_name, description,
                domain, category, subcategory, layer,
                embedding, preview_image,
                default_scale, default_rotation, attributes,
                standards, jurisdiction, code_references, is_active
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
            ON CONFLICT (block_name, domain)
            DO UPDATE SET
                display_name = EXCLUDED.display_name,
                description = EXCLUDED.description,
                category = EXCLUDED.category,
                subcategory = EXCLUDED.subcategory,
                layer = EXCLUDED.layer,
                embedding = EXCLUDED.embedding,
                preview_image = EXCLUDED.preview_image,
                default_scale = EXCLUDED.default_scale,
                default_rotation = EXCLUDED.default_rotation,
                attributes = EXCLUDED.attributes,
                standards = EXCLUDED.standards,
                jurisdiction = EXCLUDED.jurisdiction,
                code_references = EXCLUDED.code_references,
                is_active = EXCLUDED.is_active,
                updated_at = NOW()
            RETURNING id
        """

        return await self._pool.fetchval(
            query,
            symbol.id,
            symbol.block_name,
            symbol.display_name,
            symbol.description,
            symbol.domain,
            symbol.category,
            symbol.subcategory,
            symbol.layer,
            symbol.embedding,
            symbol.preview_image,
            symbol.default_scale,
            symbol.default_rotation,
            json.dumps(symbol.attributes) if isinstance(symbol.attributes, dict) else symbol.attributes,
            symbol.standards,
            symbol.jurisdiction,
            symbol.code_references,
            symbol.is_active,
        )

    async def add_symbols_batch(self, symbols: list[SymbolLibraryEntry]) -> int:
        """
        Add multiple symbols in a batch.

        Args:
            symbols: List of symbols to add

        Returns:
            Number of symbols added
        """
        count = 0
        async with self._pool.transaction() as conn:
            for symbol in symbols:
                try:
                    await self.add_symbol(symbol)
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to add symbol {symbol.block_name}: {e}")

        logger.info(f"Added {count} symbols to library")
        return count

    async def search_similar(
        self,
        embedding: list[float],
        domain: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 5,
        min_confidence: float = 0.0,
    ) -> list[SymbolMatch]:
        """
        Search for similar symbols using vector similarity.

        Args:
            embedding: Query embedding (512-dim)
            domain: Optional domain filter
            category: Optional category filter
            limit: Maximum results
            min_confidence: Minimum confidence threshold (1 - distance)

        Returns:
            List of matching symbols ordered by similarity
        """
        # Build WHERE clause
        conditions = ["is_active = TRUE", "embedding IS NOT NULL"]
        params = [embedding]
        param_idx = 2

        if domain:
            conditions.append(f"domain = ${param_idx}")
            params.append(domain)
            param_idx += 1

        if category:
            conditions.append(f"category = ${param_idx}")
            params.append(category)
            param_idx += 1

        # Add confidence filter (distance < 1 - min_confidence)
        max_distance = 1.0 - min_confidence
        conditions.append(f"embedding <=> $1 < {max_distance}")

        params.append(limit)
        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT
                id, block_name, display_name, domain, category, subcategory,
                layer, attributes, default_scale, default_rotation,
                embedding <=> $1 as distance
            FROM symbol_library
            WHERE {where_clause}
            ORDER BY embedding <=> $1
            LIMIT ${param_idx}
        """

        rows = await self._pool.fetch(query, *params)

        matches = []
        for row in rows:
            distance = row["distance"]
            confidence = 1.0 - distance  # Convert distance to confidence

            matches.append(SymbolMatch(
                symbol_id=row["id"],
                block_name=row["block_name"],
                display_name=row["display_name"],
                domain=SymbolDomain(row["domain"]),
                category=row["category"],
                subcategory=row["subcategory"],
                layer=row["layer"],
                confidence=confidence,
                distance=distance,
                attributes=row["attributes"] or {},
                default_scale=row["default_scale"],
                default_rotation=row["default_rotation"],
            ))

        return matches

    async def search_by_text(
        self,
        text_embedding: list[float],
        domain: Optional[str] = None,
        limit: int = 5,
    ) -> list[SymbolMatch]:
        """
        Search symbols using text embedding.

        Useful for queries like "smoke detector" or "duplex outlet".

        Args:
            text_embedding: CLIP text embedding
            domain: Optional domain filter
            limit: Maximum results

        Returns:
            List of matching symbols
        """
        return await self.search_similar(
            embedding=text_embedding,
            domain=domain,
            limit=limit,
        )

    async def get_symbol_by_id(self, symbol_id: UUID) -> Optional[SymbolLibraryEntry]:
        """Get a symbol by its ID."""
        query = """
            SELECT id, block_name, display_name, description,
                   domain, category, subcategory, layer,
                   embedding, preview_image,
                   default_scale, default_rotation, attributes,
                   standards, jurisdiction, code_references, is_active
            FROM symbol_library
            WHERE id = $1
        """
        row = await self._pool.fetchrow(query, symbol_id)
        if row:
            return self._row_to_entry(dict(row))
        return None

    async def get_symbol_by_block_name(
        self,
        block_name: str,
        domain: Optional[str] = None,
    ) -> Optional[SymbolLibraryEntry]:
        """Get a symbol by its block name."""
        if domain:
            query = """
                SELECT id, block_name, display_name, description,
                       domain, category, subcategory, layer,
                       embedding, preview_image,
                       default_scale, default_rotation, attributes,
                       standards, jurisdiction, code_references, is_active
                FROM symbol_library
                WHERE block_name = $1 AND domain = $2
            """
            row = await self._pool.fetchrow(query, block_name, domain)
        else:
            query = """
                SELECT id, block_name, display_name, description,
                       domain, category, subcategory, layer,
                       embedding, preview_image,
                       default_scale, default_rotation, attributes,
                       standards, jurisdiction, code_references, is_active
                FROM symbol_library
                WHERE block_name = $1
            """
            row = await self._pool.fetchrow(query, block_name)

        if row:
            return self._row_to_entry(dict(row))
        return None

    async def list_symbols(
        self,
        domain: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 100,
    ) -> list[SymbolLibraryEntry]:
        """List symbols with optional filters."""
        conditions = ["is_active = TRUE"]
        params = []
        param_idx = 1

        if domain:
            conditions.append(f"domain = ${param_idx}")
            params.append(domain)
            param_idx += 1

        if category:
            conditions.append(f"category = ${param_idx}")
            params.append(category)
            param_idx += 1

        params.append(limit)
        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT id, block_name, display_name, description,
                   domain, category, subcategory, layer,
                   embedding, preview_image,
                   default_scale, default_rotation, attributes,
                   standards, jurisdiction, code_references, is_active
            FROM symbol_library
            WHERE {where_clause}
            ORDER BY domain, category, block_name
            LIMIT ${param_idx}
        """

        rows = await self._pool.fetch(query, *params)
        return [self._row_to_entry(dict(row)) for row in rows]

    async def get_symbol_count(self, domain: Optional[str] = None) -> int:
        """Get total symbol count."""
        if domain:
            query = "SELECT COUNT(*) FROM symbol_library WHERE domain = $1 AND is_active = TRUE"
            return await self._pool.fetchval(query, domain)
        else:
            query = "SELECT COUNT(*) FROM symbol_library WHERE is_active = TRUE"
            return await self._pool.fetchval(query)

    async def record_usage(
        self,
        symbol_id: UUID,
        project_id: Optional[UUID],
        confidence: float,
        image_hash: Optional[str] = None,
    ) -> None:
        """Record symbol usage for learning."""
        query = """
            INSERT INTO symbol_usage (symbol_id, project_id, confidence, image_hash)
            VALUES ($1, $2, $3, $4)
        """
        await self._pool.execute(query, symbol_id, project_id, confidence, image_hash)

    async def record_feedback(
        self,
        usage_id: UUID,
        was_correct: bool,
        correct_symbol_id: Optional[UUID] = None,
    ) -> None:
        """Record user feedback on a symbol recognition."""
        query = """
            UPDATE symbol_usage
            SET was_correct = $2, correct_symbol_id = $3
            WHERE id = $1
        """
        await self._pool.execute(query, usage_id, was_correct, correct_symbol_id)

    def _row_to_entry(self, row: dict) -> SymbolLibraryEntry:
        """Convert database row to SymbolLibraryEntry."""
        from uuid import UUID as PyUUID

        return SymbolLibraryEntry(
            id=row["id"] if isinstance(row["id"], PyUUID) else PyUUID(str(row["id"])),
            block_name=row["block_name"],
            display_name=row["display_name"],
            description=row.get("description"),
            domain=row["domain"],
            category=row["category"],
            subcategory=row.get("subcategory"),
            layer=row["layer"],
            embedding=row.get("embedding"),
            preview_image=row.get("preview_image"),
            default_scale=row.get("default_scale", 1.0),
            default_rotation=row.get("default_rotation", 0.0),
            attributes=row.get("attributes") or {},
            standards=row.get("standards") or [],
            jurisdiction=row.get("jurisdiction"),
            code_references=row.get("code_references") or [],
            is_active=row.get("is_active", True),
        )


# =============================================================================
# Main Symbol RAG Service
# =============================================================================

class SymbolRAG:
    """
    RAG-based symbol recognition service.

    Combines CLIP encoding with pgvector similarity search
    to identify CAD symbols from image patches.

    Usage:
        rag = SymbolRAG(pool)
        await rag.initialize()

        # Recognize from image
        matches = await rag.recognize_symbol(image_patch, domain="electrical")

        # Recognize from text
        matches = await rag.search_by_description("smoke detector")
    """

    def __init__(
        self,
        pool,
        config: Optional[SymbolRecognitionConfig] = None,
    ):
        """
        Initialize Symbol RAG service.

        Args:
            pool: DatabasePool instance
            config: Recognition configuration
        """
        self.config = config or SymbolRecognitionConfig()
        self.encoder = CLIPEncoder.get_instance(self.config)
        self.repository = SymbolRAGRepository(pool)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the CLIP encoder."""
        if self._initialized:
            return

        await self.encoder.initialize()
        self._initialized = True
        logger.info("SymbolRAG initialized")

    async def recognize_symbol(
        self,
        image: "PILImage.Image",
        domain: Optional[SymbolDomain] = None,
        category: Optional[str] = None,
        top_k: Optional[int] = None,
        min_confidence: Optional[float] = None,
        project_id: Optional[UUID] = None,
    ) -> list[SymbolMatch]:
        """
        Recognize a symbol from an image patch.

        Args:
            image: PIL Image of the symbol region
            domain: Optional domain filter (electrical, mechanical, etc.)
            category: Optional category filter (outlet, diffuser, etc.)
            top_k: Number of results (default: config.top_k)
            min_confidence: Minimum confidence (default: config.min_confidence)
            project_id: Optional project for usage tracking

        Returns:
            List of symbol matches ordered by confidence
        """
        if not self._initialized:
            await self.initialize()

        # Encode the image
        embedding = self.encoder.encode_image(image)

        # Search for similar symbols
        domain_str = (domain.value if hasattr(domain, 'value') else str(domain)) if domain else None
        matches = await self.repository.search_similar(
            embedding=embedding.tolist(),
            domain=domain_str,
            category=category,
            limit=top_k or self.config.top_k,
            min_confidence=min_confidence or self.config.min_confidence,
        )

        # Record usage if we have a confident match
        if matches and matches[0].confidence >= self.config.min_confidence:
            image_hash = self._compute_image_hash(image)
            await self.repository.record_usage(
                symbol_id=matches[0].symbol_id,
                project_id=project_id,
                confidence=matches[0].confidence,
                image_hash=image_hash,
            )

        return matches

    async def recognize_symbols_batch(
        self,
        images: list["PILImage.Image"],
        domain: Optional[SymbolDomain] = None,
    ) -> list[list[SymbolMatch]]:
        """
        Recognize multiple symbols in a batch.

        Args:
            images: List of symbol image patches
            domain: Optional domain filter

        Returns:
            List of match lists (one per image)
        """
        if not self._initialized:
            await self.initialize()

        # Batch encode
        embeddings = self.encoder.encode_images_batch(images)

        # Search for each
        results = []
        domain_str = (domain.value if hasattr(domain, 'value') else str(domain)) if domain else None
        for embedding in embeddings:
            matches = await self.repository.search_similar(
                embedding=embedding.tolist(),
                domain=domain_str,
                limit=self.config.top_k,
                min_confidence=self.config.min_confidence,
            )
            results.append(matches)

        return results

    async def search_by_description(
        self,
        description: str,
        domain: Optional[SymbolDomain] = None,
        top_k: Optional[int] = None,
    ) -> list[SymbolMatch]:
        """
        Search for symbols by text description.

        Args:
            description: Text description (e.g., "duplex outlet")
            domain: Optional domain filter
            top_k: Number of results

        Returns:
            List of symbol matches
        """
        if not self._initialized:
            await self.initialize()

        # Encode the text
        embedding = self.encoder.encode_text(description)

        # Search
        domain_str = (domain.value if hasattr(domain, 'value') else str(domain)) if domain else None
        return await self.repository.search_by_text(
            text_embedding=embedding.tolist(),
            domain=domain_str,
            limit=top_k or self.config.top_k,
        )

    async def add_symbol_from_image(
        self,
        image: "PILImage.Image",
        block_name: str,
        display_name: str,
        domain: SymbolDomain,
        category: str,
        layer: str,
        subcategory: Optional[str] = None,
        description: Optional[str] = None,
        attributes: Optional[dict] = None,
    ) -> UUID:
        """
        Add a new symbol to the library from an image.

        Args:
            image: Symbol image (will be embedded and stored)
            block_name: AutoCAD block name
            display_name: Human-readable name
            domain: Symbol domain
            category: Symbol category
            layer: NCS layer name
            subcategory: Optional subcategory
            description: Optional description
            attributes: Optional default attributes

        Returns:
            Symbol ID
        """
        if not self._initialized:
            await self.initialize()

        from uuid import uuid4

        # Generate embedding
        embedding = self.encoder.encode_image(image)

        # Create thumbnail
        preview = self._create_preview(image)

        # Create entry
        domain_str = domain.value if hasattr(domain, 'value') else str(domain)
        entry = SymbolLibraryEntry(
            id=uuid4(),
            block_name=block_name,
            display_name=display_name,
            description=description,
            domain=domain_str,
            category=category,
            subcategory=subcategory,
            layer=layer,
            embedding=embedding.tolist(),
            preview_image=preview,
            attributes=attributes or {},
        )

        return await self.repository.add_symbol(entry)

    def _compute_image_hash(self, image: "PILImage.Image") -> str:
        """Compute hash of image for deduplication."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return hashlib.md5(buffer.getvalue()).hexdigest()

    def _create_preview(self, image: "PILImage.Image", size: int = 128) -> bytes:
        """Create a thumbnail preview of the symbol."""
        # Resize to square
        thumb = image.copy()
        thumb.thumbnail((size, size), Image.Resampling.LANCZOS)

        # Save as PNG
        buffer = io.BytesIO()
        thumb.save(buffer, format="PNG")
        return buffer.getvalue()


# =============================================================================
# Utility Functions
# =============================================================================

def is_symbol_rag_available() -> bool:
    """Check if Symbol RAG dependencies are available."""
    return TORCH_AVAILABLE and (CLIP_AVAILABLE or TRANSFORMERS_CLIP_AVAILABLE)


def extract_symbol_region(
    image: "PILImage.Image",
    bbox: tuple[int, int, int, int],
    padding: int = 10,
) -> "PILImage.Image":
    """
    Extract a symbol region from an image.

    Args:
        image: Full drawing image
        bbox: Bounding box (x1, y1, x2, y2) in pixels
        padding: Pixels to add around the bbox

    Returns:
        Cropped symbol image
    """
    x1, y1, x2, y2 = bbox

    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.width, x2 + padding)
    y2 = min(image.height, y2 + padding)

    return image.crop((x1, y1, x2, y2))


def image_to_base64(image: "PILImage.Image") -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def base64_to_image(b64_string: str) -> "PILImage.Image":
    """Convert base64 string to PIL Image."""
    image_data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_data))


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Classes
    "SymbolRAG",
    "CLIPEncoder",
    "SymbolRAGRepository",
    # Data Models
    "SymbolMatch",
    "SymbolLibraryEntry",
    "SymbolRecognitionConfig",
    "SymbolDomain",
    # Utilities
    "is_symbol_rag_available",
    "extract_symbol_region",
    "image_to_base64",
    "base64_to_image",
]
