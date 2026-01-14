"""
SQLite cache for AEC Agent.

Provides fast read access to Revit model metadata without
blocking the CAD application's UI thread.
"""

import aiosqlite
import json
from pathlib import Path
from typing import Optional, List
from datetime import datetime

import structlog

logger = structlog.get_logger(__name__)


class CacheManager:
    """
    Async SQLite cache manager.

    Caches element metadata from Revit/AutoCAD for fast queries.
    """

    def __init__(self, cache_dir: Path):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self.cache_dir / "aec_cache.sqlite"
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self):
        """Initialize database connection and schema."""
        self._db = await aiosqlite.connect(str(self._db_path))
        self._db.row_factory = aiosqlite.Row

        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS levels (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                elevation_m REAL,
                source TEXT,  -- 'revit' or 'autocad'
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS rooms (
                id INTEGER PRIMARY KEY,
                name TEXT,
                number TEXT,
                level_id INTEGER,
                area_sqm REAL,
                source TEXT,
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS layers (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                color INTEGER,
                is_on INTEGER DEFAULT 1,
                is_frozen INTEGER DEFAULT 0,
                source TEXT,
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_levels_name ON levels(name);
            CREATE INDEX IF NOT EXISTS idx_rooms_level ON rooms(level_id);
            CREATE INDEX IF NOT EXISTS idx_layers_name ON layers(name);
        """)

        await self._db.commit()
        logger.info("Cache database initialized", path=str(self._db_path))

    async def close(self):
        """Close database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    # =========================================================================
    # Level Operations
    # =========================================================================

    async def get_level_by_name(self, name: str) -> Optional[dict]:
        """
        Get level by name.

        Args:
            name: Level name

        Returns:
            Level dict or None
        """
        cursor = await self._db.execute(
            "SELECT * FROM levels WHERE name = ?",
            (name,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_level_by_id(self, level_id: int) -> Optional[dict]:
        """Get level by ID."""
        cursor = await self._db.execute(
            "SELECT * FROM levels WHERE id = ?",
            (level_id,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_all_levels(self) -> List[dict]:
        """Get all levels sorted by elevation."""
        cursor = await self._db.execute(
            "SELECT * FROM levels ORDER BY elevation_m ASC"
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def upsert_level(
        self,
        level_id: int,
        name: str,
        elevation_m: float,
        source: str = "revit"
    ):
        """Insert or update a level."""
        now = datetime.utcnow().isoformat()
        await self._db.execute(
            """
            INSERT OR REPLACE INTO levels (id, name, elevation_m, source, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (level_id, name, elevation_m, source, now)
        )
        await self._db.commit()

    # =========================================================================
    # Room Operations
    # =========================================================================

    async def get_all_rooms(self) -> List[dict]:
        """Get all rooms."""
        cursor = await self._db.execute("SELECT * FROM rooms")
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_rooms_by_level(self, level_id: int) -> List[dict]:
        """Get rooms on a specific level."""
        cursor = await self._db.execute(
            "SELECT * FROM rooms WHERE level_id = ?",
            (level_id,)
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def upsert_room(
        self,
        room_id: int,
        name: str,
        number: str,
        level_id: int,
        area_sqm: float,
        source: str = "revit"
    ):
        """Insert or update a room."""
        now = datetime.utcnow().isoformat()
        await self._db.execute(
            """
            INSERT OR REPLACE INTO rooms
            (id, name, number, level_id, area_sqm, source, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (room_id, name, number, level_id, area_sqm, source, now)
        )
        await self._db.commit()

    # =========================================================================
    # Layer Operations (AutoCAD)
    # =========================================================================

    async def get_layer_by_name(self, name: str) -> Optional[dict]:
        """Get layer by name."""
        cursor = await self._db.execute(
            "SELECT * FROM layers WHERE name = ?",
            (name,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_all_layers(self) -> List[dict]:
        """Get all layers."""
        cursor = await self._db.execute(
            "SELECT * FROM layers ORDER BY name ASC"
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def upsert_layer(
        self,
        layer_id: int,
        name: str,
        color: int = 7,
        is_on: bool = True,
        is_frozen: bool = False,
        source: str = "autocad"
    ):
        """Insert or update a layer."""
        now = datetime.utcnow().isoformat()
        await self._db.execute(
            """
            INSERT OR REPLACE INTO layers
            (id, name, color, is_on, is_frozen, source, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (layer_id, name, color, int(is_on), int(is_frozen), source, now)
        )
        await self._db.commit()

    # =========================================================================
    # Metadata Operations
    # =========================================================================

    async def get_metadata(self, key: str) -> Optional[str]:
        """Get metadata value by key."""
        cursor = await self._db.execute(
            "SELECT value FROM metadata WHERE key = ?",
            (key,)
        )
        row = await cursor.fetchone()
        return row["value"] if row else None

    async def set_metadata(self, key: str, value: str):
        """Set metadata value."""
        now = datetime.utcnow().isoformat()
        await self._db.execute(
            """
            INSERT OR REPLACE INTO metadata (key, value, updated_at)
            VALUES (?, ?, ?)
            """,
            (key, value, now)
        )
        await self._db.commit()

    async def get_last_sync_time(self, category: str) -> Optional[str]:
        """Get last sync time for a category."""
        return await self.get_metadata(f"{category}_last_sync")

    async def set_last_sync_time(self, category: str):
        """Set last sync time to now."""
        await self.set_metadata(
            f"{category}_last_sync",
            datetime.utcnow().isoformat()
        )

    # =========================================================================
    # Bulk Operations
    # =========================================================================

    async def clear_category(self, category: str):
        """Clear all data for a category."""
        table_map = {
            "levels": "levels",
            "rooms": "rooms",
            "layers": "layers",
        }
        table = table_map.get(category)
        if table:
            await self._db.execute(f"DELETE FROM {table}")
            await self._db.commit()
            logger.info("Cache cleared", category=category)

    async def clear_all(self):
        """Clear all cached data."""
        for table in ["levels", "rooms", "layers", "metadata"]:
            await self._db.execute(f"DELETE FROM {table}")
        await self._db.commit()
        logger.info("All cache cleared")
