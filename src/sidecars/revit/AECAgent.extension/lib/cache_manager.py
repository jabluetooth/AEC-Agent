"""
SQLite cache manager for AEC Agent.

Caches Revit model metadata for fast read operations.
This enables <50ms response times for queries like "list all rooms"
instead of blocking the Revit UI thread.
"""

import os
import hashlib
import sqlite3
import json
from datetime import datetime

from pyrevit import revit, DB
from pyrevit.coreutils import logger


def get_cache_dir():
    # type: () -> str
    """Get the cache directory path."""
    # Use LOCALAPPDATA for persistence across sessions
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        # Fallback for non-Windows
        local_app_data = os.path.expanduser("~/.local/share")

    cache_dir = os.path.join(local_app_data, "AECAgent", "cache")

    # Create directory if it doesn't exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    return cache_dir


def get_document_hash(doc_path):
    # type: (str) -> str
    """Generate a unique hash for a document path."""
    # Using SHA256 (more secure than MD5)
    return hashlib.sha256(doc_path.encode('utf-8')).hexdigest()[:16]


def get_cache_path(doc_path):
    # type: (str) -> str
    """Get the cache file path for a document."""
    doc_hash = get_document_hash(doc_path)
    return os.path.join(get_cache_dir(), "{}.sqlite".format(doc_hash))


class RevitCache:
    """Manages SQLite cache for a Revit document."""

    def __init__(self, doc):
        """
        Initialize cache for a Revit document.

        Args:
            doc: Revit Document object
        """
        self.doc = doc
        self.doc_path = doc.PathName or "untitled_{}".format(doc.Title)
        self.cache_path = get_cache_path(self.doc_path)
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database schema."""
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS elements (
                    id INTEGER PRIMARY KEY,
                    category TEXT,
                    name TEXT,
                    type_name TEXT,
                    level_id INTEGER,
                    data TEXT,
                    updated_at TEXT
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_elements_category
                ON elements(category)
            """)

            conn.commit()

    def sync_rooms(self):
        """Sync all rooms to cache."""
        collector = DB.FilteredElementCollector(self.doc)
        rooms = collector.OfCategory(
            DB.BuiltInCategory.OST_Rooms
        ).WhereElementIsNotElementType().ToElements()

        now = datetime.utcnow().isoformat()
        synced_count = 0

        with sqlite3.connect(self.cache_path) as conn:
            # Clear existing rooms
            conn.execute("DELETE FROM elements WHERE category = 'rooms'")

            for room in rooms:
                if room.Area > 0:  # Only placed rooms
                    try:
                        name_param = room.get_Parameter(DB.BuiltInParameter.ROOM_NAME)
                        number_param = room.get_Parameter(DB.BuiltInParameter.ROOM_NUMBER)

                        data = {
                            "name": name_param.AsString() if name_param else "",
                            "number": number_param.AsString() if number_param else "",
                            "area_sqm": room.Area * 0.092903,  # sqft to sqm
                            "level": room.Level.Name if room.Level else None,
                        }

                        level_id = room.Level.Id.IntegerValue if room.Level else None

                        conn.execute(
                            """
                            INSERT OR REPLACE INTO elements
                            (id, category, name, type_name, level_id, data, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                room.Id.IntegerValue,
                                "rooms",
                                data["name"],
                                None,
                                level_id,
                                json.dumps(data),
                                now
                            )
                        )
                        synced_count += 1
                    except Exception as e:
                        logger.warn("Failed to cache room {}: {}".format(room.Id, e))

            # Update sync timestamp
            conn.execute(
                """
                INSERT OR REPLACE INTO metadata (key, value, updated_at)
                VALUES ('rooms_synced_at', ?, ?)
                """,
                (now, now)
            )

            conn.commit()

        logger.info("Synced {} rooms to cache".format(synced_count))
        return synced_count

    def sync_levels(self):
        """Sync all levels to cache."""
        collector = DB.FilteredElementCollector(self.doc)
        levels = collector.OfClass(DB.Level).ToElements()

        now = datetime.utcnow().isoformat()
        synced_count = 0

        with sqlite3.connect(self.cache_path) as conn:
            # Clear existing levels
            conn.execute("DELETE FROM elements WHERE category = 'levels'")

            for level in levels:
                try:
                    data = {
                        "name": level.Name,
                        "elevation": level.Elevation,
                        "elevation_m": level.Elevation * 0.3048,
                    }

                    conn.execute(
                        """
                        INSERT OR REPLACE INTO elements
                        (id, category, name, type_name, level_id, data, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            level.Id.IntegerValue,
                            "levels",
                            level.Name,
                            None,
                            None,
                            json.dumps(data),
                            now
                        )
                    )
                    synced_count += 1
                except Exception as e:
                    logger.warn("Failed to cache level {}: {}".format(level.Id, e))

            # Update sync timestamp
            conn.execute(
                """
                INSERT OR REPLACE INTO metadata (key, value, updated_at)
                VALUES ('levels_synced_at', ?, ?)
                """,
                (now, now)
            )

            conn.commit()

        logger.info("Synced {} levels to cache".format(synced_count))
        return synced_count

    def sync_walls(self):
        """Sync all walls to cache."""
        collector = DB.FilteredElementCollector(self.doc)
        walls = collector.OfClass(DB.Wall).ToElements()

        now = datetime.utcnow().isoformat()
        synced_count = 0

        with sqlite3.connect(self.cache_path) as conn:
            # Clear existing walls
            conn.execute("DELETE FROM elements WHERE category = 'walls'")

            for wall in walls:
                try:
                    wall_type = self.doc.GetElement(wall.GetTypeId())
                    length_param = wall.get_Parameter(DB.BuiltInParameter.CURVE_ELEM_LENGTH)

                    data = {
                        "type_name": wall_type.Name if wall_type else "Unknown",
                        "length_m": length_param.AsDouble() * 0.3048 if length_param else 0,
                        "level_id": wall.LevelId.IntegerValue if wall.LevelId else None,
                    }

                    conn.execute(
                        """
                        INSERT OR REPLACE INTO elements
                        (id, category, name, type_name, level_id, data, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            wall.Id.IntegerValue,
                            "walls",
                            None,
                            data["type_name"],
                            data["level_id"],
                            json.dumps(data),
                            now
                        )
                    )
                    synced_count += 1
                except Exception as e:
                    logger.warn("Failed to cache wall {}: {}".format(wall.Id, e))

            # Update sync timestamp
            conn.execute(
                """
                INSERT OR REPLACE INTO metadata (key, value, updated_at)
                VALUES ('walls_synced_at', ?, ?)
                """,
                (now, now)
            )

            conn.commit()

        logger.info("Synced {} walls to cache".format(synced_count))
        return synced_count

    def sync_all(self):
        """Sync all supported element categories."""
        results = {
            "rooms": self.sync_rooms(),
            "levels": self.sync_levels(),
            "walls": self.sync_walls(),
        }
        return results

    def get_elements_from_cache(self, category):
        # type: (str) -> list
        """
        Get elements from cache by category.

        Args:
            category: Element category ('rooms', 'levels', 'walls')

        Returns:
            List of element dicts
        """
        with sqlite3.connect(self.cache_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT id, data FROM elements WHERE category = ?",
                (category,)
            )

            elements = []
            for row in cursor:
                data = json.loads(row["data"])
                data["id"] = row["id"]
                elements.append(data)

            return elements

    def get_rooms_from_cache(self):
        # type: () -> list
        """Get rooms from cache (fast read)."""
        return self.get_elements_from_cache("rooms")

    def get_levels_from_cache(self):
        # type: () -> list
        """Get levels from cache (fast read)."""
        return self.get_elements_from_cache("levels")

    def get_walls_from_cache(self):
        # type: () -> list
        """Get walls from cache (fast read)."""
        return self.get_elements_from_cache("walls")

    def get_cache_age_seconds(self, category):
        # type: (str) -> float
        """
        Get the age of cached data in seconds.

        Args:
            category: Element category

        Returns:
            Age in seconds, or infinity if never synced
        """
        with sqlite3.connect(self.cache_path) as conn:
            cursor = conn.execute(
                "SELECT value FROM metadata WHERE key = ?",
                ("{}_synced_at".format(category),)
            )
            row = cursor.fetchone()

            if not row:
                return float('inf')  # Never synced

            try:
                synced_at = datetime.fromisoformat(row[0])
                age = (datetime.utcnow() - synced_at).total_seconds()
                return age
            except (ValueError, TypeError):
                return float('inf')

    def is_cache_stale(self, category, max_age_seconds=300):
        # type: (str, int) -> bool
        """
        Check if cache is stale.

        Args:
            category: Element category
            max_age_seconds: Maximum acceptable age (default 5 minutes)

        Returns:
            True if cache is stale or missing
        """
        return self.get_cache_age_seconds(category) > max_age_seconds

    def clear_cache(self):
        """Clear all cached data."""
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute("DELETE FROM elements")
            conn.execute("DELETE FROM metadata")
            conn.commit()
        logger.info("Cache cleared")
