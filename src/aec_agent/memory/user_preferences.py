"""
User Preferences System with Windows username integration.

Stores and retrieves user-specific settings that persist across sessions,
using the Windows username as the unique identifier.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


def get_current_user() -> str:
    """
    Get the current Windows username for preference storage.

    Returns:
        Windows username (e.g., 'frelatorre')
    """
    try:
        return os.getlogin()
    except OSError:
        # Fallback for environments where getlogin() fails
        return os.environ.get("USERNAME", os.environ.get("USER", "anonymous"))


class PreferenceCategory:
    """Standard preference categories."""
    DISPLAY = "display"           # Units, decimal places, pressure units
    WORKFLOW = "workflow"         # Preferred duct shapes, routing methods
    DEFAULTS = "defaults"         # Default sizes, insulation, velocities
    SHORTCUTS = "shortcuts"       # Command shortcuts/aliases
    UI = "ui"                     # UI preferences
    MEP = "mep"                   # MEP-specific settings


@dataclass
class UserPreference:
    """A single user preference."""
    id: UUID = field(default_factory=uuid4)
    user_id: str = ""
    category: str = ""
    key: str = ""
    value: Any = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "user_id": self.user_id,
            "category": self.category,
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class UserShortcut:
    """A user-defined command shortcut."""
    id: UUID = field(default_factory=uuid4)
    user_id: str = ""
    alias: str = ""              # Short command (e.g., "rs")
    expansion: str = ""          # Full expansion (e.g., "route supply duct")
    description: str = ""
    usage_count: int = 0
    last_used_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "user_id": self.user_id,
            "alias": self.alias,
            "expansion": self.expansion,
            "description": self.description,
            "usage_count": self.usage_count,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "created_at": self.created_at.isoformat(),
        }


class UserPreferences:
    """
    Manages user preferences with Windows username integration.

    Preferences are stored per-user and persist across sessions.
    Supports categories for organization and shortcuts for quick commands.
    """

    # Default preference values (HVAC-focused)
    DEFAULT_PREFERENCES = {
        PreferenceCategory.DISPLAY: {
            "units": "imperial",           # imperial or metric
            "decimal_places": 2,
            "pressure_units": "inWG",      # inWG or Pa
            "velocity_units": "fpm",       # fpm or m/s
            "temperature_units": "F",      # F or C
        },
        PreferenceCategory.WORKFLOW: {
            "preferred_duct_shape": "rectangular",  # rectangular, round, oval
            "routing_method": "trunk_and_branch",   # trunk_and_branch, radial
            "auto_size_ducts": True,
            "show_velocity_warnings": True,
        },
        PreferenceCategory.DEFAULTS: {
            "default_duct_insulation": 1.0,        # inches
            "max_supply_velocity": 2000,           # fpm
            "max_return_velocity": 2400,           # fpm
            "max_main_duct_velocity": 4000,        # fpm
            "default_duct_material": "galvanized",
            "friction_loss_target": 0.08,          # inWG/100ft
        },
        PreferenceCategory.UI: {
            "theme": "light",
            "show_tooltips": True,
            "auto_expand_results": True,
            "confirm_destructive": True,
        },
        PreferenceCategory.MEP: {
            "primary_discipline": "hvac",
            "coordination_priority": ["plumbing", "hvac", "electrical", "fire_protection"],
            "default_clearance_check": True,
            "auto_validate_on_edit": False,
        },
    }

    def __init__(self, db_pool=None, user_id: Optional[str] = None):
        """
        Initialize user preferences.

        Args:
            db_pool: asyncpg connection pool
            user_id: Optional override for Windows username
        """
        self._db_pool = db_pool
        self._user_id = user_id or get_current_user()
        self._cache: dict[str, dict[str, Any]] = {}
        self._shortcuts: dict[str, UserShortcut] = {}
        self._initialized = False

    @property
    def user_id(self) -> str:
        """Get the current user ID (Windows username)."""
        return self._user_id

    async def initialize(self) -> None:
        """Load preferences from database or defaults."""
        if self._initialized:
            return

        if self._db_pool:
            await self._load_from_db()
        else:
            self._load_defaults()

        self._initialized = True
        logger.info(f"UserPreferences initialized for user: {self._user_id}")

    def _load_defaults(self) -> None:
        """Load default preferences into cache."""
        for category, prefs in self.DEFAULT_PREFERENCES.items():
            self._cache[category] = prefs.copy()

    async def _load_from_db(self) -> None:
        """Load preferences from PostgreSQL."""
        # Start with defaults
        self._load_defaults()

        try:
            async with self._db_pool.acquire() as conn:
                # Load preferences
                rows = await conn.fetch("""
                    SELECT category, key, value
                    FROM user_preferences
                    WHERE user_id = $1
                """, self._user_id)

                for row in rows:
                    category = row["category"]
                    if category not in self._cache:
                        self._cache[category] = {}
                    self._cache[category][row["key"]] = row["value"]

                # Load shortcuts
                shortcut_rows = await conn.fetch("""
                    SELECT id, alias, expansion, description, usage_count, last_used_at, created_at
                    FROM user_shortcuts
                    WHERE user_id = $1
                    ORDER BY usage_count DESC
                """, self._user_id)

                for row in shortcut_rows:
                    shortcut = UserShortcut(
                        id=row["id"],
                        user_id=self._user_id,
                        alias=row["alias"],
                        expansion=row["expansion"],
                        description=row["description"] or "",
                        usage_count=row["usage_count"],
                        last_used_at=row["last_used_at"],
                        created_at=row["created_at"],
                    )
                    self._shortcuts[shortcut.alias] = shortcut

        except Exception as e:
            logger.warning(f"Failed to load preferences from DB: {e}")

    # =========================================================================
    # Preference Operations
    # =========================================================================

    async def get(
        self,
        category: str,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Get a preference value.

        Args:
            category: Preference category
            key: Preference key
            default: Default value if not found

        Returns:
            Preference value or default
        """
        if not self._initialized:
            await self.initialize()

        if category in self._cache and key in self._cache[category]:
            return self._cache[category][key]

        # Check defaults
        if category in self.DEFAULT_PREFERENCES:
            return self.DEFAULT_PREFERENCES[category].get(key, default)

        return default

    async def set(
        self,
        category: str,
        key: str,
        value: Any,
    ) -> None:
        """
        Set a preference value.

        Args:
            category: Preference category
            key: Preference key
            value: Value to store
        """
        if not self._initialized:
            await self.initialize()

        # Update cache
        if category not in self._cache:
            self._cache[category] = {}
        self._cache[category][key] = value

        # Persist to database
        if self._db_pool:
            try:
                async with self._db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO user_preferences (user_id, category, key, value)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (user_id, category, key) DO UPDATE SET
                            value = EXCLUDED.value,
                            updated_at = NOW()
                    """, self._user_id, category, key, value)
            except Exception as e:
                logger.error(f"Failed to save preference: {e}")

    async def get_category(self, category: str) -> dict[str, Any]:
        """
        Get all preferences in a category.

        Args:
            category: Preference category

        Returns:
            Dict of all preferences in the category
        """
        if not self._initialized:
            await self.initialize()

        result = {}

        # Start with defaults
        if category in self.DEFAULT_PREFERENCES:
            result.update(self.DEFAULT_PREFERENCES[category])

        # Override with user settings
        if category in self._cache:
            result.update(self._cache[category])

        return result

    async def reset_category(self, category: str) -> None:
        """
        Reset a category to defaults.

        Args:
            category: Preference category to reset
        """
        if category in self.DEFAULT_PREFERENCES:
            self._cache[category] = self.DEFAULT_PREFERENCES[category].copy()

        if self._db_pool:
            try:
                async with self._db_pool.acquire() as conn:
                    await conn.execute("""
                        DELETE FROM user_preferences
                        WHERE user_id = $1 AND category = $2
                    """, self._user_id, category)
            except Exception as e:
                logger.error(f"Failed to reset category: {e}")

    async def get_all(self) -> dict[str, dict[str, Any]]:
        """
        Get all preferences for the user.

        Returns:
            Dict of all categories and their preferences
        """
        if not self._initialized:
            await self.initialize()

        result = {}

        # Start with defaults
        for category, prefs in self.DEFAULT_PREFERENCES.items():
            result[category] = prefs.copy()

        # Override with user settings
        for category, prefs in self._cache.items():
            if category not in result:
                result[category] = {}
            result[category].update(prefs)

        return result

    # =========================================================================
    # Shortcut Operations
    # =========================================================================

    async def add_shortcut(
        self,
        alias: str,
        expansion: str,
        description: str = "",
    ) -> UserShortcut:
        """
        Add a command shortcut.

        Args:
            alias: Short command (e.g., "rs")
            expansion: Full expansion (e.g., "route supply duct")
            description: Description of what the shortcut does

        Returns:
            Created UserShortcut
        """
        if not self._initialized:
            await self.initialize()

        shortcut = UserShortcut(
            user_id=self._user_id,
            alias=alias,
            expansion=expansion,
            description=description,
        )

        self._shortcuts[alias] = shortcut

        if self._db_pool:
            try:
                async with self._db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO user_shortcuts
                            (id, user_id, alias, expansion, description)
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (user_id, alias) DO UPDATE SET
                            expansion = EXCLUDED.expansion,
                            description = EXCLUDED.description
                    """, shortcut.id, self._user_id, alias, expansion, description)
            except Exception as e:
                logger.error(f"Failed to save shortcut: {e}")

        return shortcut

    async def get_shortcut(self, alias: str) -> Optional[UserShortcut]:
        """
        Get a shortcut by alias.

        Args:
            alias: Shortcut alias

        Returns:
            UserShortcut or None if not found
        """
        if not self._initialized:
            await self.initialize()

        return self._shortcuts.get(alias)

    async def expand_shortcut(self, text: str) -> str:
        """
        Expand shortcuts in text.

        Checks if text starts with a known shortcut alias
        and expands it to the full command.

        Args:
            text: Input text that may contain shortcuts

        Returns:
            Expanded text
        """
        if not self._initialized:
            await self.initialize()

        words = text.strip().split()
        if not words:
            return text

        alias = words[0].lower()
        if alias in self._shortcuts:
            shortcut = self._shortcuts[alias]

            # Record usage
            shortcut.usage_count += 1
            shortcut.last_used_at = datetime.utcnow()

            if self._db_pool:
                try:
                    async with self._db_pool.acquire() as conn:
                        await conn.execute("""
                            UPDATE user_shortcuts
                            SET usage_count = usage_count + 1,
                                last_used_at = NOW()
                            WHERE user_id = $1 AND alias = $2
                        """, self._user_id, alias)
                except Exception as e:
                    logger.debug(f"Failed to update shortcut usage: {e}")

            # Expand: replace alias with expansion, keep remaining words
            remaining = " ".join(words[1:])
            if remaining:
                return f"{shortcut.expansion} {remaining}"
            return shortcut.expansion

        return text

    async def list_shortcuts(self) -> list[UserShortcut]:
        """
        List all shortcuts for the user.

        Returns:
            List of shortcuts ordered by usage
        """
        if not self._initialized:
            await self.initialize()

        return sorted(
            self._shortcuts.values(),
            key=lambda s: s.usage_count,
            reverse=True
        )

    async def delete_shortcut(self, alias: str) -> bool:
        """
        Delete a shortcut.

        Args:
            alias: Shortcut alias to delete

        Returns:
            True if deleted, False if not found
        """
        if alias not in self._shortcuts:
            return False

        del self._shortcuts[alias]

        if self._db_pool:
            try:
                async with self._db_pool.acquire() as conn:
                    await conn.execute("""
                        DELETE FROM user_shortcuts
                        WHERE user_id = $1 AND alias = $2
                    """, self._user_id, alias)
            except Exception as e:
                logger.error(f"Failed to delete shortcut: {e}")

        return True

    # =========================================================================
    # MEP-Specific Helpers
    # =========================================================================

    async def get_mep_defaults(self) -> dict[str, Any]:
        """
        Get MEP-specific default values.

        Combines display units, workflow preferences, and default values
        into a single dict for easy access during MEP operations.

        Returns:
            Dict of MEP defaults
        """
        display = await self.get_category(PreferenceCategory.DISPLAY)
        workflow = await self.get_category(PreferenceCategory.WORKFLOW)
        defaults = await self.get_category(PreferenceCategory.DEFAULTS)
        mep = await self.get_category(PreferenceCategory.MEP)

        return {
            "units": display.get("units", "imperial"),
            "velocity_units": display.get("velocity_units", "fpm"),
            "pressure_units": display.get("pressure_units", "inWG"),
            "preferred_duct_shape": workflow.get("preferred_duct_shape", "rectangular"),
            "max_supply_velocity": defaults.get("max_supply_velocity", 2000),
            "max_return_velocity": defaults.get("max_return_velocity", 2400),
            "primary_discipline": mep.get("primary_discipline", "hvac"),
            "coordination_priority": mep.get("coordination_priority", []),
        }

    async def apply_unit_conversion(
        self,
        value: float,
        from_unit: str,
        to_preference: str = "velocity_units",
    ) -> tuple[float, str]:
        """
        Convert a value to user's preferred units.

        Args:
            value: Value to convert
            from_unit: Source unit (e.g., "fpm", "m/s")
            to_preference: Preference key to get target unit

        Returns:
            Tuple of (converted_value, unit_string)
        """
        target_unit = await self.get(PreferenceCategory.DISPLAY, to_preference, from_unit)

        if from_unit == target_unit:
            return value, target_unit

        # Velocity conversions
        if from_unit == "fpm" and target_unit == "m/s":
            return value * 0.00508, "m/s"
        elif from_unit == "m/s" and target_unit == "fpm":
            return value / 0.00508, "fpm"

        # Pressure conversions
        elif from_unit == "inWG" and target_unit == "Pa":
            return value * 249.089, "Pa"
        elif from_unit == "Pa" and target_unit == "inWG":
            return value / 249.089, "inWG"

        # Length conversions (assuming inches to mm)
        elif from_unit == "in" and target_unit == "mm":
            return value * 25.4, "mm"
        elif from_unit == "mm" and target_unit == "in":
            return value / 25.4, "in"

        return value, from_unit


# Global instance management
_preferences: Optional[UserPreferences] = None


async def get_user_preferences(
    db_pool=None,
    user_id: Optional[str] = None,
) -> UserPreferences:
    """Get or create the global UserPreferences instance."""
    global _preferences

    if _preferences is None or (user_id and _preferences.user_id != user_id):
        _preferences = UserPreferences(db_pool=db_pool, user_id=user_id)
        await _preferences.initialize()

    return _preferences


def reset_user_preferences() -> None:
    """Reset the global preferences (for testing)."""
    global _preferences
    _preferences = None
