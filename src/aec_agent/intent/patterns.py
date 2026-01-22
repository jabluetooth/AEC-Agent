"""
MEP intent patterns for classification.

Patterns are organized by domain and subdomain with weights indicating specificity.
Higher weights indicate more domain-specific terms.
"""

from typing import Optional
from aec_agent.intent.models import MEPDomain


# Pattern structure: {term: (weight, subdomain)}
# Weight scale: 1.0 = general, 2.0 = domain-specific, 3.0 = highly specific

MEP_PATTERNS: dict[MEPDomain, dict[str, tuple[float, Optional[str]]]] = {
    # =========================================================================
    # HVAC - Primary focus with most detailed patterns
    # =========================================================================
    MEPDomain.HVAC: {
        # Equipment - highly specific (weight 3.0)
        "ahu": (3.0, "equipment"),
        "air handler": (3.0, "equipment"),
        "air handling unit": (3.0, "equipment"),
        "vav": (3.0, "equipment"),
        "vav box": (3.0, "equipment"),
        "variable air volume": (3.0, "equipment"),
        "fcu": (3.0, "equipment"),
        "fan coil": (3.0, "equipment"),
        "fan coil unit": (3.0, "equipment"),
        "rtu": (3.0, "equipment"),
        "rooftop unit": (3.0, "equipment"),
        "chiller": (3.0, "equipment"),
        "boiler": (3.0, "equipment"),
        "cooling tower": (3.0, "equipment"),
        "heat pump": (2.5, "equipment"),
        "air conditioner": (2.0, "equipment"),
        "hvac unit": (3.0, "equipment"),
        "mechanical equipment": (2.5, "equipment"),
        "exhaust fan": (2.5, "equipment"),
        "supply fan": (2.5, "equipment"),
        "return fan": (2.5, "equipment"),
        # Distribution - domain specific (weight 2.5)
        "ductwork": (3.0, "distribution"),
        "duct": (2.5, "distribution"),
        "ducts": (2.5, "distribution"),
        "supply duct": (3.0, "distribution"),
        "return duct": (3.0, "distribution"),
        "exhaust duct": (3.0, "distribution"),
        "supply air": (2.5, "distribution"),
        "return air": (2.5, "distribution"),
        "exhaust air": (2.5, "distribution"),
        "fresh air": (2.0, "distribution"),
        "outside air": (2.5, "distribution"),
        "outdoor air": (2.5, "distribution"),
        "duct run": (2.5, "distribution"),
        "main trunk": (2.5, "distribution"),
        "branch duct": (2.5, "distribution"),
        "rectangular duct": (3.0, "distribution"),
        "round duct": (3.0, "distribution"),
        "spiral duct": (3.0, "distribution"),
        "oval duct": (3.0, "distribution"),
        "flex duct": (2.5, "distribution"),
        "flexible duct": (2.5, "distribution"),
        # Terminals - domain specific (weight 2.5)
        "diffuser": (3.0, "terminals"),
        "diffusers": (3.0, "terminals"),
        "grille": (2.5, "terminals"),
        "grilles": (2.5, "terminals"),
        "register": (2.5, "terminals"),
        "registers": (2.5, "terminals"),
        "linear diffuser": (3.0, "terminals"),
        "slot diffuser": (3.0, "terminals"),
        "ceiling diffuser": (3.0, "terminals"),
        "supply diffuser": (3.0, "terminals"),
        "return grille": (3.0, "terminals"),
        "air terminal": (2.5, "terminals"),
        # Controls - domain specific (weight 2.5)
        "thermostat": (2.5, "controls"),
        "damper": (2.5, "controls"),
        "fire damper": (3.0, "controls"),
        "smoke damper": (3.0, "controls"),
        "volume damper": (2.5, "controls"),
        "control damper": (2.5, "controls"),
        "control valve": (2.0, "controls"),
        "actuator": (2.0, "controls"),
        "sensor": (1.5, "controls"),
        "temperature sensor": (2.5, "controls"),
        "humidity sensor": (2.5, "controls"),
        "co2 sensor": (2.5, "controls"),
        # Parameters - highly specific (weight 3.0)
        "cfm": (3.0, "parameters"),
        "cubic feet per minute": (3.0, "parameters"),
        "airflow": (2.5, "parameters"),
        "air flow": (2.5, "parameters"),
        "static pressure": (3.0, "parameters"),
        "velocity": (2.0, "parameters"),
        "air velocity": (3.0, "parameters"),
        "ach": (3.0, "parameters"),
        "air changes": (3.0, "parameters"),
        "air changes per hour": (3.0, "parameters"),
        "fpm": (3.0, "parameters"),
        "feet per minute": (3.0, "parameters"),
        "btu": (2.5, "parameters"),
        "tons": (1.5, "parameters"),
        "cooling load": (3.0, "parameters"),
        "heating load": (3.0, "parameters"),
        "supply temperature": (2.5, "parameters"),
        # Sizing - domain specific (weight 2.5)
        "duct sizing": (3.0, "sizing"),
        "size duct": (3.0, "sizing"),
        "pressure drop": (3.0, "sizing"),
        "friction loss": (3.0, "sizing"),
        "equivalent length": (3.0, "sizing"),
        "fitting loss": (3.0, "sizing"),
        "duct calculator": (3.0, "sizing"),
        "equal friction": (3.0, "sizing"),
        "static regain": (3.0, "sizing"),
        # Fittings - domain specific
        "elbow": (2.0, "fittings"),
        "tee": (1.5, "fittings"),
        "wye": (2.5, "fittings"),
        "transition": (2.0, "fittings"),
        "reducer": (2.0, "fittings"),
        "offset": (1.5, "fittings"),
        "duct fitting": (3.0, "fittings"),
        # General HVAC terms
        "hvac": (3.0, None),
        "mechanical": (2.0, None),
        "ventilation": (2.5, None),
        "heating": (2.0, None),
        "cooling": (2.0, None),
        "air conditioning": (2.5, None),
    },
    # =========================================================================
    # ELECTRICAL - Secondary focus
    # =========================================================================
    MEPDomain.ELECTRICAL: {
        # Equipment
        "panel": (2.5, "equipment"),
        "electrical panel": (3.0, "equipment"),
        "panelboard": (3.0, "equipment"),
        "switchboard": (3.0, "equipment"),
        "transformer": (3.0, "equipment"),
        "generator": (2.5, "equipment"),
        "ups": (3.0, "equipment"),
        "mcc": (3.0, "equipment"),
        "motor control center": (3.0, "equipment"),
        # Distribution
        "conduit": (3.0, "distribution"),
        "cable tray": (3.0, "distribution"),
        "busway": (3.0, "distribution"),
        "wire": (2.0, "distribution"),
        "wiring": (2.5, "distribution"),
        "circuit": (2.5, "distribution"),
        "feeder": (3.0, "distribution"),
        "branch circuit": (3.0, "distribution"),
        # Devices
        "receptacle": (3.0, "devices"),
        "outlet": (2.5, "devices"),
        "switch": (2.0, "devices"),
        "light switch": (2.5, "devices"),
        "junction box": (2.5, "devices"),
        "disconnect": (3.0, "devices"),
        # Parameters
        "voltage": (2.5, "parameters"),
        "amperage": (3.0, "parameters"),
        "amps": (2.5, "parameters"),
        "watts": (2.0, "parameters"),
        "kva": (3.0, "parameters"),
        "load": (1.5, "parameters"),
        "electrical load": (3.0, "parameters"),
        # General
        "electrical": (2.5, None),
        "power": (1.5, None),
    },
    # =========================================================================
    # PLUMBING - Secondary focus
    # =========================================================================
    MEPDomain.PLUMBING: {
        # Equipment
        "water heater": (3.0, "equipment"),
        "booster pump": (3.0, "equipment"),
        "sump pump": (3.0, "equipment"),
        "water softener": (3.0, "equipment"),
        "backflow preventer": (3.0, "equipment"),
        "prv": (3.0, "equipment"),
        "pressure reducing valve": (3.0, "equipment"),
        # Distribution
        "pipe": (2.0, "distribution"),
        "piping": (2.5, "distribution"),
        "domestic water": (3.0, "distribution"),
        "cold water": (2.5, "distribution"),
        "hot water": (2.5, "distribution"),
        "drain": (2.5, "distribution"),
        "waste": (2.0, "distribution"),
        "vent": (2.0, "distribution"),
        "sanitary": (3.0, "distribution"),
        "storm": (2.5, "distribution"),
        "storm drain": (3.0, "distribution"),
        # Fixtures
        "fixture": (2.5, "fixtures"),
        "plumbing fixture": (3.0, "fixtures"),
        "sink": (2.5, "fixtures"),
        "lavatory": (3.0, "fixtures"),
        "toilet": (2.5, "fixtures"),
        "water closet": (3.0, "fixtures"),
        "urinal": (3.0, "fixtures"),
        "floor drain": (3.0, "fixtures"),
        "cleanout": (3.0, "fixtures"),
        # Parameters
        "gpm": (3.0, "parameters"),
        "gallons per minute": (3.0, "parameters"),
        "fixture unit": (3.0, "parameters"),
        "dfu": (3.0, "parameters"),
        "wfu": (3.0, "parameters"),
        "slope": (2.0, "parameters"),
        "pitch": (2.0, "parameters"),
        # Fittings
        "valve": (2.0, "fittings"),
        "gate valve": (3.0, "fittings"),
        "ball valve": (3.0, "fittings"),
        "check valve": (3.0, "fittings"),
        "tee": (1.5, "fittings"),
        "elbow": (2.0, "fittings"),
        # General
        "plumbing": (3.0, None),
        "domestic": (2.0, None),
    },
    # =========================================================================
    # FIRE PROTECTION - Secondary focus
    # =========================================================================
    MEPDomain.FIRE_PROTECTION: {
        # Equipment
        "fire pump": (3.0, "equipment"),
        "jockey pump": (3.0, "equipment"),
        # Distribution
        "sprinkler": (3.0, "distribution"),
        "sprinkler head": (3.0, "distribution"),
        "standpipe": (3.0, "distribution"),
        "fire riser": (3.0, "distribution"),
        "sprinkler pipe": (3.0, "distribution"),
        # Devices
        "fire alarm": (3.0, "devices"),
        "smoke detector": (3.0, "devices"),
        "heat detector": (3.0, "devices"),
        "pull station": (3.0, "devices"),
        "horn strobe": (3.0, "devices"),
        "annunciator": (3.0, "devices"),
        # Parameters
        "coverage": (2.0, "parameters"),
        "sprinkler coverage": (3.0, "parameters"),
        "flow": (1.5, "parameters"),
        "residual pressure": (3.0, "parameters"),
        # General
        "fire protection": (3.0, None),
        "fire suppression": (3.0, None),
        "life safety": (2.5, None),
    },
}

# Action detection patterns
ACTION_PATTERNS: dict[str, list[str]] = {
    "query": [
        "find", "list", "show", "get", "what", "where", "which", "search",
        "locate", "identify", "display", "look for", "how many",
    ],
    "create": [
        "draw", "create", "add", "place", "insert", "make", "new",
        "generate", "build",
    ],
    "modify": [
        "edit", "change", "update", "modify", "move", "resize", "adjust",
        "relocate", "rename", "set",
    ],
    "delete": [
        "delete", "remove", "erase", "clear",
    ],
    "analyze": [
        "check", "validate", "verify", "analyze", "report", "review",
        "inspect", "audit", "assess",
    ],
    "route": [
        "route", "connect", "run", "path", "routing", "trace",
    ],
    "schedule": [
        "schedule", "count", "tally", "summarize", "tabulate", "list all",
    ],
    "coordinate": [
        "clash", "conflict", "coordinate", "interference", "collision",
        "overlapping",
    ],
}

# App context detection patterns
APP_CONTEXT_PATTERNS: dict[str, list[str]] = {
    "autocad": [
        "autocad", "acad", "dwg", "drawing", "layer", "polyline", "block",
        "xref", "entity", "handle", "model space", "paper space",
    ],
    "revit": [
        "revit", "rvt", "family", "type", "level", "workset", "element",
        "category", "parameter", "schedule view", "worksharing",
    ],
}


def get_all_patterns_flat() -> list[tuple[str, MEPDomain, Optional[str], float]]:
    """
    Get all patterns as a flat list for embedding.

    Returns:
        List of (pattern, domain, subdomain, weight) tuples
    """
    result = []
    for domain, patterns in MEP_PATTERNS.items():
        for pattern, (weight, subdomain) in patterns.items():
            result.append((pattern, domain, subdomain, weight))
    return result


def get_pattern_embeddings(embedding_service) -> dict[str, list[float]]:
    """
    Pre-compute embeddings for all patterns.

    Args:
        embedding_service: The embedding service to use

    Returns:
        Dict mapping pattern text to embedding vector
    """
    patterns = get_all_patterns_flat()
    texts = [p[0] for p in patterns]

    # Generate embeddings in batch for efficiency
    embeddings = embedding_service.generate_embeddings_batch(texts)

    return dict(zip(texts, embeddings))
