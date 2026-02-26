"""
Seed data for the symbol library.

Contains standard CAD symbols for electrical, mechanical, plumbing,
fire alarm, and architectural domains with NCS-compliant layer mapping.

These symbols are initially seeded without embeddings. Run the
`generate_symbol_embeddings()` function after seeding to create
CLIP embeddings from the symbol descriptions.

Usage:
    python -m aec_agent.mcp.tools.gemini_first.symbol_library_seed
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from uuid import uuid4

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Symbol Definitions
# =============================================================================

@dataclass
class SymbolDefinition:
    """Definition of a CAD symbol for seeding."""
    block_name: str
    display_name: str
    description: str
    domain: str
    category: str
    subcategory: Optional[str]
    layer: str
    standards: list[str]
    code_references: list[str] = None
    attributes: dict = None

    def __post_init__(self):
        if self.code_references is None:
            self.code_references = []
        if self.attributes is None:
            self.attributes = {}


# =============================================================================
# Electrical Symbols (E- prefix)
# =============================================================================

ELECTRICAL_SYMBOLS = [
    # Receptacles / Outlets
    SymbolDefinition(
        block_name="E-OUTL-DUP",
        display_name="Duplex Outlet",
        description="Standard duplex electrical outlet receptacle with two sockets, 120V 15A or 20A residential and commercial power outlet",
        domain="electrical",
        category="outlet",
        subcategory="duplex",
        layer="E-POWR-OUTL",
        standards=["NCS", "NEC"],
        code_references=["NEC 210.52"],
    ),
    SymbolDefinition(
        block_name="E-OUTL-GFCI",
        display_name="GFCI Outlet",
        description="Ground fault circuit interrupter outlet, GFCI protected receptacle for wet locations bathroom kitchen outdoor, safety outlet with test/reset buttons",
        domain="electrical",
        category="outlet",
        subcategory="gfci",
        layer="E-POWR-OUTL",
        standards=["NCS", "NEC"],
        code_references=["NEC 210.8"],
    ),
    SymbolDefinition(
        block_name="E-OUTL-WP",
        display_name="Weatherproof Outlet",
        description="Weatherproof outdoor electrical outlet with cover, exterior rated receptacle for outdoor use",
        domain="electrical",
        category="outlet",
        subcategory="weatherproof",
        layer="E-POWR-OUTL",
        standards=["NCS", "NEC"],
    ),
    SymbolDefinition(
        block_name="E-OUTL-QUAD",
        display_name="Quad Outlet",
        description="Four-plex quadruplex electrical outlet receptacle with four sockets, multi-outlet assembly",
        domain="electrical",
        category="outlet",
        subcategory="quad",
        layer="E-POWR-OUTL",
        standards=["NCS", "NEC"],
    ),
    SymbolDefinition(
        block_name="E-OUTL-220",
        display_name="220V Outlet",
        description="High voltage 220V 240V electrical outlet for appliances, dryer range HVAC equipment receptacle",
        domain="electrical",
        category="outlet",
        subcategory="220v",
        layer="E-POWR-OUTL",
        standards=["NCS", "NEC"],
    ),
    SymbolDefinition(
        block_name="E-OUTL-FLOOR",
        display_name="Floor Outlet",
        description="Floor mounted electrical outlet receptacle, poke-through floor box outlet",
        domain="electrical",
        category="outlet",
        subcategory="floor",
        layer="E-POWR-OUTL",
        standards=["NCS", "NEC"],
    ),
    SymbolDefinition(
        block_name="E-OUTL-USB",
        display_name="USB Outlet",
        description="Electrical outlet with USB charging ports, combination receptacle with USB-A USB-C ports",
        domain="electrical",
        category="outlet",
        subcategory="usb",
        layer="E-POWR-OUTL",
        standards=["NCS"],
    ),

    # Switches
    SymbolDefinition(
        block_name="E-SWCH-1P",
        display_name="Single Pole Switch",
        description="Single pole light switch, on/off toggle switch for lighting control from one location",
        domain="electrical",
        category="switch",
        subcategory="single_pole",
        layer="E-LITE-SWCH",
        standards=["NCS", "NEC"],
    ),
    SymbolDefinition(
        block_name="E-SWCH-3W",
        display_name="Three-Way Switch",
        description="Three way light switch, toggle switch for lighting control from two locations, hallway stairway switch",
        domain="electrical",
        category="switch",
        subcategory="three_way",
        layer="E-LITE-SWCH",
        standards=["NCS", "NEC"],
    ),
    SymbolDefinition(
        block_name="E-SWCH-4W",
        display_name="Four-Way Switch",
        description="Four way light switch, toggle switch for lighting control from three or more locations",
        domain="electrical",
        category="switch",
        subcategory="four_way",
        layer="E-LITE-SWCH",
        standards=["NCS", "NEC"],
    ),
    SymbolDefinition(
        block_name="E-SWCH-DIM",
        display_name="Dimmer Switch",
        description="Dimmer light switch, adjustable brightness control for lighting, rotary or slide dimmer",
        domain="electrical",
        category="switch",
        subcategory="dimmer",
        layer="E-LITE-SWCH",
        standards=["NCS"],
    ),
    SymbolDefinition(
        block_name="E-SWCH-OCC",
        display_name="Occupancy Sensor Switch",
        description="Occupancy sensor motion sensor switch, automatic lighting control PIR sensor wall switch",
        domain="electrical",
        category="switch",
        subcategory="occupancy",
        layer="E-LITE-SWCH",
        standards=["NCS", "Title 24"],
        code_references=["Title 24"],
    ),
    SymbolDefinition(
        block_name="E-SWCH-KEY",
        display_name="Key Switch",
        description="Key operated switch, security switch requiring key for operation",
        domain="electrical",
        category="switch",
        subcategory="key",
        layer="E-LITE-SWCH",
        standards=["NCS"],
    ),

    # Lighting
    SymbolDefinition(
        block_name="E-LITE-2X4",
        display_name="2x4 Troffer",
        description="Recessed 2x4 troffer light fixture, fluorescent or LED ceiling light panel for drop ceiling",
        domain="electrical",
        category="light",
        subcategory="troffer_2x4",
        layer="E-LITE-CEIL",
        standards=["NCS"],
    ),
    SymbolDefinition(
        block_name="E-LITE-2X2",
        display_name="2x2 Troffer",
        description="Recessed 2x2 troffer light fixture, square fluorescent or LED ceiling light panel",
        domain="electrical",
        category="light",
        subcategory="troffer_2x2",
        layer="E-LITE-CEIL",
        standards=["NCS"],
    ),
    SymbolDefinition(
        block_name="E-LITE-DOWN",
        display_name="Downlight",
        description="Recessed downlight can light, circular ceiling mounted recessed lighting fixture",
        domain="electrical",
        category="light",
        subcategory="downlight",
        layer="E-LITE-CEIL",
        standards=["NCS"],
    ),
    SymbolDefinition(
        block_name="E-LITE-EXIT",
        display_name="Exit Sign",
        description="Illuminated exit sign, egress lighting emergency exit signage with battery backup",
        domain="electrical",
        category="light",
        subcategory="exit",
        layer="E-LITE-EXIT",
        standards=["NCS", "NFPA 101"],
        code_references=["NFPA 101", "IBC 1013"],
    ),
    SymbolDefinition(
        block_name="E-LITE-EMRG",
        display_name="Emergency Light",
        description="Emergency lighting unit, battery backup emergency egress lighting with dual heads",
        domain="electrical",
        category="light",
        subcategory="emergency",
        layer="E-LITE-EMER",
        standards=["NCS", "NFPA 101"],
        code_references=["NFPA 101"],
    ),
    SymbolDefinition(
        block_name="E-LITE-WALL",
        display_name="Wall Sconce",
        description="Wall mounted light fixture sconce, decorative or functional wall lighting",
        domain="electrical",
        category="light",
        subcategory="wall_sconce",
        layer="E-LITE-WALL",
        standards=["NCS"],
    ),

    # Panels
    SymbolDefinition(
        block_name="E-PANL-MAIN",
        display_name="Main Panel",
        description="Main electrical panel, main breaker panel distribution panel MDP service entrance",
        domain="electrical",
        category="panel",
        subcategory="main",
        layer="E-POWR-PANL",
        standards=["NCS", "NEC"],
    ),
    SymbolDefinition(
        block_name="E-PANL-SUB",
        display_name="Sub Panel",
        description="Sub panel branch panel, subpanel distribution board for branch circuits",
        domain="electrical",
        category="panel",
        subcategory="sub",
        layer="E-POWR-PANL",
        standards=["NCS", "NEC"],
    ),
    SymbolDefinition(
        block_name="E-XFMR",
        display_name="Transformer",
        description="Electrical transformer, voltage step-down or step-up transformer",
        domain="electrical",
        category="equipment",
        subcategory="transformer",
        layer="E-POWR-EQPM",
        standards=["NCS", "NEC"],
    ),

    # Data / Low Voltage
    SymbolDefinition(
        block_name="E-DATA-OUTL",
        display_name="Data Outlet",
        description="Data outlet network jack, RJ45 ethernet CAT5 CAT6 network connection wall plate",
        domain="electrical",
        category="data",
        subcategory="outlet",
        layer="E-COMM-DATA",
        standards=["NCS", "TIA/EIA"],
    ),
    SymbolDefinition(
        block_name="E-DATA-RACK",
        display_name="Data Rack",
        description="Network data rack server rack, 19-inch equipment rack for network switches servers",
        domain="electrical",
        category="data",
        subcategory="rack",
        layer="E-COMM-EQPM",
        standards=["NCS", "TIA/EIA"],
    ),
]


# =============================================================================
# Mechanical / HVAC Symbols (M- prefix)
# =============================================================================

MECHANICAL_SYMBOLS = [
    # Diffusers
    SymbolDefinition(
        block_name="M-DIFF-SQ-S",
        display_name="Square Supply Diffuser",
        description="Square ceiling supply air diffuser, HVAC air distribution supply register square pattern",
        domain="mechanical",
        category="diffuser",
        subcategory="supply_square",
        layer="M-HVAC-DIFF",
        standards=["NCS", "ASHRAE"],
    ),
    SymbolDefinition(
        block_name="M-DIFF-RD-S",
        display_name="Round Supply Diffuser",
        description="Round circular ceiling supply air diffuser, HVAC air distribution supply register round pattern",
        domain="mechanical",
        category="diffuser",
        subcategory="supply_round",
        layer="M-HVAC-DIFF",
        standards=["NCS", "ASHRAE"],
    ),
    SymbolDefinition(
        block_name="M-DIFF-LIN-S",
        display_name="Linear Supply Diffuser",
        description="Linear slot supply air diffuser, continuous slot ceiling air distribution linear pattern",
        domain="mechanical",
        category="diffuser",
        subcategory="supply_linear",
        layer="M-HVAC-DIFF",
        standards=["NCS", "ASHRAE"],
    ),
    SymbolDefinition(
        block_name="M-DIFF-SQ-R",
        display_name="Square Return Grille",
        description="Square ceiling return air grille, HVAC return air intake register square pattern",
        domain="mechanical",
        category="diffuser",
        subcategory="return_square",
        layer="M-HVAC-DIFF",
        standards=["NCS", "ASHRAE"],
    ),
    SymbolDefinition(
        block_name="M-DIFF-RD-R",
        display_name="Round Return Grille",
        description="Round circular ceiling return air grille, HVAC return air intake register round pattern",
        domain="mechanical",
        category="diffuser",
        subcategory="return_round",
        layer="M-HVAC-DIFF",
        standards=["NCS", "ASHRAE"],
    ),
    SymbolDefinition(
        block_name="M-DIFF-LIN-R",
        display_name="Linear Return Grille",
        description="Linear slot return air grille, continuous slot ceiling return air intake linear pattern",
        domain="mechanical",
        category="diffuser",
        subcategory="return_linear",
        layer="M-HVAC-DIFF",
        standards=["NCS", "ASHRAE"],
    ),

    # Equipment
    SymbolDefinition(
        block_name="M-AHU",
        display_name="Air Handling Unit",
        description="Air handling unit AHU, HVAC air handler with fan coil filter for air conditioning",
        domain="mechanical",
        category="equipment",
        subcategory="ahu",
        layer="M-HVAC-EQPM",
        standards=["NCS", "ASHRAE"],
    ),
    SymbolDefinition(
        block_name="M-RTU",
        display_name="Rooftop Unit",
        description="Rooftop HVAC unit RTU, packaged rooftop air conditioning unit",
        domain="mechanical",
        category="equipment",
        subcategory="rtu",
        layer="M-HVAC-EQPM",
        standards=["NCS", "ASHRAE"],
    ),
    SymbolDefinition(
        block_name="M-FCU",
        display_name="Fan Coil Unit",
        description="Fan coil unit FCU, terminal unit with fan and coil for zone heating cooling",
        domain="mechanical",
        category="equipment",
        subcategory="fcu",
        layer="M-HVAC-EQPM",
        standards=["NCS", "ASHRAE"],
    ),
    SymbolDefinition(
        block_name="M-VAV",
        display_name="VAV Box",
        description="Variable air volume box VAV, HVAC terminal unit for zone air flow control",
        domain="mechanical",
        category="equipment",
        subcategory="vav",
        layer="M-HVAC-EQPM",
        standards=["NCS", "ASHRAE"],
    ),
    SymbolDefinition(
        block_name="M-EXH-FAN",
        display_name="Exhaust Fan",
        description="Exhaust fan, ventilation fan for air exhaust bathroom kitchen mechanical exhaust",
        domain="mechanical",
        category="equipment",
        subcategory="exhaust_fan",
        layer="M-HVAC-EQPM",
        standards=["NCS", "ASHRAE"],
    ),

    # Thermostats
    SymbolDefinition(
        block_name="M-STAT",
        display_name="Thermostat",
        description="Thermostat temperature control, HVAC zone thermostat for heating cooling control",
        domain="mechanical",
        category="control",
        subcategory="thermostat",
        layer="M-HVAC-CTRL",
        standards=["NCS", "ASHRAE"],
    ),
    SymbolDefinition(
        block_name="M-SENSOR-T",
        display_name="Temperature Sensor",
        description="Temperature sensor, room temperature sensor for building automation BAS",
        domain="mechanical",
        category="control",
        subcategory="temp_sensor",
        layer="M-HVAC-CTRL",
        standards=["NCS", "ASHRAE"],
    ),
    SymbolDefinition(
        block_name="M-SENSOR-CO2",
        display_name="CO2 Sensor",
        description="CO2 sensor carbon dioxide sensor, air quality sensor for demand controlled ventilation",
        domain="mechanical",
        category="control",
        subcategory="co2_sensor",
        layer="M-HVAC-CTRL",
        standards=["NCS", "ASHRAE"],
        code_references=["ASHRAE 62.1"],
    ),
]


# =============================================================================
# Plumbing Symbols (P- prefix)
# =============================================================================

PLUMBING_SYMBOLS = [
    # Fixtures
    SymbolDefinition(
        block_name="P-FIXT-LAV",
        display_name="Lavatory",
        description="Lavatory sink bathroom vanity sink, wall or counter mounted hand washing basin",
        domain="plumbing",
        category="fixture",
        subcategory="lavatory",
        layer="P-FIXT",
        standards=["NCS", "CPC"],
    ),
    SymbolDefinition(
        block_name="P-FIXT-SINK",
        display_name="Kitchen Sink",
        description="Kitchen sink, single or double bowl counter sink for kitchen food prep",
        domain="plumbing",
        category="fixture",
        subcategory="sink",
        layer="P-FIXT",
        standards=["NCS", "CPC"],
    ),
    SymbolDefinition(
        block_name="P-FIXT-WC",
        display_name="Water Closet",
        description="Water closet toilet, floor mounted toilet fixture restroom water closet",
        domain="plumbing",
        category="fixture",
        subcategory="water_closet",
        layer="P-FIXT",
        standards=["NCS", "CPC"],
    ),
    SymbolDefinition(
        block_name="P-FIXT-URIN",
        display_name="Urinal",
        description="Urinal, wall mounted urinal fixture for men's restroom",
        domain="plumbing",
        category="fixture",
        subcategory="urinal",
        layer="P-FIXT",
        standards=["NCS", "CPC"],
    ),
    SymbolDefinition(
        block_name="P-FIXT-SHOW",
        display_name="Shower",
        description="Shower, bathroom shower stall or tub-shower combination",
        domain="plumbing",
        category="fixture",
        subcategory="shower",
        layer="P-FIXT",
        standards=["NCS", "CPC"],
    ),
    SymbolDefinition(
        block_name="P-FIXT-TUB",
        display_name="Bathtub",
        description="Bathtub, bathroom tub freestanding or built-in bathtub",
        domain="plumbing",
        category="fixture",
        subcategory="bathtub",
        layer="P-FIXT",
        standards=["NCS", "CPC"],
    ),
    SymbolDefinition(
        block_name="P-FIXT-DF",
        display_name="Drinking Fountain",
        description="Drinking fountain water fountain, wall mounted water cooler ADA accessible",
        domain="plumbing",
        category="fixture",
        subcategory="drinking_fountain",
        layer="P-FIXT",
        standards=["NCS", "CPC", "ADA"],
    ),
    SymbolDefinition(
        block_name="P-FIXT-FD",
        display_name="Floor Drain",
        description="Floor drain, floor drain with strainer for water drainage",
        domain="plumbing",
        category="fixture",
        subcategory="floor_drain",
        layer="P-FIXT",
        standards=["NCS", "CPC"],
    ),

    # Valves
    SymbolDefinition(
        block_name="P-VALV-GATE",
        display_name="Gate Valve",
        description="Gate valve, isolation shut-off valve for piping system",
        domain="plumbing",
        category="valve",
        subcategory="gate",
        layer="P-VALV",
        standards=["NCS", "CPC"],
    ),
    SymbolDefinition(
        block_name="P-VALV-BALL",
        display_name="Ball Valve",
        description="Ball valve, quarter turn ball valve for quick shut-off",
        domain="plumbing",
        category="valve",
        subcategory="ball",
        layer="P-VALV",
        standards=["NCS", "CPC"],
    ),
    SymbolDefinition(
        block_name="P-VALV-CHECK",
        display_name="Check Valve",
        description="Check valve, non-return valve to prevent backflow",
        domain="plumbing",
        category="valve",
        subcategory="check",
        layer="P-VALV",
        standards=["NCS", "CPC"],
    ),
    SymbolDefinition(
        block_name="P-VALV-PRV",
        display_name="Pressure Reducing Valve",
        description="Pressure reducing valve PRV, pressure regulator for water pressure control",
        domain="plumbing",
        category="valve",
        subcategory="prv",
        layer="P-VALV",
        standards=["NCS", "CPC"],
    ),

    # Equipment
    SymbolDefinition(
        block_name="P-WH",
        display_name="Water Heater",
        description="Water heater, gas or electric water heater for domestic hot water",
        domain="plumbing",
        category="equipment",
        subcategory="water_heater",
        layer="P-EQPM",
        standards=["NCS", "CPC"],
    ),
    SymbolDefinition(
        block_name="P-PUMP",
        display_name="Pump",
        description="Water pump, booster pump or circulation pump for plumbing system",
        domain="plumbing",
        category="equipment",
        subcategory="pump",
        layer="P-EQPM",
        standards=["NCS", "CPC"],
    ),
]


# =============================================================================
# Fire Alarm Symbols (F- prefix)
# =============================================================================

FIRE_SYMBOLS = [
    # Detection Devices
    SymbolDefinition(
        block_name="F-DETC-SMOK",
        display_name="Smoke Detector",
        description="Smoke detector, photoelectric or ionization smoke detection device ceiling mounted",
        domain="fire",
        category="detector",
        subcategory="smoke",
        layer="F-FIRE-DETC",
        standards=["NCS", "NFPA 72"],
        code_references=["NFPA 72", "CFC"],
    ),
    SymbolDefinition(
        block_name="F-DETC-HEAT",
        display_name="Heat Detector",
        description="Heat detector, fixed temperature or rate-of-rise heat detection device",
        domain="fire",
        category="detector",
        subcategory="heat",
        layer="F-FIRE-DETC",
        standards=["NCS", "NFPA 72"],
        code_references=["NFPA 72", "CFC"],
    ),
    SymbolDefinition(
        block_name="F-DETC-DUCT",
        display_name="Duct Smoke Detector",
        description="Duct smoke detector, duct mounted smoke detection for HVAC system",
        domain="fire",
        category="detector",
        subcategory="duct_smoke",
        layer="F-FIRE-DETC",
        standards=["NCS", "NFPA 72"],
        code_references=["NFPA 72", "NFPA 90A"],
    ),
    SymbolDefinition(
        block_name="F-DETC-BEAM",
        display_name="Beam Detector",
        description="Beam smoke detector, projected beam smoke detection for large spaces",
        domain="fire",
        category="detector",
        subcategory="beam",
        layer="F-FIRE-DETC",
        standards=["NCS", "NFPA 72"],
        code_references=["NFPA 72"],
    ),

    # Notification Devices
    SymbolDefinition(
        block_name="F-NOTF-HORN",
        display_name="Horn",
        description="Fire alarm horn, audible notification appliance horn speaker",
        domain="fire",
        category="notification",
        subcategory="horn",
        layer="F-FIRE-NOTF",
        standards=["NCS", "NFPA 72"],
        code_references=["NFPA 72"],
    ),
    SymbolDefinition(
        block_name="F-NOTF-STRB",
        display_name="Strobe",
        description="Fire alarm strobe, visual notification appliance flashing strobe light",
        domain="fire",
        category="notification",
        subcategory="strobe",
        layer="F-FIRE-NOTF",
        standards=["NCS", "NFPA 72", "ADA"],
        code_references=["NFPA 72", "ADA"],
    ),
    SymbolDefinition(
        block_name="F-NOTF-HS",
        display_name="Horn/Strobe",
        description="Fire alarm horn strobe combination, audible and visual notification appliance",
        domain="fire",
        category="notification",
        subcategory="horn_strobe",
        layer="F-FIRE-NOTF",
        standards=["NCS", "NFPA 72", "ADA"],
        code_references=["NFPA 72", "ADA"],
    ),
    SymbolDefinition(
        block_name="F-NOTF-SPKR",
        display_name="Speaker",
        description="Fire alarm speaker, voice evacuation speaker for mass notification",
        domain="fire",
        category="notification",
        subcategory="speaker",
        layer="F-FIRE-NOTF",
        standards=["NCS", "NFPA 72"],
        code_references=["NFPA 72"],
    ),

    # Manual Devices
    SymbolDefinition(
        block_name="F-PULL",
        display_name="Pull Station",
        description="Manual pull station fire alarm pull box, manual fire alarm initiating device",
        domain="fire",
        category="manual",
        subcategory="pull_station",
        layer="F-FIRE-MANL",
        standards=["NCS", "NFPA 72"],
        code_references=["NFPA 72", "CFC"],
    ),

    # Control Equipment
    SymbolDefinition(
        block_name="F-FACP",
        display_name="Fire Alarm Control Panel",
        description="Fire alarm control panel FACP, main fire alarm system control unit",
        domain="fire",
        category="control",
        subcategory="facp",
        layer="F-FIRE-CTRL",
        standards=["NCS", "NFPA 72"],
        code_references=["NFPA 72"],
    ),
    SymbolDefinition(
        block_name="F-NAC",
        display_name="NAC Panel",
        description="Notification appliance circuit panel NAC, booster power supply for notification devices",
        domain="fire",
        category="control",
        subcategory="nac",
        layer="F-FIRE-CTRL",
        standards=["NCS", "NFPA 72"],
        code_references=["NFPA 72"],
    ),
    SymbolDefinition(
        block_name="F-ANN",
        display_name="Annunciator",
        description="Fire alarm annunciator, remote annunciator panel for system status display",
        domain="fire",
        category="control",
        subcategory="annunciator",
        layer="F-FIRE-CTRL",
        standards=["NCS", "NFPA 72"],
        code_references=["NFPA 72"],
    ),
]


# =============================================================================
# Architectural Symbols (A- prefix)
# =============================================================================

ARCHITECTURAL_SYMBOLS = [
    # Doors
    SymbolDefinition(
        block_name="A-DOOR-SGL",
        display_name="Single Door",
        description="Single door swing door, standard hinged door with door swing arc",
        domain="architectural",
        category="door",
        subcategory="single",
        layer="A-DOOR",
        standards=["NCS", "AIA"],
    ),
    SymbolDefinition(
        block_name="A-DOOR-DBL",
        display_name="Double Door",
        description="Double door pair, double swing doors two-leaf door assembly",
        domain="architectural",
        category="door",
        subcategory="double",
        layer="A-DOOR",
        standards=["NCS", "AIA"],
    ),
    SymbolDefinition(
        block_name="A-DOOR-SLD",
        display_name="Sliding Door",
        description="Sliding door, pocket door or surface mounted sliding door",
        domain="architectural",
        category="door",
        subcategory="sliding",
        layer="A-DOOR",
        standards=["NCS", "AIA"],
    ),
    SymbolDefinition(
        block_name="A-DOOR-BFLD",
        display_name="Bifold Door",
        description="Bifold door, accordion folding door for closets or room dividers",
        domain="architectural",
        category="door",
        subcategory="bifold",
        layer="A-DOOR",
        standards=["NCS", "AIA"],
    ),

    # Windows
    SymbolDefinition(
        block_name="A-WIND-FIX",
        display_name="Fixed Window",
        description="Fixed window, non-operable stationary window glazing",
        domain="architectural",
        category="window",
        subcategory="fixed",
        layer="A-GLAZ",
        standards=["NCS", "AIA"],
    ),
    SymbolDefinition(
        block_name="A-WIND-SH",
        display_name="Single Hung Window",
        description="Single hung window, operable sash window with one moving panel",
        domain="architectural",
        category="window",
        subcategory="single_hung",
        layer="A-GLAZ",
        standards=["NCS", "AIA"],
    ),
    SymbolDefinition(
        block_name="A-WIND-DH",
        display_name="Double Hung Window",
        description="Double hung window, operable sash window with two moving panels",
        domain="architectural",
        category="window",
        subcategory="double_hung",
        layer="A-GLAZ",
        standards=["NCS", "AIA"],
    ),
    SymbolDefinition(
        block_name="A-WIND-CSMT",
        display_name="Casement Window",
        description="Casement window, side hinged operable window crank operated",
        domain="architectural",
        category="window",
        subcategory="casement",
        layer="A-GLAZ",
        standards=["NCS", "AIA"],
    ),

    # Stairs
    SymbolDefinition(
        block_name="A-STRS-UP",
        display_name="Stairs Up",
        description="Stairway going up, stairs with up direction arrow flight of stairs",
        domain="architectural",
        category="stairs",
        subcategory="up",
        layer="A-FLOR-STRS",
        standards=["NCS", "AIA"],
    ),
    SymbolDefinition(
        block_name="A-STRS-DN",
        display_name="Stairs Down",
        description="Stairway going down, stairs with down direction arrow flight of stairs",
        domain="architectural",
        category="stairs",
        subcategory="down",
        layer="A-FLOR-STRS",
        standards=["NCS", "AIA"],
    ),

    # Misc
    SymbolDefinition(
        block_name="A-ELEV",
        display_name="Elevator",
        description="Elevator, passenger elevator cab elevator shaft symbol",
        domain="architectural",
        category="vertical",
        subcategory="elevator",
        layer="A-FLOR-EVTR",
        standards=["NCS", "AIA"],
    ),
    SymbolDefinition(
        block_name="A-ROOM-NUM",
        display_name="Room Number",
        description="Room number tag, room identification label with number",
        domain="architectural",
        category="annotation",
        subcategory="room_number",
        layer="A-AREA-IDEN",
        standards=["NCS", "AIA"],
        attributes={"ROOM_NUM": "000", "ROOM_NAME": ""},
    ),
    SymbolDefinition(
        block_name="A-NORTH",
        display_name="North Arrow",
        description="North arrow, compass north direction indicator for plan orientation",
        domain="architectural",
        category="annotation",
        subcategory="north_arrow",
        layer="A-ANNO-SYMB",
        standards=["NCS", "AIA"],
    ),
]


# =============================================================================
# Combine All Symbols
# =============================================================================

ALL_SYMBOLS: list[SymbolDefinition] = (
    ELECTRICAL_SYMBOLS +
    MECHANICAL_SYMBOLS +
    PLUMBING_SYMBOLS +
    FIRE_SYMBOLS +
    ARCHITECTURAL_SYMBOLS
)


# =============================================================================
# Seeding Functions
# =============================================================================

async def seed_symbol_library(pool, generate_embeddings: bool = True) -> int:
    """
    Seed the symbol library with standard CAD symbols.

    Args:
        pool: DatabasePool instance
        generate_embeddings: Whether to generate CLIP embeddings from descriptions

    Returns:
        Number of symbols seeded
    """
    from aec_agent.mcp.tools.gemini_first.symbol_rag import (
        SymbolLibraryEntry,
        SymbolRAGRepository,
        CLIPEncoder,
        is_symbol_rag_available,
    )

    repo = SymbolRAGRepository(pool)
    encoder = None

    if generate_embeddings and is_symbol_rag_available():
        encoder = CLIPEncoder.get_instance()
        await encoder.initialize()
        logger.info("CLIP encoder initialized for text embeddings")
    elif generate_embeddings:
        logger.warning("CLIP not available, seeding without embeddings")

    count = 0
    for sym in ALL_SYMBOLS:
        # Generate embedding from description if encoder available
        embedding = None
        if encoder:
            try:
                # Use text embedding from description
                embedding = encoder.encode_text(sym.description).tolist()
            except Exception as e:
                logger.warning(f"Failed to encode {sym.block_name}: {e}")

        entry = SymbolLibraryEntry(
            id=uuid4(),
            block_name=sym.block_name,
            display_name=sym.display_name,
            description=sym.description,
            domain=sym.domain,
            category=sym.category,
            subcategory=sym.subcategory,
            layer=sym.layer,
            embedding=embedding,
            preview_image=None,  # No preview image for text-only seed
            attributes=sym.attributes,
            standards=sym.standards,
            jurisdiction="CA",  # Default to California
            code_references=sym.code_references,
        )

        try:
            await repo.add_symbol(entry)
            count += 1
            logger.debug(f"Seeded symbol: {sym.block_name}")
        except Exception as e:
            logger.warning(f"Failed to seed {sym.block_name}: {e}")

    logger.info(f"Seeded {count} symbols to library")
    return count


async def get_symbol_stats(pool) -> dict:
    """Get statistics about the symbol library."""
    query = """
        SELECT
            domain,
            COUNT(*) as count,
            COUNT(embedding) as with_embedding
        FROM symbol_library
        WHERE is_active = TRUE
        GROUP BY domain
        ORDER BY domain
    """

    rows = await pool.fetch(query)

    stats = {
        "total": 0,
        "with_embeddings": 0,
        "by_domain": {},
    }

    for row in rows:
        stats["by_domain"][row["domain"]] = {
            "count": row["count"],
            "with_embedding": row["with_embedding"],
        }
        stats["total"] += row["count"]
        stats["with_embeddings"] += row["with_embedding"]

    return stats


# =============================================================================
# CLI Entry Point
# =============================================================================

async def main():
    """CLI entry point for seeding the symbol library."""
    import argparse

    from aec_agent.config.settings import get_settings
    from aec_agent.db.connection import initialize_database_pool, close_database_pool

    parser = argparse.ArgumentParser(description="Seed the CAD symbol library")
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip generating CLIP embeddings",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show library statistics",
    )
    args = parser.parse_args()

    settings = get_settings()
    if not settings.database_url:
        print("ERROR: DATABASE_URL not set")
        return

    pool = await initialize_database_pool(settings.database_url)

    try:
        if args.stats_only:
            stats = await get_symbol_stats(pool)
            print(f"\nSymbol Library Statistics:")
            print(f"  Total symbols: {stats['total']}")
            print(f"  With embeddings: {stats['with_embeddings']}")
            print(f"\n  By domain:")
            for domain, data in stats["by_domain"].items():
                print(f"    {domain}: {data['count']} ({data['with_embedding']} with embeddings)")
        else:
            count = await seed_symbol_library(
                pool,
                generate_embeddings=not args.no_embeddings,
            )
            print(f"\nSeeded {count} symbols to library")

            stats = await get_symbol_stats(pool)
            print(f"Total in library: {stats['total']}")
    finally:
        await close_database_pool()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
