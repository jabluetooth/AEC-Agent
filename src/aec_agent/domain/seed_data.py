"""
Default MEP domain rules - HVAC focused.

These rules encode industry best practices for HVAC design.
"""

from uuid import uuid4

from aec_agent.domain.models import DomainRule, RuleSource, RuleType


def get_default_hvac_rules() -> list[DomainRule]:
    """
    Get default HVAC rules based on industry best practices.

    Returns a list of DomainRule objects covering:
    - Clearance requirements
    - Routing priorities
    - Sizing guidelines
    - Access requirements
    """
    rules = []

    # =========================================================================
    # CLEARANCE RULES
    # =========================================================================

    rules.append(DomainRule(
        id=uuid4(),
        domain="mep",
        subdomain="hvac",
        rule_type=RuleType.CLEARANCE,
        rule_name="duct_beam_clearance",
        condition={
            "element_type": ["duct", "supply_duct", "return_duct", "exhaust_duct"],
            "near_type": ["beam", "structural_beam", "w_beam"],
        },
        action={
            "min_clearance_mm": 150,  # 6 inches
            "min_clearance_in": 6,
            "message": "Maintain minimum 6\" (150mm) clearance from structural beams",
        },
        priority=10,
        source=RuleSource.BEST_PRACTICE,
        description="Ducts should maintain minimum clearance from structural beams for installation and maintenance access.",
    ))

    rules.append(DomainRule(
        id=uuid4(),
        domain="mep",
        subdomain="hvac",
        rule_type=RuleType.CLEARANCE,
        rule_name="duct_insulation_clearance",
        condition={
            "element_type": ["supply_duct", "duct"],
            "has_insulation": True,
        },
        action={
            "additional_clearance_mm": 50,  # 2 inches for insulation
            "additional_clearance_in": 2,
            "message": "Add 2\" (50mm) clearance for insulation on supply ducts",
        },
        priority=15,
        source=RuleSource.BEST_PRACTICE,
        description="Supply ducts with insulation require additional clearance for the insulation thickness.",
    ))

    rules.append(DomainRule(
        id=uuid4(),
        domain="mep",
        subdomain="hvac",
        rule_type=RuleType.CLEARANCE,
        rule_name="duct_column_clearance",
        condition={
            "element_type": ["duct"],
            "near_type": ["column", "structural_column"],
        },
        action={
            "min_clearance_mm": 100,  # 4 inches
            "min_clearance_in": 4,
            "message": "Maintain minimum 4\" (100mm) clearance from columns",
        },
        priority=12,
        source=RuleSource.BEST_PRACTICE,
        description="Ducts should maintain clearance from columns to allow for future modifications.",
    ))

    rules.append(DomainRule(
        id=uuid4(),
        domain="mep",
        subdomain="hvac",
        rule_type=RuleType.CLEARANCE,
        rule_name="diffuser_spacing",
        condition={
            "element_type": ["diffuser", "air_terminal", "supply_diffuser"],
        },
        action={
            "max_spacing_m": 4.5,  # Typical for office spaces
            "max_spacing_ft": 15,
            "throw_ratio": 0.8,  # Throw should be 80% of spacing
            "message": "Diffuser spacing should not exceed 15' (4.5m) for adequate coverage",
        },
        priority=20,
        source=RuleSource.BEST_PRACTICE,
        description="Diffuser spacing based on typical throw patterns for office environments.",
    ))

    rules.append(DomainRule(
        id=uuid4(),
        domain="mep",
        subdomain="hvac",
        rule_type=RuleType.CLEARANCE,
        rule_name="ahu_service_clearance",
        condition={
            "element_type": ["ahu", "air_handler", "air_handling_unit", "rtu"],
        },
        action={
            "service_clearance_mm": 900,  # 3 feet
            "service_clearance_ft": 3,
            "filter_access_mm": 1200,  # 4 feet for filter access
            "message": "Maintain 3' (900mm) service clearance around AHU, 4' for filter access",
        },
        priority=8,
        source=RuleSource.BEST_PRACTICE,
        description="AHUs require adequate clearance for filter replacement and maintenance.",
    ))

    # =========================================================================
    # ROUTING RULES
    # =========================================================================

    rules.append(DomainRule(
        id=uuid4(),
        domain="mep",
        subdomain="hvac",
        rule_type=RuleType.ROUTING,
        rule_name="trunk_parallel_to_structure",
        condition={
            "element_type": ["main_trunk", "trunk_duct"],
        },
        action={
            "orientation": "parallel_to_structure",
            "message": "Main trunk ducts should run parallel to structural grid",
        },
        priority=30,
        source=RuleSource.BEST_PRACTICE,
        description="Running main trunks parallel to structure simplifies coordination and provides consistent ceiling heights.",
    ))

    rules.append(DomainRule(
        id=uuid4(),
        domain="mep",
        subdomain="hvac",
        rule_type=RuleType.ROUTING,
        rule_name="branch_perpendicular",
        condition={
            "element_type": ["branch_duct", "branch"],
        },
        action={
            "orientation": "perpendicular_to_trunk",
            "message": "Branch ducts should run perpendicular to main trunk",
        },
        priority=32,
        source=RuleSource.BEST_PRACTICE,
        description="Perpendicular branches optimize space usage and simplify fitting selection.",
    ))

    rules.append(DomainRule(
        id=uuid4(),
        domain="mep",
        subdomain="hvac",
        rule_type=RuleType.ROUTING,
        rule_name="gravity_priority",
        condition={
            "conflict_type": "routing_priority",
            "systems": ["hvac", "plumbing"],
        },
        action={
            "priority_order": ["gravity_drain", "sanitary", "storm", "hvac"],
            "message": "Gravity drainage systems have routing priority over HVAC",
        },
        priority=5,
        source=RuleSource.BEST_PRACTICE,
        description="Gravity systems cannot be easily rerouted and take priority in coordination.",
    ))

    rules.append(DomainRule(
        id=uuid4(),
        domain="mep",
        subdomain="hvac",
        rule_type=RuleType.ROUTING,
        rule_name="minimize_elbows",
        condition={
            "element_type": ["duct"],
            "action": "route",
        },
        action={
            "max_elbows_per_run": 4,
            "elbow_equivalent_length_ft": 10,  # Typical 90° elbow
            "message": "Minimize elbows to reduce pressure drop (each 90° elbow ≈ 10' equivalent length)",
        },
        priority=25,
        source=RuleSource.BEST_PRACTICE,
        description="Excessive fittings increase system pressure drop and fan energy consumption.",
    ))

    # =========================================================================
    # SIZING RULES
    # =========================================================================

    rules.append(DomainRule(
        id=uuid4(),
        domain="mep",
        subdomain="hvac",
        rule_type=RuleType.SIZING,
        rule_name="supply_duct_velocity_low_noise",
        condition={
            "element_type": ["supply_duct", "duct"],
            "space_type": ["office", "conference", "classroom", "hospital"],
        },
        action={
            "max_velocity_fpm": 1500,
            "max_velocity_mps": 7.6,
            "recommended_fpm": 1200,
            "message": "Supply duct velocity should not exceed 1500 FPM in noise-sensitive areas",
        },
        priority=40,
        source=RuleSource.BEST_PRACTICE,
        description="Lower velocities reduce noise in occupied spaces. NC-35 or lower typically requires <1500 FPM.",
    ))

    rules.append(DomainRule(
        id=uuid4(),
        domain="mep",
        subdomain="hvac",
        rule_type=RuleType.SIZING,
        rule_name="main_duct_velocity",
        condition={
            "element_type": ["main_trunk", "trunk_duct"],
            "space_type": ["mechanical_room", "shaft", "above_ceiling"],
        },
        action={
            "max_velocity_fpm": 2500,
            "max_velocity_mps": 12.7,
            "message": "Main trunk velocity can be up to 2500 FPM in non-occupied spaces",
        },
        priority=42,
        source=RuleSource.BEST_PRACTICE,
        description="Higher velocities acceptable in mechanical spaces where noise is not a concern.",
    ))

    rules.append(DomainRule(
        id=uuid4(),
        domain="mep",
        subdomain="hvac",
        rule_type=RuleType.SIZING,
        rule_name="return_air_velocity",
        condition={
            "element_type": ["return_duct", "return_air"],
        },
        action={
            "velocity_factor": 1.2,  # Can be 20% higher than supply
            "max_velocity_fpm": 1800,
            "message": "Return air velocity can be 20% higher than supply (up to 1800 FPM)",
        },
        priority=41,
        source=RuleSource.BEST_PRACTICE,
        description="Return systems are less critical for noise and can operate at higher velocities.",
    ))

    rules.append(DomainRule(
        id=uuid4(),
        domain="mep",
        subdomain="hvac",
        rule_type=RuleType.SIZING,
        rule_name="branch_duct_sizing",
        condition={
            "element_type": ["branch_duct"],
        },
        action={
            "min_size_in": 6,
            "min_size_mm": 150,
            "aspect_ratio_max": 4,  # Width:Height
            "message": "Minimum branch duct size is 6\" (150mm), max aspect ratio 4:1",
        },
        priority=45,
        source=RuleSource.BEST_PRACTICE,
        description="Minimum sizes ensure adequate airflow; aspect ratio limits reduce pressure drop.",
    ))

    rules.append(DomainRule(
        id=uuid4(),
        domain="mep",
        subdomain="hvac",
        rule_type=RuleType.SIZING,
        rule_name="friction_rate_guideline",
        condition={
            "element_type": ["duct"],
            "method": "equal_friction",
        },
        action={
            "friction_rate_in_wg_per_100ft": 0.1,
            "friction_rate_pa_per_m": 0.82,
            "low_pressure_max": 0.08,
            "high_pressure_max": 0.15,
            "message": "Use 0.1\" w.g./100ft friction rate for medium pressure systems",
        },
        priority=38,
        source=RuleSource.BEST_PRACTICE,
        description="Equal friction method guideline for duct sizing. Adjust based on system pressure class.",
    ))

    # =========================================================================
    # ACCESS RULES
    # =========================================================================

    rules.append(DomainRule(
        id=uuid4(),
        domain="mep",
        subdomain="hvac",
        rule_type=RuleType.ACCESS,
        rule_name="damper_access",
        condition={
            "element_type": ["damper", "fire_damper", "smoke_damper", "volume_damper"],
        },
        action={
            "access_clearance_mm": 450,  # 18 inches
            "access_clearance_in": 18,
            "access_door_required": True,
            "message": "Dampers require 18\" (450mm) clearance and access door for adjustment/maintenance",
        },
        priority=15,
        source=RuleSource.CODE,
        description="Fire/smoke dampers require periodic inspection per code. All dampers need access for balancing.",
    ))

    rules.append(DomainRule(
        id=uuid4(),
        domain="mep",
        subdomain="hvac",
        rule_type=RuleType.ACCESS,
        rule_name="vav_box_access",
        condition={
            "element_type": ["vav", "vav_box", "terminal_unit"],
        },
        action={
            "access_clearance_mm": 600,  # 24 inches
            "access_clearance_in": 24,
            "access_door_size_in": "24x24",
            "message": "VAV boxes require 24\" (600mm) clearance and 24x24 access door",
        },
        priority=18,
        source=RuleSource.BEST_PRACTICE,
        description="VAV boxes require access for controller maintenance, actuator replacement, and balancing.",
    ))

    rules.append(DomainRule(
        id=uuid4(),
        domain="mep",
        subdomain="hvac",
        rule_type=RuleType.ACCESS,
        rule_name="coil_access",
        condition={
            "element_type": ["coil", "heating_coil", "cooling_coil"],
        },
        action={
            "pull_clearance_mm": 1200,  # Full coil pull
            "pull_clearance_ft": 4,
            "message": "Coils require clearance equal to coil depth for removal/replacement",
        },
        priority=12,
        source=RuleSource.BEST_PRACTICE,
        description="Coils may need to be pulled for cleaning or replacement.",
    ))

    # =========================================================================
    # VALIDATION RULES
    # =========================================================================

    rules.append(DomainRule(
        id=uuid4(),
        domain="mep",
        subdomain="hvac",
        rule_type=RuleType.VALIDATION,
        rule_name="check_air_balance",
        condition={
            "action": "validate",
            "scope": "system",
        },
        action={
            "supply_return_tolerance": 0.1,  # 10% tolerance
            "message": "System supply and return CFM should balance within 10%",
        },
        priority=50,
        source=RuleSource.BEST_PRACTICE,
        description="Balanced airflow prevents building pressurization issues.",
    ))

    rules.append(DomainRule(
        id=uuid4(),
        domain="mep",
        subdomain="hvac",
        rule_type=RuleType.VALIDATION,
        rule_name="diffuser_cfm_check",
        condition={
            "element_type": ["diffuser", "air_terminal"],
        },
        action={
            "min_cfm_per_diffuser": 50,
            "max_cfm_per_diffuser": 400,  # Typical ceiling diffuser
            "message": "Diffuser CFM should be between 50-400 CFM for typical ceiling diffusers",
        },
        priority=55,
        source=RuleSource.BEST_PRACTICE,
        description="CFM outside typical range may indicate sizing issues or missing diffusers.",
    ))

    return rules


def get_default_electrical_rules() -> list[DomainRule]:
    """Get default electrical rules (placeholder for future expansion)."""
    rules = []

    rules.append(DomainRule(
        id=uuid4(),
        domain="mep",
        subdomain="electrical",
        rule_type=RuleType.CLEARANCE,
        rule_name="panel_working_clearance",
        condition={
            "element_type": ["panel", "panelboard", "switchboard"],
        },
        action={
            "front_clearance_in": 36,
            "front_clearance_mm": 900,
            "message": "Electrical panels require 36\" (900mm) front working clearance per NEC",
        },
        priority=5,
        source=RuleSource.CODE,
        description="NEC 110.26 working space requirements.",
    ))

    return rules


def get_default_plumbing_rules() -> list[DomainRule]:
    """Get default plumbing rules (placeholder for future expansion)."""
    rules = []

    rules.append(DomainRule(
        id=uuid4(),
        domain="mep",
        subdomain="plumbing",
        rule_type=RuleType.ROUTING,
        rule_name="drain_slope",
        condition={
            "element_type": ["drain", "sanitary_pipe", "waste_pipe"],
        },
        action={
            "min_slope_in_per_ft": 0.25,  # 1/4" per foot
            "min_slope_percent": 2.08,
            "message": "Drainage pipes require minimum 1/4\" per foot (2%) slope",
        },
        priority=5,
        source=RuleSource.CODE,
        description="Gravity drainage requires adequate slope for proper flow.",
    ))

    return rules
