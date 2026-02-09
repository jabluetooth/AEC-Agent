"""
Default HVAC workflow templates.

These templates define multi-step workflows for common HVAC design tasks.
Each workflow executes a series of MCP tools without per-step LLM involvement.
"""

from uuid import uuid4

from aec_agent.workflows.models import WorkflowStep, WorkflowTemplate


def get_default_hvac_workflows() -> list[WorkflowTemplate]:
    """
    Get default HVAC workflow templates.

    Returns templates for:
    - Duct routing analysis
    - HVAC equipment schedule
    - Duct sizing verification
    - Diffuser coverage check
    """
    workflows = []

    # =========================================================================
    # 1. Duct Routing Analysis
    # =========================================================================
    workflows.append(WorkflowTemplate(
        id=uuid4(),
        name="hvac_duct_routing_analysis",
        domain="mep",
        subdomain="hvac",
        description="Analyze duct routing from equipment to terminals, checking clearances and suggesting paths.",
        steps=[
            WorkflowStep(
                name="sync_data",
                tool="sync_metadata",
                params_template={"source": "{source}"},
                description="Sync latest model data to database",
                on_error="abort",
            ),
            WorkflowStep(
                name="find_start",
                tool="find_elements",
                params_template={
                    "query": "{start_equipment}",
                    "source": "{source}",
                    "category": "Mechanical Equipment",
                    "limit": 5,
                },
                description="Find the starting equipment (AHU, etc.)",
                on_error="abort",
            ),
            WorkflowStep(
                name="find_terminals",
                tool="find_elements",
                params_template={
                    "query": "{end_terminals}",
                    "source": "{source}",
                    "category": "Air Terminals",
                    "limit": 50,
                },
                description="Find terminal units (VAV boxes, diffusers)",
                on_error="abort",
            ),
            WorkflowStep(
                name="check_nearby",
                tool="get_nearby_elements",
                params_template={
                    "element_id": "{find_start.elements[0].id}",
                    "distance": 3.0,
                    "limit": 20,
                },
                description="Check for nearby structural elements and potential conflicts",
                condition="len(find_start.get('elements', [])) > 0",
                on_error="skip",
            ),
            WorkflowStep(
                name="get_clearances",
                tool="get_related_elements",
                params_template={
                    "element_id": "{find_start.elements[0].id}",
                    "relation_type": "near",
                    "limit": 30,
                },
                description="Get elements in the routing path for clearance analysis",
                condition="len(find_start.get('elements', [])) > 0",
                on_error="skip",
            ),
        ],
        required_context=["source", "start_equipment", "end_terminals"],
        default_params={
            "source": "revit",
            "start_equipment": "AHU",
            "end_terminals": "VAV",
        },
        estimated_tokens=500,
    ))

    # =========================================================================
    # 2. HVAC Equipment Schedule
    # =========================================================================
    workflows.append(WorkflowTemplate(
        id=uuid4(),
        name="hvac_equipment_schedule",
        domain="mep",
        subdomain="hvac",
        description="Generate a schedule of HVAC equipment with CFM, capacity, and location.",
        steps=[
            WorkflowStep(
                name="sync_data",
                tool="sync_metadata",
                params_template={"source": "{source}"},
                description="Sync latest model data",
                on_error="skip",  # Continue even if sync fails (may have cached data)
            ),
            WorkflowStep(
                name="find_equipment",
                tool="find_elements",
                params_template={
                    "query": "mechanical equipment HVAC",
                    "source": "{source}",
                    "category": "Mechanical Equipment",
                    "limit": 100,
                },
                description="Find all HVAC equipment",
                on_error="abort",
            ),
            WorkflowStep(
                name="find_ahus",
                tool="find_elements",
                params_template={
                    "query": "air handling unit AHU",
                    "source": "{source}",
                    "limit": 20,
                },
                description="Find air handling units specifically",
                on_error="skip",
            ),
            WorkflowStep(
                name="find_vavs",
                tool="find_elements",
                params_template={
                    "query": "VAV terminal unit",
                    "source": "{source}",
                    "limit": 100,
                },
                description="Find VAV boxes",
                on_error="skip",
            ),
            WorkflowStep(
                name="find_fcus",
                tool="find_elements",
                params_template={
                    "query": "fan coil unit FCU",
                    "source": "{source}",
                    "limit": 50,
                },
                description="Find fan coil units",
                on_error="skip",
            ),
        ],
        required_context=["source"],
        default_params={"source": "revit"},
        estimated_tokens=400,
    ))

    # =========================================================================
    # 3. Duct Sizing Verification
    # =========================================================================
    workflows.append(WorkflowTemplate(
        id=uuid4(),
        name="hvac_duct_sizing_check",
        domain="mep",
        subdomain="hvac",
        description="Verify duct sizing by checking velocities against design criteria.",
        steps=[
            WorkflowStep(
                name="sync_data",
                tool="sync_metadata",
                params_template={"source": "{source}"},
                description="Sync latest duct data",
                on_error="skip",
            ),
            WorkflowStep(
                name="find_supply_ducts",
                tool="find_elements",
                params_template={
                    "query": "supply duct",
                    "source": "{source}",
                    "category": "Ducts",
                    "limit": 200,
                },
                description="Find supply ducts for velocity check",
                on_error="abort",
            ),
            WorkflowStep(
                name="find_return_ducts",
                tool="find_elements",
                params_template={
                    "query": "return duct",
                    "source": "{source}",
                    "category": "Ducts",
                    "limit": 200,
                },
                description="Find return ducts",
                on_error="skip",
            ),
            WorkflowStep(
                name="find_exhaust_ducts",
                tool="find_elements",
                params_template={
                    "query": "exhaust duct",
                    "source": "{source}",
                    "category": "Ducts",
                    "limit": 100,
                },
                description="Find exhaust ducts",
                on_error="skip",
            ),
        ],
        required_context=["source"],
        default_params={
            "source": "revit",
            "max_velocity_fpm": 1500,
        },
        estimated_tokens=350,
    ))

    # =========================================================================
    # 4. Diffuser Coverage Check
    # =========================================================================
    workflows.append(WorkflowTemplate(
        id=uuid4(),
        name="hvac_diffuser_coverage",
        domain="mep",
        subdomain="hvac",
        description="Check diffuser placement and coverage for rooms or levels.",
        steps=[
            WorkflowStep(
                name="sync_data",
                tool="sync_metadata",
                params_template={"source": "{source}"},
                description="Sync latest model data",
                on_error="skip",
            ),
            WorkflowStep(
                name="find_diffusers",
                tool="find_elements",
                params_template={
                    "query": "diffuser air terminal",
                    "source": "{source}",
                    "category": "Air Terminals",
                    "limit": 200,
                },
                description="Find all diffusers/air terminals",
                on_error="abort",
            ),
            WorkflowStep(
                name="find_rooms",
                tool="find_elements",
                params_template={
                    "query": "{room_filter}",
                    "source": "{source}",
                    "category": "Rooms",
                    "limit": 100,
                },
                description="Find rooms to check coverage",
                on_error="skip",
            ),
            WorkflowStep(
                name="check_spacing",
                tool="get_nearby_elements",
                params_template={
                    "element_id": "{find_diffusers.elements[0].id}",
                    "distance": 5.0,  # Max diffuser spacing ~15ft
                    "limit": 10,
                },
                description="Check spacing between diffusers",
                condition="len(find_diffusers.get('elements', [])) > 0",
                on_error="skip",
            ),
        ],
        required_context=["source"],
        default_params={
            "source": "revit",
            "room_filter": "room",
            "max_spacing_m": 4.5,
        },
        estimated_tokens=300,
    ))

    # =========================================================================
    # 5. Clash Detection Report
    # =========================================================================
    workflows.append(WorkflowTemplate(
        id=uuid4(),
        name="mep_clash_detection",
        domain="mep",
        subdomain=None,  # Applies to all MEP
        description="Find clashing/intersecting elements between MEP systems.",
        steps=[
            WorkflowStep(
                name="sync_data",
                tool="sync_metadata",
                params_template={"source": "{source}"},
                description="Sync latest model data",
                on_error="skip",
            ),
            WorkflowStep(
                name="find_ducts",
                tool="find_elements",
                params_template={
                    "query": "duct",
                    "source": "{source}",
                    "category": "Ducts",
                    "limit": 200,
                },
                description="Find duct elements",
                on_error="skip",
            ),
            WorkflowStep(
                name="find_pipes",
                tool="find_elements",
                params_template={
                    "query": "pipe",
                    "source": "{source}",
                    "category": "Pipes",
                    "limit": 200,
                },
                description="Find pipe elements",
                on_error="skip",
            ),
            WorkflowStep(
                name="check_duct_clashes",
                tool="get_related_elements",
                params_template={
                    "element_id": "{find_ducts.elements[0].id}",
                    "relation_type": "intersects",
                    "limit": 50,
                },
                description="Find elements intersecting with ducts",
                condition="len(find_ducts.get('elements', [])) > 0",
                on_error="skip",
            ),
        ],
        required_context=["source"],
        default_params={"source": "revit"},
        estimated_tokens=400,
    ))

    # =========================================================================
    # 6. System Balance Check
    # =========================================================================
    workflows.append(WorkflowTemplate(
        id=uuid4(),
        name="hvac_system_balance",
        domain="mep",
        subdomain="hvac",
        description="Check supply/return air balance for HVAC systems.",
        steps=[
            WorkflowStep(
                name="sync_data",
                tool="sync_metadata",
                params_template={"source": "{source}"},
                description="Sync latest model data",
                on_error="skip",
            ),
            WorkflowStep(
                name="find_supply_terminals",
                tool="find_elements",
                params_template={
                    "query": "supply diffuser air terminal",
                    "source": "{source}",
                    "category": "Air Terminals",
                    "limit": 200,
                },
                description="Find supply air terminals",
                on_error="abort",
            ),
            WorkflowStep(
                name="find_return_grilles",
                tool="find_elements",
                params_template={
                    "query": "return grille",
                    "source": "{source}",
                    "category": "Air Terminals",
                    "limit": 100,
                },
                description="Find return air grilles",
                on_error="skip",
            ),
            WorkflowStep(
                name="find_exhaust",
                tool="find_elements",
                params_template={
                    "query": "exhaust",
                    "source": "{source}",
                    "limit": 50,
                },
                description="Find exhaust terminals",
                on_error="skip",
            ),
        ],
        required_context=["source"],
        default_params={"source": "revit"},
        estimated_tokens=350,
    ))

    return workflows


def get_workflow_by_name(name: str) -> WorkflowTemplate | None:
    """Get a specific workflow template by name."""
    for workflow in get_default_hvac_workflows():
        if workflow.name == name:
            return workflow
    return None
