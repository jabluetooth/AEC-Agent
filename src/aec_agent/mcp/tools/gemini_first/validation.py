"""
Phase 6: Validation & Self-Correction - Verify and correct extracted entities.

This module uses Gemini Vision to compare created entities against the original
drawing and apply corrections. It's the quality assurance phase of the pipeline.

Key Features:
- Visual comparison of original vs created entities
- Issue detection: missing, extra, position errors, text errors, symbol errors
- Automatic correction suggestions
- Iterative refinement loop (up to max_iterations)
- Detailed validation reporting

Usage:
    >>> from .validation import validate_extraction
    >>> result = await validate_extraction(
    ...     original_image_path=image_path,
    ...     creation_result=creation_result,
    ...     analysis=analysis,
    ...     calibration=calibration,
    ... )
    >>> print(f"Status: {result.status}, Accuracy: {result.accuracy_estimate}%")
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import structlog

from aec_agent.config.settings import get_settings

from .adaptive_extraction import EntityToCreate, ExtractionResult
from .autocad_creation import (
    AutoCADCreationResult,
    create_single_entity,
    create_layer_if_needed,
)
from .coordinate_calibration import ScaleCalibration
from .gemini_understanding import DrawingAnalysis

logger = structlog.get_logger(__name__)


# =============================================================================
# Enums
# =============================================================================

class ValidationStatus(str, Enum):
    """Status of validation result."""
    APPROVED = "approved"  # All entities match, no issues
    ISSUES_FOUND = "issues_found"  # Issues detected, corrections available
    MANUAL_REVIEW = "manual_review"  # Issues found but no auto-corrections
    MAX_ITERATIONS = "max_iterations"  # Hit iteration limit
    ERROR = "error"  # Validation failed


class IssueType(str, Enum):
    """Types of validation issues."""
    MISSING_ELEMENT = "missing_element"
    EXTRA_ELEMENT = "extra_element"
    POSITION_ERROR = "position_error"
    TEXT_ERROR = "text_error"
    SYMBOL_ERROR = "symbol_error"
    CONNECTIVITY = "connectivity"
    LAYER_ERROR = "layer_error"
    ATTRIBUTE_ERROR = "attribute_error"


class IssueSeverity(str, Enum):
    """Severity levels for issues."""
    CRITICAL = "critical"  # Must fix, blocks approval
    MAJOR = "major"  # Should fix
    MINOR = "minor"  # Nice to fix


class CorrectionAction(str, Enum):
    """Actions to correct issues."""
    ADD = "ADD"  # Add missing element
    REMOVE = "REMOVE"  # Remove false positive
    MODIFY = "MODIFY"  # Change element properties
    REPLACE = "REPLACE"  # Replace with correct element


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ValidationIssue:
    """A single validation issue detected."""
    issue_type: IssueType
    description: str
    severity: IssueSeverity
    location: Optional[Tuple[float, float]] = None  # DWG coordinates
    pixel_location: Optional[Tuple[int, int]] = None  # Image pixel coords
    entity_handle: Optional[str] = None  # Handle of affected entity
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None

    def to_dict(self) -> dict:
        return {
            "type": self.issue_type.value if isinstance(self.issue_type, IssueType) else str(self.issue_type),
            "description": self.description,
            "severity": self.severity.value if isinstance(self.severity, IssueSeverity) else str(self.severity),
            "location": list(self.location) if self.location else None,
            "pixel_location": list(self.pixel_location) if self.pixel_location else None,
            "entity_handle": self.entity_handle,
            "expected_value": self.expected_value,
            "actual_value": self.actual_value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ValidationIssue":
        return cls(
            issue_type=IssueType(data.get("type", "missing_element")),
            description=data.get("description", ""),
            severity=IssueSeverity(data.get("severity", "minor")),
            location=tuple(data["location"]) if data.get("location") else None,
            pixel_location=tuple(data["pixel_location"]) if data.get("pixel_location") else None,
            entity_handle=data.get("entity_handle"),
            expected_value=data.get("expected_value"),
            actual_value=data.get("actual_value"),
        )


@dataclass
class Correction:
    """A correction action to apply."""
    action: CorrectionAction
    entity_type: str  # line, arc, circle, mtext, block
    layer: str = "0"
    properties: Dict[str, Any] = field(default_factory=dict)
    entity_handle: Optional[str] = None  # For REMOVE/MODIFY actions
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "action": self.action.value if isinstance(self.action, CorrectionAction) else str(self.action),
            "entity_type": self.entity_type,
            "layer": self.layer,
            "properties": self.properties,
            "entity_handle": self.entity_handle,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Correction":
        return cls(
            action=CorrectionAction(data.get("action", "ADD")),
            entity_type=data.get("entity_type", ""),
            layer=data.get("layer", "0"),
            properties=data.get("properties", {}),
            entity_handle=data.get("entity_handle"),
            reason=data.get("reason", ""),
        )


@dataclass
class CorrectionResult:
    """Result of applying a single correction."""
    correction: Correction
    success: bool
    handle: Optional[str] = None  # New handle for ADD/REPLACE
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "correction": self.correction.to_dict(),
            "success": self.success,
            "handle": self.handle,
            "error": self.error,
        }


@dataclass
class ValidationResult:
    """Complete validation result."""
    status: ValidationStatus
    accuracy_estimate: float = 0.0  # 0-100%
    issues: List[ValidationIssue] = field(default_factory=list)
    corrections: List[Correction] = field(default_factory=list)
    correction_results: List[CorrectionResult] = field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 3

    # Statistics
    elements_checked: int = 0
    elements_correct: int = 0
    elements_with_issues: int = 0

    # Entity counts from original analysis
    expected_lines: int = 0
    expected_arcs: int = 0
    expected_circles: int = 0
    expected_text: int = 0
    expected_symbols: int = 0

    @property
    def is_approved(self) -> bool:
        return self.status == ValidationStatus.APPROVED

    @property
    def critical_issues(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.CRITICAL]

    @property
    def corrections_applied(self) -> int:
        return sum(1 for c in self.correction_results if c.success)

    def to_dict(self) -> dict:
        return {
            "status": self.status.value if isinstance(self.status, ValidationStatus) else str(self.status),
            "accuracy_estimate": self.accuracy_estimate,
            "is_approved": self.is_approved,
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "statistics": {
                "elements_checked": self.elements_checked,
                "elements_correct": self.elements_correct,
                "elements_with_issues": self.elements_with_issues,
                "expected": {
                    "lines": self.expected_lines,
                    "arcs": self.expected_arcs,
                    "circles": self.expected_circles,
                    "text": self.expected_text,
                    "symbols": self.expected_symbols,
                },
            },
            "issues": [i.to_dict() for i in self.issues],
            "issues_by_severity": {
                "critical": len([i for i in self.issues if i.severity == IssueSeverity.CRITICAL]),
                "major": len([i for i in self.issues if i.severity == IssueSeverity.MAJOR]),
                "minor": len([i for i in self.issues if i.severity == IssueSeverity.MINOR]),
            },
            "corrections": [c.to_dict() for c in self.corrections],
            "corrections_applied": self.corrections_applied,
            "correction_results": [c.to_dict() for c in self.correction_results],
        }


# =============================================================================
# Validation Prompt
# =============================================================================

VALIDATION_PROMPT = """
You are validating an AutoCAD drawing extraction against the original image.

## ORIGINAL DRAWING
[Image attached - this is the source PDF rendered at high quality]

## CREATED ENTITIES
The following entities were extracted and created in AutoCAD:
```json
{entities_json}
```

## CREATION STATISTICS
{stats_summary}

## INSTRUCTIONS

Compare the extracted entities against the original drawing. Check for:

1. **MISSING ELEMENTS**: Elements visible in original but not in entity list
   - Look for lines, arcs, circles, text, or symbols not represented
   - Check edges of drawing, title block boundaries, dimension lines

2. **EXTRA ELEMENTS**: False positives that shouldn't exist
   - Noise interpreted as geometry
   - Duplicate entities at same location

3. **POSITION ERRORS**: Elements in wrong location (>5% deviation from expected)
   - Text placed incorrectly
   - Symbols at wrong coordinates
   - Line endpoints misplaced

4. **TEXT ERRORS**: Wrong content, missing characters, OCR-like errors
   - Room numbers misread
   - Dimension values incorrect
   - Equipment tags garbled

5. **SYMBOL ERRORS**: Wrong type, wrong attributes, wrong orientation
   - Diffuser typed as outlet
   - Valve rotation incorrect
   - Block name mismatch

6. **CONNECTIVITY ISSUES**: Gaps or overlaps in continuous geometry
   - Lines that should connect but don't
   - Duct runs with breaks
   - Pipe connections missing

For each issue, provide a correction action:
- **ADD**: Add missing element (provide full properties)
- **REMOVE**: Remove false positive (provide entity_handle if known)
- **MODIFY**: Change element properties (provide entity_handle and new_properties)
- **REPLACE**: Replace with correct element (provide entity_handle and new entity)

## RESPONSE FORMAT

Return ONLY valid JSON (no markdown, no code blocks):
{{
  "validation_status": "approved" | "issues_found",
  "accuracy_estimate": <0-100 as integer>,
  "summary": "<brief summary of validation>",
  "issues": [
    {{
      "type": "missing_element|extra_element|position_error|text_error|symbol_error|connectivity",
      "description": "<specific description of what's wrong>",
      "severity": "critical|major|minor",
      "location": [x, y] or null,
      "expected_value": "<what it should be>" or null,
      "actual_value": "<what it is>" or null
    }}
  ],
  "corrections": [
    {{
      "action": "ADD|REMOVE|MODIFY|REPLACE",
      "entity_type": "line|arc|circle|mtext|block",
      "layer": "<layer name>",
      "properties": {{
        // For ADD/REPLACE:
        // line: {{"start": [x,y], "end": [x,y], "linetype": "Continuous"}}
        // arc: {{"center": [x,y], "radius": r, "start_angle": a1, "end_angle": a2}}
        // circle: {{"center": [x,y], "radius": r}}
        // mtext: {{"content": "text", "position": [x,y], "height": h}}
        // block: {{"block_name": "name", "position": [x,y], "rotation": r, "scale": s}}
        // For MODIFY: only the properties to change
      }},
      "entity_handle": "<handle>" or null,
      "reason": "<why this correction>"
    }}
  ]
}}

IMPORTANT NOTES:
- If the extraction looks good overall (>90% accurate), return "approved"
- Focus on CRITICAL issues first (missing equipment, wrong text, connectivity breaks)
- Minor issues (slight position drift, optional elements) can be ignored for approval
- All coordinates are in DWG units (typically inches or feet)
- Do NOT suggest corrections you are not confident about
"""


# =============================================================================
# Validation Functions
# =============================================================================

def _encode_image_base64(image_path: Path) -> str:
    """Encode image to base64 for Gemini API."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _format_entities_for_validation(
    creation_result: AutoCADCreationResult,
    extraction_result: Optional[ExtractionResult] = None,
) -> str:
    """Format created entities as JSON for validation prompt."""
    entities_data = {
        "created_entities": {
            "total": creation_result.statistics.total_entities,
            "success_count": creation_result.statistics.success_count,
            "failure_count": creation_result.statistics.failure_count,
            "by_type": {
                "lines": creation_result.statistics.lines_created,
                "arcs": creation_result.statistics.arcs_created,
                "circles": creation_result.statistics.circles_created,
                "texts": creation_result.statistics.texts_created,
                "blocks": creation_result.statistics.blocks_created,
                "polylines": creation_result.statistics.polylines_created,
            },
        },
        "handles": creation_result.created_handles[:100],  # Limit for prompt size
        "failed": creation_result.failed_entities[:20],  # Include some failures
    }

    # Add extraction details if available
    if extraction_result:
        entities_data["extraction_source"] = {
            "primary_strategy": extraction_result.primary_strategy,
            "drawing_type": extraction_result.drawing_type,
            "direct_count": extraction_result.direct_count,
            "guided_count": extraction_result.guided_count,
            "opencv_count": extraction_result.opencv_count,
        }

        # Include sample entities (not all, to keep prompt manageable)
        sample_entities = []
        for entity in extraction_result.entities[:50]:
            sample_entities.append(entity.to_dict())
        entities_data["sample_entities"] = sample_entities

    return json.dumps(entities_data, indent=2)


def _format_stats_summary(
    creation_result: AutoCADCreationResult,
    analysis: Optional[DrawingAnalysis] = None,
) -> str:
    """Format statistics summary for validation prompt."""
    lines = [
        f"Total entities created: {creation_result.statistics.success_count}",
        f"Failed entities: {creation_result.statistics.failure_count}",
        f"Success rate: {creation_result.success_rate:.1%}",
        "",
        "By type:",
        f"  - Lines: {creation_result.statistics.lines_created}",
        f"  - Arcs: {creation_result.statistics.arcs_created}",
        f"  - Circles: {creation_result.statistics.circles_created}",
        f"  - Text: {creation_result.statistics.texts_created}",
        f"  - Blocks: {creation_result.statistics.blocks_created}",
    ]

    if analysis:
        lines.extend([
            "",
            "Original analysis found:",
            f"  - Drawing type: {analysis.drawing_type}",
            f"  - Complexity: {analysis.complexity}",
            f"  - Total elements: {analysis.total_elements}",
        ])

    return "\n".join(lines)


def _parse_validation_response(response_text: str) -> dict:
    """Parse Gemini's validation response, handling various formats."""
    # Try direct JSON parse
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code blocks
    json_patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
        r'\{[\s\S]*\}',
    ]

    for pattern in json_patterns:
        match = re.search(pattern, response_text)
        if match:
            try:
                json_str = match.group(1) if match.lastindex else match.group(0)
                return json.loads(json_str)
            except (json.JSONDecodeError, IndexError):
                continue

    # Return a default structure if parsing fails
    logger.warning("validation_response_parse_failed", response_preview=response_text[:200])
    return {
        "validation_status": "issues_found",
        "accuracy_estimate": 0,
        "summary": "Failed to parse validation response",
        "issues": [{
            "type": "missing_element",
            "description": "Validation response could not be parsed",
            "severity": "major",
        }],
        "corrections": [],
    }


async def validate_with_gemini(
    original_image_path: Path,
    creation_result: AutoCADCreationResult,
    extraction_result: Optional[ExtractionResult] = None,
    analysis: Optional[DrawingAnalysis] = None,
    model: str = "gemini-pro-latest",
) -> dict:
    """
    Use Gemini Vision to validate created entities against original image.

    Args:
        original_image_path: Path to original drawing image
        creation_result: Result from Phase 5 entity creation
        extraction_result: Optional extraction result from Phase 4
        analysis: Optional drawing analysis from Phase 2
        model: Gemini model to use

    Returns:
        Parsed validation response dict
    """
    try:
        import google.generativeai as genai
        from PIL import Image
    except ImportError:
        raise ImportError("google-generativeai and Pillow required for validation")

    settings = get_settings()
    api_key = settings.gemini_api_key

    if not api_key:
        raise ValueError("GEMINI_API_KEY not configured")

    # Configure Gemini
    genai.configure(api_key=api_key)

    # Map model names
    model_mapping = {
        "gemini-pro-latest": "gemini-2.0-flash",
        "gemini-flash-latest": "gemini-2.0-flash",
        "gemini-1.5-pro": "gemini-2.0-flash",
        "gemini-1.5-flash": "gemini-2.0-flash",
    }
    actual_model = model_mapping.get(model, model)

    gemini_model = genai.GenerativeModel(actual_model)

    # Load image
    image = Image.open(original_image_path)

    # Build prompt
    entities_json = _format_entities_for_validation(creation_result, extraction_result)
    stats_summary = _format_stats_summary(creation_result, analysis)

    prompt = VALIDATION_PROMPT.format(
        entities_json=entities_json,
        stats_summary=stats_summary,
    )

    logger.info(
        "validation_gemini_request",
        model=actual_model,
        image_size=(image.width, image.height),
    )

    # Send to Gemini
    response = await gemini_model.generate_content_async(
        [prompt, image],
        generation_config={
            "temperature": 0.1,  # Low for consistent validation
            "max_output_tokens": 8192,
        }
    )

    response_text = response.text
    logger.debug("validation_gemini_response", response_length=len(response_text))

    return _parse_validation_response(response_text)


async def apply_corrections(
    corrections: List[Correction],
    calibration: Optional[ScaleCalibration] = None,
    call_command: Optional[Callable] = None,
) -> List[CorrectionResult]:
    """
    Apply corrections to AutoCAD.

    Args:
        corrections: List of corrections to apply
        calibration: Optional scale calibration for coordinate conversion
        call_command: Async function to call AutoCAD commands

    Returns:
        List of correction results
    """
    if call_command is None:
        from aec_agent.mcp.sidecar_client import call_autocad_command
        call_command = call_autocad_command

    results = []

    for correction in corrections:
        try:
            if correction.action == CorrectionAction.ADD:
                # Create new entity
                entity = EntityToCreate(
                    entity_type=correction.entity_type,
                    layer=correction.layer,
                    properties=correction.properties,
                )
                entity_result = await create_single_entity(entity, call_command)

                results.append(CorrectionResult(
                    correction=correction,
                    success=entity_result.success,
                    handle=entity_result.handle,
                    error=entity_result.error,
                ))

            elif correction.action == CorrectionAction.REMOVE:
                # Delete entity by handle
                if correction.entity_handle:
                    result = await call_command(
                        "delete_entity",
                        {"handle": correction.entity_handle}
                    )
                    results.append(CorrectionResult(
                        correction=correction,
                        success=result.get("success", False),
                        error=result.get("error", {}).get("message") if not result.get("success") else None,
                    ))
                else:
                    results.append(CorrectionResult(
                        correction=correction,
                        success=False,
                        error="No entity_handle provided for REMOVE action",
                    ))

            elif correction.action == CorrectionAction.MODIFY:
                # Modify entity properties
                if correction.entity_handle:
                    result = await call_command(
                        "modify_entity",
                        {
                            "handle": correction.entity_handle,
                            "properties": correction.properties,
                        }
                    )
                    results.append(CorrectionResult(
                        correction=correction,
                        success=result.get("success", False),
                        error=result.get("error", {}).get("message") if not result.get("success") else None,
                    ))
                else:
                    results.append(CorrectionResult(
                        correction=correction,
                        success=False,
                        error="No entity_handle provided for MODIFY action",
                    ))

            elif correction.action == CorrectionAction.REPLACE:
                # Delete old and create new
                if correction.entity_handle:
                    await call_command(
                        "delete_entity",
                        {"handle": correction.entity_handle}
                    )

                # Create replacement
                entity = EntityToCreate(
                    entity_type=correction.entity_type,
                    layer=correction.layer,
                    properties=correction.properties,
                )
                entity_result = await create_single_entity(entity, call_command)

                results.append(CorrectionResult(
                    correction=correction,
                    success=entity_result.success,
                    handle=entity_result.handle,
                    error=entity_result.error,
                ))

        except Exception as e:
            logger.warning(
                "correction_failed",
                action=correction.action.value,
                error=str(e),
            )
            results.append(CorrectionResult(
                correction=correction,
                success=False,
                error=str(e),
            ))

    return results


async def validate_extraction(
    original_image_path: Path,
    creation_result: AutoCADCreationResult,
    extraction_result: Optional[ExtractionResult] = None,
    analysis: Optional[DrawingAnalysis] = None,
    calibration: Optional[ScaleCalibration] = None,
    max_iterations: int = 3,
    apply_corrections_enabled: bool = True,
    model: str = "gemini-pro-latest",
    call_command: Optional[Callable] = None,
) -> ValidationResult:
    """
    Validate extracted entities against original drawing and optionally self-correct.

    This is the main entry point for Phase 6.

    Args:
        original_image_path: Path to original drawing image from Phase 1
        creation_result: AutoCADCreationResult from Phase 5
        extraction_result: Optional ExtractionResult from Phase 4
        analysis: Optional DrawingAnalysis from Phase 2
        calibration: Optional ScaleCalibration from Phase 3
        max_iterations: Maximum validation/correction iterations
        apply_corrections_enabled: Whether to auto-apply corrections
        model: Gemini model to use for validation
        call_command: Optional async function to call AutoCAD commands

    Returns:
        ValidationResult with status, issues, and correction results

    Example:
        >>> result = await validate_extraction(
        ...     original_image_path=render_result.image_path,
        ...     creation_result=creation_result,
        ...     analysis=analysis,
        ...     calibration=calibration,
        ... )
        >>> if result.is_approved:
        ...     print("Extraction validated successfully!")
        >>> else:
        ...     print(f"Issues found: {len(result.issues)}")
    """
    logger.info(
        "validation_starting",
        image_path=str(original_image_path),
        entities_created=creation_result.statistics.success_count,
        max_iterations=max_iterations,
    )

    iteration = 0
    last_validation: Optional[dict] = None

    while iteration < max_iterations:
        iteration += 1

        logger.info("validation_iteration", iteration=iteration, max=max_iterations)

        # Call Gemini for validation
        try:
            validation_response = await validate_with_gemini(
                original_image_path=original_image_path,
                creation_result=creation_result,
                extraction_result=extraction_result,
                analysis=analysis,
                model=model,
            )
            last_validation = validation_response
        except Exception as e:
            logger.exception("validation_gemini_failed", error=str(e))
            return ValidationResult(
                status=ValidationStatus.ERROR,
                accuracy_estimate=0,
                iteration=iteration,
                max_iterations=max_iterations,
                issues=[ValidationIssue(
                    issue_type=IssueType.MISSING_ELEMENT,
                    description=f"Validation failed: {e}",
                    severity=IssueSeverity.CRITICAL,
                )],
            )

        # Check if approved
        if validation_response.get("validation_status") == "approved":
            logger.info(
                "validation_approved",
                iteration=iteration,
                accuracy=validation_response.get("accuracy_estimate", 100),
            )
            return ValidationResult(
                status=ValidationStatus.APPROVED,
                accuracy_estimate=validation_response.get("accuracy_estimate", 100),
                iteration=iteration,
                max_iterations=max_iterations,
                elements_checked=creation_result.statistics.total_entities,
                elements_correct=creation_result.statistics.success_count,
            )

        # Parse issues
        issues = []
        for issue_data in validation_response.get("issues", []):
            try:
                issues.append(ValidationIssue.from_dict(issue_data))
            except Exception as e:
                logger.warning("issue_parse_failed", error=str(e), data=issue_data)

        # Parse corrections
        corrections = []
        for corr_data in validation_response.get("corrections", []):
            try:
                corrections.append(Correction.from_dict(corr_data))
            except Exception as e:
                logger.warning("correction_parse_failed", error=str(e), data=corr_data)

        logger.info(
            "validation_issues_found",
            iteration=iteration,
            accuracy=validation_response.get("accuracy_estimate", 0),
            issues=len(issues),
            corrections=len(corrections),
        )

        # If no corrections available, return with issues
        if not corrections:
            return ValidationResult(
                status=ValidationStatus.MANUAL_REVIEW,
                accuracy_estimate=validation_response.get("accuracy_estimate", 0),
                issues=issues,
                corrections=[],
                iteration=iteration,
                max_iterations=max_iterations,
                elements_checked=creation_result.statistics.total_entities,
                elements_with_issues=len(issues),
            )

        # Apply corrections if enabled
        if apply_corrections_enabled and corrections:
            correction_results = await apply_corrections(
                corrections,
                calibration=calibration,
                call_command=call_command,
            )

            successful_corrections = sum(1 for r in correction_results if r.success)
            logger.info(
                "corrections_applied",
                total=len(corrections),
                successful=successful_corrections,
            )

            # If no corrections succeeded, stop iterating
            if successful_corrections == 0:
                return ValidationResult(
                    status=ValidationStatus.ISSUES_FOUND,
                    accuracy_estimate=validation_response.get("accuracy_estimate", 0),
                    issues=issues,
                    corrections=corrections,
                    correction_results=correction_results,
                    iteration=iteration,
                    max_iterations=max_iterations,
                    elements_checked=creation_result.statistics.total_entities,
                    elements_with_issues=len(issues),
                )

            # Continue to next iteration to re-validate
            continue
        else:
            # Corrections available but not enabled
            return ValidationResult(
                status=ValidationStatus.ISSUES_FOUND,
                accuracy_estimate=validation_response.get("accuracy_estimate", 0),
                issues=issues,
                corrections=corrections,
                iteration=iteration,
                max_iterations=max_iterations,
                elements_checked=creation_result.statistics.total_entities,
                elements_with_issues=len(issues),
            )

    # Max iterations reached
    logger.warning("validation_max_iterations", iterations=max_iterations)

    accuracy = last_validation.get("accuracy_estimate", 0) if last_validation else 0
    issues = []
    for issue_data in (last_validation.get("issues", []) if last_validation else []):
        try:
            issues.append(ValidationIssue.from_dict(issue_data))
        except Exception:
            pass

    return ValidationResult(
        status=ValidationStatus.MAX_ITERATIONS,
        accuracy_estimate=accuracy,
        issues=issues,
        iteration=max_iterations,
        max_iterations=max_iterations,
        elements_checked=creation_result.statistics.total_entities,
        elements_with_issues=len(issues),
    )


# =============================================================================
# Helper Functions
# =============================================================================

def get_critical_issues(result: ValidationResult) -> List[ValidationIssue]:
    """Get only critical issues from validation result."""
    return [i for i in result.issues if i.severity == IssueSeverity.CRITICAL]


def get_issues_by_type(result: ValidationResult) -> Dict[str, List[ValidationIssue]]:
    """Group issues by type."""
    grouped: Dict[str, List[ValidationIssue]] = {}
    for issue in result.issues:
        key = issue.issue_type.value if isinstance(issue.issue_type, IssueType) else str(issue.issue_type)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(issue)
    return grouped


def summarize_validation(result: ValidationResult) -> str:
    """Generate a human-readable summary of validation result."""
    lines = [
        f"Validation Status: {result.status.value}",
        f"Accuracy Estimate: {result.accuracy_estimate:.0f}%",
        f"Iterations: {result.iteration}/{result.max_iterations}",
        "",
    ]

    if result.is_approved:
        lines.append("Result: APPROVED - Extraction validated successfully!")
    else:
        lines.append(f"Issues Found: {len(result.issues)}")

        by_severity = {
            "critical": len([i for i in result.issues if i.severity == IssueSeverity.CRITICAL]),
            "major": len([i for i in result.issues if i.severity == IssueSeverity.MAJOR]),
            "minor": len([i for i in result.issues if i.severity == IssueSeverity.MINOR]),
        }
        lines.append(f"  - Critical: {by_severity['critical']}")
        lines.append(f"  - Major: {by_severity['major']}")
        lines.append(f"  - Minor: {by_severity['minor']}")

        if result.corrections:
            lines.append(f"\nCorrections Suggested: {len(result.corrections)}")
            lines.append(f"Corrections Applied: {result.corrections_applied}")

    return "\n".join(lines)
