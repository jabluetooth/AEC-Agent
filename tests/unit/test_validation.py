"""
Unit tests for Phase 6: Validation & Self-Correction.

Tests the gemini_first.validation module which validates created entities
against the original drawing and applies corrections.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

import pytest

# Import module under test
from aec_agent.mcp.tools.gemini_first.validation import (
    ValidationStatus,
    IssueType,
    IssueSeverity,
    CorrectionAction,
    ValidationIssue,
    Correction,
    CorrectionResult,
    ValidationResult,
    validate_extraction,
    apply_corrections,
    get_critical_issues,
    get_issues_by_type,
    summarize_validation,
    _format_entities_for_validation,
    _format_stats_summary,
    _parse_validation_response,
)
from aec_agent.mcp.tools.gemini_first.autocad_creation import (
    AutoCADCreationResult,
    CreationStatistics,
)


class TestValidationStatus:
    """Tests for ValidationStatus enum."""

    def test_all_statuses(self):
        """Test all validation statuses exist."""
        assert ValidationStatus.APPROVED == "approved"
        assert ValidationStatus.ISSUES_FOUND == "issues_found"
        assert ValidationStatus.MANUAL_REVIEW == "manual_review"
        assert ValidationStatus.MAX_ITERATIONS == "max_iterations"
        assert ValidationStatus.ERROR == "error"


class TestIssueType:
    """Tests for IssueType enum."""

    def test_all_issue_types(self):
        """Test all issue types exist."""
        expected = [
            "missing_element",
            "extra_element",
            "position_error",
            "text_error",
            "symbol_error",
            "connectivity",
            "layer_error",
            "attribute_error",
        ]
        for issue_type in expected:
            assert IssueType(issue_type).value == issue_type


class TestIssueSeverity:
    """Tests for IssueSeverity enum."""

    def test_all_severities(self):
        """Test all severities exist."""
        assert IssueSeverity.CRITICAL == "critical"
        assert IssueSeverity.MAJOR == "major"
        assert IssueSeverity.MINOR == "minor"


class TestCorrectionAction:
    """Tests for CorrectionAction enum."""

    def test_all_actions(self):
        """Test all correction actions exist."""
        assert CorrectionAction.ADD == "ADD"
        assert CorrectionAction.REMOVE == "REMOVE"
        assert CorrectionAction.MODIFY == "MODIFY"
        assert CorrectionAction.REPLACE == "REPLACE"


class TestValidationIssue:
    """Tests for ValidationIssue dataclass."""

    def test_basic_creation(self):
        """Test creating a ValidationIssue."""
        issue = ValidationIssue(
            issue_type=IssueType.MISSING_ELEMENT,
            description="Missing duct line",
            severity=IssueSeverity.CRITICAL,
            location=(100.0, 200.0),
        )

        assert issue.issue_type == IssueType.MISSING_ELEMENT
        assert issue.description == "Missing duct line"
        assert issue.severity == IssueSeverity.CRITICAL
        assert issue.location == (100.0, 200.0)

    def test_to_dict(self):
        """Test serialization to dictionary."""
        issue = ValidationIssue(
            issue_type=IssueType.TEXT_ERROR,
            description="Room number misread",
            severity=IssueSeverity.MAJOR,
            expected_value="101",
            actual_value="10I",
        )

        d = issue.to_dict()
        assert d["type"] == "text_error"
        assert d["severity"] == "major"
        assert d["expected_value"] == "101"
        assert d["actual_value"] == "10I"

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "type": "position_error",
            "description": "Symbol misplaced",
            "severity": "minor",
            "location": [150, 250],
        }

        issue = ValidationIssue.from_dict(data)
        assert issue.issue_type == IssueType.POSITION_ERROR
        assert issue.severity == IssueSeverity.MINOR
        assert issue.location == (150, 250)


class TestCorrection:
    """Tests for Correction dataclass."""

    def test_add_correction(self):
        """Test creating an ADD correction."""
        correction = Correction(
            action=CorrectionAction.ADD,
            entity_type="line",
            layer="M-DUCT",
            properties={
                "start": [0, 0],
                "end": [100, 0],
                "linetype": "Continuous",
            },
            reason="Missing duct run",
        )

        assert correction.action == CorrectionAction.ADD
        assert correction.entity_type == "line"
        assert correction.layer == "M-DUCT"

    def test_remove_correction(self):
        """Test creating a REMOVE correction."""
        correction = Correction(
            action=CorrectionAction.REMOVE,
            entity_type="line",
            entity_handle="1A3",
            reason="False positive noise",
        )

        assert correction.action == CorrectionAction.REMOVE
        assert correction.entity_handle == "1A3"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        correction = Correction(
            action=CorrectionAction.MODIFY,
            entity_type="mtext",
            entity_handle="2B4",
            properties={"content": "ROOM 101"},
            reason="Fix OCR error",
        )

        d = correction.to_dict()
        assert d["action"] == "MODIFY"
        assert d["entity_handle"] == "2B4"
        assert d["properties"]["content"] == "ROOM 101"

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "action": "REPLACE",
            "entity_type": "block",
            "layer": "M-DIFF",
            "properties": {
                "block_name": "M-DIFF-SQ",
                "position": [200, 300],
            },
            "entity_handle": "3C5",
            "reason": "Wrong diffuser type",
        }

        correction = Correction.from_dict(data)
        assert correction.action == CorrectionAction.REPLACE
        assert correction.entity_type == "block"
        assert correction.entity_handle == "3C5"


class TestCorrectionResult:
    """Tests for CorrectionResult dataclass."""

    def test_successful_result(self):
        """Test successful correction result."""
        correction = Correction(
            action=CorrectionAction.ADD,
            entity_type="line",
            layer="A-WALL",
            properties={"start": [0, 0], "end": [100, 0]},
        )
        result = CorrectionResult(
            correction=correction,
            success=True,
            handle="4D6",
        )

        assert result.success is True
        assert result.handle == "4D6"
        assert result.error is None

    def test_failed_result(self):
        """Test failed correction result."""
        correction = Correction(
            action=CorrectionAction.REMOVE,
            entity_type="line",
            entity_handle="INVALID",
        )
        result = CorrectionResult(
            correction=correction,
            success=False,
            error="Entity not found",
        )

        assert result.success is False
        assert result.error == "Entity not found"


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_approved_result(self):
        """Test approved validation result."""
        result = ValidationResult(
            status=ValidationStatus.APPROVED,
            accuracy_estimate=98.5,
            iteration=1,
            max_iterations=3,
            elements_checked=100,
            elements_correct=100,
        )

        assert result.is_approved is True
        assert result.accuracy_estimate == 98.5
        assert result.critical_issues == []

    def test_issues_found_result(self):
        """Test result with issues."""
        issues = [
            ValidationIssue(
                issue_type=IssueType.MISSING_ELEMENT,
                description="Missing line",
                severity=IssueSeverity.CRITICAL,
            ),
            ValidationIssue(
                issue_type=IssueType.TEXT_ERROR,
                description="Typo",
                severity=IssueSeverity.MINOR,
            ),
        ]

        result = ValidationResult(
            status=ValidationStatus.ISSUES_FOUND,
            accuracy_estimate=85,
            issues=issues,
            iteration=2,
            max_iterations=3,
        )

        assert result.is_approved is False
        assert len(result.critical_issues) == 1
        assert result.critical_issues[0].issue_type == IssueType.MISSING_ELEMENT

    def test_corrections_applied_count(self):
        """Test counting applied corrections."""
        corrections = [
            Correction(action=CorrectionAction.ADD, entity_type="line", layer="0"),
            Correction(action=CorrectionAction.ADD, entity_type="circle", layer="0"),
        ]
        correction_results = [
            CorrectionResult(correction=corrections[0], success=True, handle="1A"),
            CorrectionResult(correction=corrections[1], success=False, error="Failed"),
        ]

        result = ValidationResult(
            status=ValidationStatus.ISSUES_FOUND,
            corrections=corrections,
            correction_results=correction_results,
        )

        assert result.corrections_applied == 1

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = ValidationResult(
            status=ValidationStatus.APPROVED,
            accuracy_estimate=95,
            iteration=1,
            max_iterations=3,
            elements_checked=50,
            elements_correct=48,
            elements_with_issues=2,
        )

        d = result.to_dict()
        assert d["status"] == "approved"
        assert d["accuracy_estimate"] == 95
        assert d["is_approved"] is True
        assert d["statistics"]["elements_checked"] == 50
        assert d["issues_by_severity"]["critical"] == 0


class TestFormatEntitiesForValidation:
    """Tests for _format_entities_for_validation helper."""

    def test_format_basic(self):
        """Test formatting basic creation result."""
        creation_result = AutoCADCreationResult(
            success=True,
            statistics=CreationStatistics(
                total_entities=100,
                success_count=95,
                failure_count=5,
                lines_created=50,
                circles_created=20,
            ),
            created_handles=["1A", "1B", "1C"],
        )

        json_str = _format_entities_for_validation(creation_result)

        import json
        data = json.loads(json_str)

        assert data["created_entities"]["total"] == 100
        assert data["created_entities"]["success_count"] == 95
        assert "1A" in data["handles"]


class TestFormatStatsSummary:
    """Tests for _format_stats_summary helper."""

    def test_format_basic(self):
        """Test formatting basic stats."""
        creation_result = AutoCADCreationResult(
            success=True,
            statistics=CreationStatistics(
                total_entities=50,
                success_count=48,
                failure_count=2,
                lines_created=30,
                circles_created=10,
            ),
        )

        summary = _format_stats_summary(creation_result)

        assert "Total entities created: 48" in summary
        assert "Lines: 30" in summary


class TestParseValidationResponse:
    """Tests for _parse_validation_response helper."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON response."""
        response = '{"validation_status": "approved", "accuracy_estimate": 95}'

        result = _parse_validation_response(response)

        assert result["validation_status"] == "approved"
        assert result["accuracy_estimate"] == 95

    def test_parse_json_in_code_block(self):
        """Test parsing JSON in markdown code block."""
        response = '''
Here is the validation result:
```json
{"validation_status": "issues_found", "accuracy_estimate": 80}
```
        '''

        result = _parse_validation_response(response)

        assert result["validation_status"] == "issues_found"

    def test_parse_invalid_response(self):
        """Test handling invalid response."""
        response = "This is not JSON at all!"

        result = _parse_validation_response(response)

        # Should return a default structure
        assert result["validation_status"] == "issues_found"
        assert result["accuracy_estimate"] == 0


class TestApplyCorrections:
    """Tests for apply_corrections function."""

    @pytest.mark.asyncio
    async def test_apply_add_correction(self):
        """Test applying ADD correction."""
        corrections = [
            Correction(
                action=CorrectionAction.ADD,
                entity_type="line",
                layer="A-WALL",
                properties={"start": [0, 0], "end": [100, 0]},
            ),
        ]

        mock_command = AsyncMock(return_value={"success": True, "data": {"handle": "NEW1"}})

        with patch(
            "aec_agent.mcp.tools.gemini_first.validation.create_single_entity"
        ) as mock_create:
            mock_create.return_value = MagicMock(
                success=True,
                handle="NEW1",
                error=None,
            )

            results = await apply_corrections(corrections, call_command=mock_command)

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].handle == "NEW1"

    @pytest.mark.asyncio
    async def test_apply_remove_correction(self):
        """Test applying REMOVE correction."""
        corrections = [
            Correction(
                action=CorrectionAction.REMOVE,
                entity_type="line",
                entity_handle="DELETE_ME",
            ),
        ]

        mock_command = AsyncMock(return_value={"success": True})

        results = await apply_corrections(corrections, call_command=mock_command)

        assert len(results) == 1
        assert results[0].success is True
        mock_command.assert_called_once_with(
            "delete_entity",
            {"handle": "DELETE_ME"}
        )

    @pytest.mark.asyncio
    async def test_apply_remove_without_handle(self):
        """Test REMOVE correction without handle fails."""
        corrections = [
            Correction(
                action=CorrectionAction.REMOVE,
                entity_type="line",
                # Missing entity_handle
            ),
        ]

        mock_command = AsyncMock()

        results = await apply_corrections(corrections, call_command=mock_command)

        assert len(results) == 1
        assert results[0].success is False
        assert "No entity_handle" in results[0].error

    @pytest.mark.asyncio
    async def test_apply_modify_correction(self):
        """Test applying MODIFY correction."""
        corrections = [
            Correction(
                action=CorrectionAction.MODIFY,
                entity_type="mtext",
                entity_handle="MODIFY_ME",
                properties={"content": "New Text"},
            ),
        ]

        mock_command = AsyncMock(return_value={"success": True})

        results = await apply_corrections(corrections, call_command=mock_command)

        assert len(results) == 1
        assert results[0].success is True
        mock_command.assert_called_once()

    @pytest.mark.asyncio
    async def test_apply_replace_correction(self):
        """Test applying REPLACE correction."""
        corrections = [
            Correction(
                action=CorrectionAction.REPLACE,
                entity_type="block",
                layer="M-DIFF",
                entity_handle="OLD_BLOCK",
                properties={"block_name": "M-DIFF-SQ", "position": [100, 200]},
            ),
        ]

        mock_command = AsyncMock(return_value={"success": True})

        with patch(
            "aec_agent.mcp.tools.gemini_first.validation.create_single_entity"
        ) as mock_create:
            mock_create.return_value = MagicMock(
                success=True,
                handle="NEW_BLOCK",
                error=None,
            )

            results = await apply_corrections(corrections, call_command=mock_command)

        assert len(results) == 1
        assert results[0].success is True

        # Should have called delete for old entity
        mock_command.assert_called_once_with(
            "delete_entity",
            {"handle": "OLD_BLOCK"}
        )


class TestValidateExtraction:
    """Tests for validate_extraction main function."""

    @pytest.mark.asyncio
    async def test_validate_approved(self):
        """Test validation that returns approved."""
        creation_result = AutoCADCreationResult(
            success=True,
            statistics=CreationStatistics(
                total_entities=50,
                success_count=50,
            ),
            created_handles=["1A", "1B"],
        )

        with patch(
            "aec_agent.mcp.tools.gemini_first.validation.validate_with_gemini"
        ) as mock_gemini:
            mock_gemini.return_value = {
                "validation_status": "approved",
                "accuracy_estimate": 98,
                "issues": [],
                "corrections": [],
            }

            result = await validate_extraction(
                original_image_path=Path("test.png"),
                creation_result=creation_result,
                max_iterations=3,
                apply_corrections_enabled=False,
            )

        assert result.status == ValidationStatus.APPROVED
        assert result.accuracy_estimate == 98
        assert result.iteration == 1

    @pytest.mark.asyncio
    async def test_validate_issues_no_corrections(self):
        """Test validation with issues but no corrections."""
        creation_result = AutoCADCreationResult(
            success=True,
            statistics=CreationStatistics(total_entities=50, success_count=45),
        )

        with patch(
            "aec_agent.mcp.tools.gemini_first.validation.validate_with_gemini"
        ) as mock_gemini:
            mock_gemini.return_value = {
                "validation_status": "issues_found",
                "accuracy_estimate": 80,
                "issues": [
                    {
                        "type": "missing_element",
                        "description": "Missing line",
                        "severity": "major",
                    }
                ],
                "corrections": [],  # No corrections
            }

            result = await validate_extraction(
                original_image_path=Path("test.png"),
                creation_result=creation_result,
            )

        assert result.status == ValidationStatus.MANUAL_REVIEW
        assert len(result.issues) == 1

    @pytest.mark.asyncio
    async def test_validate_max_iterations(self):
        """Test validation hitting max iterations."""
        creation_result = AutoCADCreationResult(
            success=True,
            statistics=CreationStatistics(total_entities=50, success_count=40),
        )

        with patch(
            "aec_agent.mcp.tools.gemini_first.validation.validate_with_gemini"
        ) as mock_gemini, patch(
            "aec_agent.mcp.tools.gemini_first.validation.apply_corrections"
        ) as mock_apply:
            # Always return issues with corrections
            mock_gemini.return_value = {
                "validation_status": "issues_found",
                "accuracy_estimate": 85,
                "issues": [{"type": "missing_element", "description": "Missing", "severity": "minor"}],
                "corrections": [{"action": "ADD", "entity_type": "line", "layer": "0", "properties": {}}],
            }
            # Corrections always succeed
            mock_apply.return_value = [
                CorrectionResult(
                    correction=Correction(action=CorrectionAction.ADD, entity_type="line", layer="0"),
                    success=True,
                    handle="NEW",
                )
            ]

            result = await validate_extraction(
                original_image_path=Path("test.png"),
                creation_result=creation_result,
                max_iterations=2,  # Low limit
            )

        assert result.status == ValidationStatus.MAX_ITERATIONS
        assert result.iteration == 2

    @pytest.mark.asyncio
    async def test_validate_error_handling(self):
        """Test validation error handling."""
        creation_result = AutoCADCreationResult(
            success=True,
            statistics=CreationStatistics(total_entities=50, success_count=50),
        )

        with patch(
            "aec_agent.mcp.tools.gemini_first.validation.validate_with_gemini"
        ) as mock_gemini:
            mock_gemini.side_effect = Exception("Gemini API error")

            result = await validate_extraction(
                original_image_path=Path("test.png"),
                creation_result=creation_result,
            )

        assert result.status == ValidationStatus.ERROR
        assert len(result.issues) == 1
        assert "Gemini API error" in result.issues[0].description


class TestGetCriticalIssues:
    """Tests for get_critical_issues helper."""

    def test_filter_critical(self):
        """Test filtering critical issues."""
        result = ValidationResult(
            status=ValidationStatus.ISSUES_FOUND,
            issues=[
                ValidationIssue(
                    issue_type=IssueType.MISSING_ELEMENT,
                    description="Critical missing",
                    severity=IssueSeverity.CRITICAL,
                ),
                ValidationIssue(
                    issue_type=IssueType.TEXT_ERROR,
                    description="Minor typo",
                    severity=IssueSeverity.MINOR,
                ),
                ValidationIssue(
                    issue_type=IssueType.CONNECTIVITY,
                    description="Critical break",
                    severity=IssueSeverity.CRITICAL,
                ),
            ],
        )

        critical = get_critical_issues(result)

        assert len(critical) == 2
        assert all(i.severity == IssueSeverity.CRITICAL for i in critical)


class TestGetIssuesByType:
    """Tests for get_issues_by_type helper."""

    def test_group_by_type(self):
        """Test grouping issues by type."""
        result = ValidationResult(
            status=ValidationStatus.ISSUES_FOUND,
            issues=[
                ValidationIssue(
                    issue_type=IssueType.MISSING_ELEMENT,
                    description="Missing 1",
                    severity=IssueSeverity.MAJOR,
                ),
                ValidationIssue(
                    issue_type=IssueType.MISSING_ELEMENT,
                    description="Missing 2",
                    severity=IssueSeverity.MAJOR,
                ),
                ValidationIssue(
                    issue_type=IssueType.TEXT_ERROR,
                    description="Text error",
                    severity=IssueSeverity.MINOR,
                ),
            ],
        )

        grouped = get_issues_by_type(result)

        assert len(grouped["missing_element"]) == 2
        assert len(grouped["text_error"]) == 1


class TestSummarizeValidation:
    """Tests for summarize_validation helper."""

    def test_summarize_approved(self):
        """Test summarizing approved validation."""
        result = ValidationResult(
            status=ValidationStatus.APPROVED,
            accuracy_estimate=98,
            iteration=1,
            max_iterations=3,
        )

        summary = summarize_validation(result)

        assert "APPROVED" in summary
        assert "98%" in summary

    def test_summarize_issues(self):
        """Test summarizing validation with issues."""
        result = ValidationResult(
            status=ValidationStatus.ISSUES_FOUND,
            accuracy_estimate=75,
            iteration=2,
            max_iterations=3,
            issues=[
                ValidationIssue(
                    issue_type=IssueType.MISSING_ELEMENT,
                    description="Missing",
                    severity=IssueSeverity.CRITICAL,
                ),
                ValidationIssue(
                    issue_type=IssueType.TEXT_ERROR,
                    description="Typo",
                    severity=IssueSeverity.MINOR,
                ),
            ],
            corrections=[
                Correction(action=CorrectionAction.ADD, entity_type="line", layer="0"),
            ],
        )

        summary = summarize_validation(result)

        assert "Issues Found: 2" in summary
        assert "Critical: 1" in summary
        assert "Minor: 1" in summary
        assert "Corrections Suggested: 1" in summary
