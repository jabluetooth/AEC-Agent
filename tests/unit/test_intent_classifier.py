"""
Unit tests for the intent classification module.
"""

import pytest

from aec_agent.intent.models import (
    IntentResult,
    MEPDomain,
    MEPAction,
    AppContext,
    PatternMatch,
)
from aec_agent.intent.classifier import IntentClassifier, reset_intent_classifier
from aec_agent.intent.patterns import (
    MEP_PATTERNS,
    ACTION_PATTERNS,
    APP_CONTEXT_PATTERNS,
    get_all_patterns_flat,
)


class TestMEPDomain:
    """Tests for MEPDomain enum."""

    def test_from_string_valid(self):
        assert MEPDomain.from_string("hvac") == MEPDomain.HVAC
        assert MEPDomain.from_string("HVAC") == MEPDomain.HVAC
        assert MEPDomain.from_string("electrical") == MEPDomain.ELECTRICAL
        assert MEPDomain.from_string("plumbing") == MEPDomain.PLUMBING
        assert MEPDomain.from_string("fire_protection") == MEPDomain.FIRE_PROTECTION

    def test_from_string_invalid(self):
        assert MEPDomain.from_string("invalid") == MEPDomain.GENERAL
        assert MEPDomain.from_string("") == MEPDomain.GENERAL


class TestMEPAction:
    """Tests for MEPAction enum."""

    def test_from_string_valid(self):
        assert MEPAction.from_string("query") == MEPAction.QUERY
        assert MEPAction.from_string("create") == MEPAction.CREATE
        assert MEPAction.from_string("route") == MEPAction.ROUTE

    def test_from_string_invalid(self):
        assert MEPAction.from_string("invalid") == MEPAction.QUERY


class TestAppContext:
    """Tests for AppContext enum."""

    def test_get_tool_prefix(self):
        assert AppContext.AUTOCAD.get_tool_prefix() == "autocad_"
        assert AppContext.REVIT.get_tool_prefix() == "revit_"
        assert AppContext.BOTH.get_tool_prefix() is None


class TestIntentResult:
    """Tests for IntentResult dataclass."""

    def test_default_values(self):
        result = IntentResult()
        assert result.domain == MEPDomain.GENERAL
        assert result.subdomain is None
        assert result.action == MEPAction.QUERY
        assert result.app_context == AppContext.BOTH
        assert result.confidence == 0.0
        assert result.suggested_tools == []
        assert result.keywords_matched == []

    def test_is_mep_specific(self):
        # Not MEP specific - general domain
        result = IntentResult(domain=MEPDomain.GENERAL, confidence=0.8)
        assert not result.is_mep_specific

        # Not MEP specific - low confidence
        result = IntentResult(domain=MEPDomain.HVAC, confidence=0.3)
        assert not result.is_mep_specific

        # MEP specific - HVAC with high confidence
        result = IntentResult(domain=MEPDomain.HVAC, confidence=0.6)
        assert result.is_mep_specific

    def test_tool_filter_prefix(self):
        result = IntentResult(app_context=AppContext.AUTOCAD)
        assert result.tool_filter_prefix == "autocad_"

        result = IntentResult(app_context=AppContext.BOTH)
        assert result.tool_filter_prefix is None

    def test_get_tool_tier(self):
        # Analysis actions need advanced tier
        result = IntentResult(action=MEPAction.ANALYZE)
        assert result.get_tool_tier() == "advanced"

        result = IntentResult(action=MEPAction.ROUTE)
        assert result.get_tool_tier() == "advanced"

        # Schedule needs standard tier
        result = IntentResult(action=MEPAction.SCHEDULE)
        assert result.get_tool_tier() == "standard"

        # Query/create can use essential tier
        result = IntentResult(action=MEPAction.QUERY)
        assert result.get_tool_tier() == "essential"

    def test_to_dict(self):
        result = IntentResult(
            domain=MEPDomain.HVAC,
            subdomain="distribution",
            action=MEPAction.ROUTE,
            confidence=0.8,
        )
        d = result.to_dict()
        assert d["domain"] == "hvac"
        assert d["subdomain"] == "distribution"
        assert d["action"] == "route"
        assert d["confidence"] == 0.8
        assert d["tool_tier"] == "advanced"


class TestPatternMatch:
    """Tests for PatternMatch dataclass."""

    def test_score_calculation(self):
        match = PatternMatch(
            pattern="ahu",
            domain=MEPDomain.HVAC,
            subdomain="equipment",
            weight=3.0,
            similarity=0.5,
        )
        # score = weight * (1.0 + similarity) = 3.0 * 1.5 = 4.5
        assert match.score == 4.5


class TestMEPPatterns:
    """Tests for MEP pattern definitions."""

    def test_hvac_patterns_exist(self):
        assert MEPDomain.HVAC in MEP_PATTERNS
        hvac = MEP_PATTERNS[MEPDomain.HVAC]
        assert "ahu" in hvac
        assert "ductwork" in hvac
        assert "diffuser" in hvac
        assert "cfm" in hvac

    def test_pattern_weights_are_valid(self):
        for domain, patterns in MEP_PATTERNS.items():
            for pattern, (weight, subdomain) in patterns.items():
                assert 1.0 <= weight <= 3.0, f"Invalid weight for {pattern}"
                assert subdomain is None or isinstance(subdomain, str)

    def test_get_all_patterns_flat(self):
        patterns = get_all_patterns_flat()
        assert len(patterns) > 50  # Should have many patterns
        # Check structure
        for pattern, domain, subdomain, weight in patterns:
            assert isinstance(pattern, str)
            assert isinstance(domain, MEPDomain)
            assert subdomain is None or isinstance(subdomain, str)
            assert isinstance(weight, float)


class TestActionPatterns:
    """Tests for action pattern definitions."""

    def test_all_actions_have_patterns(self):
        for action in MEPAction:
            if action != MEPAction.COORDINATE:  # coordinate may not have explicit patterns
                assert action.value in ACTION_PATTERNS or action == MEPAction.COORDINATE

    def test_query_patterns(self):
        assert "find" in ACTION_PATTERNS["query"]
        assert "list" in ACTION_PATTERNS["query"]
        assert "search" in ACTION_PATTERNS["query"]

    def test_create_patterns(self):
        assert "draw" in ACTION_PATTERNS["create"]
        assert "add" in ACTION_PATTERNS["create"]
        assert "create" in ACTION_PATTERNS["create"]


class TestIntentClassifier:
    """Tests for the IntentClassifier class."""

    @pytest.fixture(autouse=True)
    def reset_classifier(self):
        """Reset global classifier before each test."""
        reset_intent_classifier()
        yield
        reset_intent_classifier()

    @pytest.fixture
    def classifier(self):
        """Create a classifier without embeddings for testing."""
        return IntentClassifier(use_embeddings=False)

    def test_classify_empty_input(self, classifier):
        result = classifier.classify("")
        assert result.domain == MEPDomain.GENERAL
        assert result.confidence == 0.0

    def test_classify_hvac_equipment(self, classifier):
        result = classifier.classify("Find all AHU units on level 2")
        assert result.domain == MEPDomain.HVAC
        assert result.subdomain == "equipment"
        assert "ahu" in [k.lower() for k in result.keywords_matched]

    def test_classify_hvac_distribution(self, classifier):
        result = classifier.classify("Route the supply ductwork from AHU-1 to the VAV boxes")
        assert result.domain == MEPDomain.HVAC
        assert result.action == MEPAction.ROUTE
        # Should match multiple HVAC terms
        assert len(result.keywords_matched) >= 2

    def test_classify_hvac_parameters(self, classifier):
        result = classifier.classify("What is the CFM for diffuser D-101?")
        assert result.domain == MEPDomain.HVAC
        assert "cfm" in [k.lower() for k in result.keywords_matched]

    def test_classify_electrical(self, classifier):
        result = classifier.classify("Show me the electrical panels on floor 3")
        assert result.domain == MEPDomain.ELECTRICAL

    def test_classify_plumbing(self, classifier):
        result = classifier.classify("List all plumbing fixtures in the restrooms")
        assert result.domain == MEPDomain.PLUMBING

    def test_classify_fire_protection(self, classifier):
        result = classifier.classify("Check sprinkler head coverage in the warehouse")
        assert result.domain == MEPDomain.FIRE_PROTECTION

    def test_classify_action_query(self, classifier):
        result = classifier.classify("Find all VAV boxes")
        assert result.action == MEPAction.QUERY

    def test_classify_action_create(self, classifier):
        result = classifier.classify("Draw a duct from point A to point B")
        assert result.action == MEPAction.CREATE

    def test_classify_action_route(self, classifier):
        result = classifier.classify("Route the ductwork through the corridor")
        assert result.action == MEPAction.ROUTE

    def test_classify_action_analyze(self, classifier):
        result = classifier.classify("Check the duct sizing and validate pressure drop")
        assert result.action == MEPAction.ANALYZE

    def test_classify_action_coordinate(self, classifier):
        result = classifier.classify("Find clash conflicts between ducts and beams")
        assert result.action == MEPAction.COORDINATE

    def test_classify_app_context_autocad(self, classifier):
        result = classifier.classify("In AutoCAD, draw a line on the HVAC layer")
        assert result.app_context == AppContext.AUTOCAD

    def test_classify_app_context_revit(self, classifier):
        result = classifier.classify("Create a duct in Revit on level 2")
        assert result.app_context == AppContext.REVIT

    def test_classify_app_context_ambiguous(self, classifier):
        result = classifier.classify("Find all diffusers")
        # Should default to BOTH when ambiguous
        assert result.app_context == AppContext.BOTH

    def test_confidence_high_for_specific_query(self, classifier):
        # Multiple specific HVAC terms should give high confidence
        result = classifier.classify(
            "Route the supply ductwork from AHU-1 to the VAV boxes and check CFM"
        )
        assert result.confidence >= 0.5

    def test_confidence_low_for_general_query(self, classifier):
        result = classifier.classify("Hello, how are you?")
        assert result.confidence < 0.3

    def test_suggested_tools_for_query(self, classifier):
        result = classifier.classify("Find all ducts on level 2")
        assert "find_elements" in result.suggested_tools

    def test_suggested_tools_for_route(self, classifier):
        result = classifier.classify("Route ductwork from AHU to terminal")
        assert "draw_line_between" in result.suggested_tools or \
               "get_nearby_elements" in result.suggested_tools

    def test_get_tool_filter(self, classifier):
        # Need at least 2 AutoCAD keywords to trigger filter (e.g., "autocad" + "layer" or "dwg")
        result = classifier.classify("Find AHU in AutoCAD dwg file on HVAC layer")
        filter_config = classifier.get_tool_filter(result)
        assert "filter_prefix" in filter_config
        assert "tool_tier" in filter_config
        assert "include_metadata" in filter_config
        assert filter_config["filter_prefix"] == "autocad_"


class TestIntentClassifierEdgeCases:
    """Edge case tests for IntentClassifier."""

    @pytest.fixture
    def classifier(self):
        reset_intent_classifier()
        return IntentClassifier(use_embeddings=False)

    def test_mixed_domain_query(self, classifier):
        # Query mentioning multiple domains - should pick dominant one
        result = classifier.classify("Find HVAC ducts and electrical panels")
        # Should pick one domain (likely HVAC due to more specific term)
        assert result.domain in (MEPDomain.HVAC, MEPDomain.ELECTRICAL)

    def test_case_insensitivity(self, classifier):
        result1 = classifier.classify("Find AHU")
        result2 = classifier.classify("find ahu")
        result3 = classifier.classify("FIND AHU")
        assert result1.domain == result2.domain == result3.domain == MEPDomain.HVAC

    def test_partial_word_not_matched(self, classifier):
        # "vahue" should not match "vav"
        result = classifier.classify("Check the vahue of this parameter")
        assert "vav" not in [k.lower() for k in result.keywords_matched]

    def test_unicode_handling(self, classifier):
        # Should not crash on unicode
        result = classifier.classify("Find ducts with temperature 20°C")
        assert result is not None

    def test_very_long_input(self, classifier):
        # Should handle long input without issues
        long_input = "Find all ducts " * 100
        result = classifier.classify(long_input)
        assert result is not None
        assert result.domain == MEPDomain.HVAC
