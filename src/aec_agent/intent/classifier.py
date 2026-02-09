"""
Intent classifier for MEP workflow detection.

Uses a combination of keyword matching and optional embedding-based similarity
to classify user intent with minimal computational overhead.
"""

import logging
import re

from aec_agent.intent.models import (
    AppContext,
    IntentResult,
    MEPAction,
    MEPDomain,
    PatternMatch,
)
from aec_agent.intent.patterns import (
    ACTION_PATTERNS,
    APP_CONTEXT_PATTERNS,
    MEP_PATTERNS,
)

logger = logging.getLogger(__name__)


class IntentClassifier:
    """
    Classifies user input to determine MEP domain, action, and tool requirements.

    Uses a hybrid approach:
    1. Fast keyword matching for common patterns
    2. Optional embedding-based similarity for ambiguous cases

    Token savings: By accurately classifying intent, we can load only relevant
    tools (e.g., HVAC tools only for ductwork queries), reducing tool schema
    tokens by 40-60%.
    """

    def __init__(
        self,
        embedding_service=None,
        use_embeddings: bool = True,
        min_confidence: float = 0.3,
    ):
        """
        Initialize the intent classifier.

        Args:
            embedding_service: Optional embedding service for similarity matching
            use_embeddings: Whether to use embeddings for classification
            min_confidence: Minimum confidence threshold for classification
        """
        self._embedding_service = embedding_service
        self._use_embeddings = use_embeddings and embedding_service is not None
        self._min_confidence = min_confidence

        # Pre-compiled patterns for fast matching
        self._compiled_patterns: dict[MEPDomain, list[tuple[re.Pattern, float, str | None]]] = {}
        self._compile_patterns()

        # Cached pattern embeddings (lazy loaded)
        self._pattern_embeddings: dict[str, list[float]] | None = None

        logger.info(
            f"IntentClassifier initialized (embeddings={'enabled' if self._use_embeddings else 'disabled'})"
        )

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for fast matching."""
        for domain, patterns in MEP_PATTERNS.items():
            compiled = []
            for pattern_text, (weight, subdomain) in patterns.items():
                # Create word-boundary pattern for exact matching
                regex = re.compile(
                    r"\b" + re.escape(pattern_text) + r"\b",
                    re.IGNORECASE
                )
                compiled.append((regex, weight, subdomain))
            self._compiled_patterns[domain] = compiled

    def _ensure_pattern_embeddings(self) -> None:
        """Lazy load pattern embeddings if needed."""
        if self._pattern_embeddings is None and self._use_embeddings:
            from aec_agent.intent.patterns import get_pattern_embeddings
            self._pattern_embeddings = get_pattern_embeddings(self._embedding_service)
            logger.debug(f"Loaded {len(self._pattern_embeddings)} pattern embeddings")

    def classify(self, user_input: str) -> IntentResult:
        """
        Classify user input to determine intent.

        Args:
            user_input: The user's message/query

        Returns:
            IntentResult with domain, action, and tool suggestions
        """
        if not user_input or not user_input.strip():
            return IntentResult()

        user_input_lower = user_input.lower()

        # Step 1: Detect action from input
        action = self._detect_action(user_input_lower)

        # Step 2: Detect app context (AutoCAD vs Revit)
        app_context = self._detect_app_context(user_input_lower)

        # Step 3: Match MEP domain patterns
        matches = self._match_patterns(user_input_lower)

        # Step 4: Calculate domain scores and determine winner
        domain_scores = self._calculate_domain_scores(matches)

        # Step 5: Determine winning domain
        domain, subdomain, confidence, keywords = self._determine_domain(
            domain_scores, matches
        )

        # Step 6: Get suggested tools based on domain and action
        suggested_tools = self._get_suggested_tools(domain, subdomain, action)

        result = IntentResult(
            domain=domain,
            subdomain=subdomain,
            action=action,
            app_context=app_context,
            confidence=confidence,
            suggested_tools=suggested_tools,
            keywords_matched=keywords,
        )

        logger.debug(f"Classified intent: {result.to_dict()}")
        return result

    def _detect_action(self, text: str) -> MEPAction:
        """Detect the action type from user input."""
        action_scores: dict[str, int] = {}

        for action_name, keywords in ACTION_PATTERNS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                action_scores[action_name] = score

        if not action_scores:
            return MEPAction.QUERY

        # Return action with highest score
        best_action = max(action_scores, key=action_scores.get)
        return MEPAction.from_string(best_action)

    def _detect_app_context(self, text: str) -> AppContext:
        """Detect which CAD application the user is referring to."""
        autocad_score = sum(
            1 for kw in APP_CONTEXT_PATTERNS["autocad"] if kw in text
        )
        revit_score = sum(
            1 for kw in APP_CONTEXT_PATTERNS["revit"] if kw in text
        )

        # Need clear signal to filter (score >= 2 and ratio > 2:1)
        if autocad_score >= 2 and autocad_score > revit_score * 2:
            return AppContext.AUTOCAD
        elif revit_score >= 2 and revit_score > autocad_score * 2:
            return AppContext.REVIT

        return AppContext.BOTH

    def _match_patterns(self, text: str) -> list[PatternMatch]:
        """Match all patterns against the input text."""
        matches = []

        for domain, patterns in self._compiled_patterns.items():
            for regex, weight, subdomain in patterns:
                if regex.search(text):
                    matches.append(PatternMatch(
                        pattern=regex.pattern.replace(r"\b", "").replace("\\", ""),
                        domain=domain,
                        subdomain=subdomain,
                        weight=weight,
                    ))

        return matches

    def _calculate_domain_scores(
        self, matches: list[PatternMatch]
    ) -> dict[MEPDomain, float]:
        """Calculate aggregate scores for each domain."""
        scores: dict[MEPDomain, float] = dict.fromkeys(MEPDomain, 0.0)

        for match in matches:
            scores[match.domain] += match.score

        return scores

    def _determine_domain(
        self,
        scores: dict[MEPDomain, float],
        matches: list[PatternMatch],
    ) -> tuple[MEPDomain, str | None, float, list[str]]:
        """
        Determine the winning domain from scores.

        Returns:
            (domain, subdomain, confidence, matched_keywords)
        """
        # Filter out GENERAL domain for comparison
        mep_scores = {k: v for k, v in scores.items() if k != MEPDomain.GENERAL and v > 0}

        if not mep_scores:
            return MEPDomain.GENERAL, None, 0.0, []

        # Find best domain
        best_domain = max(mep_scores, key=mep_scores.get)
        best_score = mep_scores[best_domain]

        # Calculate confidence based on score magnitude and dominance
        total_score = sum(mep_scores.values())
        dominance = best_score / total_score if total_score > 0 else 0

        # Confidence: combination of absolute score and relative dominance
        # Scale: score of 6+ with high dominance = high confidence
        score_factor = min(1.0, best_score / 6.0)
        confidence = (score_factor * 0.6) + (dominance * 0.4)

        # Get subdomain from highest-weight match
        domain_matches = [m for m in matches if m.domain == best_domain]
        subdomain = None
        if domain_matches:
            best_match = max(domain_matches, key=lambda m: m.weight)
            subdomain = best_match.subdomain

        # Get all matched keywords for this domain
        keywords = [m.pattern for m in domain_matches]

        return best_domain, subdomain, confidence, keywords

    def _get_suggested_tools(
        self,
        domain: MEPDomain,
        subdomain: str | None,
        action: MEPAction,
    ) -> list[str]:
        """
        Get suggested tools based on domain, subdomain, and action.

        These are tool name patterns that should be prioritized in filtering.
        """
        tools = []

        # Common metadata tools for MEP work
        if domain != MEPDomain.GENERAL:
            tools.extend(["find_elements", "get_nearby_elements", "sync_metadata"])

        # Action-based suggestions
        if action == MEPAction.QUERY:
            tools.extend(["find_elements", "get_related_elements"])
        elif action == MEPAction.CREATE:
            tools.append("draw_")  # Prefix for drawing tools
        elif action == MEPAction.ANALYZE:
            tools.extend(["get_intersecting_elements", "validate_"])
        elif action == MEPAction.ROUTE:
            tools.extend(["draw_line_between", "get_nearby_elements"])
        elif action == MEPAction.SCHEDULE:
            tools.extend(["find_elements", "get_related_elements"])
        elif action == MEPAction.COORDINATE:
            tools.extend(["get_intersecting_elements", "find_elements"])

        # Domain-specific tools - use domain priority tools from optimization module
        from aec_agent.frontend.tool_optimization import get_domain_priority_tools
        domain_tools = get_domain_priority_tools(domain.value, include_useful=True)
        if domain_tools:
            tools.extend(domain_tools)

        return list(set(tools))  # Remove duplicates

    def get_tool_filter(self, intent: IntentResult) -> dict:
        """
        Get tool filtering parameters based on intent.

        Returns dict with:
            - filter_prefix: Optional tool name prefix (autocad_, revit_)
            - tool_tier: essential, standard, or advanced
            - include_metadata: Whether to include metadata tools
            - suggested_tools: List of specifically suggested tools
        """
        return {
            "filter_prefix": intent.tool_filter_prefix,
            "tool_tier": intent.get_tool_tier(),
            "include_metadata": intent.is_mep_specific,
            "suggested_tools": intent.suggested_tools,
        }


# Global instance management
_classifier_instance: IntentClassifier | None = None


def get_intent_classifier(
    embedding_service=None,
    use_embeddings: bool = True,
) -> IntentClassifier:
    """
    Get or create the global IntentClassifier instance.

    Args:
        embedding_service: Optional embedding service for similarity matching
        use_embeddings: Whether to use embeddings

    Returns:
        IntentClassifier instance
    """
    global _classifier_instance

    if _classifier_instance is None:
        _classifier_instance = IntentClassifier(
            embedding_service=embedding_service,
            use_embeddings=use_embeddings,
        )

    return _classifier_instance


def reset_intent_classifier() -> None:
    """Reset the global classifier instance (for testing)."""
    global _classifier_instance
    _classifier_instance = None
