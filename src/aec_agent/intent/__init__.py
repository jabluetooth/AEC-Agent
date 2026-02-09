"""
Intent classification module for MEP workflow detection.

This module provides embedding-based intent classification to reduce token usage
by loading only relevant tools based on detected user intent.
"""

from aec_agent.intent.classifier import IntentClassifier
from aec_agent.intent.models import IntentResult, MEPAction, MEPDomain
from aec_agent.intent.patterns import MEP_PATTERNS, get_pattern_embeddings

__all__ = [
    "IntentResult",
    "MEPDomain",
    "MEPAction",
    "IntentClassifier",
    "MEP_PATTERNS",
    "get_pattern_embeddings",
]
