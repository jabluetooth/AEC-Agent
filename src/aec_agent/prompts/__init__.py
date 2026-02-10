"""
Prompt templates for LLM interactions.

Contains structured prompts for:
- Annotation parsing (semantic_ocr)
- Symbol classification (vision_llm)
- Document classification
"""

from .annotation_parsing import (
    get_annotation_parsing_prompt,
    get_batch_annotation_prompt,
)

from .symbol_classification import (
    get_symbol_classification_prompt,
    get_unknown_symbol_prompt,
    get_batch_symbol_prompt,
    get_valve_classification_prompt,
    get_diffuser_classification_prompt,
    get_outlet_classification_prompt,
    get_detector_classification_prompt,
    SYMBOL_SUBTYPES,
)

__all__ = [
    # Annotation parsing
    "get_annotation_parsing_prompt",
    "get_batch_annotation_prompt",
    # Symbol classification
    "get_symbol_classification_prompt",
    "get_unknown_symbol_prompt",
    "get_batch_symbol_prompt",
    "get_valve_classification_prompt",
    "get_diffuser_classification_prompt",
    "get_outlet_classification_prompt",
    "get_detector_classification_prompt",
    "SYMBOL_SUBTYPES",
]
