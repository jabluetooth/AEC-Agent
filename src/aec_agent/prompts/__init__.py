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

__all__ = [
    "get_annotation_parsing_prompt",
    "get_batch_annotation_prompt",
]
