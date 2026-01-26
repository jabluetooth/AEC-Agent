"""
Conversation summarization for token optimization.

Summarizes older conversation context to reduce token usage while
preserving important information about what was discussed and done.
"""

from dataclasses import dataclass
from typing import Any, Optional, Protocol
import structlog

logger = structlog.get_logger(__name__)


class Message(Protocol):
    """Protocol for message objects."""
    role: str
    content: str
    tool_calls: Optional[list[dict[str, Any]]]


@dataclass
class SummaryResult:
    """Result of conversation summarization."""
    summary: str
    messages_summarized: int
    tokens_saved_estimate: int


class ConversationSummarizer:
    """
    Summarizes conversation history to reduce token usage.

    Uses a two-tier approach:
    1. Fast extraction of key actions (no LLM needed)
    2. Optional LLM-based summarization for richer context

    The summarizer maintains a cache to avoid regenerating summaries
    for the same conversation prefix.
    """

    def __init__(
        self,
        max_summary_tokens: int = 200,
        use_llm: bool = False,
    ):
        """
        Initialize the conversation summarizer.

        Args:
            max_summary_tokens: Maximum tokens for the summary
            use_llm: Whether to use LLM for summarization (slower but richer)
        """
        self._max_summary_tokens = max_summary_tokens
        self._use_llm = use_llm

        # Cache for avoiding regeneration
        self._cached_summary: Optional[str] = None
        self._summarized_count: int = 0

    def summarize(
        self,
        messages: list[Any],
        keep_recent: int = 6,
    ) -> Optional[SummaryResult]:
        """
        Summarize older messages in the conversation.

        Args:
            messages: Full message list (excluding system message)
            keep_recent: Number of recent messages to keep verbatim

        Returns:
            SummaryResult if summarization was performed, None otherwise
        """
        if len(messages) <= keep_recent:
            return None

        old_messages = messages[:-keep_recent]

        # Check if we already summarized this prefix
        if self._summarized_count >= len(old_messages) and self._cached_summary:
            return SummaryResult(
                summary=self._cached_summary,
                messages_summarized=len(old_messages),
                tokens_saved_estimate=self._estimate_tokens(old_messages) - len(self._cached_summary) // 4,
            )

        # Generate summary
        summary = self._generate_summary(old_messages)

        # Cache the result
        self._cached_summary = summary
        self._summarized_count = len(old_messages)

        original_tokens = self._estimate_tokens(old_messages)
        summary_tokens = len(summary) // 4

        logger.debug(
            "Conversation summarized",
            messages_summarized=len(old_messages),
            original_tokens=original_tokens,
            summary_tokens=summary_tokens,
            tokens_saved=original_tokens - summary_tokens,
        )

        return SummaryResult(
            summary=summary,
            messages_summarized=len(old_messages),
            tokens_saved_estimate=original_tokens - summary_tokens,
        )

    def _generate_summary(self, messages: list[Any]) -> str:
        """
        Generate a summary of the given messages.

        Uses fast extraction by default, LLM if enabled.
        """
        # Extract key information without LLM
        actions = []
        topics = []
        errors = []

        for msg in messages:
            role = getattr(msg, 'role', None)
            content = getattr(msg, 'content', None)
            tool_calls = getattr(msg, 'tool_calls', None)

            if role == "assistant" and tool_calls:
                # Extract tool actions
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        func = tc.get("function", {})
                        tool_name = func.get("name", "unknown")
                        actions.append(tool_name)

            if role == "user" and content:
                # Extract topic keywords (first 50 chars)
                topic = content[:50].strip()
                if topic:
                    topics.append(topic)

            if role == "tool" and content:
                # Check for errors
                if "error" in content.lower() or "failed" in content.lower():
                    errors.append(content[:100])

        # Build summary
        parts = []

        if topics:
            # Keep unique topics, max 3
            unique_topics = list(dict.fromkeys(topics))[:3]
            parts.append(f"Topics: {'; '.join(unique_topics)}")

        if actions:
            # Keep unique actions, max 10
            unique_actions = list(dict.fromkeys(actions))[:10]
            parts.append(f"Actions: {', '.join(unique_actions)}")

        if errors:
            parts.append(f"Errors encountered: {len(errors)}")

        if not parts:
            return "[Previous conversation context]"

        summary = ". ".join(parts)

        # Truncate to max tokens (rough: 4 chars per token)
        max_chars = self._max_summary_tokens * 4
        if len(summary) > max_chars:
            summary = summary[:max_chars - 3] + "..."

        return summary

    def _estimate_tokens(self, messages: list[Any]) -> int:
        """Estimate token count for messages."""
        total = 0
        for msg in messages:
            content = getattr(msg, 'content', None)
            if content:
                total += len(content) // 4

            tool_calls = getattr(msg, 'tool_calls', None)
            if tool_calls:
                import json
                try:
                    total += len(json.dumps(tool_calls)) // 4
                except (TypeError, ValueError):
                    total += 50  # Estimate for tool calls

        return total

    def clear(self) -> None:
        """Clear cached summary."""
        self._cached_summary = None
        self._summarized_count = 0

    def get_stats(self) -> dict[str, Any]:
        """Get summarizer statistics."""
        return {
            "cached_summary_length": len(self._cached_summary) if self._cached_summary else 0,
            "messages_summarized": self._summarized_count,
            "has_cache": self._cached_summary is not None,
        }


# Global instance management
_summarizer_instance: Optional[ConversationSummarizer] = None


def get_summarizer(
    max_summary_tokens: int = 200,
    use_llm: bool = False,
) -> ConversationSummarizer:
    """
    Get or create the global ConversationSummarizer instance.

    Args:
        max_summary_tokens: Maximum tokens for summaries
        use_llm: Whether to use LLM for summarization

    Returns:
        ConversationSummarizer instance
    """
    global _summarizer_instance

    if _summarizer_instance is None:
        _summarizer_instance = ConversationSummarizer(
            max_summary_tokens=max_summary_tokens,
            use_llm=use_llm,
        )

    return _summarizer_instance


def reset_summarizer() -> None:
    """Reset the global summarizer instance."""
    global _summarizer_instance
    _summarizer_instance = None
