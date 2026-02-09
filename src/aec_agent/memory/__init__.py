"""
Memory module for project context and user preferences.

Provides persistent storage and retrieval of:
- Project facts and decisions
- Conversation summaries
- User preferences with Windows username integration
"""

from aec_agent.memory.models import (
    ConversationSummary,
    FactSource,
    FactType,
    MemoryContext,
    ProjectFact,
)
from aec_agent.memory.project_memory import ProjectMemory, get_project_memory
from aec_agent.memory.summarizer import (
    ConversationSummarizer,
    SummaryResult,
    get_summarizer,
    reset_summarizer,
)
from aec_agent.memory.user_preferences import (
    PreferenceCategory,
    UserPreferences,
    UserShortcut,
    get_current_user,
    get_user_preferences,
)

__all__ = [
    # Models
    "ProjectFact",
    "FactType",
    "FactSource",
    "ConversationSummary",
    "MemoryContext",
    # Project Memory
    "ProjectMemory",
    "get_project_memory",
    # User Preferences
    "UserPreferences",
    "UserShortcut",
    "PreferenceCategory",
    "get_user_preferences",
    "get_current_user",
    # Summarizer
    "ConversationSummarizer",
    "SummaryResult",
    "get_summarizer",
    "reset_summarizer",
]
