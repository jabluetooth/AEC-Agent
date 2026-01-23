"""
LLM Agent for AEC Agent.

Provides the intelligence loop that drives conversations with tool calling.
Supports multiple LLM providers: OpenAI, Anthropic, Azure OpenAI, HuggingFace.
"""

import json
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Optional
from dataclasses import dataclass, field

import structlog

from aec_agent.config.settings import get_settings, LLMProvider
from aec_agent.frontend.mcp_client import MCPClient, ToolResult
from aec_agent.intent.classifier import IntentClassifier, get_intent_classifier
from aec_agent.intent.models import IntentResult, MEPDomain

logger = structlog.get_logger(__name__)

# System prompt - optimized for minimal tokens (~60 tokens vs ~100)
SYSTEM_PROMPT = """AEC assistant for AutoCAD/Revit automation.
Tools: layers, drawing, levels, walls, rooms, queries.
Rules: Brief explanations. Confirm deletes. Report errors with alternatives.
Units: meters (Revit), drawing units (AutoCAD). One operation at a time."""


@dataclass
class Message:
    """Represents a conversation message."""

    role: str  # "user", "assistant", "system", "tool"
    content: str
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None
    name: Optional[str] = None  # For tool messages


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class AgentResponse:
    """Response from the agent."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finished: bool = True


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> AgentResponse:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> AsyncGenerator[str, None]:
        """Stream a response from the LLM."""
        pass


class OpenAIBackend(LLMBackend):
    """OpenAI GPT backend."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model
        self._client = None

    async def _get_client(self):
        """Lazy-load the OpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> AgentResponse:
        """Generate a response using OpenAI."""
        client = await self._get_client()

        # Convert messages to OpenAI format
        openai_messages = []
        for msg in messages:
            if msg.role == "tool":
                openai_messages.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id,
                })
            elif msg.tool_calls:
                openai_messages.append({
                    "role": msg.role,
                    "content": msg.content or "",
                    "tool_calls": msg.tool_calls,
                })
            else:
                openai_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        # Build request
        kwargs = {
            "model": self.model,
            "messages": openai_messages,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message

        # Extract tool calls
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=arguments,
                ))

        return AgentResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            finished=choice.finish_reason == "stop",
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> AsyncGenerator[str, None]:
        """Stream a response using OpenAI."""
        client = await self._get_client()

        # Convert messages to OpenAI format
        openai_messages = []
        for msg in messages:
            if msg.role == "tool":
                openai_messages.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id,
                })
            elif msg.tool_calls:
                openai_messages.append({
                    "role": msg.role,
                    "content": msg.content or "",
                    "tool_calls": msg.tool_calls,
                })
            else:
                openai_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        kwargs = {
            "model": self.model,
            "messages": openai_messages,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        stream = await client.chat.completions.create(**kwargs)

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AnthropicBackend(LLMBackend):
    """Anthropic Claude backend."""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.api_key = api_key
        self.model = model
        self._client = None

    async def _get_client(self):
        """Lazy-load the Anthropic client."""
        if self._client is None:
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(api_key=self.api_key)
        return self._client

    async def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> AgentResponse:
        """Generate a response using Anthropic."""
        client = await self._get_client()

        # Convert messages to Anthropic format
        system_content = ""
        anthropic_messages = []

        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
            elif msg.role == "tool":
                anthropic_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": msg.content,
                        }
                    ],
                })
            elif msg.tool_calls:
                # Assistant message with tool calls
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": json.loads(tc["function"]["arguments"]),
                    })
                anthropic_messages.append({
                    "role": "assistant",
                    "content": content,
                })
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        # Build request with optional prompt caching
        kwargs = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": anthropic_messages,
        }

        # Check if prompt caching is enabled
        settings = get_settings()
        use_caching = settings.enable_prompt_caching

        # Use prompt caching for system prompt (reduces repeated token costs)
        if system_content:
            if use_caching:
                kwargs["system"] = [
                    {
                        "type": "text",
                        "text": system_content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                kwargs["system"] = system_content

        # Use prompt caching for tool definitions
        if tools:
            if use_caching:
                # Add cache_control to the last tool for efficient caching
                cached_tools = []
                for i, tool in enumerate(tools):
                    if i == len(tools) - 1:
                        # Add cache control to last tool
                        cached_tools.append({
                            **tool,
                            "cache_control": {"type": "ephemeral"},
                        })
                    else:
                        cached_tools.append(tool)
                kwargs["tools"] = cached_tools
            else:
                kwargs["tools"] = tools

        response = await client.messages.create(**kwargs)

        # Extract content and tool calls
        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))

        return AgentResponse(
            content=content,
            tool_calls=tool_calls,
            finished=response.stop_reason == "end_turn",
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> AsyncGenerator[str, None]:
        """Stream a response using Anthropic."""
        client = await self._get_client()

        # Convert messages (same as generate)
        system_content = ""
        anthropic_messages = []

        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
            elif msg.role == "tool":
                anthropic_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": msg.content,
                        }
                    ],
                })
            elif msg.tool_calls:
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": json.loads(tc["function"]["arguments"]),
                    })
                anthropic_messages.append({
                    "role": "assistant",
                    "content": content,
                })
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        kwargs = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": anthropic_messages,
            "stream": True,
        }
        if system_content:
            kwargs["system"] = system_content
        if tools:
            kwargs["tools"] = tools

        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text


class AzureOpenAIBackend(LLMBackend):
    """Azure OpenAI backend."""

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        deployment: str,
    ):
        self.api_key = api_key
        self.endpoint = endpoint
        self.deployment = deployment
        self._client = None

    async def _get_client(self):
        """Lazy-load the Azure OpenAI client."""
        if self._client is None:
            from openai import AsyncAzureOpenAI
            self._client = AsyncAzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.endpoint,
                api_version="2024-02-15-preview",
            )
        return self._client

    async def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> AgentResponse:
        """Generate using Azure OpenAI (same format as OpenAI)."""
        client = await self._get_client()

        # Convert messages to OpenAI format
        openai_messages = []
        for msg in messages:
            if msg.role == "tool":
                openai_messages.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id,
                })
            elif msg.tool_calls:
                openai_messages.append({
                    "role": msg.role,
                    "content": msg.content or "",
                    "tool_calls": msg.tool_calls,
                })
            else:
                openai_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        kwargs = {
            "model": self.deployment,
            "messages": openai_messages,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message

        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=arguments,
                ))

        return AgentResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            finished=choice.finish_reason == "stop",
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> AsyncGenerator[str, None]:
        """Stream using Azure OpenAI."""
        client = await self._get_client()

        openai_messages = []
        for msg in messages:
            if msg.role == "tool":
                openai_messages.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id,
                })
            elif msg.tool_calls:
                openai_messages.append({
                    "role": msg.role,
                    "content": msg.content or "",
                    "tool_calls": msg.tool_calls,
                })
            else:
                openai_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        kwargs = {
            "model": self.deployment,
            "messages": openai_messages,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        stream = await client.chat.completions.create(**kwargs)

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class HuggingFaceBackend(LLMBackend):
    """Hugging Face Inference API backend (OpenAI compatible)."""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        # Use simple model name or full path
        self.model = model
        self._client = None

    async def _get_client(self):
        """Lazy-load the OpenAI client configured for Hugging Face."""
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url="https://router.huggingface.co/v1"
            )
        return self._client

    async def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> AgentResponse:
        """Generate using Hugging Face (via OpenAI protocol)."""
        client = await self._get_client()

        # Convert messages to OpenAI format
        openai_messages = []
        for msg in messages:
            if msg.role == "tool":
                openai_messages.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id,
                })
            elif msg.tool_calls:
                openai_messages.append({
                    "role": msg.role,
                    "content": msg.content or "",
                    "tool_calls": msg.tool_calls,
                })
            else:
                openai_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        kwargs = {
            "model": self.model,
            "messages": openai_messages,
            "max_tokens": 1024,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            response = await client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            message = choice.message

            # Extract tool calls
            tool_calls = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                    tool_calls.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments,
                    ))

            return AgentResponse(
                content=message.content or "",
                tool_calls=tool_calls,
                finished=choice.finish_reason == "stop",
            )
        except Exception as e:
            logger.error("Hugging Face API error", error=str(e))
            return AgentResponse(
                content=f"Error calling Hugging Face API: {str(e)}",
                finished=True
            )

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> AsyncGenerator[str, None]:
        """Stream using Hugging Face."""
        client = await self._get_client()

        # Convert messages to OpenAI format
        openai_messages = []
        for msg in messages:
            if msg.role == "tool":
                openai_messages.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id,
                })
            elif msg.tool_calls:
                openai_messages.append({
                    "role": msg.role,
                    "content": msg.content or "",
                    "tool_calls": msg.tool_calls,
                })
            else:
                openai_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        kwargs = {
            "model": self.model,
            "messages": openai_messages,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        stream = await client.chat.completions.create(**kwargs)

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class GroqBackend(LLMBackend):
    """Groq Inference API backend (OpenAI compatible)."""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self._client = None

    async def _get_client(self):
        """Lazy-load the OpenAI client configured for Groq."""
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url="https://api.groq.com/openai/v1"
            )
        return self._client

    async def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> AgentResponse:
        """Generate using Groq (via OpenAI protocol)."""
        client = await self._get_client()

        # Convert messages to OpenAI format
        openai_messages = []
        for msg in messages:
            if msg.role == "tool":
                openai_messages.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id,
                })
            elif msg.tool_calls:
                openai_messages.append({
                    "role": msg.role,
                    "content": msg.content or "",
                    "tool_calls": msg.tool_calls,
                })
            else:
                openai_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        kwargs = {
            "model": self.model,
            "messages": openai_messages,
            "max_tokens": 4096,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            response = await client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            message = choice.message

            # Extract tool calls
            tool_calls = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                    tool_calls.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments,
                    ))

            return AgentResponse(
                content=message.content or "",
                tool_calls=tool_calls,
                finished=choice.finish_reason == "stop",
            )
        except Exception as e:
            logger.error("Groq API error", error=str(e))
            return AgentResponse(
                content=f"Error calling Groq API: {str(e)}",
                finished=True
            )

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> AsyncGenerator[str, None]:
        """Stream using Groq."""
        client = await self._get_client()

        # Convert messages to OpenAI format
        openai_messages = []
        for msg in messages:
            if msg.role == "tool":
                openai_messages.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id,
                })
            elif msg.tool_calls:
                openai_messages.append({
                    "role": msg.role,
                    "content": msg.content or "",
                    "tool_calls": msg.tool_calls,
                })
            else:
                openai_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        kwargs = {
            "model": self.model,
            "messages": openai_messages,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        stream = await client.chat.completions.create(**kwargs)

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AECAgent:
    """
    Main agent class that orchestrates LLM interactions and tool calls.

    Manages the conversation loop, tool execution, and message history.
    """

    def __init__(
        self,
        mcp_client: MCPClient,
        backend: Optional[LLMBackend] = None,
        max_history_messages: Optional[int] = None,
        app_context: str = "both",
        intent_classifier: Optional[IntentClassifier] = None,
    ):
        """
        Initialize the AEC Agent.

        Args:
            mcp_client: Connected MCP client for tool calls.
            backend: LLM backend to use. Auto-detected from settings if not provided.
            max_history_messages: Maximum conversation messages to keep (excluding system).
                                  Older messages are trimmed to reduce token usage.
                                  Defaults to settings.max_history_messages.
            app_context: Application context ("both", "autocad", or "revit").
                        Controls which tools are available to reduce token usage.
            intent_classifier: Optional IntentClassifier for MEP-aware tool filtering.
                              If not provided, uses global instance.
        """
        settings = get_settings()
        self.mcp_client = mcp_client
        self.backend = backend or self._create_backend()
        self.max_history_messages = max_history_messages or settings.max_history_messages
        self.app_context = app_context
        self.messages: list[Message] = [
            Message(role="system", content=SYSTEM_PROMPT)
        ]
        # Intent classifier for MEP-aware tool filtering
        self._intent_classifier = intent_classifier
        self._last_intent: Optional[IntentResult] = None

        # Conversation summarizer for long conversations
        from aec_agent.memory.summarizer import ConversationSummarizer
        self._summarizer = ConversationSummarizer(
            max_summary_tokens=settings.max_summary_tokens,
        )
        self._context_summary: Optional[str] = None

    def set_app_context(self, app_context: str) -> None:
        """
        Update the application context.

        Args:
            app_context: New context ("both", "autocad", or "revit")
        """
        self.app_context = app_context
        logger.info("App context updated", context=app_context)

    def _trim_history(self) -> None:
        """Trim conversation history to reduce token usage.

        Keeps the system message and the most recent messages up to max_history_messages.
        """
        if len(self.messages) <= self.max_history_messages + 1:  # +1 for system
            return

        # Keep system message (index 0) and most recent messages
        system_msg = self.messages[0]
        recent_msgs = self.messages[-(self.max_history_messages):]
        self.messages = [system_msg] + recent_msgs
        logger.debug("Trimmed history", kept_messages=len(self.messages))

    def _optimize_context(self) -> None:
        """Optimize context using summarization for long conversations.

        When enabled, summarizes older messages instead of just trimming them,
        preserving important context while reducing tokens.
        """
        settings = get_settings()

        # Fall back to simple trimming if summarization disabled
        if not settings.enable_conversation_summarization:
            self._trim_history()
            return

        # Get non-system messages
        if len(self.messages) <= 1:
            return

        non_system = self.messages[1:]
        keep_recent = settings.summarization_keep_recent

        # Not enough messages to summarize
        if len(non_system) <= keep_recent:
            return

        # Generate summary of older messages
        result = self._summarizer.summarize(non_system, keep_recent=keep_recent)

        if result is None:
            return

        # Store the summary
        self._context_summary = result.summary

        # Rebuild messages: system + summary message + recent
        system_msg = self.messages[0]
        recent_msgs = self.messages[-(keep_recent):]

        # Create summary message as a system reminder
        summary_msg = Message(
            role="system",
            content=f"[Previous context summary: {result.summary}]"
        )

        self.messages = [system_msg, summary_msg] + recent_msgs

        logger.info(
            "Context optimized with summarization",
            messages_summarized=result.messages_summarized,
            tokens_saved=result.tokens_saved_estimate,
            new_message_count=len(self.messages),
        )

    def _estimate_context_tokens(self) -> int:
        """Estimate current context token count.

        Uses rough approximation of ~4 characters per token.
        """
        total = 0
        for msg in self.messages:
            if msg.content:
                total += len(msg.content) // 4
            if msg.tool_calls:
                total += len(json.dumps(msg.tool_calls)) // 4
        return total

    def _get_dynamic_compression_mode(self) -> str:
        """Get compression mode based on context size.

        Automatically escalates compression as context grows to stay
        within token limits while preserving quality when possible.

        Returns:
            Compression mode: "standard", "minimal", or "ultra"
        """
        settings = get_settings()

        # If dynamic compression disabled, use static setting
        if not settings.enable_dynamic_compression:
            return settings.tool_compression_mode

        # Calculate context ratio
        tokens = self._estimate_context_tokens()
        ratio = tokens / settings.max_context_tokens

        # Escalate compression based on usage
        if ratio < 0.3:
            return "standard"
        elif ratio < 0.6:
            return "minimal"
        else:
            return "ultra"

    def _log_token_usage(self, tools_loaded: int, compression_mode: str) -> None:
        """Log token usage for monitoring.

        Args:
            tools_loaded: Number of tools in current request
            compression_mode: Current compression mode
        """
        settings = get_settings()
        if not settings.enable_token_logging:
            return

        estimated_tokens = self._estimate_context_tokens()
        logger.info(
            "Token usage",
            estimated_input_tokens=estimated_tokens,
            tools_loaded=tools_loaded,
            compression_mode=compression_mode,
            history_messages=len(self.messages),
            context_ratio=round(estimated_tokens / settings.max_context_tokens, 2),
        )

    def _create_backend(self) -> LLMBackend:
        """Create an LLM backend based on settings."""
        settings = get_settings()

        if settings.llm_provider == LLMProvider.OPENAI:
            return OpenAIBackend(
                api_key=settings.get_llm_api_key(),
                model="gpt-4o",
            )
        elif settings.llm_provider == LLMProvider.ANTHROPIC:
            return AnthropicBackend(
                api_key=settings.get_llm_api_key(),
                model="claude-3-5-sonnet-20241022",
            )
        elif settings.llm_provider == LLMProvider.AZURE_OPENAI:
            return AzureOpenAIBackend(
                api_key=settings.get_llm_api_key(),
                endpoint=settings.azure_openai_endpoint or "",
                deployment=settings.azure_openai_deployment or "",
            )
        elif settings.llm_provider == LLMProvider.HUGGINGFACE:
            return HuggingFaceBackend(
                api_key=settings.get_llm_api_key(),
                model=settings.huggingface_model,
            )
        elif settings.llm_provider == LLMProvider.GROQ:
            return GroqBackend(
                api_key=settings.get_llm_api_key(),
                model=settings.groq_model,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

    def _get_intent_classifier(self) -> IntentClassifier:
        """Get or create the intent classifier."""
        if self._intent_classifier is None:
            settings = get_settings()
            # Use embeddings only if enabled in settings
            use_embeddings = getattr(settings, 'use_intent_embeddings', False)
            self._intent_classifier = get_intent_classifier(
                use_embeddings=use_embeddings
            )
        return self._intent_classifier

    def _classify_intent(self, user_input: str) -> IntentResult:
        """
        Classify user intent for smart tool filtering.

        This replaces the simple keyword matching with the full IntentClassifier,
        enabling MEP-aware tool filtering that can reduce token usage by 40-60%.

        Args:
            user_input: The user's message

        Returns:
            IntentResult with domain, action, and tool suggestions
        """
        classifier = self._get_intent_classifier()
        intent = classifier.classify(user_input)
        self._last_intent = intent

        logger.debug(
            "Classified user intent",
            domain=intent.domain.value,
            subdomain=intent.subdomain,
            action=intent.action.value,
            confidence=intent.confidence,
            app_context=intent.app_context.value,
            keywords=intent.keywords_matched,
        )

        return intent

    def _detect_tool_context(self, user_input: str) -> Optional[str]:
        """Detect which tool context (autocad/revit) based on user input.

        This is a legacy method that uses the new IntentClassifier.

        Returns:
            'autocad_' or 'revit_' prefix, or None for all tools.
        """
        intent = self._classify_intent(user_input)
        return intent.tool_filter_prefix

    def get_last_intent(self) -> Optional[IntentResult]:
        """Get the last classified intent for debugging/logging."""
        return self._last_intent

    def _get_tools(self, user_input: Optional[str] = None) -> list[dict[str, Any]]:
        """Get tools in the format expected by the current backend.

        Uses intent classification for intelligent tool filtering:
        - Detects MEP domain (HVAC, electrical, plumbing, fire protection)
        - Determines appropriate tool tier based on action
        - Filters by app context (AutoCAD vs Revit)

        This can reduce token usage by 40-60% by loading only relevant tools.

        Args:
            user_input: Optional user message to detect context for smart filtering.
        """
        settings = get_settings()

        # Default values
        filter_prefix = None
        tool_tier = settings.tool_tier if settings.tool_tier != "standard" else None
        include_metadata = settings.enable_metadata_tools and settings.has_database

        # Classify intent if smart routing is enabled and we have user input
        intent: Optional[IntentResult] = None
        if settings.smart_tool_routing and user_input:
            intent = self._classify_intent(user_input)

        # Determine tool filter prefix based on app context
        if self.app_context == "autocad":
            filter_prefix = "autocad_"
        elif self.app_context == "revit":
            filter_prefix = "revit_"
        elif intent:
            # Use intent-based app context for "both" mode
            filter_prefix = intent.tool_filter_prefix

        # Use intent to determine tool tier and domain-specific loading
        domain_priority_tools = None
        if intent and intent.is_mep_specific:
            # MEP-specific intent detected - use domain-aware tier
            intent_tier = intent.get_tool_tier()
            if intent_tier != "standard":
                tool_tier = intent_tier

            # Get domain priority tools for more focused loading
            from aec_agent.frontend.tool_optimization import get_domain_priority_tools
            domain_priority_tools = get_domain_priority_tools(
                intent.domain.value,
                include_useful=(intent.confidence > 0.7)
            )

            logger.debug(
                "MEP domain-specific tool loading",
                tier=tool_tier,
                domain=intent.domain.value,
                action=intent.action.value,
                confidence=intent.confidence,
                priority_tools=list(domain_priority_tools) if domain_priority_tools else None,
            )

            # Always include metadata tools for MEP queries
            if settings.has_database:
                include_metadata = True

        if filter_prefix:
            logger.debug(
                "Filtered tools by context",
                prefix=filter_prefix,
                app_context=self.app_context,
                mep_domain=intent.domain.value if intent else None,
            )

        # Determine compression settings
        compress = settings.compress_tool_schemas or settings.llm_provider in (
            LLMProvider.HUGGINGFACE,
            LLMProvider.GROQ,
        )

        # Use dynamic compression based on context size
        compression_mode = self._get_dynamic_compression_mode()

        # Override with minimal for budget providers
        if settings.llm_provider in (LLMProvider.HUGGINGFACE, LLMProvider.GROQ):
            compression_mode = "minimal"

        if settings.llm_provider == LLMProvider.ANTHROPIC:
            tools = self.mcp_client.get_tools_for_anthropic(
                compress=compress,
                compression_mode=compression_mode,
                filter_prefix=filter_prefix,
                tool_tier=tool_tier,
                include_metadata=include_metadata,
            )
        else:
            tools = self.mcp_client.get_tools_for_openai(
                compress=compress,
                compression_mode=compression_mode,
                filter_prefix=filter_prefix,
                tool_tier=tool_tier,
                include_metadata=include_metadata,
            )

        # Log token usage
        self._log_token_usage(len(tools), compression_mode)

        return tools

    async def process_message(self, user_input: str) -> AsyncGenerator[str, None]:
        """
        Process a user message and yield the response.

        Handles tool calls automatically in a loop until the LLM
        produces a final response.

        Args:
            user_input: The user's message.

        Yields:
            Response text chunks.
        """
        # Optimize context (summarization or trimming based on settings)
        self._optimize_context()

        # Add user message
        self.messages.append(Message(role="user", content=user_input))

        # Get tools with smart filtering based on user input
        tools = self._get_tools(user_input)
        max_iterations = 10  # Prevent infinite loops

        for _ in range(max_iterations):
            logger.debug(
                "Agent iteration",
                message_count=len(self.messages),
                tool_count=len(tools),
            )

            # Get response from LLM
            response = await self.backend.generate(self.messages, tools)

            # If there's content, yield it
            if response.content:
                yield response.content

            # Handle tool calls
            if response.tool_calls:
                # Add assistant message with tool calls
                tool_calls_data = []
                for tc in response.tool_calls:
                    tool_calls_data.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    })

                self.messages.append(Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=tool_calls_data,
                ))

                # Execute tool calls
                for tc in response.tool_calls:
                    logger.info(
                        "Executing tool",
                        tool=tc.name,
                        arguments=tc.arguments,
                    )

                    yield f"\n\n*Executing {tc.name}...*\n"

                    try:
                        result = await self.mcp_client.call_tool(
                            tc.name,
                            tc.arguments,
                        )

                        # Add tool result to messages
                        self.messages.append(Message(
                            role="tool",
                            content=result.to_message_content(),
                            tool_call_id=tc.id,
                            name=tc.name,
                        ))

                        if result.success:
                            yield f"*{tc.name} completed successfully*\n"
                        else:
                            yield f"*{tc.name} failed: {result.error}*\n"

                    except Exception as e:
                        logger.error(
                            "Tool execution error",
                            tool=tc.name,
                            error=str(e),
                        )
                        self.messages.append(Message(
                            role="tool",
                            content=f"Error: {str(e)}",
                            tool_call_id=tc.id,
                            name=tc.name,
                        ))
                        yield f"*{tc.name} error: {e}*\n"

                # Continue loop to get next response
                continue

            # No tool calls and finished - we're done
            if response.finished:
                # Add final assistant message if we haven't already
                if not response.tool_calls:
                    self.messages.append(Message(
                        role="assistant",
                        content=response.content,
                    ))
                break

        logger.debug(
            "Agent finished",
            total_messages=len(self.messages),
        )

    async def stream_response(self, user_input: str) -> AsyncGenerator[str, None]:
        """
        Stream a response for a user message (without tool support).

        This is a simplified streaming method for responses that
        don't require tool calls.

        Args:
            user_input: The user's message.

        Yields:
            Response text chunks.
        """
        self.messages.append(Message(role="user", content=user_input))

        async for chunk in self.backend.stream(self.messages, []):
            yield chunk

    def clear_history(self) -> None:
        """Clear conversation history, keeping only the system prompt."""
        self.messages = [Message(role="system", content=SYSTEM_PROMPT)]

    def get_history(self) -> list[Message]:
        """Get the conversation history."""
        return self.messages.copy()
