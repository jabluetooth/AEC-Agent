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

logger = structlog.get_logger(__name__)

# System prompt that defines the agent's behavior
SYSTEM_PROMPT = """You are an expert AEC (Architecture, Engineering, Construction) Specialist AI assistant.

Your primary role is to help users work with AutoCAD and Revit through natural language commands. You have access to specialized tools that allow you to:

**AutoCAD Operations:**
- Manage layers (create, list, toggle visibility/frozen state)
- Draw entities (lines, circles, rectangles)
- Query drawing information and entity counts

**Revit Operations:**
- Manage levels (create, list)
- Work with walls (create, list)
- Manage rooms (list, query by level)
- Get document status
- Delete elements

**Guidelines:**
1. Always explain what you're about to do before executing commands
2. Confirm destructive operations (like deleting elements) before proceeding
3. Provide clear feedback on the results of operations
4. If an operation fails, explain the error and suggest alternatives
5. Use appropriate units (typically feet for Revit, drawing units for AutoCAD)
6. When creating geometry, ask for dimensions if not specified

**Important Notes:**
- AutoCAD and Revit are single-threaded applications; operations run one at a time
- Complex operations may take several seconds to complete
- Always verify the target application (AutoCAD vs Revit) is running before operations

Be helpful, precise, and proactive in suggesting improvements to the user's workflow."""


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

        # Build request
        kwargs = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": anthropic_messages,
        }
        if system_content:
            kwargs["system"] = system_content
        if tools:
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


class AECAgent:
    """
    Main agent class that orchestrates LLM interactions and tool calls.

    Manages the conversation loop, tool execution, and message history.
    """

    def __init__(
        self,
        mcp_client: MCPClient,
        backend: Optional[LLMBackend] = None,
    ):
        """
        Initialize the AEC Agent.

        Args:
            mcp_client: Connected MCP client for tool calls.
            backend: LLM backend to use. Auto-detected from settings if not provided.
        """
        self.mcp_client = mcp_client
        self.backend = backend or self._create_backend()
        self.messages: list[Message] = [
            Message(role="system", content=SYSTEM_PROMPT)
        ]

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
        else:
            raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

    def _get_tools(self) -> list[dict[str, Any]]:
        """Get tools in the format expected by the current backend."""
        settings = get_settings()

        if settings.llm_provider == LLMProvider.ANTHROPIC:
            return self.mcp_client.get_tools_for_anthropic()
        else:
            # OpenAI and Azure use the same format
            return self.mcp_client.get_tools_for_openai()

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
        # Add user message
        self.messages.append(Message(role="user", content=user_input))

        tools = self._get_tools()
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
