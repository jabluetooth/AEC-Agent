"""
Chainlit Application for AEC Agent.

Provides the user-facing chat interface with real-time tool execution display.
"""

import chainlit as cl
import structlog

from aec_agent.config.settings import get_settings
from aec_agent.frontend.mcp_client import MCPClient
from aec_agent.frontend.agent import AECAgent

logger = structlog.get_logger(__name__)


@cl.on_chat_start
async def on_chat_start():
    """
    Initialize the chat session.

    Called when a new user connects or starts a new conversation.
    """
    settings = get_settings()

    # Send welcome message
    await cl.Message(
        content=(
            "Welcome to **AEC Agent**!\n\n"
            "I'm your AI assistant for AutoCAD and Revit automation. "
            "I can help you:\n\n"
            "**AutoCAD:**\n"
            "- Create and manage layers\n"
            "- Draw lines, circles, and rectangles\n"
            "- Query drawing information\n\n"
            "**Revit:**\n"
            "- Create and manage levels\n"
            "- Draw walls\n"
            "- Query rooms and elements\n\n"
            "How can I help you today?"
        ),
    ).send()

    # Initialize MCP client
    mcp_client = MCPClient()
    logger.info("MCP client created", base_url=mcp_client.base_url)

    try:
        await mcp_client.connect()
        logger.info("MCP client connected", tool_count=len(mcp_client.get_tools()))

        # Check server health
        if await mcp_client.ping():
            await cl.Message(
                content="*Connected to MCP server successfully.*",
                author="System",
            ).send()
        else:
            await cl.Message(
                content="*Warning: MCP server ping failed. Some features may not work.*",
                author="System",
            ).send()

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error("Failed to connect to MCP server", error=str(e), traceback=error_details)
        await cl.Message(
            content=(
                f"*Error: Could not connect to MCP server: {e}*\n\n"
                "Please ensure the MCP server is running.\n\n"
                f"Details: Check the terminal for full error logs."
            ),
            author="System",
        ).send()
        # Store client anyway for retry
        cl.user_session.set("mcp_client", mcp_client)
        return

    # Initialize agent
    try:
        agent = AECAgent(mcp_client=mcp_client)
        # Store in session
        cl.user_session.set("mcp_client", mcp_client)
        cl.user_session.set("agent", agent)
    except Exception as e:
        logger.error("Failed to initialize agent", error=str(e))
        await cl.Message(
            content=f"*Error: Failed to initialize AI agent: {e}*\n\nPlease checks your settings and API keys.",
            author="System",
        ).send()
        return

    # Show available tools
    tools = mcp_client.get_tools()
    if tools:
        tool_list = "\n".join([f"- `{t.name}`: {t.description}" for t in tools[:10]])
        if len(tools) > 10:
            tool_list += f"\n- *...and {len(tools) - 10} more tools*"

        await cl.Message(
            content=f"**Available Tools ({len(tools)}):**\n{tool_list}",
            author="System",
        ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handle incoming user messages.

    Processes the message through the agent and streams back the response.
    """
    agent: AECAgent = cl.user_session.get("agent")

    if not agent:
        await cl.Message(
            content="*Error: Agent not initialized. Please refresh the page.*",
        ).send()
        return

    # Create response message for streaming
    response_msg = cl.Message(content="")
    await response_msg.send()

    # Track tool executions with Steps
    full_response = ""

    try:
        async for chunk in agent.process_message(message.content):
            # Check if this is a tool execution marker
            if chunk.startswith("\n\n*Executing "):
                # Extract tool name
                tool_name = chunk.replace("\n\n*Executing ", "").replace("...*\n", "")

                # Create a step for the tool execution
                async with cl.Step(name=tool_name, type="tool") as step:
                    step.input = f"Executing {tool_name}"

                    # Get next chunk which should be the result
                    # For now, just continue streaming
                    full_response += chunk
                    await response_msg.stream_token(chunk)

            elif chunk.startswith("*") and chunk.endswith("*\n"):
                # This is a tool result marker - update the step
                full_response += chunk
                await response_msg.stream_token(chunk)

            else:
                # Regular response text
                full_response += chunk
                await response_msg.stream_token(chunk)

        # Update final message
        response_msg.content = full_response
        await response_msg.update()

    except Exception as e:
        logger.error("Error processing message", error=str(e))
        await cl.Message(
            content=f"*Error processing message: {e}*",
        ).send()


@cl.on_chat_end
async def on_chat_end():
    """
    Clean up when the chat session ends.
    """
    mcp_client: MCPClient = cl.user_session.get("mcp_client")

    if mcp_client:
        try:
            await mcp_client.close()
            logger.info("MCP client closed")
        except Exception as e:
            logger.error("Error closing MCP client", error=str(e))


@cl.on_stop
async def on_stop():
    """
    Handle user stopping the current generation.
    """
    logger.info("User stopped generation")


# Action handlers for common operations
@cl.action_callback("clear_history")
async def on_clear_history(action: cl.Action):
    """Clear the conversation history."""
    agent: AECAgent = cl.user_session.get("agent")

    if agent:
        agent.clear_history()
        await cl.Message(
            content="*Conversation history cleared.*",
            author="System",
        ).send()


@cl.action_callback("check_sidecar")
async def on_check_sidecar(action: cl.Action):
    """Check sidecar connection status."""
    mcp_client: MCPClient = cl.user_session.get("mcp_client")

    if not mcp_client:
        await cl.Message(
            content="*MCP client not connected.*",
            author="System",
        ).send()
        return

    # Check AutoCAD sidecar
    autocad_result = await mcp_client.call_tool(
        "check_sidecar",
        {"sidecar_type": "autocad"},
    )

    # Check Revit sidecar
    revit_result = await mcp_client.call_tool(
        "check_sidecar",
        {"sidecar_type": "revit"},
    )

    status_msg = "**Sidecar Status:**\n"
    status_msg += f"- AutoCAD: {'Connected' if autocad_result.success else 'Not connected'}\n"
    status_msg += f"- Revit: {'Connected' if revit_result.success else 'Not connected'}\n"

    await cl.Message(
        content=status_msg,
        author="System",
    ).send()


# Settings panel
@cl.set_chat_profiles
async def chat_profiles():
    """Define chat profiles for different LLM providers."""
    return [
        cl.ChatProfile(
            name="Default",
            markdown_description="Use the default LLM provider configured in settings.",
            icon="https://cdn.jsdelivr.net/npm/heroicons@2.0.18/24/outline/cpu-chip.svg",
        ),
    ]


# Header actions
@cl.on_settings_update
async def settings_update(settings: dict):
    """Handle settings updates from the UI."""
    logger.info("Settings updated", settings=settings)


# Custom CSS for the UI (via chainlit.md config)
# The actual styling is controlled by Chainlit's configuration
