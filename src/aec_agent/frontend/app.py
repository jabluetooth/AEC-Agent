"""
Chainlit Application for AEC Agent.

Provides the user-facing chat interface with real-time tool execution display.
"""

import chainlit as cl
from chainlit.input_widget import Select
import structlog

from aec_agent.config.settings import get_settings
from aec_agent.frontend.mcp_client import MCPClient
from aec_agent.frontend.agent import AECAgent

logger = structlog.get_logger(__name__)

# App selector options
APP_OPTIONS = {
    "both": "Both (AutoCAD + Revit)",
    "autocad": "AutoCAD Only",
    "revit": "Revit Only",
}

# Selection indicators with emojis for visual feedback
APP_INDICATORS = {
    "both": "🔗 Both (AutoCAD + Revit)",
    "autocad": "🔷 AutoCAD Only",
    "revit": "🏠 Revit Only",
}


def _get_selection_indicator(app_context: str) -> str:
    """Get a visual indicator for the current app selection."""
    return APP_INDICATORS.get(app_context, APP_OPTIONS.get(app_context, app_context))


@cl.on_chat_start
async def on_chat_start():
    """
    Initialize the chat session.

    Called when a new user connects or starts a new conversation.
    """
    settings = get_settings()

    # Set up chat settings with app selector
    chat_settings = await cl.ChatSettings(
        [
            Select(
                id="app_context",
                label="Application",
                description="Select which CAD application to use",
                values=list(APP_OPTIONS.keys()),
                initial_value="both",
            ),
        ]
    ).send()

    # Store initial app context
    app_context = chat_settings.get("app_context", "both")
    cl.user_session.set("app_context", app_context)

    # Create app selection action buttons
    actions = [
        cl.Action(
            name="select_autocad",
            payload={"app": "autocad"},
            label="🔷 AutoCAD",
            description="Use AutoCAD tools only (saves tokens)",
        ),
        cl.Action(
            name="select_revit",
            payload={"app": "revit"},
            label="🏠 Revit",
            description="Use Revit tools only (saves tokens)",
        ),
        cl.Action(
            name="select_both",
            payload={"app": "both"},
            label="🔗 Both",
            description="Use all tools (AutoCAD + Revit)",
        ),
    ]

    # Get current selection indicator
    selection_indicator = _get_selection_indicator(app_context)

    # Send welcome message with action buttons
    await cl.Message(
        content=(
            "Welcome to **AEC Agent**!\n\n"
            "I'm your AI assistant for AutoCAD and Revit automation.\n\n"
            f"**Current Mode:** {selection_indicator}\n\n"
            "**Select your application to optimize token usage:**"
        ),
        actions=actions,
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
                content=f"*Connected to MCP server. App: **{APP_OPTIONS[app_context]}***",
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

    # Initialize agent with app context
    try:
        agent = AECAgent(mcp_client=mcp_client, app_context=app_context)
        # Store in session
        cl.user_session.set("mcp_client", mcp_client)
        cl.user_session.set("agent", agent)
    except Exception as e:
        logger.error("Failed to initialize agent", error=str(e))
        await cl.Message(
            content=f"*Error: Failed to initialize AI agent: {e}*\n\nPlease check your settings and API keys.",
            author="System",
        ).send()
        return

    # Show available tools for selected context
    await _show_tools_for_context(mcp_client, app_context)


async def _show_tools_for_context(mcp_client: MCPClient, app_context: str):
    """Show tools available for the selected app context."""
    filter_prefix = None
    if app_context == "autocad":
        filter_prefix = "autocad_"
    elif app_context == "revit":
        filter_prefix = "revit_"

    tools = mcp_client.get_tools(filter_prefix=filter_prefix)
    if tools:
        tool_list = "\n".join([f"- `{t.name}`" for t in tools[:10]])
        if len(tools) > 10:
            tool_list += f"\n- *...and {len(tools) - 10} more*"

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
        import traceback
        logger.error("Error processing message", error=str(e), traceback=traceback.format_exc())
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


@cl.action_callback("select_autocad")
@cl.action_callback("select_revit")
@cl.action_callback("select_both")
async def on_select_app(action: cl.Action):
    """Handle app selection button clicks."""
    new_context = action.payload.get("app", "both")
    old_context = cl.user_session.get("app_context")

    if new_context == old_context:
        await cl.Message(
            content=f"*Already using {_get_selection_indicator(new_context)}*",
            author="System",
        ).send()
        return

    # Update session
    cl.user_session.set("app_context", new_context)

    # Update agent's app context
    agent: AECAgent = cl.user_session.get("agent")
    if agent:
        agent.set_app_context(new_context)

    # Create new action buttons with updated selection
    actions = [
        cl.Action(
            name="select_autocad",
            payload={"app": "autocad"},
            label="🔷 AutoCAD" + (" ✓" if new_context == "autocad" else ""),
            description="Use AutoCAD tools only (saves tokens)",
        ),
        cl.Action(
            name="select_revit",
            payload={"app": "revit"},
            label="🏠 Revit" + (" ✓" if new_context == "revit" else ""),
            description="Use Revit tools only (saves tokens)",
        ),
        cl.Action(
            name="select_both",
            payload={"app": "both"},
            label="🔗 Both" + (" ✓" if new_context == "both" else ""),
            description="Use all tools (AutoCAD + Revit)",
        ),
    ]

    # Notify user with new buttons
    await cl.Message(
        content=f"**Switched to {_get_selection_indicator(new_context)}**\n\nTools filtered accordingly. This reduces token usage for your LLM calls.",
        author="System",
        actions=actions,
    ).send()

    # Show new available tools
    mcp_client: MCPClient = cl.user_session.get("mcp_client")
    if mcp_client:
        await _show_tools_for_context(mcp_client, new_context)

    logger.info("App context changed via button", old=old_context, new=new_context)


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

    # Update app context if changed
    new_context = settings.get("app_context")
    old_context = cl.user_session.get("app_context")

    if new_context and new_context != old_context:
        cl.user_session.set("app_context", new_context)

        # Update agent's app context
        agent: AECAgent = cl.user_session.get("agent")
        if agent:
            agent.set_app_context(new_context)

        # Notify user
        await cl.Message(
            content=f"*Switched to **{APP_OPTIONS.get(new_context, new_context)}**. Tools filtered accordingly.*",
            author="System",
        ).send()

        # Show new available tools
        mcp_client: MCPClient = cl.user_session.get("mcp_client")
        if mcp_client:
            await _show_tools_for_context(mcp_client, new_context)


# Custom CSS for the UI (via chainlit.md config)
# The actual styling is controlled by Chainlit's configuration
