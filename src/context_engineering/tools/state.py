from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv(".env")


@tool
def public_search() -> str:
    """Public search."""
    pass


@tool
def private_search() -> str:
    """Private search."""
    pass


@tool
def advanced_search() -> str:
    """Advanced search."""
    pass


@wrap_model_call
def state_based_tools(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Filter tools based on conversation State."""
    # Read from State: check if user has authenticated
    state = request.state
    is_authenticated = state.get("authenticated", False)
    message_count = len(state["messages"])

    # Only enable sensitive tools after authentication
    if not is_authenticated:
        tools = [t for t in request.tools if t.name.startswith("public_")]
        request = request.override(tools=tools)
    elif message_count < 5:
        # Limit tools early in conversation
        tools = [t for t in request.tools if t.name != "advanced_search"]
        request = request.override(tools=tools)

    return handler(request)


agent = create_agent(
    model="gpt-4o",
    tools=[public_search, private_search, advanced_search],
    middleware=[state_based_tools],
)

response = agent.invoke({"messages": [{"role": "user", "content": "what is 2 + 2?"}]})
print(response)
