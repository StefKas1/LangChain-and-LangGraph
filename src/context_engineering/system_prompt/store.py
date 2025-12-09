from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langgraph.store.memory import InMemoryStore
from dotenv import load_dotenv

load_dotenv(".env")


@dataclass
class Context:
    user_id: str


@dynamic_prompt
def store_aware_prompt(request: ModelRequest) -> str:
    user_id = request.runtime.context.user_id

    # Read from Store: get user preferences
    store = request.runtime.store
    user_prefs = store.get(("preferences",), user_id)

    base = "You are a helpful assistant."

    if user_prefs:
        style = user_prefs.value.get("communication_style", "balanced")
        base += f"\nUser prefers {style} responses."

    return base


agent = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[store_aware_prompt],
    context_schema=Context,
    store=InMemoryStore(),
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is 2 + 2?"}]},
    context=Context(user_id="user-123"),
)
print(response)
