from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable
from langgraph.store.memory import InMemoryStore
from dotenv import load_dotenv

load_dotenv(".env")


@dataclass
class Context:
    user_id: str


@wrap_model_call
def inject_writing_style(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Inject user's email writing style from Store."""
    user_id = request.runtime.context.user_id

    # Read from Store: get user's writing style examples
    store = request.runtime.store
    writing_style = store.get(("writing_style",), user_id)

    if writing_style:
        style = writing_style.value
        # Build style guide from stored examples
        style_context = f"""Your writing style:
- Tone: {style.get("tone", "professional")}
- Typical greeting: "{style.get("greeting", "Hi")}"
- Typical sign-off: "{style.get("sign_off", "Best")}"
- Example email you've written:
{style.get("example_email", "")}"""

        # Append at end - models pay more attention to final messages
        messages = [*request.messages, {"role": "user", "content": style_context}]
        request = request.override(messages=messages)

    return handler(request)


agent = create_agent(
    model="gpt-5",
    tools=[],
    middleware=[inject_writing_style],
    context_schema=Context,
    store=InMemoryStore(),
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is 2 + 2?"}]},
    context=Context(user_id="user-123"),
)
print(response)
