from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from dotenv import load_dotenv

load_dotenv(".env")


@dataclass
class Context:
    user_role: str
    deployment_env: str


@dynamic_prompt
def context_aware_prompt(request: ModelRequest) -> str:
    # Read from Runtime Context: user role and environment
    user_role = request.runtime.context.user_role
    env = request.runtime.context.deployment_env

    base = "You are a helpful assistant."

    if user_role == "admin":
        base += "\nYou have admin access. You can perform all operations."
    elif user_role == "viewer":
        base += "\nYou have read-only access. Guide users to read operations only."

    if env == "production":
        base += "\nBe extra careful with any data modifications."

    return base


agent = create_agent(
    model="gpt-4o", tools=[], middleware=[context_aware_prompt], context_schema=Context
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is 2 + 2?"}]},
    context=Context(user_role="admin", deployment_env="production"),
)
print(response)
