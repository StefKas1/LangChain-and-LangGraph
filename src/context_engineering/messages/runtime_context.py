from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable
from dotenv import load_dotenv

load_dotenv(".env")


@dataclass
class Context:
    user_jurisdiction: str
    industry: str
    compliance_frameworks: list[str]


@wrap_model_call
def inject_compliance_rules(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Inject compliance constraints from Runtime Context."""
    # Read from Runtime Context: get compliance requirements
    jurisdiction = request.runtime.context.user_jurisdiction
    industry = request.runtime.context.industry
    frameworks = request.runtime.context.compliance_frameworks

    # Build compliance constraints
    rules = []
    if "GDPR" in frameworks:
        rules.append("- Must obtain explicit consent before processing personal data")
        rules.append("- Users have right to data deletion")
    if "HIPAA" in frameworks:
        rules.append("- Cannot share patient health information without authorization")
        rules.append("- Must use secure, encrypted communication")
    if industry == "finance":
        rules.append("- Cannot provide financial advice without proper disclaimers")

    if rules:
        compliance_context = f"""Compliance requirements for {jurisdiction}:
{chr(10).join(rules)}"""

        # Append at end - models pay more attention to final messages
        messages = [*request.messages, {"role": "user", "content": compliance_context}]
        request = request.override(messages=messages)

    return handler(request)


agent = create_agent(
    model="gpt-5",
    tools=[],
    middleware=[inject_compliance_rules],
    context_schema=Context,
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is 2 + 2?"}]},
    context=Context(
        user_jurisdiction="US", industry="finance", compliance_frameworks=["GDPR"]
    ),
)
print(response)
