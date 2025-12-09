from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv(".env")


@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


if __name__ == "__main__":
    agent = create_agent(
        model="gpt-4o",
        tools=[get_weather],
        system_prompt="You are a helpful assistant",
    )

    # Run the agent
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
    )

    print(response)
