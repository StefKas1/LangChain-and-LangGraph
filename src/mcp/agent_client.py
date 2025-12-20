from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
import asyncio

# Might not work in Windows
client = MultiServerMCPClient(
    {
        "math": {
            "transport": "stdio",  # Local subprocess communication
            "command": "python",
            # Absolute path to your math_server.py file
            "args": [
                "/path/to/math_server.py"
            ],  # Client/agent will start the math server
        },
        "weather": {
            "transport": "http",  # HTTP-based remote server
            # Ensure you start your weather server on port 8000
            "url": "http://localhost:8000/mcp",  # Weather server must already be started
        },
    }
)


async def main() -> None:
    tools = await client.get_tools()
    agent = create_agent("gpt-5", tools)
    math_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
    )
    messages = math_response.get("messages", [])
    for message in messages:
        message.pretty_print()

    weather_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what is the weather in nyc?"}]}
    )
    messages = weather_response.get("messages", [])
    for message in messages:
        message.pretty_print()


if __name__ == "__main__":
    asyncio.run(main())
