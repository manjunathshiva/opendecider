"""Call OpenDecider's MCP server the way an AI assistant does: start `opendecider mcp` and use its tools over stdio.

    pip install "opendecider[mcp]"
    python examples/agent_frameworks/mcp_client.py

Claude Code, Claude Desktop and Cursor do exactly this once the server is registered
(`claude mcp add opendecider -- opendecider mcp`). This script shows what the assistant sees: the tools, their
answers with a probability for every option, and an invalid call coming back as a message the assistant can act on.
"""
import asyncio
import sys

from mcp import Client, StdioServerParameters

SERVER = StdioServerParameters(command=sys.executable, args=["-m", "opendecider.cli", "mcp", *sys.argv[1:]])
TICKET = "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."


async def main():
    async with Client(SERVER) as client:
        tools = (await client.list_tools()).tools
        print("tools:", ", ".join(t.name for t in tools))

        r = await client.call_tool("choose", {"state": TICKET, "question": "Which department should handle this?",
                                              "options": {"billing": "invoices, payments, refunds",
                                                          "technical": "bugs, outages, system errors",
                                                          "other": "everything else"}})
        print("choose:", r.structured_content)
        r = await client.call_tool("yes_no", {"state": TICKET, "question": "Does the customer threaten to leave?"})
        print("yes_no:", r.structured_content)
        r = await client.call_tool("score", {"state": TICKET, "question": "How urgent is this?",
                                             "levels": ["not urgent", "soon", "blocking"]})
        print("score: ", r.structured_content)

        r = await client.call_tool("choose", {"state": TICKET, "question": "Which team?", "options": ["billing"]})
        print("invalid call ->", r.content[0].text)   # the assistant reads this and fixes its call


asyncio.run(main())
