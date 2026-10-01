"""Agent-framework integrations, one module per framework, each behind its own extra:

    langchain (LangChain, LangGraph), llamaindex, agno, crewai, agent_framework (Microsoft Agent Framework),
    google_adk, pydantic_ai, strands

All use the shared tool core in `opendecider.tools`, so their answers match the MCP server's. TypeScript frameworks such
as Mastra use the MCP server (`opendecider mcp`).
"""
