"""Give a PydanticAI agent OpenDecider's decisions as tools: it routes a ticket with `choose` and checks churn risk
with `yes_no`, each answer with a calibrated probability.

    pip install "opendecider[pydantic-ai]"
    python examples/agent_frameworks/pydantic_ai_agent.py

`FunctionModel` stands in for your LLM so the script runs without an API key: it makes the tool calls a real model
would, including one invalid call, which comes back as a retry prompt saying what to fix. Use your model instead:
`Agent("openai:gpt-5", toolsets=[decision_toolset()])`. To route your own code or a graph's nodes, call
`DecisionRouter` directly.
"""
import sys

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from opendecider.integrations.pydantic_ai import decision_toolset

TICKET = "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
CALLS = [
    ToolCallPart("choose", {"state": TICKET, "question": "Which team?", "options": ["billing"]}),   # invalid
    ToolCallPart("choose", {"state": TICKET, "question": "Which department should handle this?",
                            "options": {"billing": "invoices, payments, refunds",
                                        "technical": "bugs, outages, system errors",
                                        "other": "everything else"}}),
    ToolCallPart("yes_no", {"state": TICKET, "question": "Does the customer threaten to leave?"}),
]


def llm(messages, info):
    """Prints what the agent got back from each tool, then makes the next call."""
    replies = [p for m in messages for p in getattr(m, "parts", []) if isinstance(p, (ToolReturnPart, RetryPromptPart))]
    if replies:
        last = replies[-1]
        print(f"{last.tool_name:<7} ->", last.content if isinstance(last, ToolReturnPart) else f"retry: {last.content}")
    if len(replies) < len(CALLS):
        return ModelResponse(parts=[CALLS[len(replies)]])
    return ModelResponse(parts=[TextPart("Routed to billing; churn risk flagged.")])


agent = Agent(FunctionModel(llm), toolsets=[decision_toolset(sys.argv[1] if len(sys.argv) > 1
                                                             else "manjunathshiva/opendecider-nano")])
print("agent:", agent.run_sync("Triage this ticket.").output)
