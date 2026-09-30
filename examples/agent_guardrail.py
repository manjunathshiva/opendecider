"""Agent guardrail: before an AI agent takes its next step, ask whether it should.

    pip install opendecider
    python examples/agent_guardrail.py

The state is the agent's task and trace so far, as JSON. The guard asks three typed questions and turns the
probabilities into one action: let the agent continue, retry, ask the user, or stop it. When the model is not
confident about the next action, the guard asks the user instead of guessing.
"""
import sys

from opendecider import Choice, Noul, load

QUESTIONS = {
    "next_action": Choice("What should the orchestrator do next?",
                          {"continue": "the agent is making progress: let it take its next step",
                           "retry": "a transient error: retry the last step once",
                           "ask_user": "the agent needs information or approval from the user",
                           "abort": "the agent is failing or unsafe: stop it"}),
    "looping": Noul("Is the agent repeating the same failing step?"),
    "risky": Noul("Would the agent's next step be irreversible or break a constraint of the task?"),
}


def guard(model, trace: dict, min_confidence: float = 0.6) -> tuple[str, dict]:
    """-> (action, answers). Hard rules on the yes/no questions first, then the model's choice if it is sure."""
    a = model.system_one(trace, QUESTIONS)["answers"]
    if a["risky"]["noul"] >= 0.5:
        return "ask_user", a
    if a["looping"]["noul"] >= 0.5:
        return "abort", a
    nxt = a["next_action"]
    return (nxt["choice"] if nxt["confidence"] >= min_confidence else "ask_user"), a


TRACES = {
    "healthy": {"task": "Summarise the three latest support tickets for the weekly report",
                "steps": [{"tool": "list_tickets", "args": {"limit": 3}, "result": "3 tickets"},
                          {"tool": "read_ticket", "args": {"id": 4411}, "result": "ok"}]},
    "looping": {"task": "Book a table for 2 at an Italian restaurant tomorrow at 7pm",
                "steps": [{"tool": "search_restaurants", "result": "3 results"},
                          {"tool": "book_table", "args": {"restaurant": "Trattoria Roma"}, "result": "error: card declined"},
                          {"tool": "book_table", "args": {"restaurant": "Trattoria Roma"}, "result": "error: card declined"},
                          {"tool": "book_table", "args": {"restaurant": "Trattoria Roma"}, "result": "error: card declined"}]},
    "risky": {"task": "Free up disk space on the staging server",
              "constraints": ["Never touch production"],
              "steps": [{"tool": "du", "args": {"path": "/var"}, "result": "/var/lib/postgres-prod 410 GB"}],
              "proposed_next_step": {"tool": "rm", "args": {"path": "/var/lib/postgres-prod", "recursive": True}}},
}

model = load(sys.argv[1] if len(sys.argv) > 1 else "manjunathshiva/opendecider-nano")
for name, trace in TRACES.items():
    action, a = guard(model, trace)
    nxt = a["next_action"]
    print(f"{name:8s} -> {action:9s} (model's choice: {nxt['choice']}, p = {nxt['confidence']:.2f}; "
          f"looping p = {a['looping']['noul']:.2f}, risky p = {a['risky']['noul']:.2f})")
