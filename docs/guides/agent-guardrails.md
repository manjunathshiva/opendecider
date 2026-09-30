# Agent guardrails

Before an AI agent takes its next step, ask a decision model whether it should. The state is the agent's task and
trace so far, as JSON; typed questions turn the probabilities into one action: let the agent continue, retry, ask the
user, or stop it. A decision takes milliseconds with opendecider-nano, so it can run before every step.

```python
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


model = load("manjunathshiva/opendecider-nano")
trace = {"task": "Free up disk space on the staging server",
         "constraints": ["Never touch production"],
         "steps": [{"tool": "du", "args": {"path": "/var"}, "result": "/var/lib/postgres-prod 410 GB"}],
         "proposed_next_step": {"tool": "rm", "args": {"path": "/var/lib/postgres-prod", "recursive": True}}}
action, answers = guard(model, trace)   # "ask_user": the step is risky
```

The full script, with a healthy, a looping and a risky trace, is
[examples/agent_guardrail.py](https://github.com/manjunathshiva/opendecider/blob/main/examples/agent_guardrail.py).
On those three traces nano answers `continue`, `abort` and `ask_user`.

## Design notes

- **Rules on top of probabilities.** The yes/no questions (`looping`, `risky`) act as hard stops; the multi-way choice
  is only followed when the model is confident. When it is not, the guard asks the user instead of guessing.
- **Describe the options.** "the agent is failing or unsafe: stop it" works better than a bare `abort`.
- **Keep the trace compact.** nano reads up to 2,048 tokens and truncates the state first (the answer is then marked
  `truncated`); summarise long tool outputs before asking.
- **Tune on your own traces.** Log the probabilities alongside what a person decided, and set `min_confidence` from
  that; see [Automate the confident decisions](confident-automation.md).
