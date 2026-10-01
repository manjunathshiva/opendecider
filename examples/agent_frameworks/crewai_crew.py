"""Assign a CrewAI crew's tasks with OpenDecider: each task goes to the crew member whose role and goal fit it, unclear
ones to the team lead, with no manager LLM.

    pip install "opendecider[crewai]"
    python examples/agent_frameworks/crewai_crew.py

`TaskAssigner` reads each task's description and expected output and picks a member in one forward pass. When the top
member's probability is below `min_confidence`, the task goes to the fallback member instead. It sets each task's
`agent`, so the crew then runs as a sequential crew. This script stops before `kickoff()`, so it runs without an API
key: the agents' LLMs are only needed to do the work.
"""
import os
import sys

# CrewAI 1.0 and 1.1 build each agent's LLM as the agent is created, and fail without a key. This script never calls
# the LLM (assignment needs none), so a placeholder is enough; set your real key to kickoff() the crew.
os.environ.setdefault("OPENAI_API_KEY", "not-used-to-assign-tasks")

from crewai import Agent, Task  # noqa: E402

from opendecider.integrations.crewai import TaskAssigner  # noqa: E402

billing = Agent(role="Billing specialist", goal="Resolve invoices, duplicate charges and refunds",
                backstory="Five years in finance operations.")
engineer = Agent(role="Support engineer", goal="Diagnose and fix bugs, outages and API errors",
                 backstory="Former site reliability engineer.")
sales = Agent(role="Account executive", goal="Prepare pricing, quotes and new contracts",
              backstory="Closes enterprise deals.")
lead = Agent(role="Team lead", goal="Handle anything unclear or out of scope", backstory="Runs the support team.")

tasks = [
    Task(description="A customer was charged twice for the March invoice.", expected_output="A refund decision"),
    Task(description="The API has returned 500 errors since 9am for every request.",
         expected_output="The root cause and a fix"),
    Task(description="Prepare a quote for 200 enterprise seats.", expected_output="A price quote"),
    Task(description="Look into this.", expected_output="Something"),
]

assigner = TaskAssigner([billing, engineer, sales], fallback=lead, min_confidence=0.6,
                        model=sys.argv[1] if len(sys.argv) > 1 else "manjunathshiva/opendecider-nano")
for task in tasks:
    assigner.assign(task)
    d = assigner.last
    print(f"{task.description[:56]:<56} -> {task.agent.role:<18} (top: {d.choice}, p = {d.confidence:.2f})")

# Crew(agents=[billing, engineer, sales, lead], tasks=tasks, process=Process.sequential).kickoff()
