"""Agno, CrewAI and Microsoft Agent Framework integrations, through each framework's own API, with a fake model.
Each section skips when its framework is not installed (CI runs each in its own environment)."""
import asyncio
import os

import pytest

from opendecider import OpenDecider
from opendecider.tools import Decider

for var, off in (("AGNO_TELEMETRY", "false"), ("CREWAI_DISABLE_TELEMETRY", "true"), ("CREWAI_TRACING_ENABLED", "false"),
                 ("OTEL_SDK_DISABLED", "true")):
    os.environ.setdefault(var, off)   # the frameworks report usage by default; tests send nothing


class Fake:
    """The option whose name appears in the state gets 0.8 (else the first option); records every call."""

    def __init__(self):
        self.calls = 0

    def decide_many(self, items, info=None):
        self.calls += 1
        out = []
        for state, _, opts in items:
            names = list(opts)
            top = next((n for n in names if n.lower() in str(state).lower()), names[0])
            out.append({n: 0.8 if n == top else 0.2 / (len(names) - 1) for n in names})
            if info is not None:
                info.append({"input_tokens": 10, "truncated": False})
        return out


def decider(fake=None):
    return Decider(OpenDecider(fake or Fake(), {"name": "opendecider-test", "kind": "nano"}))


ROUTES = {"billing": "charges, refunds", "tech": "bugs, outages"}


# ---- Agno ----------------------------------------------------------------------------------------------------------

def test_agno_toolkit():
    pytest.importorskip("agno")
    from opendecider.integrations.agno import decision_toolkit
    kit = decision_toolkit(decider())
    fns = kit.get_functions() if hasattr(kit, "get_functions") else kit.functions
    assert set(fns) == {"decide", "choose", "yes_no", "score"}
    assert "calibrated probability" in kit.instructions
    choose = fns["choose"]
    assert choose.entrypoint(state="refund my billing", question="Which team?",
                  options=["billing", "tech"])["choice"] == "billing"
    with pytest.raises(ValueError, match="at least 2 options"):
        choose.entrypoint(state="x", question="Which?", options=["only"])


def test_agno_workflow_router():
    pytest.importorskip("agno")
    from agno.workflow import Router, Step, StepOutput, Workflow
    from opendecider.integrations.agno import DecisionRouter

    def handler(name):
        return Step(name=name, executor=lambda step_input, name=name: StepOutput(content=name))

    steps = {name: handler(name) for name in ("billing", "tech", "human")}
    route = DecisionRouter(ROUTES, "Which team?", model=decider(), fallback="human", min_confidence=0.9)
    wf = Workflow(name="support", steps=[Router(name="triage", choices=list(steps.values()),
                                                selector=route.selector(steps))])
    out = wf.run(input="the tech dashboard is down")
    assert out.content == "human"   # 0.8 < 0.9: the fallback
    route.min_confidence = 0.5
    assert wf.run(input="the tech dashboard is down").content == "tech"
    with pytest.raises(ValueError, match="no step for route"):
        route.selector({"billing": steps["billing"]})
    first = DecisionRouter(ROUTES, "Which team?", model=decider(), state="previous")
    from agno.workflow.types import StepInput
    with pytest.raises(ValueError, match="no previous step content"):
        first.selector({**steps})(StepInput(input="x"))


# ---- CrewAI --------------------------------------------------------------------------------------------------------

def test_crewai_tools():
    pytest.importorskip("crewai")
    from opendecider.integrations.crewai import decision_tools
    tools = {t.name: t for t in decision_tools(decider())}
    assert set(tools) == {"decide", "choose", "yes_no", "score"}
    assert tools["choose"].run(state="refund my billing", question="Which team?",
                  options=["billing", "tech"])["choice"] == "billing"
    assert tools["score"].run(state="x", question="How bad?", levels=["low", "high"])["label"] in ("low", "high")
    assert set(tools["choose"].args_schema.model_fields) == {"state", "question", "options"}
    bad = tools["choose"].run(state="x", question="Which?", options=["only"])
    assert "at least 2 options" in bad["error"]   # returned to the agent, not raised
    if hasattr(tools["yes_no"], "arun"):   # async crews (CrewAI 1.0 has no async tool path)
        r = asyncio.run(tools["yes_no"].arun(state="refund my billing", question="Refund?"))
        assert r["answer"] in ("yes", "no")


def test_crewai_flow_router():
    pytest.importorskip("crewai")
    from crewai.flow.flow import Flow, listen, router, start
    from pydantic import BaseModel
    from opendecider.integrations.crewai import DecisionRouter

    route = DecisionRouter(ROUTES, "Which team?", model=decider(), fallback="human", min_confidence=0.5)

    class Ticket(BaseModel):
        text: str = ""
        handled_by: str = ""

    class Support(Flow[Ticket]):
        @start()
        def intake(self):
            pass

        @router(intake)
        def triage(self):
            return route(self.state.text)

        @listen("billing")
        def handle_billing(self):
            self.state.handled_by = "billing"

        @listen("tech")
        def handle_tech(self):
            self.state.handled_by = "tech"

    flow = Support()
    flow.kickoff(inputs={"text": "please refund my billing error"})
    assert flow.state.handled_by == "billing"


# ---- Microsoft Agent Framework -------------------------------------------------------------------------------------

def test_agent_framework_tools():
    pytest.importorskip("agent_framework")
    from opendecider.integrations.agent_framework import decision_tools
    tools = {t.name: t for t in decision_tools(decider())}
    assert set(tools) == {"decide", "choose", "yes_no", "score"}
    r = asyncio.run(tools["choose"].invoke(arguments={"state": "refund my billing", "question": "Which team?",
                                                     "options": ["billing", "tech"]}))
    text = "".join(getattr(c, "text", "") or "" for c in r)
    assert '"choice": "billing"' in text or "'choice': 'billing'" in text
    r = asyncio.run(tools["choose"].invoke(arguments={"state": "x", "question": "Which?", "options": ["only"]}))
    assert "at least 2 options" in "".join(getattr(c, "text", "") or "" for c in r)   # the model sees why


def test_agent_framework_switch_case_runs_the_model_once_per_message():
    pytest.importorskip("agent_framework")
    from agent_framework import WorkflowBuilder, WorkflowContext, executor
    from opendecider.integrations.agent_framework import DecisionRouter

    @executor(id="triage")
    async def triage(text: str, ctx: WorkflowContext[str]) -> None:
        await ctx.send_message(text)

    def team(name):
        @executor(id=name)
        async def handle(text: str, ctx: WorkflowContext[str, str]) -> None:
            await ctx.yield_output(name)
        return handle

    billing, tech, human = team("billing"), team("tech"), team("human")
    fake = Fake()
    route = DecisionRouter(ROUTES, "Which team?", model=decider(fake), fallback="human", min_confidence=0.5)
    wf = (WorkflowBuilder(start_executor=triage)
          .add_switch_case_edge_group(triage, route.cases({"billing": billing, "tech": tech}, default=human))
          .build())
    out = asyncio.run(wf.run("the tech dashboard is down"))
    assert out.get_outputs() == ["tech"]
    assert fake.calls == 1   # two cases, one forward pass
    with pytest.raises(ValueError, match="needs a default target"):
        DecisionRouter(ROUTES, "Which team?", model=decider()).cases({"billing": billing, "tech": tech})


def test_agent_framework_routes_on_an_agents_reply():
    pytest.importorskip("agent_framework")
    from agent_framework import AgentExecutorResponse, AgentResponse, Message
    from opendecider.integrations.agent_framework import DecisionRouter, _text

    reply = AgentExecutorResponse(executor_id="triage_agent",
                                  agent_response=AgentResponse(messages=[Message("assistant", ["a billing dispute"])]),
                                  full_conversation=[Message("user", ["long history that must not be routed on"])])
    assert _text(reply) == "a billing dispute"   # the agent's reply, not the whole conversation
    assert _text(Message("user", ["the tech stack is down"])) == "the tech stack is down"
    route = DecisionRouter(ROUTES, "Which team?", model=decider())
    assert route.pick(reply) == "billing"


def test_agent_framework_failed_decision_takes_the_default_once(caplog):
    pytest.importorskip("agent_framework")
    from opendecider.integrations.agent_framework import DecisionRouter

    class Broken(Fake):
        def decide_many(self, items, info=None):
            self.calls += 1
            raise RuntimeError("model gone")

    broken = Broken()
    route = DecisionRouter(ROUTES, "Which team?", model=decider(broken), fallback="human")
    cases = route.cases({"billing": "B", "tech": "T"}, default="H")
    message = "refund my billing"
    assert [c.condition(message) for c in cases[:-1]] == [False, False]   # no case matches: the Default takes it
    assert broken.calls == 1 and "routing failed" in caplog.text


# ---- Google ADK ----------------------------------------------------------------------------------------------------

def test_google_adk_tools():
    pytest.importorskip("google.adk")
    from opendecider.integrations.google_adk import decision_tools
    tools = {t.name: t for t in decision_tools(decider())}
    assert set(tools) == {"decide", "choose", "yes_no", "score"}
    assert tools["choose"].func(state="refund my billing", question="Which team?",
                  options=["billing", "tech"])["choice"] == "billing"
    assert "at least 2 options" in tools["choose"].func(state="x", question="Which?", options=["only"])["error"]
    assert tools["yes_no"].description.startswith("Answer a yes/no question")


def test_google_adk_router_agent_hands_over_to_a_sub_agent():
    pytest.importorskip("google.adk")
    from google.adk.agents import BaseAgent
    from google.adk.events import Event
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
    from opendecider.integrations.google_adk import DecisionRouterAgent

    class Team(BaseAgent):
        async def _run_async_impl(self, ctx):
            yield Event(author=self.name, content=types.Content(role="model", parts=[types.Part(text=self.name)]))

    def ask(router, text):
        sessions = InMemorySessionService()
        runner = Runner(agent=router, app_name="t", session_service=sessions)

        async def go():
            await sessions.create_session(app_name="t", user_id="u", session_id="s")
            msg = types.Content(role="user", parts=[types.Part(text=text)])
            events = runner.run_async(user_id="u", session_id="s", new_message=msg)
            return [e.content.parts[0].text async for e in events if e.content and e.content.parts]
        return asyncio.run(go())

    teams = [Team(name="billing"), Team(name="tech"), Team(name="human")]
    router = DecisionRouterAgent(name="triage", sub_agents=teams,
                                 routes=ROUTES, instructions="Which team?", fallback="human", min_confidence=0.5,
                                 decision_model=decider())
    assert ask(router, "please refund my billing error") == ["billing"]
    assert router.last.choice == "billing"
    with pytest.raises(ValueError, match="no sub-agent named"):
        DecisionRouterAgent(name="t2", sub_agents=[Team(name="billing")], routes=ROUTES, instructions="Which team?",
                            decision_model=decider())


# ---- PydanticAI ----------------------------------------------------------------------------------------------------

def test_pydantic_ai_toolset_in_an_agent():
    pytest.importorskip("pydantic_ai")
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart, RetryPromptPart
    from pydantic_ai.models.function import FunctionModel
    from opendecider.integrations.pydantic_ai import decision_toolset

    seen = []

    def llm(messages, info):   # behaves like a model: a bad call first, then a good one, then answers
        returned = [p for m in messages for p in getattr(m, "parts", [])
                    if isinstance(p, (ToolReturnPart, RetryPromptPart))]
        seen[:] = returned
        calls = len(returned)
        if calls == 0:
            bad = {"state": "x", "question": "Which?", "options": ["only"]}
            return ModelResponse(parts=[ToolCallPart("choose", bad)])
        if calls == 1:
            return ModelResponse(parts=[ToolCallPart("choose", {"state": "refund my billing", "question": "Which team?",
                                                                "options": ["billing", "tech"]})])
        return ModelResponse(parts=[TextPart("done")])

    agent = Agent(FunctionModel(llm), toolsets=[decision_toolset(decider())])
    assert agent.run_sync("route this").output == "done"
    retry = next(p for p in seen if isinstance(p, RetryPromptPart))
    assert "at least 2 options" in str(retry.content)   # ModelRetry: the model saw what to fix
    result = next(p for p in seen if isinstance(p, ToolReturnPart))
    assert result.content["choice"] == "billing"


# ---- Strands -------------------------------------------------------------------------------------------------------

def test_strands_tools():
    pytest.importorskip("strands")
    from opendecider.integrations.strands import decision_tools
    tools = {t.tool_name: t for t in decision_tools(decider())}
    assert set(tools) == {"decide", "choose", "yes_no", "score"}
    assert tools["choose"](state="refund my billing", question="Which team?",
                  options=["billing", "tech"])["choice"] == "billing"
    assert "at least 2 options" in tools["choose"](state="x", question="Which?", options=["only"])["error"]
    props = tools["choose"].tool_spec["inputSchema"]["json"]["properties"]
    assert set(props) == {"state", "question", "options"}


def test_an_empty_state_takes_the_fallback_without_a_decision():
    from opendecider.tools import Router
    fake = Fake()
    route = Router(ROUTES, "Which team?", model=decider(fake), fallback="human")
    assert [route(s) for s in ("", "  \n", {}, [], None)] == ["human"] * 5
    assert fake.calls == 0 and route.last.reason == "empty_input" and route.last.choice is None   # no guess
    with pytest.raises(ValueError, match="nothing to route on"):
        Router(ROUTES, "Which team?", model=decider())("")


def test_concurrent_routes_never_take_each_others_answer():
    """One router shared by concurrent requests (a server): a call routes on its own answer, even when another
    request overwrites `.last` between this call's decision and its return."""
    from opendecider.tools import Router

    class Racing(Router):
        def __setattr__(self, name, value):
            object.__setattr__(self, name, value)
            if name == "last" and value and value.choice == "billing" and not self.__dict__.get("raced"):
                self.raced = True
                self.route("the tech stack is down")   # another request lands mid-call and overwrites .last

    route = Racing(ROUTES, "Which team?", model=decider(), fallback="human", min_confidence=0.5)
    assert route("refund my billing") == "billing"
    assert route.last.choice == "tech"   # the race happened: .last is the other request's answer


def test_google_adk_on_error_fallback_hands_the_request_to_the_fallback_agent():
    pytest.importorskip("google.adk")
    from google.adk.agents import BaseAgent
    from google.adk.events import Event
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
    from opendecider.integrations.google_adk import DecisionRouterAgent

    class Broken(Fake):
        def decide_many(self, items, info=None):
            raise RuntimeError("model gone")

    class Team(BaseAgent):
        async def _run_async_impl(self, ctx):
            yield Event(author=self.name, content=types.Content(role="model", parts=[types.Part(text=self.name)]))

    seen = []
    router = DecisionRouterAgent(name="triage", sub_agents=[Team(name="billing"), Team(name="tech"),
                                                            Team(name="human")],
                                 routes=ROUTES, instructions="Which team?", fallback="human", on_error="fallback",
                                 on_decision=seen.append, decision_model=decider(Broken()))
    sessions = InMemorySessionService()
    runner = Runner(agent=router, app_name="t", session_service=sessions)

    async def go():
        await sessions.create_session(app_name="t", user_id="u", session_id="s")
        msg = types.Content(role="user", parts=[types.Part(text="refund my billing")])
        return [e.author async for e in runner.run_async(user_id="u", session_id="s", new_message=msg)]

    assert asyncio.run(go()) == ["human"]
    assert seen[0].reason == "error" and router.last is seen[0]


def test_agent_framework_reports_a_failed_decision_to_hooks():
    pytest.importorskip("agent_framework")
    from opendecider.integrations.agent_framework import DecisionRouter

    class Broken(Fake):
        def decide_many(self, items, info=None):
            raise RuntimeError("model gone")

    seen = []
    route = DecisionRouter(ROUTES, "Which team?", model=decider(Broken()), fallback="human", on_decision=seen.append)
    cases = route.cases({"billing": "B", "tech": "T"}, default="H")
    assert [c.condition("refund") for c in cases[:-1]] == [False, False]   # the Default takes it
    assert len(seen) == 1 and seen[0].reason == "error"                   # one decision for all the cases


def test_crewai_task_assigner_assigns_each_task_from_roles_and_goals():
    pytest.importorskip("crewai")
    from types import SimpleNamespace as NS
    from opendecider.integrations.crewai import TaskAssigner

    billing = NS(role="billing", goal="charges and refunds")
    tech = NS(role="tech", goal="bugs and outages")
    tech2 = NS(role="tech", goal="mobile apps")         # a duplicate role gets its own label
    lead = NS(role="team lead", goal="anything unclear")
    seen = []
    assigner = TaskAssigner([billing, tech, tech2], fallback=lead, min_confidence=0.5, model=decider(),
                            on_decision=seen.append)
    assert list(assigner.members) == ["billing", "tech", "tech (3)", "team lead"]
    t1 = NS(description="refund the billing error", expected_output="a refund", agent=None)
    t2 = NS(description="the tech stack is down", expected_output="a fix", agent=None)
    assert assigner.assign_all([t1, t2]) == [billing, tech] and t1.agent is billing and t2.agent is tech
    assert seen[0].route == "billing" and assigner.last.route == "tech"
    states = []

    class Recording(Fake):
        def decide_many(self, items, info=None):
            states.extend(state for state, _, _ in items)
            return super().decide_many(items, info)

    TaskAssigner([billing, tech], model=decider(Recording())).assign(t2)
    assert states == ["the tech stack is down\n\nExpected output: a fix"]   # the task's description and output
    unsure = TaskAssigner([billing, tech], fallback=lead, min_confidence=0.9, model=decider())
    assert unsure.assign("refund the billing error") is lead              # below 0.9: the fallback member
    assert asyncio.run(assigner.aassign({"description": "tech outage"})) is tech
    with pytest.raises(ValueError, match="at least 2 crew members"):
        TaskAssigner([billing], model=decider())


def test_crewai_task_assigner_never_lets_a_generated_label_overwrite_a_member():
    pytest.importorskip("crewai")
    from types import SimpleNamespace as NS
    from opendecider.integrations.crewai import TaskAssigner
    crew = [NS(role="tech", goal="a"), NS(role="tech (3)", goal="b"), NS(role="tech", goal="c")]
    members = TaskAssigner(crew, model=decider()).members
    assert list(members) == ["tech", "tech (3)", "tech (4)"] and list(members.values()) == crew


def test_crewai_task_assigner_sets_a_real_crewai_tasks_agent():
    pytest.importorskip("crewai")
    from crewai import Agent, Task
    from opendecider.integrations.crewai import TaskAssigner
    def build():
        try:
            return (Agent(role="billing", goal="charges and refunds", backstory="finance"),
                    Agent(role="tech", goal="bugs and outages", backstory="sre"))
        except Exception:   # noqa: BLE001 -- CrewAI 1.0 builds the agent's LLM eagerly and needs a key
            return None

    agents = build()
    if agents is None:
        pytest.skip("this CrewAI version needs an LLM key to build an Agent")
    billing, tech = agents
    task = Task(description="the tech stack is down", expected_output="a fix")
    assert TaskAssigner([billing, tech], model=decider()).assign(task) is tech and task.agent is tech


def test_unique_labels():
    from opendecider.tools import unique_labels
    assert unique_labels(["a", "b"]) == ["a", "b"]
    assert unique_labels(["tech", "tech (3)", "tech"]) == ["tech", "tech (3)", "tech (4)"]
    assert unique_labels(["x", "x", "x (2)"]) == ["x", "x (2)", "x (2) (3)"]
    assert len(set(unique_labels(["a"] * 50))) == 50


def test_agno_unreadable_step_input_takes_the_fallback_step():
    pytest.importorskip("agno")
    from agno.workflow.types import StepInput
    from opendecider.integrations.agno import DecisionRouter
    steps = {n: n for n in ("billing", "tech", "human")}
    route = DecisionRouter(ROUTES, "Which team?", model=decider(), fallback="human", on_error="fallback",
                           state="previous")
    assert route.selector(steps)(StepInput(input="x")) == ["human"]   # no previous step content
    assert route.last.reason == "error" and "no previous step content" in route.last.error
