# Agent frameworks

OpenDecider plugs into agent frameworks in two ways: as **tools** an agent calls (`decide`, `choose`, `yes_no`,
`score`, the same as the [MCP server](mcp.md)), and as a **router** that picks the next step without an LLM call. A
routing decision with opendecider-nano takes milliseconds and comes with a calibrated probability, so low-confidence
cases can go to a fallback (a person, a slower model) instead of the wrong branch.

!!! note "Not on PyPI yet"
    These integrations ship in opendecider 0.4.0, which is not released yet. Until then, install from GitHub:
    `pip install "opendecider[agno] @ git+https://github.com/manjunathshiva/opendecider"` (any extra in place of
    `agno`). The `>=0.4.0` install lines below work once 0.4.0 is on PyPI.

| framework | install | tools | router | example |
|---|---|---|---|---|
| [LangGraph](#langgraph-route-on-confidence) | `opendecider[langchain]` + `langgraph` | `decision_tools()` | `DecisionRouter`: a conditional edge | [langgraph_router.py](https://github.com/manjunathshiva/opendecider/blob/main/examples/agent_frameworks/langgraph_router.py) |
| [LangChain](#langchain-tools-for-an-agent) | `opendecider[langchain]` | `decision_tools()` | (use LangGraph) | |
| [LlamaIndex](#llamaindex-a-selector-for-routerqueryengine) | `opendecider[llamaindex]` | `decision_tools()` | `DecisionSelector`: a RouterQueryEngine selector | [llamaindex_selector.py](https://github.com/manjunathshiva/opendecider/blob/main/examples/agent_frameworks/llamaindex_selector.py) |
| [Agno](#agno-a-workflow-router-and-a-toolkit) | `opendecider[agno]` | `decision_toolkit()` | `DecisionRouter.selector()`: a workflow Router's selector | [agno_workflow.py](https://github.com/manjunathshiva/opendecider/blob/main/examples/agent_frameworks/agno_workflow.py) |
| [CrewAI](#crewai-a-flow-router-task-assignment-and-crew-tools) | `opendecider[crewai]` | `decision_tools()` | `DecisionRouter`: returns a Flow `@router` label; `TaskAssigner`: picks the crew member for each task | [crewai_flow.py](https://github.com/manjunathshiva/opendecider/blob/main/examples/agent_frameworks/crewai_flow.py), [crewai_crew.py](https://github.com/manjunathshiva/opendecider/blob/main/examples/agent_frameworks/crewai_crew.py) |
| [Microsoft Agent Framework](#microsoft-agent-framework-a-switch-case-edge) | `opendecider[agent-framework]` | `decision_tools()` | `DecisionRouter.cases()`: a switch-case edge group | [agent_framework_workflow.py](https://github.com/manjunathshiva/opendecider/blob/main/examples/agent_frameworks/agent_framework_workflow.py) |
| [Google ADK](#google-adk-a-router-agent) | `opendecider[google-adk]` | `decision_tools()` | `DecisionRouterAgent`: hands over to a sub-agent | [google_adk_router.py](https://github.com/manjunathshiva/opendecider/blob/main/examples/agent_frameworks/google_adk_router.py) |
| [PydanticAI](#pydanticai-a-toolset) | `opendecider[pydantic-ai]` | `decision_toolset()` | `DecisionRouter`: call it from your code | [pydantic_ai_agent.py](https://github.com/manjunathshiva/opendecider/blob/main/examples/agent_frameworks/pydantic_ai_agent.py) |
| [Strands Agents](#strands-agents-tools-and-a-router) | `opendecider[strands]` | `decision_tools()` | `DecisionRouter`: call it from your code | [strands_agent.py](https://github.com/manjunathshiva/opendecider/blob/main/examples/agent_frameworks/strands_agent.py) |
| [Mastra](#mastra-typescript-through-mcp) (TypeScript) | `opendecider[mcp]` + `@mastra/mcp` | the MCP server's tools | (route with the `choose` tool) | [mastra/index.mjs](https://github.com/manjunathshiva/opendecider/blob/main/examples/agent_frameworks/mastra/index.mjs) |

Every example runs on a laptop with opendecider-nano, without an LLM API key, and CI runs each of them against the
released model. The routing examples (LangGraph, Agno, CrewAI, Agent Framework, Google ADK, Strands) route the same
four support tickets, and every framework routes them the same way, since the decision is the model's:

| ticket | routed to | top route, probability |
|---|---|---|
| "Our API has returned 500 errors since 9am and every request fails." | `tech_support` | tech_support, 0.94 |
| "I was charged twice for the March invoice. Please refund the duplicate." | `billing_agent` | billing_agent, 0.94 |
| "Can I get a quote for the enterprise plan for 200 seats?" | `sales_agent` | sales_agent, 0.91 |
| "Hmm, not sure, something feels off." | `human_agent` (fallback, `min_confidence=0.6`) | tech_support, 0.51 |

Every component loads its model on the first call: opendecider-nano by default, or any model name you pass
(`manjunathshiva/opendecider-small-td`, or a served model such as
`ollama:hf.co/manjunathshiva/opendecider-small-td-GGUF:Q8_0`). Components that name the same model share one loaded
copy. To set loading options such as the device, pass a `Decider`:

```python
from opendecider.tools import Decider
nano_cpu = Decider("manjunathshiva/opendecider-nano", device="cpu")
```

Every `DecisionRouter` takes the same arguments: `routes` (names, or {"name": "when to take it"}), the routing
question, `model`, `fallback`, `min_confidence`, `on_error` and `on_decision` (Google ADK's `DecisionRouterAgent`
takes them as fields, with `instructions=` for the question and `decision_model=` for the model). `.last` holds the
last [`Decision`](#production), for logging; under concurrent calls it is whichever call finished last, while each
call still routes on its own answer. An empty input (blank text, `{}` or `[]`, such as an image-only message) takes the
fallback without asking the model, which would otherwise guess; without a fallback it raises `ValueError`.

## LangGraph: route on confidence

```bash
pip install "opendecider[langchain]>=0.4.0" langgraph
```

```python
from typing import TypedDict
from langgraph.graph import END, StateGraph
from opendecider.integrations.langchain import DecisionRouter

class State(TypedDict):
    input: str
    handled_by: str

route = DecisionRouter(
    {"billing_agent": "invoices, payment methods, duplicate charges, refunds",
     "tech_support": "system errors, bugs, API downtime, stack traces",
     "sales_agent": "pricing plans, new contracts, demo requests"},
    "Which specialist agent should answer this user query?",
    state_key="input", fallback="human_agent", min_confidence=0.6)

graph = StateGraph(State)
graph.add_node("triage", lambda s: {})
for node in route.path_map:                       # the routes, plus the fallback
    graph.add_node(node, lambda s, node=node: {"handled_by": node})
    graph.add_edge(node, END)
graph.set_entry_point("triage")
graph.add_conditional_edges("triage", route, route.path_map)
app = graph.compile()

app.invoke({"input": "Our API has returned 500 errors since 9am and every request fails."})
```

`DecisionRouter` reads `state[state_key]`; without `state_key` it reads the last message's text when the state has
`messages` (as in `MessagesState`), otherwise the whole state. `router.last` holds the last answer, for logging. Pick
`min_confidence` from your own labelled examples; see [Automate the confident decisions](confident-automation.md).

The whole graph, runnable on a laptop: [examples/agent_frameworks/langgraph_router.py](https://github.com/manjunathshiva/opendecider/blob/main/examples/agent_frameworks/langgraph_router.py).

## LangChain: tools for an agent

```python
from opendecider.integrations.langchain import decision_tools

tools = decision_tools()     # [decide, choose, yes_no, score]; pass them to any tool-calling agent
tools[1].invoke({"state": "Hi, we were billed twice for March. Please refund the duplicate today.",
                 "question": "Which department should handle this?",
                 "options": {"billing": "invoices, payments, refunds", "technical": "bugs, outages", "other": "anything else"}})
# {'choice': 'billing', 'probabilities': {...}, 'confidence': ...}
```

Invalid input (an unknown question type, a single option, repeated score levels) and a model that cannot load come back
to the agent as the tool's message, so it can correct the call. Async agents get an async tool that runs inference off
the event loop.

## LlamaIndex: a selector for RouterQueryEngine

```bash
pip install "opendecider[llamaindex]>=0.4.0"
```

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from opendecider.integrations.llamaindex import DecisionSelector

engine = RouterQueryEngine(
    selector=DecisionSelector(),
    query_engine_tools=[
        QueryEngineTool(query_engine=sales_engine,
                        metadata=ToolMetadata(name="sales_db", description="SQL database of orders, revenue and customers")),
        QueryEngineTool(query_engine=docs_engine,
                        metadata=ToolMetadata(name="product_docs", description="product manuals, setup guides and FAQs")),
    ])
engine.query("What was our revenue in March?")
```

`DecisionSelector` picks one engine from the tools' names and descriptions, in place of an LLM selector. Each
description is cut to 400 characters (`max_description_chars`), so long descriptions cannot push the query out of
the model's input. With
opendecider-nano, "What was our revenue in March?" goes to `sales_db` (0.94) and "How do I reset the device to factory
settings?" to `product_docs` (0.95). The selection's `reason` carries the probability, so it shows in LlamaIndex traces.

Runnable without data or an API key: [examples/agent_frameworks/llamaindex_selector.py](https://github.com/manjunathshiva/opendecider/blob/main/examples/agent_frameworks/llamaindex_selector.py).

The same four tools are available as `FunctionTool`s for LlamaIndex agents:

```python
from opendecider.integrations.llamaindex import decision_tools
tools = decision_tools()
```

## Agno: a workflow Router and a toolkit

```bash
pip install "opendecider[agno]>=0.4.0"
```

```python
from agno.agent import Agent
from agno.workflow import Router, Step, Workflow
from opendecider.integrations.agno import DecisionRouter, decision_toolkit

route = DecisionRouter(
    {"billing_agent": "invoices, payment methods, duplicate charges, refunds",
     "tech_support": "system errors, bugs, API downtime, stack traces",
     "sales_agent": "pricing plans, new contracts, demo requests"},
    "Which specialist agent should answer this user query?",
    fallback="human_agent", min_confidence=0.6)

agents = {"billing_agent": billing_agent, "tech_support": tech_agent,     # your Agno agents (or teams)
          "sales_agent": sales_agent, "human_agent": human_agent}
steps = {name: Step(name=name, agent=agent) for name, agent in agents.items()}
workflow = Workflow(name="support", steps=[
    Router(name="triage", choices=list(steps.values()), selector=route.selector(steps))])
workflow.run(input="Our API has returned 500 errors since 9am and every request fails.")

agent = Agent(model=..., tools=[decision_toolkit()])     # or: the decisions as tools, with instructions on confidence
```

The selector decides on the workflow's input; `DecisionRouter(..., state="previous")` decides on the previous step's
output instead, and `state=` also takes a function of the `StepInput`. `selector()` checks that every route (and the
fallback) has a step. Works with Agno 2.x and 3.x.

## CrewAI: a Flow router, task assignment and crew tools

```bash
pip install "opendecider[crewai]>=0.4.0"
```

```python
from crewai import Agent
from crewai.flow.flow import Flow, listen, router, start
from opendecider.integrations.crewai import DecisionRouter, decision_tools

route = DecisionRouter({...}, "Which specialist agent should answer this user query?",
                       fallback="human_agent", min_confidence=0.6)     # the same routes as above

class Support(Flow[Ticket]):
    @start()
    def intake(self): ...

    @router(intake)
    def triage(self):
        return route(self.state.ticket)        # the label a @listen method waits for

    @listen("billing_agent")
    def billing(self):
        BillingCrew().crew().kickoff(inputs={"ticket": self.state.ticket})
    ...

agent = Agent(role=..., goal=..., backstory=..., tools=decision_tools())   # or: the decisions as crew tools
```

`TaskAssigner` picks the crew member for each task from the members' roles and goals, in place of a hierarchical
crew's manager LLM (or assigning every task by hand). It reads each task's description and expected output, sets
`task.agent`, and sends tasks no member fits confidently to a fallback member:

```python
from crewai import Crew, Process
from opendecider.integrations.crewai import TaskAssigner

assigner = TaskAssigner([billing, engineer, sales], fallback=lead, min_confidence=0.6)
assigner.assign_all(tasks)
Crew(agents=[billing, engineer, sales, lead], tasks=tasks, process=Process.sequential).kickoff()
```

With opendecider-nano, a duplicate charge goes to the billing specialist (0.92), a 500 error to the support engineer
(0.91), a quote request to the account executive (0.90), and "Look into this." to the team lead (top member only
0.36). Runnable without an API key: [examples/agent_frameworks/crewai_crew.py](https://github.com/manjunathshiva/opendecider/blob/main/examples/agent_frameworks/crewai_crew.py).

## Microsoft Agent Framework: a switch-case edge

```bash
pip install "opendecider[agent-framework]>=0.4.0"
```

```python
from agent_framework import Agent, WorkflowBuilder
from opendecider.integrations.agent_framework import DecisionRouter, decision_tools

route = DecisionRouter({...}, "Which specialist agent should answer this user query?",
                       fallback="human_agent", min_confidence=0.6)

workflow = (WorkflowBuilder(start_executor=triage)
            .add_switch_case_edge_group(triage, route.cases(
                {"billing_agent": billing, "tech_support": tech, "sales_agent": sales, "human_agent": human}))
            .build())

agent = Agent(client=..., tools=decision_tools())   # or: the decisions as function tools
```

`cases()` gives one `Case` per route and a `Default` (the fallback's executor, or `default=`). The model runs once per
message, however many cases the group has. The router decides on the message's text: a string as is, an agent's
response or a chat message as its text, JSON as is; `state=` takes a function of the message instead.

## Google ADK: a router agent

```bash
pip install "opendecider[google-adk]>=0.4.0"
```

```python
from google.adk.agents import LlmAgent
from opendecider.integrations.google_adk import DecisionRouterAgent, decision_tools

root_agent = DecisionRouterAgent(
    name="triage", sub_agents=[billing_agent, tech_support, sales_agent, human_agent],   # LlmAgents, by name
    routes={"billing_agent": "invoices, payment methods, duplicate charges, refunds",
            "tech_support": "system errors, bugs, API downtime, stack traces",
            "sales_agent": "pricing plans, new contracts, demo requests"},
    instructions="Which specialist agent should answer this user query?",
    fallback="human_agent", min_confidence=0.6)

assistant = LlmAgent(name="assistant", model="gemini-...", tools=decision_tools())   # or: the decisions as tools
```

`DecisionRouterAgent` decides on the user's message and hands the whole turn to one sub-agent, in place of an LLM
coordinator's `transfer_to_agent` call. It checks at construction that every route has a sub-agent.
`decision_model=` picks the OpenDecider model.

## PydanticAI: a toolset

```bash
pip install "opendecider[pydantic-ai]>=0.4.0"
```

```python
from pydantic_ai import Agent
from opendecider.integrations.pydantic_ai import decision_toolset

agent = Agent("openai:gpt-5", toolsets=[decision_toolset()])
```

An invalid call raises `ModelRetry` with what to fix, so the model corrects it and tries again. To branch your own code
(or pick a pydantic-graph node), call `DecisionRouter(...)(text)`: it returns a route name.

## Strands Agents: tools and a router

```bash
pip install "opendecider[strands]>=0.4.0"
```

```python
from strands import Agent
from opendecider.integrations.strands import DecisionRouter, decision_tools

agent = Agent(model=..., tools=decision_tools())

route = DecisionRouter({...}, "Which specialist agent should answer this user query?",
                       fallback="human_agent", min_confidence=0.6)
SPECIALISTS[route(ticket)](ticket)          # call the chosen specialist agent
```

## Mastra (TypeScript): through MCP

Mastra connects to OpenDecider's [MCP server](mcp.md) through its MCP client; there is no TypeScript package to install
from OpenDecider.

```bash
pip install "opendecider[mcp]"
npm install @mastra/mcp @mastra/core
```

```ts
import { Agent } from "@mastra/core/agent";
import { MCPClient } from "@mastra/mcp";

const mcp = new MCPClient({ servers: { opendecider: { command: "opendecider", args: ["mcp"] } } });
const agent = new Agent({ name: "support", instructions: "...", model: ...,
                          tools: await mcp.listTools() });   // opendecider_decide, _choose, _yes_no, _score
```

The example calls the tools the way the agent would, so it runs without an API key. Other frameworks with an MCP
client connect the same way.

## Production

**The model on its own server.** Pass an `opendecider serve` URL as the model, in any integration, the MCP server or
`load`: the process that routes then holds no model, and the server runs it for every client, batches their requests
and reports `/metrics`. Answers are the server model's own, the same as in-process. Set `OPENDECIDER_REMOTE_API_KEY`
(or `api_key=` on `tools.Decider`) when the server needs a key; it is sent only to that server, and never over plain
HTTP to another host.

```python
route = DecisionRouter(routes, "Which specialist?", model="https://decider.internal:8000",
                       fallback="human_agent", min_confidence=0.6)
```

Each request waits at most 30 seconds by default; for routing, set a limit you can afford with
`tools.Decider(url, timeout=5)` and pass that as the model. After a failed load (the server down, a wrong key), calls
fail at once for 5 seconds and then try again, so an outage costs one slow call rather than one per request.

**Every decision, recorded.** `router.decide(state)` returns a `Decision`, and `on_decision=` receives every one,
failed ones included, for logs, metrics or audits:

| field | meaning |
|---|---|
| `route` | the route taken (`None` when the decision failed and the error was raised) |
| `reason` | `top_choice` (the model's top route), `low_confidence` (below `min_confidence`: the fallback), `empty_input` (nothing to decide on: the fallback) or `error` |
| `choice`, `confidence`, `probabilities` | the model's top route, its probability, and every route's probability |
| `model`, `latency_ms`, `truncated`, `error` | the model that answered, the time taken, whether the input was cut to fit, and what failed |

```python
def audit(d):
    log.info(json.dumps(d.to_dict()))     # one line per decision

route = DecisionRouter(routes, "Which specialist?", fallback="human_agent", on_decision=audit)
```

A hook that raises is logged and never breaks routing. Hooks run inline, in the routing call, so keep them fast
(hand slow work to a queue). A `Decision` holds no input text, so audit logs and spans carry no ticket or message
contents.

**Traces.** With `opentelemetry-api` installed (`pip install "opendecider[otel]"`), every decision is an
`opendecider.route` span with the route, choice, confidence, reason and latency as attributes, and the model call an
`opendecider.decide` span inside it. A failed decision is an error span with the exception. Spans go wherever your
OpenTelemetry SDK sends them; without one configured, they cost nothing.

**When the decision fails.** `on_error="raise"` (the default) lets a failed decision raise, as any other step that
fails would. `on_error="fallback"` takes the fallback route instead and logs the error, so a model outage sends
requests to people rather than failing them. In an Agent Framework switch, a failed decision always goes to the
default executor, since the framework would swallow the error anyway.

**Concurrency.** One router serves concurrent requests: each call routes on its own answer, and a shared model loads
once. Local models run one inference at a time per process; for higher request rates, serve the model.

A router set up this way (served model, JSON audit log, fallback on error, spans):
[examples/production_router.py](https://github.com/manjunathshiva/opendecider/blob/main/examples/production_router.py).

## Other frameworks

`opendecider.tools.Router` is the router every integration builds on: `Router(routes, instructions, fallback=...,
min_confidence=...)(state)` returns a route name for any text or JSON, so it drops into any framework's branching. The
plain functions in `opendecider.tools` (`decide`, `choose`, `yes_no`, `score`) are what every tool wraps.

## When a tool call is invalid

Invalid input (an unknown question type, a single option, repeated score levels) and a model that cannot load go back
to the model with a message saying what to fix, in each framework's own way:

| framework | the model sees |
|---|---|
| LangChain, LlamaIndex, Agno, MCP (and Mastra) | the error message, as the tool's result |
| CrewAI, Microsoft Agent Framework, Google ADK, Strands | `{"error": "<what to fix>"}` as the tool's result, not an exception: Agent Framework hides exception text from the model by default, Google ADK ends the run when a tool raises, and CrewAI and Strands follow suit |
| PydanticAI | a retry prompt with the message (`ModelRetry`); a model that cannot load raises `ModelError` instead |

## Notes

- Write the routes' descriptions the way you would brief a person; descriptions matter more than the labels.
- One model serves every component that names it; calls run one at a time. For high request rates from
  services, use [`opendecider serve`](serve.md).
- Routers decide in the thread that calls them. In async workflows (Agno's `arun`, CrewAI's async flows, Agent
  Framework switches) that is the event loop, which waits for the decision: milliseconds with opendecider-nano, longer
  with a 4B model on CPU. Google ADK's router agent, and the LangChain, LlamaIndex and CrewAI tools when called
  asynchronously, run the model off the event loop.
- CrewAI and Strands require `mcp` 1.x, and the MCP server needs 2.2 or later: install `opendecider[mcp]` in its own
  environment (`uvx --from "opendecider[mcp]" opendecider mcp`). The CrewAI and Strands integrations do not need it.
- `DecisionSelector` returns a single selection; for multi-engine fan-out, keep an LLM multi-selector.
