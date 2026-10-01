# LangChain, LangGraph and LlamaIndex

OpenDecider plugs into agent frameworks in two ways: as **tools** an agent calls (`decide`, `choose`, `yes_no`,
`score`, the same as the [MCP server](mcp.md)), and as a **router** that picks the next step without an LLM call. A
routing decision with opendecider-nano takes milliseconds and comes with a calibrated probability, so low-confidence
cases can go to a fallback (a person, a slower model) instead of the wrong branch.

Every component loads its model on the first call: opendecider-nano by default, or any model name you pass
(`manjunathshiva/opendecider-small-td`, or a served model such as
`ollama:hf.co/manjunathshiva/opendecider-small-td-GGUF:Q8_0`). Components that name the same model share one loaded
copy. To set loading options such as the device, pass a `Decider`:

```python
from opendecider.tools import Decider
nano_cpu = Decider("manjunathshiva/opendecider-nano", device="cpu")
```

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

With opendecider-nano:

| input | routed to | top route, probability |
|---|---|---|
| "Our API has returned 500 errors since 9am and every request fails." | `tech_support` | tech_support, 0.94 |
| "Can I get a quote for the enterprise plan for 200 seats?" | `sales_agent` | sales_agent, 0.91 |
| "Hmm, not sure, something feels off." | `human_agent` (fallback) | tech_support, 0.51 |

`DecisionRouter` reads `state[state_key]`; without `state_key` it reads the last message's text when the state has
`messages` (as in `MessagesState`), otherwise the whole state. `router.last` holds the last answer, for logging. Pick
`min_confidence` from your own labelled examples; see [Automate the confident decisions](confident-automation.md).

The whole graph, runnable on a laptop: [examples/langgraph_router.py](https://github.com/manjunathshiva/opendecider/blob/main/examples/langgraph_router.py).

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

Runnable without data or an API key: [examples/llamaindex_selector.py](https://github.com/manjunathshiva/opendecider/blob/main/examples/llamaindex_selector.py).

The same four tools are available as `FunctionTool`s for LlamaIndex agents:

```python
from opendecider.integrations.llamaindex import decision_tools
tools = decision_tools()
```

## Notes

- Write the routes' descriptions the way you would brief a person; descriptions matter more than the labels.
- One model serves every component that names it; calls run one at a time. For high request rates from
  services, use [`opendecider serve`](serve.md).
- `DecisionSelector` returns a single selection; for multi-engine fan-out, keep an LLM multi-selector.
