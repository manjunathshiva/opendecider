"""LangChain / LangGraph and LlamaIndex integrations, through each framework's own API, with a fake model."""
import asyncio

import pytest

from opendecider import OpenDecider
from opendecider.tools import Decider


class Fake:
    """The option whose name appears in the state gets 0.8 (else the first option); the rest share the remainder."""

    def __init__(self):
        self.calls = 0

    def decide_many(self, items, info=None):
        self.calls += 1
        out = []
        for state, _, opts in items:
            names = list(opts)
            text = str(state).lower()
            top = next((n for n in names if n.lower() in text), names[0])
            rest = 0.2 / (len(names) - 1)
            out.append({n: 0.8 if n == top else rest for n in names})
            if info is not None:
                info.append({"input_tokens": 10, "truncated": False})
        return out


def decider(fake=None):
    return Decider(OpenDecider(fake or Fake(), {"name": "opendecider-test", "kind": "nano"}))


# ---- LangChain ----------------------------------------------------------------------------------------------------

def test_langchain_tools_answer_and_report_errors():
    pytest.importorskip("langchain_core")
    from opendecider.integrations.langchain import decision_tools
    tools = {t.name: t for t in decision_tools(decider())}
    assert set(tools) == {"decide", "choose", "yes_no", "score"}
    assert set(tools["choose"].args) == {"state", "question", "options"}   # schema from the typed signature

    r = tools["choose"].invoke({"state": "please refund me", "question": "Which team?",
                                "options": {"billing": "refunds", "tech": "bugs"}})
    assert r["choice"] == "billing"   # neither name in the state: the fake picks the first option
    assert set(r["probabilities"]) == {"billing", "tech"} and "confidence" in r
    r = tools["yes_no"].invoke({"state": "x", "question": "Is it?"})
    assert r["answer"] in ("yes", "no") and 0 <= r["probability_yes"] <= 1
    r = tools["decide"].invoke({"state": "x", "questions": {"u": {"type": "score", "instructions": "How bad?",
                                                                 "criteria": ["low", "high"]}}})
    assert r["answers"]["u"]["label"] in ("low", "high")
    r = asyncio.run(tools["score"].ainvoke({"state": "x", "question": "How bad?", "levels": ["low", "high"]}))
    assert r["label"] in ("low", "high")   # async path

    # invalid input comes back to the agent as the tool's message, not as an exception
    msg = tools["choose"].invoke({"state": "x", "question": "Which?", "options": ["only-one"]})
    assert isinstance(msg, str) and "at least 2 options" in msg
    msg = tools["score"].invoke({"state": "x", "question": "How bad?", "levels": ["low", "low"]})
    assert "score levels must be distinct" in msg


def test_langchain_router_routes_and_falls_back():
    pytest.importorskip("langchain_core")
    from opendecider.integrations.langchain import DecisionRouter
    routes = {"billing": "charges, refunds", "tech": "bugs, outages"}
    route = DecisionRouter(routes, "Which team?", model=decider(), state_key="input")
    assert route({"input": "the tech dashboard is down"}) == "tech"
    assert route.last["confidence"] == 0.8
    assert route.path_map == ["billing", "tech"]

    unsure = DecisionRouter(routes, "Which team?", model=decider(), state_key="input", fallback="human",
                            min_confidence=0.9)
    assert unsure({"input": "the tech dashboard is down"}) == "human"   # 0.8 < 0.9
    assert unsure.path_map == ["billing", "tech", "human"]
    with pytest.raises(ValueError):
        DecisionRouter(routes, "Which team?", model=decider(), min_confidence=0.5)   # a threshold needs a fallback


def test_langchain_router_reads_the_last_message():
    pytest.importorskip("langchain_core")
    from langchain_core.messages import AIMessage, HumanMessage
    from opendecider.integrations.langchain import DecisionRouter
    route = DecisionRouter(["billing", "tech"], "Which team?", model=decider())
    state = {"messages": [AIMessage(content="How can I help?"), HumanMessage(content="I need a refund for billing")]}
    assert route(state) == "billing"


def test_langgraph_conditional_edge():
    pytest.importorskip("langgraph")
    from typing import TypedDict

    from langgraph.graph import END, StateGraph
    from opendecider.integrations.langchain import DecisionRouter

    class State(TypedDict):
        input: str
        handled_by: str

    route = DecisionRouter({"billing": "charges, refunds", "tech": "bugs, outages"}, "Which team?",
                           model=decider(), state_key="input", fallback="human", min_confidence=0.5)
    g = StateGraph(State)
    g.add_node("triage", lambda s: {})
    for node in route.path_map:
        g.add_node(node, lambda s, node=node: {"handled_by": node})
        g.add_edge(node, END)
    g.set_entry_point("triage")
    g.add_conditional_edges("triage", route, route.path_map)
    app = g.compile()
    assert app.invoke({"input": "billing: charged twice"})["handled_by"] == "billing"
    assert app.invoke({"input": "the tech api is down"})["handled_by"] == "tech"
    assert asyncio.run(app.ainvoke({"input": "the tech api is down"}))["handled_by"] == "tech"   # async graphs too


# ---- LlamaIndex ---------------------------------------------------------------------------------------------------

def test_llamaindex_tools():
    pytest.importorskip("llama_index.core")
    from opendecider.integrations.llamaindex import decision_tools
    tools = {t.metadata.name: t for t in decision_tools(decider())}
    assert set(tools) == {"decide", "choose", "yes_no", "score"}
    out = tools["choose"].call(state="refund please", question="Which team?", options=["billing", "tech"])
    assert out.raw_output["choice"] in ("billing", "tech")
    out = asyncio.run(tools["yes_no"].acall(state="x", question="Is it?"))
    assert out.raw_output["answer"] in ("yes", "no")
    with pytest.raises(ValueError, match="at least 2 options"):
        tools["choose"].call(state="x", question="Which?", options=["only-one"])


def test_llamaindex_selector_in_a_router_query_engine():
    pytest.importorskip("llama_index.core")
    from llama_index.core.base.response.schema import Response
    from llama_index.core.llms.mock import MockLLM
    from llama_index.core.query_engine import CustomQueryEngine, RouterQueryEngine
    from llama_index.core.tools import QueryEngineTool, ToolMetadata
    from opendecider.integrations.llamaindex import DecisionSelector

    class Echo(CustomQueryEngine):
        label: str

        def custom_query(self, query_str: str):
            return Response(response=self.label)

    sql = QueryEngineTool(query_engine=Echo(label="sql"),
                          metadata=ToolMetadata(name="sql", description="sales figures"))
    docs = QueryEngineTool(query_engine=Echo(label="docs"), metadata=ToolMetadata(name="docs", description="manuals"))
    selector = DecisionSelector(model=decider())
    engine = RouterQueryEngine(selector=selector, query_engine_tools=[sql, docs], llm=MockLLM())   # no LLM call made
    assert str(engine.query("what do the docs say about setup?")) == "docs"
    assert str(engine.query("sql: total sales last month")) == "sql"
    assert selector.last["confidence"] == 0.8
    r = selector.select([sql.metadata, docs.metadata], "manuals for the docs")
    assert r.ind == 1 and "probability" in r.reason
