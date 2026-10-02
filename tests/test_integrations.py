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
    assert route.last.confidence == 0.8
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
    assert selector.last.confidence == 0.8
    r = selector.select([sql.metadata, docs.metadata], "manuals for the docs")
    assert r.ind == 1 and "probability" in r.reason



def test_llamaindex_multi_selector_picks_every_helpful_source():
    pytest.importorskip("llama_index.core")
    from llama_index.core.base.response.schema import Response
    from llama_index.core.llms.mock import MockLLM
    from llama_index.core.query_engine import CustomQueryEngine, RouterQueryEngine
    from llama_index.core.tools import QueryEngineTool, ToolMetadata
    from opendecider.integrations.llamaindex import DecisionMultiSelector

    class Relevance:
        """'yes' gets 0.9 when the source's name is in the query, else 0.2; records the questions asked."""
        def __init__(self):
            self.calls = 0

        def decide_many(self, items, info=None):
            self.calls += 1
            out = []
            for state, question, _ in items:
                name = question.split("Source: ")[1].split(":")[0]
                p = 0.9 if name in state else 0.2
                out.append({"yes": p, "no": 1 - p})
                if info is not None:
                    info.append({"input_tokens": 5, "truncated": False})
            return out

    class Echo(CustomQueryEngine):
        label: str

        def custom_query(self, query_str: str):
            return Response(response=self.label)

    tools = [QueryEngineTool(query_engine=Echo(label=n), metadata=ToolMetadata(name=n, description=d))
             for n, d in (("sql", "sales figures"), ("docs", "manuals"), ("web", "news"))]
    fake = Relevance()
    selector = DecisionMultiSelector(model=decider(fake))
    r = selector.select([t.metadata for t in tools], "compare sql totals with the docs")
    assert sorted(r.inds) == [0, 1] and fake.calls == 1   # one forward pass for every source
    assert selector.last == {"sql": 0.9, "docs": 0.9, "web": 0.2}
    assert selector.select([t.metadata for t in tools], "nothing relevant").inds == [0]   # at least the most likely
    assert DecisionMultiSelector(model=decider(fake), max_outputs=1).select(
        [t.metadata for t in tools], "sql and docs and web").inds == [0]
    engine = RouterQueryEngine(selector=selector, query_engine_tools=tools, llm=MockLLM())   # combines the answers
    assert engine.query("what do sql and the web say?").metadata["selector_result"].inds in ([0, 2], [2, 0])
    with pytest.raises(ValueError, match="min_probability"):
        DecisionMultiSelector(model=decider(), min_probability=0)
    with pytest.raises(ValueError, match="max_outputs"):
        DecisionMultiSelector(model=decider(), max_outputs=0)


def test_langchain_decision_runnable_answers_and_batches():
    pytest.importorskip("langchain_core")
    from langchain_core.documents import Document
    from langchain_core.messages import HumanMessage
    from opendecider.integrations.langchain import decision_runnable
    from opendecider.questions import Choice, Noul
    fake = Fake()
    qs = {"team": Choice("Which team?", {"billing": "charges", "tech": "bugs"}), "urgent": Noul("Is it urgent?")}
    triage = decision_runnable(qs, model=decider(fake))
    a = triage.invoke("the tech dashboard is down")
    assert a["team"]["choice"] == "tech" and "probability_yes" in a["urgent"]
    assert triage.invoke([HumanMessage("refund my billing")])["team"]["choice"] == "billing"   # a message list
    assert triage.invoke(Document(page_content="tech outage"))["team"]["choice"] == "tech"
    keyed = decision_runnable(qs, model=decider(fake), key="body")
    assert keyed.invoke({"body": "tech", "subject": "billing"})["team"]["choice"] == "tech"

    fake.calls = 0
    out = triage.batch(["billing please", "tech is down", "billing again"])
    assert [o["team"]["choice"] for o in out] == ["billing", "tech", "billing"] and fake.calls == 1   # one call
    out = keyed.batch([{"body": "tech"}, {"nobody": 1}], return_exceptions=True)
    assert out[0]["team"]["choice"] == "tech" and isinstance(out[1], KeyError)   # each input gets its own result
    with pytest.raises(KeyError):
        keyed.batch([{"body": "tech"}, {"nobody": 1}])
    assert asyncio.run(triage.abatch(["tech"]))[0]["team"]["choice"] == "tech"
    with pytest.raises(ValueError, match="at least 2 options"):
        decision_runnable({"q": Choice("Which?", ["only"])}, model=decider())   # checked when it is made


def test_langchain_decision_evaluator_scores_runs():
    pytest.importorskip("langchain_core")
    from opendecider.integrations.langchain import decision_evaluator
    yes_no = decision_evaluator("grounded", "Is `output` supported by `input`?", model=decider())
    r = yes_no(inputs={"q": "2+2"}, outputs={"a": "4"})
    assert r["key"] == "grounded" and 0 <= r["score"] <= 1 and "probability" in r["comment"]
    rating = decision_evaluator("quality", "How good is `output`?", levels=["poor", "ok", "great"], model=decider())
    r = rating(inputs={"q": "x"}, outputs={"a": "y"}, reference_outputs={"a": "y"})
    assert r["value"] in ("poor", "ok", "great") and 0 <= r["score"] <= 1
    with pytest.raises(ValueError):
        decision_evaluator("bad", "How?", levels=["one"], model=decider())
    langsmith = pytest.importorskip("langsmith")
    import uuid

    from langsmith.schemas import Example
    data = [Example(id=uuid.uuid4(), dataset_id=uuid.uuid4(), inputs={"q": "2+2"}, outputs={"a": "4"})]
    results = langsmith.evaluate(lambda inputs: {"answer": "4"}, data=data, evaluators=[yes_no], upload_results=False)
    assert [e.key for r in results for e in r["evaluation_results"]["results"]] == ["grounded"]

# ---- review fixes --------------------------------------------------------------------------------------------------

def test_router_reads_text_from_content_blocks():
    pytest.importorskip("langchain_core")
    from langchain_core.messages import HumanMessage
    from opendecider.integrations.langchain import DecisionRouter
    class Recording(Fake):
        def decide_many(self, items, info=None):
            self.states = [st for st, _, _ in items]
            return super().decide_many(items, info)

    fake = Recording()
    route = DecisionRouter(["billing", "tech"], "Which team?", model=decider(fake))
    msg = HumanMessage(content=[{"type": "text", "text": "please refund my billing error"},
                                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}])
    assert route({"messages": [msg]}) == "billing"
    assert fake.states == ["please refund my billing error"]   # the text block only, not the image or the structure


def test_router_validates_at_construction():
    pytest.importorskip("langchain_core")
    from opendecider.integrations.langchain import DecisionRouter
    with pytest.raises(ValueError, match="between 0 and 1"):
        DecisionRouter(["a", "b"], "Which?", model=decider(), fallback="h", min_confidence=60)
    with pytest.raises(ValueError, match="at least 2 options"):
        DecisionRouter(["only"], "Which?", model=decider())


def test_non_json_state_is_a_clear_error():
    pytest.importorskip("langchain_core")
    import datetime
    from opendecider.integrations.langchain import decision_tools
    tools = {t.name: t for t in decision_tools(decider())}
    msg = tools["yes_no"].invoke({"state": "x", "question": "Is it?"})   # fine
    assert isinstance(msg, dict)
    from opendecider import tools as core
    with pytest.raises(ValueError, match="text or JSON-serialisable"):
        core.yes_no(decider(), {"when": datetime.datetime(2026, 10, 1)}, "Is it late?")


def test_selector_caps_long_descriptions():
    pytest.importorskip("llama_index.core")
    from llama_index.core.tools import ToolMetadata
    from opendecider.integrations.llamaindex import DecisionSelector

    class Recording(Fake):
        def decide_many(self, items, info=None):
            self.items = items
            return super().decide_many(items, info)

    fake = Recording()
    sel = DecisionSelector(model=decider(fake), max_description_chars=50)
    sel.select([ToolMetadata(name="a", description="x" * 5000), ToolMetadata(name="b", description="short")], "q")
    (_, _, opts), = fake.items
    assert len(opts["a"]) == 50 and opts["b"] == "short"


def test_components_naming_the_same_model_share_it():
    pytest.importorskip("langchain_core")
    from opendecider import tools as core
    from opendecider.integrations.langchain import DecisionRouter
    name = "test/shared-model"   # never loaded: nothing is called
    a = DecisionRouter(["x", "y"], "Which?", model=name)
    b = DecisionRouter(["x", "y"], "Which?", model=name)
    assert a.decider is b.decider is core.shared(name)
    assert core.shared(decider()) is not core.shared(decider())   # distinct objects keep distinct Deciders


def test_agents_see_typed_argument_schemas():
    """The schemas the LLM driving an agent sees (deferred annotations must still resolve to real types)."""
    pytest.importorskip("langchain_core")
    pytest.importorskip("llama_index.core")
    from opendecider.integrations.langchain import decision_tools as lc
    from opendecider.integrations.llamaindex import decision_tools as li
    options = {"anyOf": [{"items": {"type": "string"}, "type": "array"},
                         {"additionalProperties": {"type": "string"}, "type": "object"}], "title": "Options"}
    lt = {t.name: t for t in lc(decider())}
    assert lt["choose"].args["options"] == options and lt["score"].args["levels"]["type"] == "array"
    it = {t.metadata.name: t for t in li(decider())}
    params = it["choose"].metadata.get_parameters_dict()
    assert params["properties"]["options"] == options and params["required"] == ["state", "question", "options"]


def test_selector_picks_from_its_own_answer_under_concurrency():
    """A selector shared by concurrent queries picks from its own answer, even when another query overwrites
    `.last` between this query's decision and its selection."""
    pytest.importorskip("llama_index.core")
    from llama_index.core.tools import ToolMetadata
    from opendecider.integrations.llamaindex import DecisionSelector

    tools = [ToolMetadata(name="sql", description="sales figures"), ToolMetadata(name="docs", description="manuals")]

    class Racing(DecisionSelector):
        def __setattr__(self, name, value):
            object.__setattr__(self, name, value)
            if name == "last" and value and value.choice == "sql" and not self.__dict__.get("raced"):
                object.__setattr__(self, "raced", True)
                self.select(tools, "what do the docs say?")   # another query lands mid-call

    sel = Racing(model=decider())
    assert sel.select(tools, "sql: total sales").ind == 0
    assert sel.last.choice == "docs"   # the race happened


def test_selector_labels_never_collide():
    pytest.importorskip("llama_index.core")
    from llama_index.core.tools import ToolMetadata
    from opendecider.integrations.llamaindex import DecisionSelector

    class Recording(Fake):
        def decide_many(self, items, info=None):
            self.opts = list(items[0][2])
            return super().decide_many(items, info)

    fake = Recording()
    tools = [ToolMetadata(name="tech", description="a"), ToolMetadata(name="tech (3)", description="b"),
             ToolMetadata(name="tech", description="c")]
    DecisionSelector(model=decider(fake)).select(tools, "q")
    assert fake.opts == ["tech", "tech (3)", "tech (4)"]   # three options, none lost
