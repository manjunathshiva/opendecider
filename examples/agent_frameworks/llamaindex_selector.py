"""Pick a LlamaIndex RouterQueryEngine's source with OpenDecider, in place of an LLM selector.

    pip install "opendecider[llamaindex]"
    python examples/agent_frameworks/llamaindex_selector.py

`DecisionSelector` reads each query engine's name and description and picks one in a single forward pass. The two
engines here only report their name, so the script runs without data or an API key: swap in your own query engines
(a SQL engine, a vector index, …). No LLM is called; `MockLLM` only satisfies RouterQueryEngine's constructor, which
uses an LLM for multi-engine summaries this single-choice selector never asks for.
"""
import sys

from llama_index.core.base.response.schema import Response
from llama_index.core.llms.mock import MockLLM
from llama_index.core.query_engine import CustomQueryEngine, RouterQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata

from opendecider.integrations.llamaindex import DecisionSelector


class Named(CustomQueryEngine):
    """Stands in for a real query engine: answers with its own name."""
    name: str

    def custom_query(self, query_str: str):
        return Response(response=self.name)


sources = [
    QueryEngineTool(query_engine=Named(name="sales_db"),
                    metadata=ToolMetadata(name="sales_db",
                                          description="SQL database of orders, revenue and customers")),
    QueryEngineTool(query_engine=Named(name="product_docs"),
                    metadata=ToolMetadata(name="product_docs", description="product manuals, setup guides and FAQs")),
]
selector = DecisionSelector(model=sys.argv[1] if len(sys.argv) > 1 else "manjunathshiva/opendecider-nano")
engine = RouterQueryEngine(selector=selector, query_engine_tools=sources, llm=MockLLM())

for query in ["What was our revenue in March?",
              "How do I reset the device to factory settings?",
              "Which customers ordered more than 100 units last quarter?"]:
    answered_by = engine.query(query)
    print(f"{query:<58} -> {str(answered_by):<12} (p = {selector.last['confidence']:.2f})")
