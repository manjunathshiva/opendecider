# Examples and notebook

## Runnable examples

Short scripts, one per use case, in
[examples/](https://github.com/manjunathshiva/opendecider/tree/main/examples). Each runs on a laptop CPU with
opendecider-nano (a 0.8 GB download on first use), and CI runs them against the released model whenever the package
or the examples change (`remote_backends.py`, which needs a model server, is compile-checked).

| script | what it shows |
|---|---|
| [`quickstart.py`](https://github.com/manjunathshiva/opendecider/blob/main/examples/quickstart.py) | three typed questions (`choice`, `score`, `noul`) about one ticket |
| [`support_triage.py`](https://github.com/manjunathshiva/opendecider/blob/main/examples/support_triage.py) | a queue of tickets in one call: route by team, flag churn risk, send unsure answers to a person |
| [`agent_guardrail.py`](https://github.com/manjunathshiva/opendecider/blob/main/examples/agent_guardrail.py) | check an AI agent's JSON trace before its next step: continue, retry, ask the user or stop |
| [`confident_automation.py`](https://github.com/manjunathshiva/opendecider/blob/main/examples/confident_automation.py) | on 2,000 labelled business decisions: how many you can automate at a given accuracy |
| [`serve_client.py`](https://github.com/manjunathshiva/opendecider/blob/main/examples/serve_client.py) | call `opendecider serve` over HTTP (Jev's `/v1/systemone` protocol), with retries |
| [`remote_backends.py`](https://github.com/manjunathshiva/opendecider/blob/main/examples/remote_backends.py) | the same decision through LM Studio, Ollama or vLLM |

```bash
git clone https://github.com/manjunathshiva/opendecider && cd opendecider
pip install opendecider pandas pyarrow
python examples/support_triage.py
```

### Agent frameworks

One script per framework, in
[examples/agent_frameworks/](https://github.com/manjunathshiva/opendecider/tree/main/examples/agent_frameworks). Each
runs without an LLM API key; [Agent frameworks](guides/agent-frameworks.md) explains each integration.

| script | framework | what it shows |
|---|---|---|
| [`langgraph_router.py`](https://github.com/manjunathshiva/opendecider/blob/main/examples/agent_frameworks/langgraph_router.py) | LangGraph | `DecisionRouter` as a graph's conditional edge: each ticket to a specialist, unsure ones to a person |
| [`llamaindex_selector.py`](https://github.com/manjunathshiva/opendecider/blob/main/examples/agent_frameworks/llamaindex_selector.py) | LlamaIndex | `DecisionSelector` picks a RouterQueryEngine's source without an LLM call |
| [`agno_workflow.py`](https://github.com/manjunathshiva/opendecider/blob/main/examples/agent_frameworks/agno_workflow.py) | Agno | `DecisionRouter.selector()` as a workflow Router's selector |
| [`crewai_flow.py`](https://github.com/manjunathshiva/opendecider/blob/main/examples/agent_frameworks/crewai_flow.py) | CrewAI | `DecisionRouter` in a Flow's `@router` |
| [`agent_framework_workflow.py`](https://github.com/manjunathshiva/opendecider/blob/main/examples/agent_frameworks/agent_framework_workflow.py) | Microsoft Agent Framework | `DecisionRouter.cases()` as a switch-case edge group |
| [`google_adk_router.py`](https://github.com/manjunathshiva/opendecider/blob/main/examples/agent_frameworks/google_adk_router.py) | Google ADK | `DecisionRouterAgent` hands each request to a sub-agent |
| [`pydantic_ai_agent.py`](https://github.com/manjunathshiva/opendecider/blob/main/examples/agent_frameworks/pydantic_ai_agent.py) | PydanticAI | an agent calling `decision_toolset()`, including a retry on an invalid call |
| [`strands_agent.py`](https://github.com/manjunathshiva/opendecider/blob/main/examples/agent_frameworks/strands_agent.py) | Strands Agents | `DecisionRouter` plus `decision_tools()` |
| [`mcp_client.py`](https://github.com/manjunathshiva/opendecider/blob/main/examples/agent_frameworks/mcp_client.py) | MCP | what an AI assistant sees: `opendecider mcp`'s tools called over stdio |
| [`mastra/`](https://github.com/manjunathshiva/opendecider/tree/main/examples/agent_frameworks/mastra) | Mastra (TypeScript) | the MCP tools through Mastra's `MCPClient` |

## Colab notebook

[Open in Colab](https://colab.research.google.com/github/manjunathshiva/opendecider/blob/main/notebooks/opendecider_colab.ipynb){ .md-button .md-button--primary }

On a free NVIDIA T4, about 15 minutes: nano on a support ticket and an agent trace, many states in one call, confident
automation on the typed-decisions test split (with a chart), speed at 1–50 questions per call, `opendecider serve`,
small and small-td, and rebuilding the benchmark tables. The notebook in the repository is saved with the outputs of a
real T4 run, so you can read it without running it.

## Live demo

[Try it in your browser](https://huggingface.co/spaces/manjunathshiva/opendecider-demo){ .md-button }

opendecider-nano on a basic CPU Space: edit the state and the questions and see every option's probability.
