# Examples

Short scripts, one per use case. Each runs on a laptop CPU with
[opendecider-nano](https://huggingface.co/manjunathshiva/opendecider-nano) (a 0.8 GB download on first use) unless
noted. CI runs each of them against the released model whenever the package or the examples change, except
`remote_backends.py`, which needs LM Studio, Ollama or vLLM running and is only compile-checked there.

| script | what it shows | run it |
|---|---|---|
| [`quickstart.py`](quickstart.py) | three typed questions (`choice`, `score`, `noul`) about one ticket | `python examples/quickstart.py` |
| [`support_triage.py`](support_triage.py) | a queue of tickets in one call: route by team, flag churn risk, send unsure answers to a person | `python examples/support_triage.py` |
| [`agent_guardrail.py`](agent_guardrail.py) | check an AI agent's JSON trace before its next step: continue, retry, ask the user or stop | `python examples/agent_guardrail.py` |
| [`confident_automation.py`](confident_automation.py) | on 2,000 labelled business decisions: how many you can automate at a given accuracy | `python examples/confident_automation.py` |
| [`serve_client.py`](serve_client.py) | call `opendecider serve` over HTTP (Jev's `/v1/systemone` protocol), with retries | start the server, then `python examples/serve_client.py` |
| [`remote_backends.py`](remote_backends.py) | the same decision through LM Studio, Ollama or vLLM | `python examples/remote_backends.py lmstudio:opendecider-small` |

```bash
pip install opendecider                    # nano
pip install "opendecider[serve]"           # serve_client.py: the server
pip install pandas pyarrow                 # confident_automation.py: reads the typed-decisions test split
pip install "opendecider[small]"           # the 4B models: pass e.g. manjunathshiva/opendecider-small-td as the model
```

The scripts that take a model name accept any OpenDecider model (`quickstart.py`, `support_triage.py` and
`agent_guardrail.py` as the first argument, `confident_automation.py` as `--model`).

## Agent frameworks

[`agent_frameworks/`](agent_frameworks/) has one script per framework. Each routes the same four support tickets (to
a specialist, or to a person when unsure) or calls the decision tools as an agent would, without an LLM API key. See
[Agent frameworks](https://manjunathshiva.github.io/opendecider/guides/agent-frameworks/) for each integration.

| script | framework | what it shows | install |
|---|---|---|---|
| [`langgraph_router.py`](agent_frameworks/langgraph_router.py) | LangGraph | `DecisionRouter` as a graph's conditional edge | `pip install "opendecider[langchain]" langgraph` |
| [`llamaindex_selector.py`](agent_frameworks/llamaindex_selector.py) | LlamaIndex | `DecisionSelector` picks a RouterQueryEngine's source | `pip install "opendecider[llamaindex]"` |
| [`agno_workflow.py`](agent_frameworks/agno_workflow.py) | Agno | `DecisionRouter.selector()` as a workflow Router's selector | `pip install "opendecider[agno]"` |
| [`crewai_flow.py`](agent_frameworks/crewai_flow.py) | CrewAI | `DecisionRouter` in a Flow's `@router` | `pip install "opendecider[crewai]"` |
| [`agent_framework_workflow.py`](agent_frameworks/agent_framework_workflow.py) | Microsoft Agent Framework | `DecisionRouter.cases()` as a switch-case edge group | `pip install "opendecider[agent-framework]"` |
| [`google_adk_router.py`](agent_frameworks/google_adk_router.py) | Google ADK | `DecisionRouterAgent` hands each request to a sub-agent | `pip install "opendecider[google-adk]"` |
| [`pydantic_ai_agent.py`](agent_frameworks/pydantic_ai_agent.py) | PydanticAI | an agent calling `decision_toolset()`, including a retry on an invalid call | `pip install "opendecider[pydantic-ai]"` |
| [`strands_agent.py`](agent_frameworks/strands_agent.py) | Strands Agents | `DecisionRouter` plus `decision_tools()` | `pip install "opendecider[strands]"` |
| [`mcp_client.py`](agent_frameworks/mcp_client.py) | MCP | what an AI assistant sees: `opendecider mcp`'s tools over stdio | `pip install "opendecider[mcp]"` |
| [`mastra/`](agent_frameworks/mastra/) | Mastra (TypeScript) | the MCP tools through Mastra's `MCPClient` | `pip install "opendecider[mcp]"`, then `npm ci` in the folder |

These integrations ship in 0.4.0, which is not on PyPI yet: from a clone, install with `pip install -e ".[agno]"`
(the extra each script needs) instead of the table's install column. Run each with
`python examples/agent_frameworks/<script>.py` (the Mastra one with `node index.mjs` in its folder).
The Python ones take a model name as the first argument (`mcp_client.py` and the Mastra one: `--model <name>`). CrewAI
and Strands require `mcp` 1.x, so keep them out of the environment that runs `opendecider mcp`.

## What `confident_automation.py` prints for nano

The full test split (400 cases, 2,000 decisions; no OpenDecider model was trained on it), on a laptop CPU in about a
minute:

```
manjunathshiva/opendecider-nano: 400 cases, 2000 decisions, accuracy 0.796

automate when p >= | automated | accuracy of automated | to a person
              0.0 |     100% |                 0.796 |         0%
              0.5 |      76% |                 0.873 |        24%
              0.6 |      51% |                 0.941 |        49%
              0.7 |      32% |                 0.972 |        68%
              0.8 |      21% |                 0.998 |        79%
              0.9 |      13% |                 0.996 |        87%

most confident | accuracy
          100% | 0.796
           70% | 0.894
           50% | 0.943
```

Automating the answers with p ≥ 0.6 handles half the decisions at 94% accuracy; the rest go to a person. The gold
labels come from a teacher model and are themselves imperfect (the dataset card has the details), so read the top
rows as "agrees with the labels", not as a ceiling.

## Notebook

[`notebooks/opendecider_colab.ipynb`](../notebooks/opendecider_colab.ipynb) walks through the same ground on a free
Colab GPU, with the outputs from a real run:
[open it in Colab](https://colab.research.google.com/github/manjunathshiva/opendecider/blob/main/notebooks/opendecider_colab.ipynb).
