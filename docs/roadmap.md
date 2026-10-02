# Roadmap

**In progress**

* **Native in Ollama.** opendecider-small-td retrained on Ollama's own `/v1/systemone` prompt as well as ours: 0.793 on
  typed-decisions through Ollama's endpoint, up from 0.719, and still 0.794 through the opendecider package. It goes on
  ollama.com once Ollama 0.35.1 (the first release that accepts third-party decision models) is out.
* **Guardrails for agent workflows (done, in the next release).** A guard that screens prompts for jailbreaks and
  prompt injection in each framework's own hook (LangChain, CrewAI, Agno, Google ADK, Microsoft Agent Framework,
  Strands, PydanticAI and MCP). On three public datasets opendecider-small-td catches as many attacks as Laya with
  half the false alarms; Laya is ahead on jailbreak-classification ([Agent guardrails](guides/agent-guardrails.md)). Also LangChain triage and
  evaluator runnables, and a LlamaIndex selector that picks several engines.
* **Beyond Python and PyTorch.** A client for `opendecider serve` that installs without PyTorch, a TypeScript client
  with tools for Mastra and the Vercel AI SDK, and opendecider-nano as ONNX in the browser.

**Next**

* **More than 26 options:** shortlist-then-letters for the Qwen-based models, measured on every benchmark before it
  ships (0.5).
* **Rule-labelled evaluation:** every model on tasksource/procedural-typed-decisions, whose answers are computed exactly
  from rules, so it measures correctness rather than agreement with a teacher model.
* **Fine-tune nano on your own labels:** a script and a guide for adapting opendecider-nano to your decisions.
* **Multilingual evaluation.**

Want to help with one of these? See [where to help](https://github.com/manjunathshiva/opendecider/blob/main/CONTRIBUTING.md)
and open an issue to agree on the approach first.
