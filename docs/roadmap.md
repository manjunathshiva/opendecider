# Roadmap

**In progress**

* **Native in Ollama.** opendecider-small-td retrained on Ollama's own `/v1/systemone` prompt as well as ours: 0.793 on
  typed-decisions through Ollama's endpoint, up from 0.719, and still 0.794 through the opendecider package. It goes on
  ollama.com once Ollama 0.35.1 (the first release that accepts third-party decision models) is out.
* **A browser extension.** A Chrome extension on `@opendecider/web` (first: a YouTube feed that keeps only what you
  choose, decided on the device), and a guide for building your own (`@opendecider/web` shipped in 0.7.0).

**Next**

* **More than 26 options:** shortlist-then-letters for the Qwen-based models, measured on every benchmark before it
  ships.
* **Rule-labelled evaluation:** every model on tasksource/procedural-typed-decisions, whose answers are computed exactly
  from rules, so it measures correctness rather than agreement with a teacher model.
* **Fine-tune nano on your own labels:** a script and a guide for adapting opendecider-nano to your decisions.
* **Multilingual evaluation.**

Want to help with one of these? See [where to help](https://github.com/manjunathshiva/opendecider/blob/main/CONTRIBUTING.md)
and open an issue to agree on the approach first.
