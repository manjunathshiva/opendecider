# Contributing

Thanks for helping. Issues and pull requests are welcome; for anything large, open an issue first so we can agree
on the approach.

## Development setup

```bash
git clone https://github.com/manjunathshiva/opendecider && cd opendecider
python -m venv .venv && source .venv/bin/activate
pip install -e ".[small,serve,dev]"
python -m pytest -q tests        # fast, no model download
```

## What a pull request needs

- **Tests** for the change. `tests/` runs without downloading a model (a fake model stands in); CI runs it on
  Linux, macOS and Windows, and `docker.yml` builds the image and answers a real request with opendecider-nano.
- **No silent change to answers.** A change to how a model reads or scores inputs must come with before/after numbers
  from the benchmark harness (`python benchmarks/run.py --model ... --suites general,typed,laya`, then
  `python benchmarks/report.py`), and a CHANGELOG entry.
- **Compatibility.** The HTTP server must keep answering Jev's `/v1/systemone` request and response shape; the
  public Python API follows the versioning rules in CHANGELOG.md.
- **Licences.** Training data and teachers must allow commercial use; see NOTICE for what the released models used.

## Reporting a problem

Include the model name and revision, the package version (`python -c "import opendecider; print(opendecider.__version__)"`),
your device (CPU, CUDA GPU, Apple Silicon), and a minimal state and questions that reproduce it. Security issues go
through SECURITY.md, not public issues.
