# Use it from AI assistants (MCP)

`opendecider mcp` runs OpenDecider as an [MCP](https://modelcontextprotocol.io) server, so AI assistants and agents
(Claude Code, Claude Desktop, Cursor and any other MCP client) can call it as a tool. Instead of reasoning a
classification out in text, the agent gets a typed answer with a probability for every option, and can act on the
confident ones and ask you about the rest.

```bash
pip install "opendecider[mcp]"
opendecider mcp                      # opendecider-nano over stdio; the client starts it for you
```

The model loads on the first call (nano: about 5 seconds once downloaded, then milliseconds per question), so
connecting is instant.

## Connect a client

=== "Claude Code"

    ```bash
    claude mcp add opendecider -- opendecider mcp
    ```

    Run it from the environment where you installed opendecider, or give the full path to the command (see below).

=== "Claude Desktop"

    In `claude_desktop_config.json` (Settings → Developer → Edit Config):

    ```json
    {
      "mcpServers": {
        "opendecider": {
          "command": "/full/path/to/.venv/bin/opendecider",
          "args": ["mcp"]
        }
      }
    }
    ```

=== "Cursor"

    In `.cursor/mcp.json` (one project) or `~/.cursor/mcp.json` (all projects):

    ```json
    {
      "mcpServers": {
        "opendecider": {
          "command": "/full/path/to/.venv/bin/opendecider",
          "args": ["mcp"]
        }
      }
    }
    ```

Desktop apps do not see your shell's virtual environment, so give them the full path to the `opendecider` command
(`which opendecider` prints it). To run without installing anything into a project, use
[uv](https://docs.astral.sh/uv/): `"command": "uvx", "args": ["--from", "opendecider[mcp]", "opendecider", "mcp"]`.

## Tools

| tool | arguments | returns |
|---|---|---|
| `decide` | `state` (text or JSON), `questions` (any number of typed questions, as in the [Python API](../reference/python-api.md#questions)) | one answer per question |
| `choose` | `state`, `question`, `options` (a list of labels, or `{"label": "description"}`) | `choice`, `probabilities`, `confidence` |
| `yes_no` | `state`, `question` | `answer` (yes / no), `probability_yes`, `confidence` |
| `score` | `state`, `question`, `levels` (lowest first) | `level`, `label`, `expected_level`, `probabilities`, `confidence` |

Every tool is read-only and idempotent. Answers carry `truncated: true` when the state was cut to fit the model.
Invalid input (an unknown question type, a single option, repeated score levels, more than 64 questions or 256
options, a state over 200,000 characters) comes back as a tool error that names the problem, so the agent can fix
the call. So does a model that cannot load, with the reason (for example a mistyped `--model`).

For example, an agent asked to triage your inbox might call:

```json
{"tool": "choose",
 "arguments": {"state": {"from": "billing@vendor.example", "subject": "Invoice 4471 overdue"},
               "question": "Which folder should this email go to?",
               "options": {"finance": "invoices, payments", "support": "customer issues", "other": "everything else"}}}
```

and get back `{"choice": "finance", "probabilities": {...}, "confidence": ...}`.

## Choosing the model

```bash
opendecider mcp --model manjunathshiva/opendecider-small-td        # pip install "opendecider[small,mcp]"
opendecider mcp --model ollama:hf.co/manjunathshiva/opendecider-small-GGUF:Q8_0   # a model Ollama serves
```

`--model` takes anything [`load`](../reference/python-api.md#load) does (or set `OPENDECIDER_MODEL`); `--device`,
`--dtype` and `--revision` work as for [`opendecider serve`](../reference/cli.md). nano is the default: it is fast on
any machine. Use small or small-td on a GPU or a Mac with 16 GB for higher accuracy; see
[Choose a model](../models.md).

A model downloads on its first call (small: about 8 GB), which can take longer than a client waits for a tool. Download it
first (`hf download manjunathshiva/opendecider-small-td`), or run one decision with it from Python.

## Notes

- The server speaks MCP over stdio: logs and download progress go to stderr, which clients show in their MCP logs.
- Calls run one at a time; for many decisions per second from services, use [`opendecider serve`](serve.md).
- `confidence` is calibrated, so a threshold means something: see
  [Automate the confident decisions](confident-automation.md).
