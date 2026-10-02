"""Build opendecider-client: the same `opendecider` package without PyTorch, for models served elsewhere.

    python packaging/build_client.py            # sdist and wheel in dist/
    python packaging/build_client.py --outdir out

The source is the same `opendecider/` folder; only the distribution changes. The name is opendecider-client, and the
dependencies drop torch, transformers, safetensors and huggingface_hub (what runs a model on this machine) and the
extras that only make sense with a local model (small, mlx, serve; dev is for this repository). Routers, tools, the
guard and `opendecider mcp` work as in opendecider, against an `opendecider serve` URL, Ollama or LM Studio. Like
opencv-python and opencv-python-headless, install one of the two packages, not both.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL_ONLY_EXTRAS = ("small", "mlx", "serve", "dev")
FILES = ("README.md", "LICENSE", "NOTICE")
README_NOTE = """> **opendecider-client** is the PyTorch-free build of [opendecider](https://pypi.org/project/opendecider/): the same
> `import opendecider`, routers, tools, guard and MCP server, for models served elsewhere (an `opendecider serve` URL,
> Ollama, LM Studio). It cannot run a model on this machine; for that, install `opendecider` instead. Install one of
> the two packages, not both: they share the `opendecider` files, so uninstalling one removes them for the other (to
> recover, reinstall the one you keep with `pip install --force-reinstall`).

"""


def client_pyproject(text: str) -> str:
    """The client's pyproject.toml, from the main one. Every edit must apply, or the build stops."""
    def sub(pattern: str, repl: str, s: str, flags: int = 0) -> str:
        out, n = re.subn(pattern, repl, s, count=1, flags=flags)
        if n != 1:
            sys.exit(f"build_client: pyproject.toml no longer matches {pattern!r}; update packaging/build_client.py")
        return out

    text = sub(r'^name = "opendecider"$', 'name = "opendecider-client"', text, re.M)
    text = sub(r'^description = "[^"]*"$',
               'description = "OpenDecider without PyTorch: routers, tools, the prompt guard and the MCP server for '
               'decision models served by opendecider serve, Ollama or LM Studio."', text, re.M)
    text = sub(r"^dependencies = \[\n(?:    .*\n)*?\]$", "dependencies = []", text, re.M)
    for extra in LOCAL_ONLY_EXTRAS:   # the extra and the comment line above it, if any
        text = sub(rf"^(?:#.*\n)?{re.escape(extra)} = \[[^\]]*\]\n", "", text, re.M)
    return text


def build(outdir: Path) -> list[Path]:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp)
        shutil.copytree(ROOT / "opendecider", src / "opendecider",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for name in FILES:
            shutil.copy2(ROOT / name, src / name)
        (src / "README.md").write_text(README_NOTE + (ROOT / "README.md").read_text())
        (src / "pyproject.toml").write_text(client_pyproject((ROOT / "pyproject.toml").read_text()))
        outdir.mkdir(parents=True, exist_ok=True)
        before = set(outdir.iterdir())
        subprocess.run([sys.executable, "-m", "build", "--outdir", str(outdir), str(src)], check=True)
        return sorted(set(outdir.iterdir()) - before)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(ROOT / "dist"))
    for f in build(Path(ap.parse_args().outdir).resolve()):
        print(f)
