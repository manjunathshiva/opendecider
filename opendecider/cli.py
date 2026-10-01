"""`opendecider` command line.

    opendecider serve --model manjunathshiva/opendecider-nano --port 8000
    opendecider mcp --model manjunathshiva/opendecider-nano            # MCP server over stdio, for AI assistants
    opendecider bench-speed manjunathshiva/opendecider-nano
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="opendecider", description="OpenDecider decision models")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sv = sub.add_parser("serve", help="HTTP server speaking the Jev /v1/systemone protocol (pip install 'opendecider[serve]')",
                        description="Every flag can also be set as OPENDECIDER_<NAME> in the environment; flags win.")
    sv.add_argument("--model", help="Hub name or local folder (default: manjunathshiva/opendecider-nano)")
    sv.add_argument("--revision", help="pin a Hub revision (tag, branch or commit)")
    sv.add_argument("--device", help="cuda, mps or cpu (default: auto)")
    sv.add_argument("--host")
    sv.add_argument("--port", type=int)
    sv.add_argument("--max-batch", type=int, help="questions per inference batch (default 32)")
    sv.add_argument("--batch-wait-ms", type=float, help="how long to wait to fill a batch (default 2)")
    sv.add_argument("--max-in-flight", type=int, help="admitted requests before answering 503 (default 256)")
    sv.add_argument("--request-timeout-s", type=float, help="504 after this long (default 30)")
    sv.add_argument("--threads", type=int, help="torch CPU threads (default: torch's choice)")
    sv.add_argument("--dtype", choices=["float32", "bfloat16"], help="nano: bfloat16 is faster on CPUs with bf16 units (default float32)")
    sv.add_argument("--small-batch", type=int, help="Qwen-based models: questions per forward pass (default 1 = exact)")
    sv.add_argument("--log-level")

    mp = sub.add_parser("mcp", help="MCP server over stdio, for AI assistants and agents "
                                    "(pip install 'opendecider[mcp]')")
    mp.add_argument("--model", help="Hub name, local folder, or lmstudio: / ollama: / openai: model "
                    "(default: OPENDECIDER_MODEL, else manjunathshiva/opendecider-nano)")
    mp.add_argument("--revision", help="pin a Hub revision (tag, branch or commit)")
    mp.add_argument("--device", help="cuda, mps or cpu (default: auto)")
    mp.add_argument("--dtype", choices=["float32", "bfloat16"], help="nano only (default float32)")

    bs = sub.add_parser("bench-speed", help="latency by questions per call on this machine")
    bs.add_argument("model")

    a = ap.parse_args(argv)
    if a.cmd == "serve":
        try:
            import fastapi  # noqa: F401
            import uvicorn  # noqa: F401
        except ImportError:
            sys.exit('the server needs extra packages: pip install "opendecider[serve]"')
        from .serve import Settings, run
        run(Settings.from_env(model=a.model, revision=a.revision, device=a.device, host=a.host, port=a.port, max_batch=a.max_batch,
                              batch_wait_ms=a.batch_wait_ms, max_in_flight=a.max_in_flight,
                              request_timeout_s=a.request_timeout_s, threads=a.threads, small_batch=a.small_batch, dtype=a.dtype, log_level=a.log_level))
    elif a.cmd == "mcp":
        try:
            import mcp  # noqa: F401
        except ImportError:
            sys.exit('the MCP server needs extra packages: pip install "opendecider[mcp]"')
        import os
        from .mcp_server import run as run_mcp
        kw = {k: v for k, v in (("revision", a.revision), ("device", a.device), ("dtype", a.dtype)) if v}
        run_mcp(a.model or os.environ.get("OPENDECIDER_MODEL") or "manjunathshiva/opendecider-nano", **kw)
    elif a.cmd == "bench-speed":
        from .bench_speed import main as bench
        sys.argv = ["bench_speed", a.model]
        bench()


if __name__ == "__main__":
    main()
