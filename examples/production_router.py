"""A router set up for production: the model behind `opendecider serve`, one JSON log line per decision, a fallback
when the decision fails, and OpenTelemetry spans.

    pip install opendecider                                   # this script
    opendecider serve --model manjunathshiva/opendecider-nano # the model, elsewhere (pip install "opendecider[serve]")
    python examples/production_router.py http://127.0.0.1:8000
    python examples/production_router.py http://127.0.0.1:8000 --otel   # with spans: pip install opentelemetry-sdk

With a URL as the model, the process that routes holds no model: `opendecider serve` runs it, batches requests from
every client and reports /metrics. The same router works in any framework integration (`model=` takes the URL).
Every decision reaches `on_decision`, failed ones included; `on_error="fallback"` sends a request the model could not
decide (server down, timeout) to the fallback route instead of failing it. Set OPENDECIDER_REMOTE_API_KEY when the
server needs a key.
"""
import json
import logging
import sys

from opendecider.tools import Decision, Router

url = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "manjunathshiva/opendecider-nano"
if "--otel" in sys.argv:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
audit = logging.getLogger("routing.audit")
audit.setLevel(logging.INFO)
audit.addHandler(logging.StreamHandler(sys.stdout))
audit.propagate = False


def log_decision(d: Decision) -> None:
    """One structured line per decision: ship it to your log pipeline, or count reasons as metrics."""
    audit.info(json.dumps({"route": d.route, "reason": d.reason, "choice": d.choice,
                           "confidence": None if d.confidence is None else round(d.confidence, 3),
                           "latency_ms": d.latency_ms, "model": d.model, "error": d.error}))


route = Router(
    {"billing_agent": "invoices, payment methods, duplicate charges, refunds",
     "tech_support": "system errors, bugs, API downtime, stack traces",
     "sales_agent": "pricing plans, new contracts, demo requests"},
    "Which specialist agent should answer this user query?",
    model=url, fallback="human_agent", min_confidence=0.6, on_error="fallback", on_decision=log_decision)

for ticket in ["Our API has returned 500 errors since 9am and every request fails.",
               "I was charged twice for the March invoice. Please refund the duplicate.",
               "Hmm, not sure, something feels off.",
               ""]:
    route(ticket)
