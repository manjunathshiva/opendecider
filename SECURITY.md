# Security policy

## Reporting a vulnerability

Please report security issues privately through GitHub's
[private vulnerability reporting](https://github.com/manjunathshiva/opendecider/security/advisories/new), not in a
public issue. You'll get an acknowledgement within 3 working days and a fix or mitigation plan within 14 days for
confirmed issues. Fixes ship in a patch release, with credit if you want it.

Supported versions: the latest minor release (currently 0.2.x) receives security fixes.

## Running the server safely

- **Set an API key** (`OPENDECIDER_API_KEY`) whenever the server is reachable beyond localhost; without one it
  accepts any request. Put TLS in front of it (a reverse proxy or load balancer); the server itself speaks plain HTTP.
- **Keep the limits on.** The defaults bound the request body (1 MiB), the questions per state (64), the options per
  question (256), the state length (200,000 characters), the admitted requests (256) and the time per request (30 s).
  Raise them deliberately, not globally.
- **Pin model revisions** in production (`load(..., revision=...)`) and download weights from the Hub repositories
  you trust. OpenDecider loads safetensors weights and never runs code from a model repository.
- **Inputs are data.** A state is only ever tokenised and scored; nothing in it is executed. As with any model,
  though, a decision can be wrong or steered by adversarial text, so gate irreversible actions on confidence and keep
  a human or a rule-based check for them.
- The container runs as a non-root user (uid 10001). Server errors never return tracebacks or paths to the client;
  they are logged server-side with the request id.
