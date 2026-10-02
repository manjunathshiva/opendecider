/** HTTP to one server: the URL and credential rules, timeouts, and retries for a busy or restarting server. */
import { InputError, ServerError } from "./errors.js";

export const DOWN_RETRY_MS = 5000; // after a server times out or cannot be reached, calls fail at once for this long
const LOOPBACK = new Set(["127.0.0.1", "localhost", "[::1]", "::1"]);
const MAX_TIMEOUT_MS = 2_147_483_647; // the largest delay a timer takes

/** How to reach a model server. */
export interface ConnectionOptions {
  /** The server's base URL for `ollama:`, `lmstudio:` and `openai:` models (default: OPENDECIDER_REMOTE_URL, else
   * LM Studio's or Ollama's local address). */
  baseUrl?: string;
  /** Sent as `Authorization: Bearer <apiKey>` (default: OPENDECIDER_REMOTE_API_KEY). */
  apiKey?: string;
  /** Milliseconds per request (default 30 s for opendecider serve, 120 s for other servers). */
  timeoutMs?: number;
  /** Requests in flight at once when one call needs several (default 4). */
  workers?: number;
  /** Send the API key over plain HTTP to another host (default: OPENDECIDER_REMOTE_ALLOW_HTTP=1). Only on a trusted
   * network path. */
  allowInsecureHttp?: boolean;
  /** Extra request headers (e.g. for a gateway). */
  headers?: Record<string, string>;
  /** The fetch implementation (default: the global fetch). */
  fetch?: typeof fetch;
}

/** An environment variable where the runtime has them (Node, Bun, Deno with --allow-env); otherwise undefined. */
export function env(name: string): string | undefined {
  try {
    const value = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env?.[name];
    return value || undefined;
  } catch {
    // Deno without env permission
    return undefined;
  }
}

/** `text` without its trailing slashes (a loop: a regular expression such as /\/+$/ takes quadratic time on a long run
 * of slashes). */
export function trimSlashes(text: string): string {
  let end = text.length;
  while (end > 0 && text.charCodeAt(end - 1) === 47) end--;
  return text.slice(0, end);
}

/** Resolve after `ms`, or reject when `signal` fires. */
export function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(signal.reason);
    const done = () => {
      signal?.removeEventListener("abort", stop);
      resolve();
    };
    const timer = setTimeout(done, ms);
    const stop = () => {
      clearTimeout(timer);
      reject(signal!.reason);
    };
    signal?.addEventListener("abort", stop, { once: true });
  });
}

/** `fn` over `items` with at most `limit` running at once; results in order. After a failure no new item starts
 * (the call has failed: more requests would only add load to a server in trouble). */
export async function mapLimit<T, R>(items: readonly T[], limit: number, fn: (item: T) => Promise<R>): Promise<R[]> {
  const out = new Array<R>(items.length);
  let next = 0;
  const run = async () => {
    while (next < items.length) {
      const i = next++;
      try {
        out[i] = await fn(items[i]!);
      } catch (e) {
        next = items.length;
        throw e;
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(Math.max(1, limit), items.length) }, run));
  return out;
}

export class HttpClient {
  readonly baseUrl: string;
  readonly apiKey: string | undefined;
  timeoutMs: number;
  readonly workers: number;
  private readonly fetchFn: typeof fetch;
  private readonly headers: Record<string, string>;
  private down: { until: number; message: string } | null = null;
  private readonly credentials: boolean;

  constructor(
    baseUrl: string,
    options: ConnectionOptions,
    defaultTimeoutMs: number,
    private readonly retryStatus: ReadonlySet<number>,
    private readonly onHttpError: (status: number, text: string) => Error,
  ) {
    this.baseUrl = trimSlashes(baseUrl);
    let url: URL;
    try {
      url = new URL(this.baseUrl);
    } catch {
      throw new InputError(`baseUrl must be an http or https URL, got '${baseUrl}'`);
    }
    if ((url.protocol !== "http:" && url.protocol !== "https:") || !url.hostname) {
      throw new InputError(`baseUrl must be an http or https URL, got '${baseUrl}'`);
    }
    if (url.username || url.password) {
      // (not echoed: it would carry the credentials into messages and logs)
      throw new InputError("the server URL must not contain a user name or password; pass the key as apiKey");
    }
    if (url.search || url.hash) {
      // (not echoed either: a query string often carries a token) the paths are appended to it, so it must end there
      throw new InputError("the server URL must not have a query string or a fragment; pass a key as apiKey");
    }
    this.baseUrl = trimSlashes(url.origin + url.pathname); // e.g. https://host/decider behind a proxy
    this.apiKey = options.apiKey || env("OPENDECIDER_REMOTE_API_KEY");
    const insecure = options.allowInsecureHttp ?? env("OPENDECIDER_REMOTE_ALLOW_HTTP") === "1";
    if (this.apiKey && url.protocol === "http:" && !LOOPBACK.has(url.hostname) && !insecure) {
      throw new InputError(
        `refusing to send an API key over plain HTTP to ${url.hostname}; use https, or set ` +
          "allowInsecureHttp (OPENDECIDER_REMOTE_ALLOW_HTTP=1) if the network path is trusted",
      );
    }
    this.timeoutMs = options.timeoutMs ?? defaultTimeoutMs;
    if (!(this.timeoutMs > 0 && this.timeoutMs <= MAX_TIMEOUT_MS)) {
      throw new InputError(`timeoutMs must be above 0 and at most ${MAX_TIMEOUT_MS} (got ${options.timeoutMs})`);
    }
    this.workers = options.workers ?? 4;
    if (!Number.isInteger(this.workers) || this.workers < 1) {
      throw new InputError(`workers must be a whole number of at least 1 (got ${options.workers})`);
    }
    this.fetchFn = options.fetch ?? ((...args: Parameters<typeof fetch>) => globalThis.fetch(...args));
    this.headers = { ...options.headers };
    // checked now, by name only: fetch's own error would quote the value, and these errors reach agents and logs
    const sent = { ...this.headers, ...(this.apiKey ? { authorization: `Bearer ${this.apiKey}` } : {}) };
    for (const [name, value] of Object.entries(sent)) {
      try {
        new Headers([[name, value]]);
      } catch {
        throw new InputError(
          name.toLowerCase() === "authorization" && this.apiKey
            ? "the API key is not a valid header value (a line break or a control character in it?)"
            : `the ${JSON.stringify(name)} header is not a valid header (a line break or a control character in it?)`,
        );
      }
    }
    // a credential, given as apiKey or as a header, must reach only this origin: never follow a redirect with one
    this.credentials =
      Boolean(this.apiKey) ||
      Object.keys(this.headers).some((h) => /^(authorization|proxy-authorization|cookie|x-api-key)$/i.test(h));
  }

  private request(method: "GET" | "POST", body: string | undefined, signal: AbortSignal): RequestInit {
    const headers: Record<string, string> = { ...this.headers };
    if (body !== undefined) headers["content-type"] = "application/json";
    if (this.apiKey) headers["authorization"] = `Bearer ${this.apiKey}`;
    return { method, headers, body, signal, redirect: this.credentials ? "manual" : "follow" };
  }

  private seconds(ms: number): string {
    return `${Number((ms / 1000).toPrecision(6))} s`;
  }

  /** One request, no retries: the response and its body, or a ServerError for a timeout or a server that cannot be
   * reached (the timeout covers reading the body too). */
  async send(
    method: "GET" | "POST",
    path: string,
    body: string | undefined,
    timeoutMs: number,
    signal?: AbortSignal,
  ): Promise<{ response: Response; text: string }> {
    // One controller per request, its timer cleared and its listener removed once the body is read: a timeout signal
    // (AbortSignal.timeout) would stay in memory until its timer fires, long after the request is done.
    const controller = new AbortController();
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      controller.abort(new DOMException("the request timed out", "TimeoutError"));
    }, timeoutMs);
    const cancel = () => controller.abort(signal!.reason);
    if (signal?.aborted) cancel();
    else signal?.addEventListener("abort", cancel, { once: true });
    let response: Response;
    let text: string;
    try {
      response = await this.fetchFn(this.baseUrl + path, this.request(method, body, controller.signal));
      if (
        this.credentials &&
        (response.type === "opaqueredirect" || (response.status >= 300 && response.status < 400))
      ) {
        await response.body?.cancel().catch(() => {});
        throw new ServerError(
          `${this.baseUrl} redirected the request${
            response.headers.get("location") ? ` to ${response.headers.get("location")}` : ""
          }; credentials are sent only to the URL given, so use the final URL`,
        );
      }
      text = await response.text();
    } catch (e) {
      if (e instanceof ServerError) throw e;
      if (signal?.aborted && !timedOut) throw signal.reason;
      if (timedOut) {
        const reason = `gave no answer within ${this.seconds(timeoutMs)}`;
        throw new ServerError(`${this.baseUrl} ${reason}`, { unreachable: true, reason });
      }
      const cause = (e as { cause?: { code?: string; message?: string } }).cause;
      const reason = cause?.code ?? cause?.message ?? (e as Error).message;
      throw new ServerError(`cannot reach ${this.baseUrl} (${reason}); is the server running and the model loaded?`, {
        unreachable: true,
        reason,
        cause: e,
      });
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener("abort", cancel);
    }
    return { response, text };
  }

  /** POST JSON with retries (429 and the server's retryable codes: two retries, 0.5 s then 1 s apart, or the server's
   * Retry-After up to 5 s) and the fail-fast window after an unreachable server. */
  async postJson(path: string, payload: unknown, signal?: AbortSignal): Promise<unknown> {
    const down = this.down;
    if (down && performance.now() < down.until) throw new ServerError(down.message, { unreachable: true });
    try {
      const out = await this.retrying(path, JSON.stringify(payload), signal);
      this.down = null;
      return out;
    } catch (e) {
      if (e instanceof ServerError && e.unreachable) {
        this.down = {
          until: performance.now() + DOWN_RETRY_MS,
          message: `${e.message} (calls fail at once for ${DOWN_RETRY_MS / 1000} s)`,
        };
      }
      throw e;
    }
  }

  private async retrying(path: string, body: string, signal?: AbortSignal): Promise<unknown> {
    for (let attempt = 0; ; attempt++) {
      let response: Response;
      let text: string;
      try {
        ({ response, text } = await this.send("POST", path, body, this.timeoutMs, signal));
      } catch (e) {
        if (e instanceof ServerError && e.reason === "ECONNRESET" && attempt < 2) {
          // a restarting server
          await sleep(500 * 2 ** attempt, signal);
          continue;
        }
        throw e;
      }
      if (response.ok) {
        try {
          return JSON.parse(text);
        } catch {
          throw new ServerError(
            `${this.baseUrl} answered with something that is not JSON: ${JSON.stringify(text.slice(0, 200))}`,
          );
        }
      }
      if (this.retryStatus.has(response.status) && attempt < 2) {
        await sleep(retryAfter(response.headers.get("retry-after"), 500 * 2 ** attempt), signal);
        continue;
      }
      throw this.onHttpError(response.status, text.slice(0, 300));
    }
  }
}

/** The server's Retry-After in milliseconds (capped at 5 s), else `fallback`. */
export function retryAfter(header: string | null, fallback: number): number {
  const s = header === null || header.trim() === "" ? NaN : Number(header);
  return Number.isFinite(s) ? Math.min(5000, Math.max(0, s * 1000)) : fallback;
}
