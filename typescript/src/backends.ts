/**
 * Where a model runs: `opendecider serve` (by URL), or LM Studio, Ollama, vLLM and other OpenAI-compatible servers that
 * return token log-probabilities (a port of opendecider/remote.py).
 */
import { InputError, ServerError } from "./errors.js";
import { HttpClient, mapLimit, type ConnectionOptions } from "./http.js";
import { pyJson, type JsonValue } from "./json.js";
import { LETTERS, SYSTEM, render, type Option } from "./prompt.js";
import type { Prepared } from "./questions.js";

/** One question to answer: the state, the instructions and the options. */
export interface Item {
  state: JsonValue;
  instructions: string;
  options: readonly Option[];
}
/** Per item: input tokens (the request's total, on its first item) and whether the state was shortened. */
export interface Info {
  input_tokens: number;
  truncated: boolean;
}
export interface Decided {
  probs: Record<string, number>[];
  info: Info[];
}

export interface Backend {
  readonly kind: "served" | "remote";
  /** Where the model runs, for status reports. */
  readonly device: string;
  decideMany(items: readonly Item[], signal?: AbortSignal): Promise<Decided>;
  /** The same questions about many states (a batch endpoint where there is one). */
  decideBatch?(states: readonly JsonValue[], questions: readonly Prepared[], signal?: AbortSignal): Promise<Decided>;
  /** True when the server answers. */
  ping(signal?: AbortSignal): Promise<boolean>;
}

export const DEFAULT_URLS: Readonly<Record<string, string>> = {
  lmstudio: "http://127.0.0.1:1234/v1",
  ollama: "http://127.0.0.1:11434/v1",
};

const isRecord = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);

/** A model behind `opendecider serve`: each question goes to the server as the named options the local model would
 * see, so answers match the server model's own exactly. */
export class ServedBackend implements Backend {
  readonly kind = "served";
  readonly device: string;
  /** Questions per request (the server's default limit). */
  maxQuestions = 64;
  /** States per batch request (the server's default limit). */
  maxBatchStates = 256;
  /** Request body size to stay under (the server's default limit is 1 MiB). */
  maxBodyBytes = 1_000_000;

  private constructor(
    readonly http: HttpClient,
    readonly name: string,
    readonly servedKind: string | null,
  ) {
    this.device = `served (${http.baseUrl})`;
  }

  /** Connect to `opendecider serve` at `baseUrl`: fails now, with a clear message, rather than on the first
   * decision. */
  static async connect(baseUrl: string, options: ConnectionOptions = {}, signal?: AbortSignal): Promise<ServedBackend> {
    const http = new HttpClient(
      baseUrl,
      options,
      30_000,
      new Set([429, 502, 503]),
      (status, text) => servedError(http.baseUrl)(status, text), // (called only once `http` exists)
    );
    let r: Response;
    let text: string;
    try {
      ({ response: r, text } = await http.send(
        "GET",
        "/v1/models",
        undefined,
        Math.min(http.timeoutMs, 10_000),
        signal,
      ));
    } catch (e) {
      if (!(e instanceof ServerError) || !e.unreachable) throw e; // the caller cancelled, or a redirect with a key
      throw new ServerError(
        `cannot reach opendecider serve at ${http.baseUrl} (${e.reason ?? e.message}); is it running?`,
        { unreachable: true },
      );
    }
    if (r.status === 401 || r.status === 403) throw servedError(http.baseUrl)(r.status, "");
    if (!r.ok)
      throw new ServerError(`cannot reach opendecider serve at ${http.baseUrl} (HTTP ${r.status}); is it running?`);
    let body: unknown;
    try {
      body = JSON.parse(text);
    } catch {
      throw new ServerError(
        `${http.baseUrl} is not opendecider serve: GET /v1/models answered with something that is not JSON`,
      );
    }
    if (isRecord(body) && Array.isArray(body["data"]) && !Array.isArray(body["models"])) {
      // an OpenAI-style listing: Ollama, LM Studio or vLLM, which this client reaches through the model's own name
      const model = (body["data"][0] as { id?: unknown } | undefined)?.id;
      const example = typeof model === "string" ? model : "<model>";
      throw new InputError(
        `${http.baseUrl} is an OpenAI-compatible server (Ollama, LM Studio or vLLM), not opendecider serve: ` +
          `load "openai:${example}" with baseUrl "${http.baseUrl}${http.baseUrl.endsWith("/v1") ? "" : "/v1"}" ` +
          `(or "ollama:${example}" / "lmstudio:${example}" at their default local addresses)`,
      );
    }
    const listed = isRecord(body) && Array.isArray(body["models"]) ? body["models"][0] : undefined;
    const meta = isRecord(listed) ? listed : {}; // an unexpected listing: use the URL as the name
    return new ServedBackend(
      http,
      typeof meta["name"] === "string" && meta["name"] ? meta["name"] : http.baseUrl,
      typeof meta["kind"] === "string" ? meta["kind"] : null,
    );
  }

  private static questions(group: readonly { instructions: string; options: readonly Option[] }[]) {
    return Object.fromEntries(
      group.map((q, j) => [
        `q${j}`,
        {
          type: "choice",
          instructions: q.instructions,
          criteria: Object.fromEntries(q.options),
        },
      ]),
    );
  }

  private read(r: unknown, group: readonly { options: readonly Option[] }[]): Decided {
    const probs: Record<string, number>[] = [];
    const info: Info[] = [];
    group.forEach((q, j) => {
      const a = isRecord(r) && isRecord(r["answers"]) ? r["answers"][`q${j}`] : undefined;
      const p = isRecord(a) && isRecord(a["probabilities"]) ? a["probabilities"] : null;
      const out: [string, number][] = [];
      for (const [name] of q.options) {
        const v = p && Object.hasOwn(p, name) ? p[name] : undefined;
        if (typeof v !== "number") {
          throw new ServerError(
            `${this.http.baseUrl} returned no probabilities for every option; is it opendecider ` +
              "serve (a Jev or Laya server returns only the top answer)?",
          );
        }
        out.push([name, v]);
      }
      probs.push(Object.fromEntries(out)); // (fromEntries: a label such as "__proto__" stays a plain key)
      info.push({ input_tokens: 0, truncated: isRecord(a) && Boolean(a["truncated"]) });
    });
    const usage = isRecord(r) && isRecord(r["usage"]) ? Number(r["usage"]["input_tokens"]) || 0 : 0;
    if (info[0]) info[0].input_tokens = Math.trunc(usage); // the request's total
    return { probs, info };
  }

  async decideMany(items: readonly Item[], signal?: AbortSignal): Promise<Decided> {
    const groups: Item[][] = []; // consecutive questions about the same state share a request
    for (const it of items) {
      const last = groups.at(-1);
      if (last && last[0]!.state === it.state && last.length < this.maxQuestions) last.push(it);
      else groups.push([it]);
    }
    const res = await mapLimit(groups, this.http.workers, async (g) =>
      this.read(
        await this.http.postJson(
          "/v1/systemone",
          { state: g[0]!.state, questions: ServedBackend.questions(g) },
          signal,
        ),
        g,
      ),
    );
    return { probs: res.flatMap((r) => r.probs), info: res.flatMap((r) => r.info) };
  }

  async decideBatch(
    states: readonly JsonValue[],
    questions: readonly Prepared[],
    signal?: AbortSignal,
  ): Promise<Decided> {
    if (questions.length > this.maxQuestions) {
      return this.decideMany(
        states.flatMap((state) => questions.map((q) => ({ state, instructions: q.instructions, options: q.options }))),
        signal,
      );
    }
    const qs = ServedBackend.questions(questions);
    const overhead = new TextEncoder().encode(JSON.stringify({ states: [], questions: qs })).length;
    const chunks: JsonValue[][] = []; // up to maxBatchStates states and maxBodyBytes per request
    let size = overhead;
    for (const st of states) {
      const n = new TextEncoder().encode(JSON.stringify(st)).length + 1;
      const last = chunks.at(-1);
      if (last && last.length < this.maxBatchStates && size + n <= this.maxBodyBytes) {
        last.push(st);
        size += n;
      } else {
        chunks.push([st]);
        size = overhead + n;
      }
    }
    const res = await mapLimit(chunks, this.http.workers, async (chunk) => {
      const r = await this.http.postJson("/v1/systemone/batch", { states: chunk, questions: qs }, signal);
      const results = isRecord(r) && Array.isArray(r["results"]) ? r["results"] : [];
      if (results.length !== chunk.length) {
        throw new ServerError(`${this.http.baseUrl} returned ${results.length} results for ${chunk.length} states`);
      }
      return results.map((x) => this.read(x, questions));
    });
    const flat = res.flat();
    return { probs: flat.flatMap((r) => r.probs), info: flat.flatMap((r) => r.info) };
  }

  async ping(signal?: AbortSignal): Promise<boolean> {
    try {
      return (await this.http.send("GET", "/ready", undefined, 3000, signal)).response.status === 200;
    } catch {
      return false;
    }
  }
}

function servedError(baseUrl: string) {
  return (status: number, text: string): Error => {
    let detail = text;
    try {
      const body = JSON.parse(text) as unknown;
      if (isRecord(body) && body["detail"])
        detail = typeof body["detail"] === "string" ? body["detail"] : pyJson(body["detail"] as JsonValue);
    } catch {
      /* not JSON (a proxy's HTML page, say): report the text as it came */
    }
    if (status === 400 || status === 413 || status === 422)
      return new InputError(`${baseUrl} rejected the request: ${detail}`);
    if (status === 401 || status === 403) {
      return new ServerError(
        `${baseUrl} rejected the credentials (HTTP ${status}); set OPENDECIDER_REMOTE_API_KEY or pass apiKey`,
      );
    }
    return new ServerError(`${baseUrl} answered HTTP ${status}: ${detail}`);
  };
}

/** An OpenDecider model (opendecider-small or -small-td: the GGUF build, or the base model with the LoRA adapter)
 * served by LM Studio, Ollama, vLLM or another OpenAI-compatible server that returns `top_logprobs`. The client builds
 * the prompt the model was trained on and reads each option letter's probability from the next-token
 * log-probabilities. Up to 26 options per question. */
export class LogprobBackend implements Backend {
  readonly kind = "remote";
  readonly device: string;
  readonly http: HttpClient;

  constructor(
    readonly model: string,
    baseUrl: string,
    options: ConnectionOptions = {},
  ) {
    this.http = new HttpClient(
      baseUrl,
      options,
      120_000,
      new Set([429, 500, 502, 503, 504]),
      (status, text) => new ServerError(`${this.http.baseUrl} answered HTTP ${status}: ${text}`),
    );
    this.device = `remote (${this.http.baseUrl})`;
  }

  /** Option probabilities for one item, from the next-token log-probabilities. */
  async decide(item: Item, signal?: AbortSignal): Promise<[Record<string, number>, number]> {
    const names = item.options.map(([n]) => n);
    if (names.length > LETTERS.length) {
      throw new InputError(
        `at most ${LETTERS.length} options per question through a model server (got ${names.length}); ` +
          "use the PyTorch model for more",
      );
    }
    const r = await this.http.postJson(
      "/chat/completions",
      {
        model: this.model,
        messages: [
          { role: "system", content: SYSTEM },
          { role: "user", content: render(item.state, item.instructions, item.options) },
        ],
        max_tokens: 1,
        temperature: 0,
        logprobs: true,
        top_logprobs: 20,
      },
      signal,
    );
    return [
      letterProbabilities(r, names, this.http.baseUrl, this.model),
      Number((r as { usage?: { prompt_tokens?: unknown } })?.usage?.prompt_tokens) || 0,
    ];
  }

  async decideMany(items: readonly Item[], signal?: AbortSignal): Promise<Decided> {
    const res = await mapLimit(items, this.http.workers, (it) => this.decide(it, signal));
    return { probs: res.map(([p]) => p), info: res.map(([, t]) => ({ input_tokens: t, truncated: false })) };
  }

  async ping(signal?: AbortSignal): Promise<boolean> {
    try {
      return (await this.http.send("GET", "/models", undefined, 3000, signal)).response.status === 200;
    } catch {
      return false;
    }
  }
}

/** Each option's probability from a chat completion's first-token `top_logprobs`, normalised over the options. */
export function letterProbabilities(
  r: unknown,
  names: readonly string[],
  baseUrl: string,
  model: string,
): Record<string, number> {
  const top = (r as { choices?: { logprobs?: { content?: { top_logprobs?: unknown }[] } }[] })?.choices?.[0]?.logprobs
    ?.content?.[0]?.top_logprobs;
  if (!Array.isArray(top) || top.length === 0) {
    throw new ServerError(
      `${baseUrl} returned no token log-probabilities for '${model}'. Use a server and engine that ` +
        "supports them (LM Studio and Ollama with the GGUF build, and vLLM, do; LM Studio's MLX engine does not)",
    );
  }
  // "A" and " A" are different tokens that both mean the letter A: add their probabilities
  const prob = new Map<string, number>();
  let least = Infinity;
  for (const t of top as { token?: unknown; logprob?: unknown }[]) {
    if (typeof t?.token !== "string" || typeof t.logprob !== "number") {
      throw new ServerError(`${baseUrl} returned malformed token log-probabilities for '${model}'`);
    }
    const letter = t.token.trim();
    const p = Math.exp(t.logprob);
    prob.set(letter, (prob.get(letter) ?? 0) + p);
    least = Math.min(least, p);
  }
  const wanted = [...LETTERS.slice(0, names.length)];
  if (!wanted.some((c) => prob.has(c))) {
    throw new ServerError(
      `none of the option letters ${wanted[0]}-${wanted.at(-1)} is among the most likely next tokens ` +
        `from '${model}'; is this an OpenDecider model (opendecider-small or -small-td)?`,
    );
  }
  const floor = least * Math.exp(-5.0); // a letter outside the top 20: well below the least likely token returned
  const w = wanted.map((c) => prob.get(c) ?? floor);
  const z = w.reduce((a, b) => a + b, 0);
  return Object.fromEntries(names.map((n, i) => [n, w[i]! / z]));
}
