/** Loading a served model, and typed questions about a state: `systemOne` and `systemOneBatch`. */
import { DEFAULT_URLS, LogprobBackend, ServedBackend, type Backend, type Info, type Item } from "./backends.js";
import { InputError } from "./errors.js";
import { env, type ConnectionOptions } from "./http.js";
import { jsonValue, pyRepr, type JsonValue } from "./json.js";
import { answer, prepare, type AnswersFor, type AnyAnswer, type Prepared, type Questions } from "./questions.js";

/** The answers to typed questions about one state. */
export interface SystemOneResult<Q extends Questions = Questions> {
  /** The model that answered. */
  model: string;
  /** One answer per question, by name, typed by its question (`choice(...)` gets a ChoiceAnswer, and so on). */
  answers: AnswersFor<Q>;
  usage: { input_tokens: number; output_tokens: number };
  /** Present when an input was shortened to fit the model. */
  warnings?: string[];
  latency_ms?: number;
}

/** Per-call options. */
export interface CallOptions {
  /** Cancels the call (e.g. the AbortSignal an agent framework passes to its tools). */
  signal?: AbortSignal;
}

/** What the client runs (served models only), for the error a local model name gets. */
export const LOCAL_MODELS =
  "@opendecider/client calls served models only: an `opendecider serve` URL (http://... or https://...), " +
  "ollama:<model>, lmstudio:<model>, or openai:<model> with baseUrl. To run a model on this machine, use the Python " +
  "package (pip install opendecider) or its server (opendecider serve, also as a Docker image)";

/** A loaded model: `systemOne(state, questions)` answers typed questions about one state. */
export class OpenDecider {
  constructor(
    readonly backend: Backend,
    readonly meta: Readonly<Record<string, unknown>>,
  ) {}

  get name(): string {
    return typeof this.meta["name"] === "string" && this.meta["name"] ? this.meta["name"] : "opendecider";
  }

  /** Validate typed questions without a model (throws InputError with a readable message). */
  static prepare(questions: Questions): Map<string, Prepared> {
    return prepare(questions);
  }

  private assemble<Q extends Questions>(
    qs: Map<string, Prepared>,
    probs: Record<string, number>[],
    info: Info[],
  ): SystemOneResult<Q> {
    const entries: [string, AnyAnswer][] = [];
    const warnings: string[] = [];
    let i = 0;
    for (const [k, q] of qs) {
      const a = answer(q, probs[i]!);
      if (info[i]?.truncated) {
        a.truncated = true;
        warnings.push(`question ${pyRepr(k)}: the state was truncated to fit the model's input length`);
      }
      entries.push([k, a]);
      i++;
    }
    const out: SystemOneResult<Q> = {
      model: this.name,
      answers: Object.fromEntries(entries) as AnswersFor<Q>,
      usage: { input_tokens: info.reduce((s, x) => s + (x.input_tokens || 0), 0), output_tokens: 0 },
    };
    if (warnings.length) out.warnings = warnings;
    return out;
  }

  private static items(state: JsonValue, qs: Map<string, Prepared>): Item[] {
    return [...qs.values()].map((q) => ({ state, instructions: q.instructions, options: q.options }));
  }

  /** Answer typed questions about one state (text, or any JSON). */
  async systemOne<Q extends Questions>(
    state: unknown,
    questions: Q,
    options: CallOptions = {},
  ): Promise<SystemOneResult<Q>> {
    const qs = prepare(questions);
    const st = typeof state === "string" ? state : jsonValue(state);
    const t0 = performance.now();
    const { probs, info } = await this.backend.decideMany(OpenDecider.items(st, qs), options.signal);
    const out = this.assemble<Q>(qs, probs, info);
    out.latency_ms = Math.round((performance.now() - t0) * 10) / 10;
    return out;
  }

  /** The same questions about many states, in one call (the server's batch endpoint for opendecider serve). */
  async systemOneBatch<Q extends Questions>(
    states: readonly unknown[],
    questions: Q,
    options: CallOptions = {},
  ): Promise<SystemOneResult<Q>[]> {
    if (!Array.isArray(states)) throw new InputError("states must be a list");
    const qs = prepare(questions);
    const sts = states.map((s, i) => (typeof s === "string" ? s : jsonValue(s, `state ${i}`)));
    if (sts.length === 0) return [];
    const n = qs.size;
    const { probs, info } = this.backend.decideBatch
      ? await this.backend.decideBatch(sts, [...qs.values()], options.signal)
      : await this.backend.decideMany(
          sts.flatMap((s) => OpenDecider.items(s, qs)),
          options.signal,
        );
    return sts.map((_, i) => this.assemble<Q>(qs, probs.slice(i * n, (i + 1) * n), info.slice(i * n, (i + 1) * n)));
  }

  /** True when the server answers (opendecider serve: GET /ready; other servers: GET /models). */
  ping(options: CallOptions = {}): Promise<boolean> {
    return this.backend.ping(options.signal);
  }
}

/** The error for a model this client cannot call. A URL of another scheme is not quoted: it may carry a password. */
function notServed(model: string): InputError {
  if (/^[a-z][a-z0-9+.-]*:\/\//i.test(model)) {
    return new InputError(`the server URL must be an http or https URL; ${LOCAL_MODELS}`);
  }
  return new InputError(`cannot load ${pyRepr(model)}: ${LOCAL_MODELS}`);
}

/** True for an `opendecider serve` URL given as the model name. */
export function isUrl(name: string): boolean {
  return /^https?:\/\//i.test(name);
}

/** "lmstudio:model" / "ollama:model" / "openai:model" -> [model, base URL]; null when it is not such a name. */
export function parse(name: string, baseUrl?: string): [string, string] | null {
  const i = name.indexOf(":");
  if (i < 0) return null;
  const scheme = name.slice(0, i);
  const model = name.slice(i + 1);
  if (!(Object.hasOwn(DEFAULT_URLS, scheme) || scheme === "openai") || !model || model.startsWith("//")) return null;
  const url = baseUrl || env("OPENDECIDER_REMOTE_URL") || DEFAULT_URLS[scheme];
  if (!url) throw new InputError("openai:<model> needs baseUrl (or OPENDECIDER_REMOTE_URL)");
  return [model, url];
}

/**
 * Connect to a served model:
 *
 *     await load("http://localhost:8000")             // opendecider serve
 *     await load("ollama:opendecider-small")          // Ollama on http://127.0.0.1:11434
 *     await load("lmstudio:opendecider-small")        // LM Studio on http://127.0.0.1:1234
 *     await load("openai:opendecider-small", { baseUrl: "http://gpu-box:8000/v1" })   // e.g. vLLM
 */
export async function load(model: string, options: ConnectionOptions & CallOptions = {}): Promise<OpenDecider> {
  if (typeof model !== "string" || !model) throw new InputError(`model must be a name or URL, got ${pyRepr(model)}`);
  if (isUrl(model)) {
    const b = await ServedBackend.connect(model, options, options.signal);
    return new OpenDecider(b, { name: b.name, kind: "served", served_kind: b.servedKind, base_url: b.http.baseUrl });
  }
  const target = parse(model, options.baseUrl);
  if (!target) throw notServed(model);
  const b = new LogprobBackend(target[0], target[1], options);
  return new OpenDecider(b, { name: target[0], kind: "remote", base_url: b.http.baseUrl });
}

/** After a failed connection, a Decider fails at once for this long before the next attempt. */
export const LOAD_RETRY_MS = 5000;

/** A model by name (connected on the first call) or an already-loaded OpenDecider: what the tools, Router and Guard
 * share. Pass one Decider to several of them to share one connection. */
export class Decider {
  readonly name: string;
  private loaded: OpenDecider | null;
  private loading: Promise<OpenDecider> | null = null;
  private failed: { until: number; error: unknown } | null = null;

  constructor(
    model: string | OpenDecider,
    readonly options: ConnectionOptions = {},
  ) {
    if (model instanceof OpenDecider) {
      this.name = model.name;
      this.loaded = model;
    } else {
      this.name = model;
      this.loaded = null;
      if (!isUrl(model) && !parse(model, options.baseUrl)) throw notServed(model);
    }
  }

  /** The loaded model's own name once it is loaded (what answered), else the name it was asked for. */
  get label(): string {
    return this.loaded?.name ?? this.name;
  }

  get isLoaded(): boolean {
    return this.loaded !== null;
  }

  /** The model, connecting on the first call; after a failed connection, calls fail at once for 5 s. */
  async model(): Promise<OpenDecider> {
    if (this.loaded) return this.loaded;
    if (this.failed && performance.now() < this.failed.until) throw this.failed.error;
    this.loading ??= load(this.name, this.options)
      .then(
        (m) => {
          this.loaded = m;
          this.failed = null;
          return m;
        },
        (e: unknown) => {
          this.failed = { until: performance.now() + LOAD_RETRY_MS, error: e };
          throw e;
        },
      )
      .finally(() => {
        this.loading = null;
      });
    return this.loading;
  }

  /** What is behind this Decider, without connecting. */
  status(): Record<string, unknown> {
    const out: Record<string, unknown> = { model: this.name, loaded: this.loaded !== null };
    if (this.loaded) {
      out["kind"] = this.loaded.meta["kind"] ?? null;
      out["device"] = this.loaded.backend.device;
    }
    return out;
  }
}

/** The Decider for `model`: a Decider passes through; a name or a loaded OpenDecider gets a new one. */
export function asDecider(model: string | OpenDecider | Decider, options?: ConnectionOptions): Decider {
  if (model instanceof Decider) {
    if (options && Object.keys(options).length)
      throw new InputError("pass connection options to the Decider itself, not with it");
    return model;
  }
  return new Decider(model, options);
}
