/** Routing: pick one route for a state, with a fallback for low confidence and errors (a port of tools.Router). */
import { InputError } from "./errors.js";
import type { ConnectionOptions } from "./http.js";
import { log, logOnce } from "./log.js";
import { asDecider, type CallOptions, type Decider, type OpenDecider } from "./model.js";
import { choice, prepare, type Criteria } from "./questions.js";
import { choose } from "./tools.js";

/** The reasons a Decision gives. */
export const REASONS = ["top_choice", "low_confidence", "empty_input", "error"] as const;

/** One routing decision: what the router returned and why. */
export interface Decision {
  /** The route taken (null when the decision failed and the error was thrown). */
  route: string | null;
  /** "top_choice" (the model's top route), "low_confidence" (below `minConfidence`: the fallback), "empty_input"
   * (nothing to decide on: the fallback), or "error" (the decision failed: the fallback, or thrown). */
  reason: (typeof REASONS)[number];
  /** The model's top route, its probability, and every route's probability (empty when no decision was made). */
  choice: string | null;
  confidence: number | null;
  probabilities: Record<string, number>;
  model: string;
  latencyMs: number;
  truncated: boolean;
  error: string | null;
}

/** The settings of a Router. */
export interface RouterOptions {
  /** The route names, as a list or as {route: "when to take it"} (descriptions help). */
  routes: Criteria;
  /** The routing question, e.g. "Which agent should handle this request?". */
  instructions: string;
  /** An `opendecider serve` URL, `ollama:...`, `lmstudio:...`, `openai:...`, or a loaded OpenDecider / Decider. */
  model: string | OpenDecider | Decider;
  /** How to reach the model, when `model` is a name. */
  connection?: ConnectionOptions;
  /** Take this route when the top route's probability is below `minConfidence`. */
  fallback?: string;
  minConfidence?: number;
  /** "raise" (the default) lets a failed decision throw; "fallback" takes the fallback route and logs the error. */
  onError?: "raise" | "fallback";
  /** Called with every Decision, failed ones included: for logs, metrics or audits. A hook that throws is logged
   * and never breaks routing. */
  onDecision?: ((d: Decision) => unknown) | ((d: Decision) => unknown)[];
}

const ms = (t0: number) => Math.round((performance.now() - t0) * 10) / 10;

function isEmpty(state: unknown): boolean {
  return (
    state === null ||
    state === undefined ||
    (typeof state === "string" && !state.trim()) ||
    (Array.isArray(state) && state.length === 0) ||
    (typeof state === "object" &&
      !Array.isArray(state) &&
      Object.getPrototypeOf(state) === Object.prototype &&
      Object.keys(state).length === 0)
  );
}

/** Picks one route for a state: the most likely route, or `fallback` when its probability is below `minConfidence`. */
export class Router {
  readonly routes: Criteria;
  readonly instructions: string;
  readonly fallback: string | null;
  readonly minConfidence: number;
  readonly onError: "raise" | "fallback";
  readonly hooks: ((d: Decision) => unknown)[];
  readonly decider: Decider;
  /** The last decision, for logging; concurrent calls overwrite it. */
  last: Decision | null = null;

  constructor(options: RouterOptions) {
    const { minConfidence = 0, onError = "raise" } = options;
    const fallback = options.fallback ?? null;
    if (!(minConfidence >= 0 && minConfidence <= 1)) {
      throw new InputError(`minConfidence is a probability, between 0 and 1 (got ${minConfidence})`);
    }
    if (fallback === null && minConfidence > 0) throw new InputError("minConfidence needs a fallback route");
    if (onError !== "raise" && onError !== "fallback")
      throw new InputError(`onError must be "raise" or "fallback" (got '${String(onError)}')`);
    if (onError === "fallback" && fallback === null) throw new InputError('onError: "fallback" needs a fallback route');
    prepare({ routes: choice(options.instructions, options.routes) }); // at least 2 routes and an instruction, now
    const hooks =
      options.onDecision === undefined
        ? []
        : Array.isArray(options.onDecision)
          ? options.onDecision
          : [options.onDecision];
    if (!hooks.every((h) => typeof h === "function"))
      throw new InputError("onDecision must be a function or a list of functions");
    this.routes = options.routes;
    this.instructions = options.instructions;
    this.fallback = fallback;
    this.minConfidence = minConfidence;
    this.onError = onError;
    this.hooks = [...hooks];
    this.decider = asDecider(options.model, options.connection);
  }

  /** The full decision for `state` (text or JSON). An empty state (blank text, {}, [] or null) has nothing to decide
   * on: it takes the fallback, or throws InputError without one. */
  async decide(state: unknown, options: CallOptions = {}): Promise<Decision> {
    const t0 = performance.now();
    let decision: Decision;
    try {
      decision = await this.decideOrThrow(state, t0, options);
    } catch (e) {
      if (options.signal?.aborted) throw e; // cancelled: not a decision
      decision = {
        route: this.onError === "fallback" ? this.fallback : null,
        reason: "error",
        choice: null,
        confidence: null,
        probabilities: {},
        model: this.decider.label,
        latencyMs: ms(t0),
        truncated: false,
        error: `${(e as Error).name ?? "Error"}: ${(e as Error).message ?? String(e)}`,
      };
      this.report(decision);
      if (this.onError === "raise") throw e;
      logOnce("error", `routing failed, taking the fallback route '${this.fallback}': ${decision.error}`);
      return decision;
    }
    this.report(decision);
    return decision;
  }

  private async decideOrThrow(state: unknown, t0: number, options: CallOptions): Promise<Decision> {
    const base = { model: this.decider.label, error: null };
    if (isEmpty(state)) {
      if (this.fallback === null) throw new InputError("nothing to route on: the state is empty");
      logOnce("warn", `nothing to route on (the state is empty): taking the fallback route '${this.fallback}'`);
      return {
        ...base,
        route: this.fallback,
        reason: "empty_input",
        choice: null,
        confidence: null,
        probabilities: {},
        latencyMs: ms(t0),
        truncated: false,
      };
    }
    const a = await choose(this.decider, state, this.instructions, this.routes, options);
    if (a.truncated)
      logOnce("warn", "the routing input was shortened to fit the model's input; route on a shorter field");
    const top = a["choice"] as string;
    const low = this.fallback !== null && a.confidence < this.minConfidence;
    return {
      route: low ? this.fallback : top,
      reason: low ? "low_confidence" : "top_choice",
      choice: top,
      confidence: a.confidence,
      probabilities: a["probabilities"] as Record<string, number>,
      model: this.decider.label,
      latencyMs: ms(t0),
      truncated: Boolean(a.truncated),
      error: null,
    };
  }

  private report(decision: Decision): void {
    this.last = decision;
    for (const hook of this.hooks) {
      try {
        const r = hook(decision);
        if (r instanceof Promise) r.catch((e: unknown) => log("error", `an onDecision hook failed: ${String(e)}`));
      } catch (e) {
        log("error", `an onDecision hook failed: ${String(e)}`);
      }
    }
  }

  /** The route name for `state`: `(await decide(state)).route`. */
  async route(state: unknown, options: CallOptions = {}): Promise<string | null> {
    return (await this.decide(state, options)).route;
  }

  /** Every name the router can return: the routes, then the fallback. */
  get names(): string[] {
    const names = Array.isArray(this.routes) ? [...this.routes] : Object.keys(this.routes);
    return this.fallback !== null && !names.includes(this.fallback) ? [...names, this.fallback] : names;
  }
}
