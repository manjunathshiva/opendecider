// The TypeScript client against reference outputs from the Python package (test/fixtures/make_fixtures.py), so the
// two cannot drift apart: the prompt, typed answers, agent answers, validation messages, limits, guard windows,
// log-probability reading, tool descriptions and constants.
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { letterProbabilities } from "../src/backends.js";
import {
  ATTACK_CHECKS,
  BLOCKED_MESSAGE,
  DEFAULT_THRESHOLD,
  Guard,
  THRESHOLDS,
  WINDOW_CHARS,
  WINDOW_OVERLAP,
  thresholdKey,
} from "../src/guard.js";
import { SYSTEM, forServer, render } from "../src/prompt.js";
import { answer, prepare, prepareQuestion } from "../src/questions.js";
import * as tools from "../src/tools.js";
import { VERSION } from "../src/version.js";

const fx = JSON.parse(readFileSync(new URL("./fixtures/parity.json", import.meta.url), "utf8"));

/** `actual` equals `expected`, with numbers equal to 12 significant digits: floating-point sums can differ in the last
 * bit between languages (Python 3.12+ compensates its float sums) and C libraries. */
function expectClose(actual: unknown, expected: unknown, path = "answer"): void {
  if (typeof expected === "number" && typeof actual === "number") {
    expect(
      Math.abs(actual - expected) <= 1e-12 * Math.max(1, Math.abs(expected)),
      `${path}: ${actual} vs ${expected}`,
    ).toBe(true);
  } else if (expected !== null && typeof expected === "object") {
    expect(Object.keys(actual as object), path).toEqual(Object.keys(expected));
    for (const [k, v] of Object.entries(expected))
      expectClose((actual as Record<string, unknown>)[k], v, `${path}.${k}`);
  } else {
    expect(actual, path).toEqual(expected);
  }
}

type Spec = { text: string; count: number; tail?: string; key?: string };
const build = (s: Spec) => {
  const text = s.text.repeat(s.count) + (s.tail ?? "");
  return s.key === undefined ? text : { [s.key]: text };
};

describe("parity with the Python package", () => {
  it("has the same version", () => {
    const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"));
    expect(VERSION).toBe(fx.version);
    if (pkg.name === "@opendecider/client") expect(pkg.version).toBe(fx.version); // (compat/ has its own package.json)
  });

  it("renders the prompt the models were trained on", () => {
    expect(SYSTEM).toBe(fx.system);
    for (const r of fx.renders) {
      const q = prepareQuestion(r.question);
      expect(render(r.state, q.instructions, q.options)).toBe(r.prompt);
      expect(forServer(r.prompt)).toBe(r.sent); // what Ollama, LM Studio and vLLM receive
    }
  });

  it("turns questions into the same options", () => {
    for (const a of fx.answers) {
      const prep = prepareQuestion(a.question);
      expect(Object.keys(a.probabilities)).toEqual(prep.options.map(([n]) => n));
    }
  });

  it("builds the same typed and agent answers", () => {
    for (const a of fx.answers) {
      const typed = answer(prepareQuestion(a.question), a.probabilities);
      expectClose(typed, a.answer);
      expect(tools.agentAnswer(typed)).toEqual(a.agent); // rounded to 4 places: exactly equal
    }
  });

  it("rejects invalid questions with the same messages", () => {
    for (const c of fx.invalid) expect(() => prepare({ q: c.question })).toThrow(c.error);
  });

  it("enforces the same limits", () => {
    for (const c of fx.limits) {
      const run = () => tools.check(build(c.state), prepare(c.questions));
      if (c.error === null) expect(run).not.toThrow();
      else expect(run).toThrow(c.error);
    }
  });

  it("splits long prompts into the same windows", () => {
    const guard = new Guard({ model: "ollama:x" });
    for (const c of fx.windows) {
      const got = (guard as unknown as { windows(p: string): string[] }).windows(build(c.prompt) as string);
      expect(
        got.map((w) => ({
          length: [...w].length,
          head: [...w].slice(0, 12).join(""),
          tail: [...w].slice(-12).join(""),
        })),
      ).toEqual(c.windows);
    }
  });

  it("keys model names for their thresholds the same way", () => {
    for (const c of fx.threshold_keys) expect(thresholdKey(c.name), c.name).toBe(c.key);
  });

  it("reads option probabilities from log-probabilities the same way", () => {
    for (const c of fx.letters) {
      const p = letterProbabilities(
        { choices: [{ logprobs: { content: [{ top_logprobs: c.top_logprobs }] } }] },
        c.names,
        "http://x",
        "m",
      );
      for (const n of c.names) expect(p[n]).toBeCloseTo(c.probabilities[n], 12);
    }
  });

  it("describes the tools and the guard the same way", () => {
    expect(tools.INSTRUCTIONS).toBe(fx.instructions);
    expect(tools.DESCRIPTIONS).toEqual(fx.descriptions);
    expect(ATTACK_CHECKS).toEqual(fx.attack_checks);
    expect(THRESHOLDS).toEqual(fx.thresholds);
    expect({
      MAX_QUESTIONS: tools.MAX_QUESTIONS,
      MAX_OPTIONS: tools.MAX_OPTIONS,
      MAX_STATE_CHARS: tools.MAX_STATE_CHARS,
      MAX_BATCH_STATES: tools.MAX_BATCH_STATES,
      MAX_BATCH_ITEMS: tools.MAX_BATCH_ITEMS,
      WINDOW_CHARS,
      WINDOW_OVERLAP,
      DEFAULT_THRESHOLD,
      BLOCKED_MESSAGE,
    }).toEqual(fx.constants);
  });
});
