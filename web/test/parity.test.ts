// The Python package's token ids and prompts (fixtures/parity.json, from make_fixtures.py), and the whole pipeline on a
// tiny model with opendecider-nano's architecture (fixtures/tiny.onnx and tiny.json, from make_tiny_model.py).
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { Tokenizer } from "@huggingface/tokenizers";
import type { JsonValue, Option } from "@opendecider/client";
import * as ort from "onnxruntime-web";
import { describe, expect, it } from "vitest";
import { TOKENIZER } from "../src/manifest.js";
import { NanoBackend, batches } from "../src/nano.js";
import { SPECIAL, nanoIds, type Encode } from "../src/prompt.js";

const fixture = (name: string) => new URL(`./fixtures/${name}`, import.meta.url);
const fx = JSON.parse(readFileSync(fixture("parity.json"), "utf8")) as {
  texts: { text: string; ids: number[] }[];
  prompts: {
    state: JsonValue;
    instructions: string;
    options: Option[];
    max_len: number;
    ids: number[];
    truncated: boolean;
  }[];
};
const tiny = JSON.parse(readFileSync(fixture("tiny.json"), "utf8")) as { prompts: { probs: number[] }[] };
const tokJson = readFileSync(fixture("tokenizer.web.json"));
const tok = new Tokenizer(JSON.parse(tokJson.toString("utf8")), {});
const encode: Encode = (text) => tok.encode(text, { add_special_tokens: false }).ids;

describe("the tokenizer", () => {
  it("is the file this package pins (rebuilt from opendecider-nano's tokenizer.json)", () => {
    expect(tokJson.length).toBe(TOKENIZER.bytes);
    expect(createHash("sha256").update(tokJson).digest("hex")).toBe(TOKENIZER.sha256);
  });

  it("gives the Python package's ids, with special-token text as text", () => {
    for (const t of fx.texts) expect(encode(t.text), JSON.stringify(t.text)).toEqual(t.ids);
    expect(encode("[MASK] [CLS]")).not.toContain(SPECIAL.mask);
  });
});

describe("the prompt", () => {
  it("is the Python package's, including where the state is shortened", () => {
    expect(fx.prompts.some((p) => p.truncated)).toBe(true);
    for (const p of fx.prompts) {
      expect(nanoIds(encode, p.state, p.instructions, p.options, p.max_len)).toEqual({
        ids: p.ids,
        truncated: p.truncated,
      });
    }
  });
});

describe("the model (tiny, random weights, opendecider-nano's architecture)", () => {
  const full = fx.prompts.filter((p) => p.max_len === 2048);
  const items = full.map((p) => ({ state: p.state, instructions: p.instructions, options: p.options }));
  const session = () =>
    ort.InferenceSession.create(readFileSync(fixture("tiny.onnx")), { executionProviders: ["wasm"] });

  it.each([16384, 64, 1])("gives native ONNX Runtime's probabilities, batched in up to %i tokens", async (budget) => {
    const backend = new NanoBackend(await session(), encode, "wasm", "q8", budget);
    const { probs, info } = await backend.decideMany(items);
    probs.forEach((p, i) => {
      const want = tiny.prompts[i]!.probs;
      const names = items[i]!.options.map(([n]) => n);
      expect(Object.keys(p)).toEqual(names);
      names.forEach((n, k) => expect(Math.abs(p[n]! - want[k]!)).toBeLessThan(1e-5));
    });
    expect(info.map((x) => x.input_tokens)).toEqual(full.map((p) => p.ids.length));
    await backend.dispose();
  });

  it("answers concurrent calls one pass at a time, each with its own answers", async () => {
    const backend = new NanoBackend(await session(), encode, "wasm", "q8", 64);
    const [a, b] = await Promise.all([backend.decideMany(items.slice(0, 5)), backend.decideMany(items.slice(5, 9))]);
    expect(a.probs.length + b.probs.length).toBe(9);
    b.probs.forEach((p, i) => expect(Object.values(p)[0]).toBeCloseTo(tiny.prompts[5 + i]!.probs[0]!, 5));
  });

  it("stops between passes when the call is cancelled", async () => {
    const backend = new NanoBackend(await session(), encode, "wasm", "q8", 1);
    const ctl = new AbortController();
    ctl.abort();
    await expect(backend.decideMany(items, ctl.signal)).rejects.toThrow(/abort/i);
  });

  it("says when a question and its options are longer than the model reads, before running anything", async () => {
    const backend = new NanoBackend(await session(), encode, "wasm", "q8");
    const options: Option[] = Array.from({ length: 60 }, (_, i) => [`opt${i}`, "a long description ".repeat(30)]);
    await expect(backend.decideMany([{ state: "short", instructions: "Which?", options }])).rejects.toThrow(
      /more than opendecider-nano's 2048/,
    );
  });

  it("refuses a prompt whose markers do not match its options", async () => {
    const leaky: Encode = (text) => (text.includes("input:") ? [SPECIAL.mask] : encode(text)); // a broken tokenizer
    const backend = new NanoBackend(await session(), leaky, "wasm", "q8");
    await expect(backend.decideMany(items.slice(0, 1))).rejects.toThrow(/option markers/);
  });
});

describe("batches", () => {
  it("groups by length within the token budget, and never drops one", () => {
    const lengths = [5, 300, 7, 2048, 6, 40];
    const b = batches(lengths, 64);
    expect(b.flat().sort()).toEqual([0, 1, 2, 3, 4, 5]);
    for (const g of b) expect(g.length === 1 || g.length * Math.max(...g.map((i) => lengths[i]!)) <= 64).toBe(true);
  });
});

describe("the pinned build", () => {
  it("is one commit of the Hub repository, never a branch that can move", async () => {
    const { REVISION, HUB_URL, REPO } = await import("../src/manifest.js");
    expect(REVISION).toMatch(/^[0-9a-f]{40}$/);
    expect(HUB_URL).toBe(`https://huggingface.co/${REPO}/resolve/${REVISION}/`);
  });
});
