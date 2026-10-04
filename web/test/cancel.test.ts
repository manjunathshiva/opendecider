// Cancelling: loadNano while ONNX Runtime creates the session (it cannot stop that, so loadNano waits, frees the session
// and rejects), and a question while it waits for, or runs, a forward pass. ONNX Runtime and the file loader are
// replaced here, so this runs without the model.
import { readFileSync } from "node:fs";
import { beforeEach, describe, expect, it, vi } from "vitest";

const tokenizer = readFileSync(new URL("./fixtures/tokenizer.web.json", import.meta.url));
const ort = vi.hoisted(() => ({ create: vi.fn() }));
const files = vi.hoisted(() => ({ loaded: [] as string[], tokenizerFails: false }));
vi.mock("../src/ort.js", () => ({
  ort: {
    env: { wasm: {} },
    InferenceSession: { create: ort.create },
    Tensor: class {
      constructor(
        readonly type: string,
        readonly data: unknown,
        readonly dims: number[],
      ) {}
    },
  },
}));
vi.mock("../src/hub.js", async (actual) => ({
  ...(await actual<typeof import("../src/hub.js")>()),
  loadFile: async (spec: { path: string }) => {
    files.loaded.push(spec.path);
    if (!spec.path.endsWith(".json")) return new Uint8Array(8);
    if (files.tokenizerFails) throw new Error("tokenizer unavailable");
    return tokenizer;
  },
}));
const { loadNano, NanoBackend } = await import("../src/nano.js");

/** A session creation that finishes when the test says so. */
function pending() {
  let finish!: (s: unknown) => void;
  let fail!: (e: unknown) => void;
  const promise = new Promise((res, rej) => ((finish = res), (fail = rej)));
  return { promise, finish, fail };
}

beforeEach(() => {
  ort.create.mockReset();
  files.loaded = [];
  files.tokenizerFails = false;
});

describe("loadNano cancelled while the session is created", () => {
  it("rejects and frees the session", async () => {
    const creating = pending();
    ort.create.mockReturnValueOnce(creating.promise);
    const ctl = new AbortController();
    const loading = loadNano({ device: "wasm", signal: ctl.signal });
    await vi.waitFor(() => expect(ort.create).toHaveBeenCalledTimes(1));
    ctl.abort();
    const session = { release: vi.fn(async () => undefined) };
    creating.finish(session);
    await expect(loading).rejects.toThrow(/abort/i);
    expect(session.release).toHaveBeenCalledTimes(1);
  });

  it("reports the cancellation even when freeing the session fails", async () => {
    const creating = pending();
    ort.create.mockReturnValueOnce(creating.promise);
    const ctl = new AbortController();
    const loading = loadNano({ device: "wasm", signal: ctl.signal });
    await vi.waitFor(() => expect(ort.create).toHaveBeenCalledTimes(1));
    ctl.abort();
    creating.finish({ release: vi.fn(async () => Promise.reject(new Error("release failed"))) });
    await expect(loading).rejects.toThrow(/abort/i);
  });

  it("does not fall back to WebAssembly once cancelled", async () => {
    const creating = pending();
    ort.create.mockReturnValueOnce(creating.promise);
    const gpu = { requestAdapter: async () => ({}) };
    vi.stubGlobal("navigator", { gpu });
    try {
      const ctl = new AbortController();
      const loading = loadNano({ signal: ctl.signal }); // auto: WebGPU first
      await vi.waitFor(() => expect(ort.create).toHaveBeenCalledTimes(1));
      ctl.abort();
      creating.fail(new Error("WebGPU could not run the model"));
      await expect(loading).rejects.toThrow(/abort/i);
      expect(ort.create).toHaveBeenCalledTimes(1);
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("returns the model when nothing was cancelled", async () => {
    ort.create.mockResolvedValueOnce({ release: vi.fn() });
    const model = await loadNano({ device: "wasm" });
    expect(model.name).toBe("opendecider-nano-onnx-q8");
  });
});

describe("loadNano's files", () => {
  it("does not download the model when the tokenizer cannot be loaded", async () => {
    files.tokenizerFails = true;
    await expect(loadNano({ device: "wasm" })).rejects.toThrow(/tokenizer unavailable/);
    expect(files.loaded).toEqual(["tokenizer.web.json"]);
    expect(ort.create).not.toHaveBeenCalled();
  });
});

describe("a question cancelled at inference", () => {
  const question = {
    state: "hi",
    instructions: "Pick",
    options: [
      ["a", ""],
      ["b", ""],
    ] as [string, string][],
  };
  /** A session whose passes finish when the test says so; each pass's logits are zeros (equal probabilities). */
  function session() {
    const passes: { finish: () => void }[] = [];
    const run = vi.fn(
      (feeds: { input_ids: { dims: number[] } }) =>
        new Promise((res) => {
          const [B, L] = feeds.input_ids.dims as [number, number];
          passes.push({ finish: () => res({ logits: { dims: [B, L], data: new Float32Array(B * L) } }) });
        }),
    );
    return { run, passes, release: vi.fn() };
  }
  const encode = () => [10, 11];

  it("never runs a pass that was cancelled while queued", async () => {
    const s = session();
    const backend = new NanoBackend(s as never, encode, "wasm", "q8");
    const first = backend.decideMany([question]);
    await vi.waitFor(() => expect(s.run).toHaveBeenCalledTimes(1));
    const ctl = new AbortController();
    const second = backend.decideMany([question], ctl.signal); // queued behind the first
    ctl.abort();
    s.passes[0]!.finish();
    await expect(first).resolves.toMatchObject({ probs: [{ a: 0.5, b: 0.5 }] });
    await expect(second).rejects.toThrow(/abort/i);
    expect(s.run).toHaveBeenCalledTimes(1);
    const third = backend.decideMany([question]); // the queue still works
    await vi.waitFor(() => expect(s.run).toHaveBeenCalledTimes(2));
    s.passes[1]!.finish();
    await expect(third).resolves.toMatchObject({ probs: [{ a: 0.5, b: 0.5 }] });
  });

  it("gives no answer when cancelled during the pass", async () => {
    const s = session();
    const backend = new NanoBackend(s as never, encode, "wasm", "q8");
    const ctl = new AbortController();
    const asking = backend.decideMany([question], ctl.signal);
    await vi.waitFor(() => expect(s.run).toHaveBeenCalledTimes(1));
    ctl.abort();
    s.passes[0]!.finish();
    await expect(asking).rejects.toThrow(/abort/i);
  });
});
