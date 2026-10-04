// Cancelling loadNano while ONNX Runtime creates the session: ONNX Runtime cannot stop that, so loadNano waits, frees
// the session and rejects. ONNX Runtime and the file loader are replaced here, so this runs without the model.
import { readFileSync } from "node:fs";
import { beforeEach, describe, expect, it, vi } from "vitest";

const tokenizer = readFileSync(new URL("./fixtures/tokenizer.web.json", import.meta.url));
const ort = vi.hoisted(() => ({ create: vi.fn() }));
vi.mock("../src/ort.js", () => ({ ort: { env: { wasm: {} }, InferenceSession: { create: ort.create } } }));
vi.mock("../src/hub.js", async (actual) => ({
  ...(await actual<typeof import("../src/hub.js")>()),
  loadFile: async (spec: { path: string }) => (spec.path.endsWith(".json") ? tokenizer : new Uint8Array(8)),
}));
const { loadNano } = await import("../src/nano.js");

/** A session creation that finishes when the test says so. */
function pending() {
  let finish!: (s: unknown) => void;
  let fail!: (e: unknown) => void;
  const promise = new Promise((res, rej) => ((finish = res), (fail = rej)));
  return { promise, finish, fail };
}

beforeEach(() => ort.create.mockReset());

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
