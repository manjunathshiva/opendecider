// loadNano's own checks, which run before anything is downloaded (the model itself: parity.test.ts and the live demo).
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { ModelFileError } from "../src/hub.js";
import { loadNano } from "../src/nano.js";

const tokenizer = readFileSync(new URL("./fixtures/tokenizer.web.json", import.meta.url));

const never = (() => Promise.reject(new Error("should not download"))) as unknown as typeof globalThis.fetch;

describe("loadNano", () => {
  it("refuses an unknown device or build before downloading anything", async () => {
    await expect(loadNano({ device: "cuda" as never, fetch: never })).rejects.toThrow(/device must be/);
    await expect(loadNano({ device: "wasm", dtype: "q4" as never, fetch: never })).rejects.toThrow(/dtype must be/);
  });

  it("reads each build's own pinned file, and refuses one that does not match it", async () => {
    for (const [dtype, path] of [
      ["q8", "onnx/model_q8.onnx"],
      ["q8f16", "onnx/model_q8f16.onnx"],
      ["fp16", "onnx/model_fp16.onnx"],
    ] as const) {
      const read: string[] = [];
      const files = async (p: string) => (read.push(p), p === "tokenizer.web.json" ? tokenizer : new Uint8Array(16));
      await expect(loadNano({ device: "wasm", dtype, files, cache: false })).rejects.toThrow(ModelFileError);
      expect(read).toEqual(["tokenizer.web.json", path]);
    }
  });

  it("reads no file and builds no model when the call is already cancelled", async () => {
    const ctl = new AbortController();
    ctl.abort();
    let read = 0;
    const files = async () => (read++, new Uint8Array(1));
    await expect(loadNano({ device: "wasm", files, signal: ctl.signal })).rejects.toThrow(/abort/i);
    expect(read).toBe(0);
  });

  it("downloads nothing when the call is already cancelled", async () => {
    const ctl = new AbortController();
    ctl.abort();
    let calls = 0;
    const fetch = ((_: string, init: RequestInit) => {
      calls++;
      init.signal?.throwIfAborted();
      return Promise.reject(new Error("unreachable"));
    }) as unknown as typeof globalThis.fetch;
    await expect(loadNano({ device: "wasm", fetch, cache: false, signal: ctl.signal })).rejects.toThrow(/abort/i);
    expect(calls).toBeLessThanOrEqual(2); // each file's request saw the cancellation at once
  });
});
