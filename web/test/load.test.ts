// loadNano's own checks, which run before anything is downloaded (the model itself: parity.test.ts and the live demo).
import { describe, expect, it } from "vitest";
import { loadNano } from "../src/nano.js";

const never = (() => Promise.reject(new Error("should not download"))) as unknown as typeof globalThis.fetch;

describe("loadNano", () => {
  it("refuses an unknown device or build before downloading anything", async () => {
    await expect(loadNano({ device: "cuda" as never, fetch: never })).rejects.toThrow(/device must be/);
    await expect(loadNano({ device: "wasm", dtype: "q4" as never, fetch: never })).rejects.toThrow(/dtype must be/);
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
