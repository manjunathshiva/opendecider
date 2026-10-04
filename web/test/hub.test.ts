// Model files: the pinned size and SHA-256 are checked on every load, from the network, the cache or your own reader.
import { createHash } from "node:crypto";
import { afterEach, describe, expect, it } from "vitest";
import { ModelFileError, loadFile, type FileSpec, type LoadProgress } from "../src/hub.js";

const BYTES = new TextEncoder().encode("opendecider-nano, pretend weights ".repeat(100));
const SPEC: FileSpec = { path: "onnx/model.onnx", bytes: BYTES.length, sha256: sha(BYTES) };
const BASE = "https://models.example/nano/";

function sha(b: Uint8Array): string {
  return createHash("sha256").update(b).digest("hex");
}

/** A server that sends `body` in small chunks, optionally with its own content-length. */
function server(body: Uint8Array, init: { status?: number; length?: string | null } = {}) {
  const calls: string[] = [];
  const fetch = (async (url: string) => {
    calls.push(String(url));
    const headers = new Headers();
    const length = init.length === undefined ? String(body.length) : init.length;
    if (length !== null) headers.set("content-length", length);
    const stream = new ReadableStream<Uint8Array>({
      start(c) {
        for (let i = 0; i < body.length; i += 512) c.enqueue(body.slice(i, i + 512));
        c.close();
      },
    });
    return new Response(stream, { status: init.status ?? 200, headers });
  }) as unknown as typeof globalThis.fetch;
  return { fetch, calls };
}

/** An in-memory Cache API. */
function fakeCaches() {
  const store = new Map<string, Uint8Array>();
  const cache = {
    match: async (k: string) => (store.has(k) ? new Response(store.get(k)! as Uint8Array<ArrayBuffer>) : undefined),
    put: async (k: string, r: Response) => void store.set(k, new Uint8Array(await r.arrayBuffer())),
    delete: async (k: string) => store.delete(k),
  };
  (globalThis as { caches?: unknown }).caches = { open: async () => cache };
  return store;
}

afterEach(() => {
  delete (globalThis as { caches?: unknown }).caches;
});

describe("loadFile", () => {
  it("downloads the file from baseUrl, reporting progress, and checks it", async () => {
    const s = server(BYTES);
    const seen: LoadProgress[] = [];
    const got = await loadFile(SPEC, {
      baseUrl: "https://models.example/nano",
      fetch: s.fetch,
      onProgress: (p) => seen.push(p),
    });
    expect(got).toEqual(BYTES);
    expect(s.calls).toEqual(["https://models.example/nano/onnx/model.onnx"]);
    expect(seen.at(-1)).toEqual({ file: SPEC.path, loaded: BYTES.length, total: BYTES.length, cached: false });
  });

  it("refuses a file whose SHA-256 differs, even at the right size", async () => {
    const tampered = BYTES.slice();
    tampered[10] = tampered[10]! ^ 1;
    await expect(loadFile(SPEC, { baseUrl: BASE, fetch: server(tampered).fetch })).rejects.toThrow(
      /does not match its pinned SHA-256/,
    );
  });

  it("stops reading a file larger than its pinned size", async () => {
    const big = new Uint8Array(BYTES.length + 4096);
    await expect(loadFile(SPEC, { baseUrl: BASE, fetch: server(big, { length: null }).fetch })).rejects.toThrow(
      /larger than the expected/,
    );
    await expect(loadFile(SPEC, { baseUrl: BASE, fetch: server(big).fetch })).rejects.toThrow(ModelFileError);
  });

  it("refuses a short file and an HTTP error", async () => {
    const short = BYTES.slice(0, 100);
    await expect(loadFile(SPEC, { baseUrl: BASE, fetch: server(short, { length: null }).fetch })).rejects.toThrow(
      /is 100 bytes/,
    );
    await expect(loadFile(SPEC, { baseUrl: BASE, fetch: server(BYTES, { status: 404 }).fetch })).rejects.toThrow(
      /HTTP 404/,
    );
  });

  it("keeps the file in the cache, checks it again on the next load, and replaces a damaged copy", async () => {
    const store = fakeCaches();
    const s = server(BYTES);
    await loadFile(SPEC, { baseUrl: BASE, fetch: s.fetch });
    const seen: LoadProgress[] = [];
    expect(await loadFile(SPEC, { baseUrl: BASE, fetch: s.fetch, onProgress: (p) => seen.push(p) })).toEqual(BYTES);
    expect(s.calls.length).toBe(1);
    expect(seen.at(-1)!.cached).toBe(true);
    const [key] = store.keys();
    store.get(key!)![0] = store.get(key!)![0]! ^ 1; // the cached copy is damaged
    expect(await loadFile(SPEC, { baseUrl: BASE, fetch: s.fetch })).toEqual(BYTES);
    expect(s.calls.length).toBe(2);
    expect(await loadFile(SPEC, { baseUrl: BASE, fetch: s.fetch, cache: false })).toEqual(BYTES);
    expect(s.calls.length).toBe(3);
  });

  it("never quotes the URL in an error: it may carry a token", async () => {
    const base = "https://user:secret@models.example/nano/";
    for (const fetch of [
      server(BYTES, { status: 403 }).fetch,
      (async () => Promise.reject(new TypeError("offline"))) as unknown as typeof globalThis.fetch,
    ]) {
      const e = (await loadFile(SPEC, { baseUrl: base, fetch }).catch((x: Error) => x)) as Error;
      expect(e).toBeInstanceOf(ModelFileError);
      expect(e.message).not.toContain("secret");
      expect(e.message).not.toContain("models.example");
    }
  });

  it("resolves a relative baseUrl against the page, and says when there is no page", async () => {
    const s = server(BYTES);
    (globalThis as { location?: unknown }).location = { href: "https://app.example/inbox/today" };
    try {
      expect(await loadFile(SPEC, { baseUrl: "/models/nano", fetch: s.fetch })).toEqual(BYTES);
      expect(s.calls).toEqual(["https://app.example/models/nano/onnx/model.onnx"]);
    } finally {
      delete (globalThis as { location?: unknown }).location;
    }
    await expect(loadFile(SPEC, { baseUrl: "/models/nano", fetch: s.fetch })).rejects.toThrow(/absolute URL/);
  });

  it("stops before reading anything when the call is already cancelled", async () => {
    const ctl = new AbortController();
    ctl.abort();
    let read = 0;
    const files = async () => (read++, BYTES);
    await expect(loadFile(SPEC, { baseUrl: BASE, files, signal: ctl.signal })).rejects.toThrow(/abort/i);
    expect(read).toBe(0);
  });

  it("checks a file you read yourself", async () => {
    expect(await loadFile(SPEC, { baseUrl: BASE, files: async () => BYTES.buffer.slice(0) })).toEqual(BYTES);
    await expect(loadFile(SPEC, { baseUrl: BASE, files: async () => BYTES.slice(1) })).rejects.toThrow(ModelFileError);
  });

  it("passes a cancellation through, not as a download error", async () => {
    const ctl = new AbortController();
    ctl.abort();
    const fetch = ((_: string, init: RequestInit) => {
      init.signal?.throwIfAborted();
      return Promise.resolve(new Response(BYTES as Uint8Array<ArrayBuffer>));
    }) as unknown as typeof globalThis.fetch;
    const e = await loadFile(SPEC, { baseUrl: BASE, fetch, signal: ctl.signal }).catch((x: unknown) => x);
    expect(e).not.toBeInstanceOf(ModelFileError);
    expect(String(e)).toMatch(/abort/i);
  });
});
