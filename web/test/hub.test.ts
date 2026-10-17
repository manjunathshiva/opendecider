// Model files: the pinned size and SHA-256 are checked on every load, from the network, the cache or your own reader.
import { createHash } from "node:crypto";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ModelFileError, loadFile, prune, type FileSpec, type LoadProgress } from "../src/hub.js";

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

/** An in-memory Cache API; `refuse`: every put rejects, as a full cache or one that will not keep a large file. */
function fakeCaches(init: { refuse?: boolean } = {}) {
  const store = new Map<string, Uint8Array>();
  const opened: string[] = [];
  const cache = {
    match: async (k: string) => (store.has(k) ? new Response(store.get(k)! as Uint8Array<ArrayBuffer>) : undefined),
    put: async (k: string, r: Response) => {
      if (init.refuse) throw new DOMException("refused", "QuotaExceededError");
      store.set(k, new Uint8Array(await r.arrayBuffer()));
    },
    delete: async (k: string) => store.delete(k),
    keys: async () => [...store.keys()].map((k) => new Request(k)),
  };
  (globalThis as { caches?: unknown }).caches = { open: async (name: string) => (opened.push(name), cache) };
  return Object.assign(store, { opened });
}

/** An in-memory origin private file system, where a write lands when it is closed (none when it fails, or is aborted);
 * `refuse`: every write rejects, as a full disk. */
function fakeFolder(init: { refuse?: boolean } = {}) {
  const files = new Map<string, Uint8Array>();
  const opened: string[] = [];
  let aborted = 0;
  class FileSystemFileHandle {
    constructor(readonly name: string) {}
    async getFile() {
      if (!files.has(this.name)) throw new DOMException("gone", "NotFoundError");
      return new Blob([files.get(this.name)! as Uint8Array<ArrayBuffer>]);
    }
    async createWritable() {
      const parts: Uint8Array[] = [];
      return {
        write: async (b: Uint8Array) => {
          if (init.refuse) throw new DOMException("full", "QuotaExceededError");
          parts.push(b.slice());
        },
        close: async () => void files.set(this.name, new Uint8Array(await new Blob(parts as BlobPart[]).arrayBuffer())),
        abort: async () => void aborted++,
      };
    }
  }
  const dir = {
    getFileHandle: async (name: string, o?: { create?: boolean }) => {
      if (!files.has(name) && !o?.create) throw new DOMException("missing", "NotFoundError");
      if (!files.has(name)) files.set(name, new Uint8Array(0)); // created empty, as the browser does
      return new FileSystemFileHandle(name);
    },
    removeEntry: async (name: string) => {
      if (!files.delete(name)) throw new DOMException("missing", "NotFoundError");
    },
    keys: async function* () {
      yield* [...files.keys()];
    },
  };
  const root = { getDirectoryHandle: async (name: string) => (opened.push(name), dir) };
  vi.stubGlobal("FileSystemFileHandle", FileSystemFileHandle);
  vi.stubGlobal("navigator", { storage: { getDirectory: async () => root } });
  return Object.assign(files, { opened, aborted: () => aborted });
}

afterEach(() => {
  delete (globalThis as { caches?: unknown }).caches;
  vi.unstubAllGlobals();
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
    // the last event says where it was kept: nowhere here (Node has neither the Cache API nor a private file system)
    expect(seen.at(-1)).toEqual({
      file: SPEC.path,
      loaded: BYTES.length,
      total: BYTES.length,
      cached: false,
      stored: false,
    });
    expect(seen.slice(0, -1).every((p) => !("stored" in p))).toBe(true);
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
    const first: LoadProgress[] = [];
    await loadFile(SPEC, { baseUrl: BASE, fetch: s.fetch, onProgress: (p) => first.push(p) });
    expect(first.at(-1)!.stored).toBe("cache");
    expect(store.opened).toEqual(["opendecider-web-v1"]);
    const seen: LoadProgress[] = [];
    expect(await loadFile(SPEC, { baseUrl: BASE, fetch: s.fetch, onProgress: (p) => seen.push(p) })).toEqual(BYTES);
    expect(s.calls.length).toBe(1);
    expect(seen).toEqual([{ file: SPEC.path, loaded: BYTES.length, total: BYTES.length, cached: true }]);
    const [key] = store.keys();
    store.get(key!)![0] = store.get(key!)![0]! ^ 1; // the cached copy is damaged
    expect(await loadFile(SPEC, { baseUrl: BASE, fetch: s.fetch })).toEqual(BYTES);
    expect(s.calls.length).toBe(2);
    expect(await loadFile(SPEC, { baseUrl: BASE, fetch: s.fetch, cache: false })).toEqual(BYTES);
    expect(s.calls.length).toBe(3);
  });

  it("keeps the file in the private file system when the Cache API refuses it, and reads it from there", async () => {
    const cache = fakeCaches({ refuse: true });
    const folder = fakeFolder();
    const s = server(BYTES);
    const first: LoadProgress[] = [];
    expect(await loadFile(SPEC, { baseUrl: BASE, fetch: s.fetch, onProgress: (p) => first.push(p) })).toEqual(BYTES);
    expect(first.at(-1)).toEqual({
      file: SPEC.path,
      loaded: BYTES.length,
      total: BYTES.length,
      cached: false,
      stored: "opfs",
    });
    expect(cache.size).toBe(0);
    expect(folder.opened).toEqual(["opendecider-web-v1"]);
    expect([...folder.keys()]).toEqual([SPEC.sha256]);
    const seen: LoadProgress[] = [];
    expect(await loadFile(SPEC, { baseUrl: BASE, fetch: s.fetch, onProgress: (p) => seen.push(p) })).toEqual(BYTES);
    expect(s.calls.length).toBe(1);
    expect(seen).toEqual([{ file: SPEC.path, loaded: BYTES.length, total: BYTES.length, cached: true }]);
  });

  it("uses the origin private file system where there is no Cache API", async () => {
    const folder = fakeFolder();
    const s = server(BYTES);
    const seen: LoadProgress[] = [];
    await loadFile(SPEC, { baseUrl: BASE, fetch: s.fetch, onProgress: (p) => seen.push(p) });
    expect(seen.at(-1)!.stored).toBe("opfs");
    expect(folder.get(SPEC.sha256)).toEqual(BYTES);
    expect(await loadFile(SPEC, { baseUrl: BASE, fetch: s.fetch })).toEqual(BYTES);
    expect(s.calls.length).toBe(1);
  });

  it("still returns the file when neither keeps it, and leaves no part of it behind", async () => {
    fakeCaches({ refuse: true });
    const folder = fakeFolder({ refuse: true });
    const s = server(BYTES);
    const seen: LoadProgress[] = [];
    expect(await loadFile(SPEC, { baseUrl: BASE, fetch: s.fetch, onProgress: (p) => seen.push(p) })).toEqual(BYTES);
    expect(seen.at(-1)!.stored).toBe(false);
    expect(folder.aborted()).toBe(1);
    expect(folder.get(SPEC.sha256)?.length ?? 0).toBe(0); // the write was abandoned: nothing of it was kept
    expect(await loadFile(SPEC, { baseUrl: BASE, fetch: s.fetch })).toEqual(BYTES);
    expect(s.calls.length).toBe(2); // downloaded again, as without a cache
  });

  it("replaces a damaged copy in the origin private file system", async () => {
    fakeCaches({ refuse: true });
    const folder = fakeFolder();
    const s = server(BYTES);
    await loadFile(SPEC, { baseUrl: BASE, fetch: s.fetch });
    folder.get(SPEC.sha256)![0] = folder.get(SPEC.sha256)![0]! ^ 1; // the kept copy is damaged
    const seen: LoadProgress[] = [];
    expect(await loadFile(SPEC, { baseUrl: BASE, fetch: s.fetch, onProgress: (p) => seen.push(p) })).toEqual(BYTES);
    expect(s.calls.length).toBe(2);
    expect(seen.at(-1)).toMatchObject({ cached: false, stored: "opfs" });
    expect(folder.get(SPEC.sha256)).toEqual(BYTES);
    folder.set(SPEC.sha256, BYTES.slice(0, 10)); // a short copy too
    expect(await loadFile(SPEC, { baseUrl: BASE, fetch: s.fetch })).toEqual(BYTES);
    expect(s.calls.length).toBe(3);
  });

  it("loads the same file twice at once", async () => {
    fakeCaches({ refuse: true });
    const folder = fakeFolder();
    const s = server(BYTES);
    const both = await Promise.all([1, 2].map(() => loadFile(SPEC, { baseUrl: BASE, fetch: s.fetch })));
    expect(both).toEqual([BYTES, BYTES]);
    expect(folder.get(SPEC.sha256)).toEqual(BYTES);
  });

  it("with cache: false, reads and writes neither the Cache API nor the private file system", async () => {
    const cache = fakeCaches();
    const folder = fakeFolder();
    const s = server(BYTES);
    const seen: LoadProgress[] = [];
    expect(
      await loadFile(SPEC, { baseUrl: BASE, fetch: s.fetch, cache: false, onProgress: (p) => seen.push(p) }),
    ).toEqual(BYTES);
    expect(s.calls.length).toBe(1);
    expect(seen.some((p) => "stored" in p)).toBe(false);
    expect([cache.opened, cache.size, folder.opened, folder.size]).toEqual([[], 0, [], 0]);
  });

  it("prunes the files no longer pinned from both stores, and nothing else", async () => {
    const cache = fakeCaches();
    const folder = fakeFolder();
    const [a, b, old] = ["a", "b", "c"].map((c) => c.repeat(64));
    for (const sha of [a!, b!, old!]) {
      cache.set(`https://opendecider.invalid/sha256/${sha}`, BYTES);
      folder.set(sha, BYTES);
    }
    cache.set("https://app.example/other", BYTES); // not ours (never there in practice: the cache is ours alone)
    folder.set(`${old}.crswap`, BYTES); // nor a name that is not a SHA-256
    await prune([a!, b!]);
    expect([...cache.keys()].sort()).toEqual(
      [
        "https://app.example/other",
        `https://opendecider.invalid/sha256/${a}`,
        `https://opendecider.invalid/sha256/${b}`,
      ].sort(),
    );
    expect([...folder.keys()].sort()).toEqual([a!, b!, `${old}.crswap`].sort());
    expect(cache.opened).toEqual(["opendecider-web-v1"]);
    expect(folder.opened).toEqual(["opendecider-web-v1"]);
  });

  it("never fails to prune, whatever the stores do", async () => {
    await expect(prune([])).resolves.toBeUndefined(); // neither exists
    (globalThis as { caches?: unknown }).caches = { open: async () => Promise.reject(new Error("blocked")) };
    vi.stubGlobal("navigator", { storage: { getDirectory: async () => Promise.reject(new Error("blocked")) } });
    await expect(prune([])).resolves.toBeUndefined();
    (globalThis as { caches?: unknown }).caches = {
      open: async () => ({ keys: async () => Promise.reject(new Error("cleared")) }),
    };
    const folder = fakeFolder();
    folder.set("d".repeat(64), BYTES);
    await expect(prune([])).resolves.toBeUndefined();
    expect(folder.size).toBe(0); // one store failing does not stop the other's
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
