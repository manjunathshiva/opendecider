/** The model files: downloaded once, checked against the SHA-256 pinned in this package, kept in the Cache API (or,
 * where it refuses them, the origin private file system). */

/** One file of a model build: where it is, and its exact size and SHA-256. */
export interface FileSpec {
  path: string;
  bytes: number;
  sha256: string;
}

/** Download progress, per file. */
export interface LoadProgress {
  file: string;
  loaded: number;
  total: number;
  /** True when the file came from the cache (the Cache API or the origin private file system). */
  cached: boolean;
  /** On the last event of a file that was downloaded: where it was kept for the next load, "cache" (the Cache API),
   * "opfs" (the origin private file system, where the Cache API refused it) or false (neither kept it: the next load
   * downloads it again). Absent otherwise, and with `files` or `cache: false`. */
  stored?: "cache" | "opfs" | false;
}

/** A model file that could not be loaded, or did not match its pinned size or SHA-256 (it is never used then). */
export class ModelFileError extends Error {
  override readonly name = "ModelFileError";
}

export interface FetchOptions {
  /** Where the files are: a folder URL (your own origin, for a strict connect-src), default the Hugging Face Hub. */
  baseUrl: string;
  /** Read a file yourself instead of downloading it (Node, Bun, Deno: a local folder). It is still checked. */
  files?: ((path: string) => Promise<ArrayBuffer | Uint8Array>) | undefined;
  /** Keep the files in the Cache API, or where it refuses them the origin private file system (browsers, workers,
   * extensions). Default true where either exists; false reads and writes neither. */
  cache?: boolean | undefined;
  fetch?: typeof globalThis.fetch | undefined;
  onProgress?: ((p: LoadProgress) => void) | undefined;
  signal?: AbortSignal | undefined;
}

const CACHE = "opendecider-web-v1"; // the Cache API's cache, and the origin private file system's folder
const KEY = "https://opendecider.invalid/sha256/"; // + the SHA-256: the content, wherever it was downloaded from
const SHA256 = /^[0-9a-f]{64}$/;

/** Where downloaded files are kept, by SHA-256. Any call may reject (storage full, blocked, or cleared meanwhile). */
interface Store {
  name: "cache" | "opfs";
  get(sha256: string): Promise<Uint8Array | undefined>;
  put(sha256: string, bytes: Uint8Array): Promise<void>;
  delete(sha256: string): Promise<unknown>;
  /** The SHA-256 of every file kept. */
  list(): Promise<string[]>;
}

/** The file's bytes, after its size and SHA-256 matched the spec. */
export async function loadFile(spec: FileSpec, o: FetchOptions): Promise<Uint8Array> {
  o.signal?.throwIfAborted();
  const progress = (loaded: number, cached: boolean, stored?: LoadProgress["stored"]) =>
    o.onProgress?.({ file: spec.path, loaded, total: spec.bytes, cached, ...(stored === undefined ? {} : { stored }) });
  if (o.files) {
    const got = await o.files(spec.path);
    o.signal?.throwIfAborted();
    const bytes = got instanceof Uint8Array ? got : new Uint8Array(got);
    await check(spec, bytes);
    progress(bytes.length, false);
    return bytes;
  }
  const stores = o.cache !== false ? await openStores() : null;
  for (const store of stores ?? []) {
    const hit = await store.get(spec.sha256).catch(() => undefined);
    if (!hit) continue;
    o.signal?.throwIfAborted();
    try {
      await check(spec, hit);
      progress(hit.length, true);
      return hit;
    } catch {
      await store.delete(spec.sha256).catch(() => false); // damaged or replaced: download it again
    }
  }
  const bytes = await download(spec, o, (n) => progress(n, false));
  await check(spec, bytes);
  if (stores) progress(bytes.length, false, await save(stores, spec.sha256, bytes));
  return bytes;
}

/** Deletes every kept file but `keep` (SHA-256s), such as the builds an older version of the package pinned. Never
 * fails: a file left behind only takes space. */
export async function prune(keep: readonly string[]): Promise<void> {
  for (const store of await openStores()) {
    for (const sha256 of await store.list().catch(() => [])) {
      if (!keep.includes(sha256)) await store.delete(sha256).catch(() => false);
    }
  }
}

/** The first store that keeps the bytes, or false: a full or blocked store is no error (the next load downloads). */
async function save(stores: Store[], sha256: string, bytes: Uint8Array): Promise<Store["name"] | false> {
  for (const store of stores) {
    try {
      await store.put(sha256, bytes);
      return store.name;
    } catch {
      // refused, full or blocked: the next store
    }
  }
  return false;
}

/** The Cache API first, then the origin private file system: whichever of them exist here. */
async function openStores(): Promise<Store[]> {
  return (await Promise.all([openCache(), openFolder()])).filter((s) => s !== null);
}

async function openCache(): Promise<Store | null> {
  let cache: Cache;
  try {
    if (typeof caches === "undefined") return null;
    cache = await caches.open(CACHE);
  } catch {
    return null; // e.g. an opaque origin, or storage blocked
  }
  return {
    name: "cache",
    get: async (sha256) => {
      const hit = await cache.match(KEY + sha256);
      return hit && new Uint8Array(await hit.arrayBuffer());
    },
    put: (sha256, bytes) => cache.put(KEY + sha256, new Response(bytes as Uint8Array<ArrayBuffer>)),
    delete: (sha256) => cache.delete(KEY + sha256),
    list: async () =>
      (await cache.keys())
        .filter((r) => r.url.startsWith(KEY))
        .map((r) => r.url.slice(KEY.length))
        .filter((k) => SHA256.test(k)),
  };
}

/** The origin private file system's folder (navigator.storage.getDirectory), where there is one. */
async function openFolder(): Promise<Store | null> {
  let dir: FileSystemDirectoryHandle;
  try {
    const storage = (globalThis.navigator as { storage?: Partial<StorageManager> } | undefined)?.storage;
    if (typeof storage?.getDirectory !== "function") return null;
    dir = await (await storage.getDirectory()).getDirectoryHandle(CACHE, { create: true });
  } catch {
    return null; // e.g. a private window without one
  }
  return {
    name: "opfs",
    get: async (sha256) => {
      const file = await (await dir.getFileHandle(sha256)).getFile();
      return new Uint8Array(await file.arrayBuffer());
    },
    put: async (sha256, bytes) => {
      // where a page cannot write a file this way (older Safari's main thread): skip it, before an empty one exists
      if (typeof FileSystemFileHandle === "undefined" || !("createWritable" in FileSystemFileHandle.prototype))
        throw new Error("no createWritable");
      const w = await (await dir.getFileHandle(sha256, { create: true })).createWritable();
      try {
        await w.write(bytes as Uint8Array<ArrayBuffer>);
        await w.close(); // the file changes here, at once: a failed or concurrent write never leaves part of one
      } catch (e) {
        await w.abort().catch(() => undefined);
        throw e;
      }
    },
    delete: (sha256) => dir.removeEntry(sha256),
    list: async () => {
      const names: string[] = [];
      for await (const n of (dir as unknown as { keys(): AsyncIterable<string> }).keys()) names.push(n);
      return names.filter((n) => SHA256.test(n));
    },
  };
}

async function download(spec: FileSpec, o: FetchOptions, onBytes: (n: number) => void): Promise<Uint8Array> {
  let url: string;
  try {
    // a relative baseUrl ("/models/") is the page's: resolved against its address, where there is a page
    const base = new URL(o.baseUrl.endsWith("/") ? o.baseUrl : `${o.baseUrl}/`, globalThis.location?.href);
    url = new URL(spec.path, base).href;
  } catch {
    throw new ModelFileError(
      "baseUrl must be an absolute URL here (there is no page to resolve a relative one against)",
    );
  }
  let res: Response;
  try {
    res = await (o.fetch ?? globalThis.fetch)(url, { signal: o.signal ?? null });
  } catch (e) {
    if (o.signal?.aborted) throw e;
    throw new ModelFileError(`could not download ${spec.path}: ${(e as Error).message}`); // (never the URL: it may carry a token)
  }
  if (!res.ok) throw new ModelFileError(`could not download ${spec.path}: HTTP ${res.status}`);
  const length = res.headers.get("content-length");
  if (length !== null && Number(length) !== spec.bytes && !res.headers.get("content-encoding")) {
    await res.body?.cancel().catch(() => undefined);
    throw new ModelFileError(`${spec.path} is ${length} bytes, expected ${spec.bytes}`);
  }
  if (!res.body) throw new ModelFileError(`could not download ${spec.path}: the response has no body`);
  const out = new Uint8Array(spec.bytes); // never more than the pinned size in memory
  let n = 0;
  const reader = res.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (n + value.length > spec.bytes) {
      await reader.cancel().catch(() => undefined);
      throw new ModelFileError(`${spec.path} is larger than the expected ${spec.bytes} bytes`);
    }
    out.set(value, n);
    n += value.length;
    onBytes(n);
  }
  if (n !== spec.bytes) throw new ModelFileError(`${spec.path} is ${n} bytes, expected ${spec.bytes}`);
  return out;
}

async function check(spec: FileSpec, bytes: Uint8Array): Promise<void> {
  if (bytes.length !== spec.bytes)
    throw new ModelFileError(`${spec.path} is ${bytes.length} bytes, expected ${spec.bytes}`);
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes as Uint8Array<ArrayBuffer>));
  const hex = Array.from(digest, (b) => b.toString(16).padStart(2, "0")).join("");
  if (hex !== spec.sha256) {
    throw new ModelFileError(`${spec.path} does not match its pinned SHA-256 (got ${hex}, expected ${spec.sha256})`);
  }
}
