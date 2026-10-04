/** The model files: downloaded once, checked against the SHA-256 pinned in this package, kept in the Cache API. */

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
  /** True when the file came from the cache. */
  cached: boolean;
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
  /** Keep the files in the Cache API (browsers, workers, extensions). Default true where it exists. */
  cache?: boolean | undefined;
  fetch?: typeof globalThis.fetch | undefined;
  onProgress?: ((p: LoadProgress) => void) | undefined;
  signal?: AbortSignal | undefined;
}

const CACHE = "opendecider-web-v1";

/** The file's bytes, after its size and SHA-256 matched the spec. */
export async function loadFile(spec: FileSpec, o: FetchOptions): Promise<Uint8Array> {
  o.signal?.throwIfAborted();
  const progress = (loaded: number, cached: boolean) =>
    o.onProgress?.({ file: spec.path, loaded, total: spec.bytes, cached });
  if (o.files) {
    const got = await o.files(spec.path);
    o.signal?.throwIfAborted();
    const bytes = got instanceof Uint8Array ? got : new Uint8Array(got);
    await check(spec, bytes);
    progress(bytes.length, false);
    return bytes;
  }
  const cache = o.cache !== false ? await openCache() : null;
  const key = `https://opendecider.invalid/sha256/${spec.sha256}`; // the content, wherever it was downloaded from
  if (cache) {
    const hit = await cache.match(key).catch(() => undefined);
    if (hit) {
      const bytes = new Uint8Array(await hit.arrayBuffer());
      o.signal?.throwIfAborted();
      try {
        await check(spec, bytes);
        progress(bytes.length, true);
        return bytes;
      } catch {
        await cache.delete(key).catch(() => false); // damaged or replaced: download it again
      }
    }
  }
  const bytes = await download(spec, o, (n) => progress(n, false));
  await check(spec, bytes);
  if (cache) await cache.put(key, new Response(bytes as Uint8Array<ArrayBuffer>)).catch(() => undefined); // a full or blocked cache is no error
  return bytes;
}

async function openCache(): Promise<Cache | null> {
  try {
    return typeof caches === "undefined" ? null : await caches.open(CACHE);
  } catch {
    return null; // e.g. an opaque origin, or storage blocked
  }
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
