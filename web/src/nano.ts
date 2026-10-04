/** opendecider-nano on this device: ONNX Runtime Web (WebGPU or WebAssembly) behind @opendecider/client's API. */
import { Tokenizer } from "@huggingface/tokenizers";
import {
  InputError,
  OpenDecider,
  type Backend,
  type CallOptions,
  type Decided,
  type Info,
  type Item,
} from "@opendecider/client";
import type { InferenceSession, Tensor } from "onnxruntime-web";
import { loadFile, type FetchOptions, type LoadProgress } from "./hub.js";
import { ort } from "./ort.js";
import { HUB_URL, MAX_TOKENS, MODELS, REPO, REVISION, TOKENIZER, type Dtype } from "./manifest.js";
import { SPECIAL, nanoIds, type Encode } from "./prompt.js";

/** Where the model runs: WebGPU (a GPU through the browser) or WebAssembly (the CPU, in browsers and in Node). */
export type Device = "webgpu" | "wasm";

/** How to load opendecider-nano. */
export interface LoadOptions extends CallOptions {
  /** "auto" (default): WebGPU when the browser has a GPU adapter, else WebAssembly. */
  device?: Device | "auto";
  /** The build: q8f16 (default on WebGPU, 450 MiB) or q8 (default on WebAssembly, 569 MiB). */
  dtype?: Dtype;
  /** A folder URL with the files (self-hosting, for a strict connect-src); default the pinned Hub revision. */
  baseUrl?: string;
  /** Read a file yourself instead of downloading it, e.g. from a local folder in Node. Its SHA-256 is still checked. */
  files?: (path: string) => Promise<ArrayBuffer | Uint8Array>;
  /** Keep the files in the browser's Cache API after the first download (default true where it exists). */
  cache?: boolean;
  /** Called as the files arrive (about 450-570 MiB on the first load, then from the cache). */
  onProgress?: (p: LoadProgress) => void;
  /** ONNX Runtime's WebAssembly settings: where its .wasm and .mjs files are (default: bundled with your app, never a
   * CDN unless you say so), threads (default 1; more need a cross-origin-isolated page and wasmPaths pointing at ONNX
   * Runtime's own files), and proxy (run inference in a worker). */
  wasm?: { wasmPaths?: string; numThreads?: number; proxy?: boolean };
  /** Most tokens in one forward pass (questions are batched by length up to this; one longer question still runs).
   * Default 4096: attention's memory grows with each question's length squared. */
  maxBatchTokens?: number;
  fetch?: typeof globalThis.fetch;
}

/** The model, as an @opendecider/client Backend: what `OpenDecider` calls for each question. */
export class NanoBackend implements Backend {
  readonly kind = "local" as const;
  private queue: Promise<unknown> = Promise.resolve();

  constructor(
    private readonly session: InferenceSession,
    private readonly encode: Encode,
    readonly device: Device,
    readonly dtype: Dtype,
    private readonly maxBatchTokens = 4096,
  ) {}

  async decideMany(items: readonly Item[], signal?: AbortSignal): Promise<Decided> {
    const built = items.map((it) => nanoIds(this.encode, it.state, it.instructions, it.options, MAX_TOKENS));
    for (const b of built) {
      // the state is shortened to fit, the question and its options are not; past 2,048 tokens the browser runs out of
      // memory (Python runs the longer input), so say so before anything runs
      if (b.ids.length > MAX_TOKENS) {
        throw new InputError(
          `a question and its options need ${b.ids.length} tokens, more than opendecider-nano's ${MAX_TOKENS}: ` +
            "shorten the option descriptions or split the question",
        );
      }
    }
    const probs: Record<string, number>[] = new Array(items.length);
    const info: Info[] = built.map((b) => ({ input_tokens: b.ids.length, truncated: b.truncated }));
    for (const batch of batches(
      built.map((b) => b.ids.length),
      this.maxBatchTokens,
    )) {
      signal?.throwIfAborted();
      const logits = await this.run(batch.map((i) => built[i]!.ids));
      batch.forEach((i, r) => {
        const ids = built[i]!.ids;
        const L = logits.dims[1]!;
        const z: number[] = [];
        ids.forEach((t, j) => t === SPECIAL.mask && z.push((logits.data as Float32Array)[r * L + j]!));
        const opts = items[i]!.options;
        if (z.length !== opts.length) throw new Error(`expected ${opts.length} option markers, found ${z.length}`);
        const top = Math.max(...z);
        const e = z.map((x) => Math.exp(x - top));
        const sum = e.reduce((a, b) => a + b, 0);
        probs[i] = Object.fromEntries(opts.map(([name], k) => [name, e[k]! / sum]));
      });
    }
    return { probs, info };
  }

  async ping(): Promise<boolean> {
    return true;
  }

  /** Free the model's memory (the GPU's, on WebGPU). */
  async dispose(): Promise<void> {
    await this.queue.catch(() => undefined);
    await this.session.release();
  }

  /** One padded forward pass. A session runs one call at a time, so calls queue. */
  private run(rows: readonly number[][]): Promise<Tensor> {
    const go = async () => {
      const B = rows.length;
      const L = Math.max(...rows.map((r) => r.length));
      const ids = new BigInt64Array(B * L).fill(BigInt(SPECIAL.pad));
      const mask = new BigInt64Array(B * L);
      rows.forEach((r, b) => r.forEach((t, j) => ((ids[b * L + j] = BigInt(t)), (mask[b * L + j] = 1n))));
      const out = await this.session.run({
        input_ids: new ort.Tensor("int64", ids, [B, L]),
        attention_mask: new ort.Tensor("int64", mask, [B, L]),
      });
      return out["logits"]!;
    };
    const p = this.queue.then(go, go);
    this.queue = p.catch(() => undefined);
    return p;
  }
}

/** Indices grouped into forward passes: by length, so padding stays small, at most `budget` padded tokens each. */
export function batches(lengths: readonly number[], budget: number): number[][] {
  const order = lengths.map((_, i) => i).sort((a, b) => lengths[a]! - lengths[b]!);
  const out: number[][] = [];
  let cur: number[] = [];
  for (const i of order) {
    const L = lengths[i]!; // the longest so far, since the order is by length
    if (cur.length && (cur.length + 1) * L > budget) {
      out.push(cur);
      cur = [];
    }
    cur.push(i);
  }
  if (cur.length) out.push(cur);
  return out;
}

/**
 * opendecider-nano, running here:
 *
 *     const model = await loadNano({ onProgress: (p) => console.log(p.file, p.loaded / p.total) });
 *     const r = await model.systemOne(ticket, { team: choice("Which team?", { billing: "charges", tech: "bugs" }) });
 *
 * The first load downloads the build (its SHA-256 is checked against this package's pin), later loads read the
 * Cache API. The result is an @opendecider/client `OpenDecider`, so Router, Guard and the agent tools take it as is.
 */
export async function loadNano(options: LoadOptions = {}): Promise<OpenDecider> {
  const asked = options.device ?? "auto";
  if (!["auto", "webgpu", "wasm"].includes(asked)) throw new Error(`device must be "auto", "webgpu" or "wasm"`);
  let device: Device = asked === "auto" ? await bestDevice() : (asked as Device);
  const dtype: Dtype = options.dtype ?? (device === "webgpu" ? "q8f16" : "q8");
  if (!Object.hasOwn(MODELS, dtype)) throw new Error(`dtype must be one of ${Object.keys(MODELS).join(", ")}`);
  if (options.wasm?.wasmPaths !== undefined) ort.env.wasm.wasmPaths = options.wasm.wasmPaths;
  // One thread unless asked: ONNX Runtime starts its threads from its own script, which a bundler has merged into the
  // page's, so they would never start (see the browser guide for threads with wasmPaths)
  ort.env.wasm.numThreads = options.wasm?.numThreads ?? 1;
  if (options.wasm?.proxy !== undefined) ort.env.wasm.proxy = options.wasm.proxy;
  const fo: FetchOptions = {
    baseUrl: options.baseUrl ?? HUB_URL,
    files: options.files,
    cache: options.cache,
    fetch: options.fetch,
    onProgress: options.onProgress,
    signal: options.signal,
  };
  options.signal?.throwIfAborted();
  const [tokJson, modelBytes] = await Promise.all([loadFile(TOKENIZER, fo), loadFile(MODELS[dtype], fo)]);
  options.signal?.throwIfAborted(); // cancelled while the files arrived: no model is built
  const tok = new Tokenizer(JSON.parse(new TextDecoder().decode(tokJson)), {});
  const encode: Encode = (text) => tok.encode(text, { add_special_tokens: false }).ids;
  let session: InferenceSession;
  try {
    session = await ort.InferenceSession.create(modelBytes, { executionProviders: [device] });
  } catch (e) {
    if (asked !== "auto" || device === "wasm") throw e;
    // a GPU adapter, but its WebGPU could not run the model: the same build on the CPU, without a second download
    device = "wasm";
    session = await ort.InferenceSession.create(modelBytes, { executionProviders: ["wasm"] });
  }
  const backend = new NanoBackend(session, encode, device, dtype, options.maxBatchTokens);
  return new OpenDecider(backend, {
    name: `opendecider-nano-onnx-${dtype}`,
    kind: "local",
    device,
    dtype,
    repo: REPO,
    revision: REVISION,
  });
}

/** WebGPU when this browser offers a GPU adapter, else WebAssembly. */
async function bestDevice(): Promise<Device> {
  const gpu = (globalThis.navigator as { gpu?: { requestAdapter(): Promise<unknown> } } | undefined)?.gpu;
  if (!gpu) return "wasm";
  try {
    return (await gpu.requestAdapter()) ? "webgpu" : "wasm";
  } catch {
    return "wasm";
  }
}
