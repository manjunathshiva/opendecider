/** The extension's hidden page (chrome.offscreen, reason WORKERS): it holds opendecider-nano, on WebGPU when there is a
 * GPU, else WebAssembly. Only the runtime API exists here; the service worker does everything else. */
import { Guard, loadNano, type GuardResult, type OpenDecider } from "@opendecider/web";
import { pickBuild } from "./build.js";
import { KIND_QUESTION, ruleQuestion, videoState, type Video } from "./questions.js";

/** "" (the pinned Hub revision), or a test build's local server (build.mjs --model-url). */
declare const __MODEL_URL__: string;

let model: Promise<OpenDecider> | null = null;
let lastProgress = 0;

function load(): Promise<OpenDecider> {
  model ??= pickBuild()
    .then(({ device, dtype }) =>
      loadNano({
        device,
        dtype,
        ...(__MODEL_URL__ ? { baseUrl: __MODEL_URL__ } : {}),
        // ONNX Runtime's own files, shipped in the extension: threads need them, and nothing is fetched from a CDN
        wasm: {
          wasmPaths: chrome.runtime.getURL("ort/"),
          numThreads: Math.max(1, Math.min(8, navigator.hardwareConcurrency - 2)),
        },
        onProgress: (p) => {
          const now = performance.now();
          if (now - lastProgress < 250 && p.loaded < p.total) return;
          lastProgress = now;
          void chrome.runtime.sendMessage({
            type: "progress",
            file: p.file,
            loaded: p.loaded,
            total: p.total,
            cached: p.cached,
          });
        },
      }),
    )
    .catch((e: unknown) => {
      model = null; // the next request tries again
      throw e;
    });
  return model;
}

export interface Decided {
  kinds?: Record<string, number>;
  rule?: number;
}

async function decide(videos: Video[], kinds: boolean, rule: string): Promise<Decided[]> {
  const m = await load();
  const questions: Record<string, object> = {};
  if (kinds) questions.kind = KIND_QUESTION;
  if (rule) questions.rule = ruleQuestion(rule);
  if (!videos.length || !Object.keys(questions).length) return videos.map(() => ({}));
  const rs = await m.systemOneBatch(videos.map(videoState), questions as never);
  return rs.map((r) => {
    const a = r.answers as Record<string, { probabilities?: Record<string, number>; noul?: number }>;
    return { kinds: a.kind?.probabilities, rule: a.rule?.noul };
  });
}

async function guardCheck(text: string): Promise<GuardResult> {
  return new Guard({ model: await load() }).check(text);
}

async function status() {
  const m = model && (await model.catch(() => null));
  return { loaded: !!m, device: m?.meta["device"] ?? null, dtype: m?.meta["dtype"] ?? null };
}

chrome.runtime.onMessage.addListener((m, sender, reply) => {
  if (m?.target !== "offscreen" || sender.id !== chrome.runtime.id) return;
  const run =
    m.type === "load"
      ? load().then(status)
      : m.type === "decide"
        ? decide(m.videos, !!m.kinds, typeof m.rule === "string" ? m.rule : "")
        : m.type === "guard"
          ? guardCheck(String(m.text))
          : m.type === "status"
            ? status()
            : Promise.reject(new Error(`unknown request ${String(m.type)}`));
  run.then(
    (r) => reply({ ok: true, r }),
    (e: unknown) => reply({ ok: false, error: e instanceof Error ? `${e.name}: ${e.message}` : String(e) }),
  );
  return true;
});
