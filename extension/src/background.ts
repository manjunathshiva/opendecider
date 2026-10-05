/** The service worker: settings, the answer cache, the offscreen page's lifetime, and the right-click guard check.
 * A YouTube page only ever gets "show" or "hide" back; the model never sees anything but titles and channels. */
import { REVISION } from "@opendecider/web";
import { AnswerCache, kindKey, packKinds, ruleKey, unpackKinds } from "./cache.js";
import { GUARD_CHARS, readVideos } from "./messages.js";
import { hash, hides, type Answers } from "./questions.js";
import { KIND_NAMES, readSettings } from "./settings.js";

const IDLE_MINUTES = 10; // the model holds about 3 GiB: give it back when the feed has not been filtered for this long
/** The answers of the build this computer runs (stored when it loads; see build.ts), at the pinned revision: another
 * build or revision is another model, whose answers are asked again. */
async function answers(): Promise<AnswerCache> {
  let { build } = await chrome.storage.local.get("build");
  if (typeof build !== "string") {
    await load();
    ({ build } = await chrome.storage.local.get("build"));
  }
  return new AnswerCache(chrome.storage.local, `${String(build)}@${REVISION}`);
}
const prune = async () =>
  void (await new AnswerCache(chrome.storage.local, "").prune(await chrome.storage.local.get(null)));

// --- the offscreen page -----------------------------------------------------------------------------------------

let creating: Promise<void> | null = null;

async function ensureOffscreen() {
  if (await chrome.offscreen.hasDocument()) return;
  creating ??= chrome.offscreen
    .createDocument({
      url: "offscreen.html",
      reasons: [chrome.offscreen.Reason.WORKERS],
      justification: "Runs the opendecider-nano model on this computer to sort YouTube videos.",
    })
    .finally(() => (creating = null));
  await creating;
}

async function offscreen<T>(msg: Record<string, unknown>): Promise<T> {
  await ensureOffscreen();
  await chrome.storage.session.set({ lastUsed: Date.now() });
  const res = await chrome.runtime.sendMessage({ target: "offscreen", ...msg });
  if (!res?.ok) throw new Error(res?.error ?? "the model page did not answer");
  return res.r as T;
}

chrome.alarms.create("idle", { periodInMinutes: 1 });
chrome.alarms.create("prune", { periodInMinutes: 360 });
chrome.alarms.onAlarm.addListener(async (a) => {
  if (a.name === "prune") return prune();
  if (a.name !== "idle") return;
  const { lastUsed = 0 } = await chrome.storage.session.get("lastUsed");
  // never while the model loads: a slow connection can take longer than IDLE_MINUTES to download it
  if (loading || Date.now() - (lastUsed as number) < IDLE_MINUTES * 60_000) return;
  if (await chrome.offscreen.hasDocument()) {
    ready = false;
    await chrome.offscreen.closeDocument();
    await chrome.storage.session.set({ model: { state: "unloaded" } });
  }
});

// --- the model's state, for the popup -----------------------------------------------------------------------------

/** Consent: nothing is downloaded until the viewer presses Download in the popup. */
const consented = async () => (await chrome.storage.local.get("consent")).consent === true;

const RETRY_MS = 60_000; // after a failed load (offline, say), each new batch of tiles does not try again at once
let ready = false; // the offscreen page holds a loaded model (reset when it is closed, or this worker restarts)
let failedAt = 0;
let loading: Promise<void> | null = null;

async function load(retry = false): Promise<void> {
  if (ready && (await chrome.offscreen.hasDocument())) return;
  if (!retry && Date.now() - failedAt < RETRY_MS) throw new Error("the model failed to load a moment ago");
  loading ??= (async () => {
    await chrome.storage.session.set({ model: { state: "loading" } });
    try {
      // the build that loaded last time: q8 where WebGPU could not run the model, so it is not tried on every load
      const { build } = await chrome.storage.local.get("build");
      const s = await offscreen<{ device: string; dtype: string }>({ type: "load", build });
      ready = true;
      await chrome.storage.local.set({ build: s.dtype });
      await chrome.storage.session.set({ model: { state: "ready", device: s.device, dtype: s.dtype } });
    } catch (e) {
      failedAt = Date.now();
      await chrome.storage.session.set({
        model: { state: "error", error: e instanceof Error ? e.message : String(e) },
      });
      throw e;
    } finally {
      loading = null;
    }
  })();
  return loading;
}

// --- deciding a page's videos -------------------------------------------------------------------------------------

async function triage(list: unknown): Promise<Record<string, "ok" | "hide">> {
  const settings = readSettings((await chrome.storage.sync.get("settings")).settings);
  const videos = readVideos(list);
  const rule = settings.rule ? hash(`${settings.rule}`) : "";
  const keys = videos.flatMap((v) => [
    ...(settings.hide.length ? [kindKey(v.id)] : []),
    ...(rule ? [ruleKey(rule, v.id)] : []),
  ]);
  const cache = await answers();
  const known = await cache.get(keys);
  const ask = videos.filter(
    (v) => (settings.hide.length && !known.has(kindKey(v.id))) || (rule && !known.has(ruleKey(rule, v.id))),
  );
  if (ask.length) {
    await load();
    const got = await offscreen<{ kinds?: Record<string, number>; rule?: number }[]>({
      type: "decide",
      videos: ask,
      kinds: settings.hide.length > 0,
      rule: settings.rule,
    });
    const fresh = new Map<string, number[] | number>();
    ask.forEach((v, i) => {
      const a = got[i];
      if (a?.kinds) fresh.set(kindKey(v.id), packKinds(KIND_NAMES, a.kinds));
      if (rule && typeof a?.rule === "number") fresh.set(ruleKey(rule, v.id), a.rule);
    });
    await cache.set(fresh);
    fresh.forEach((p, k) => known.set(k, p));
  }
  return Object.fromEntries(
    videos.map((v) => {
      const k = known.get(kindKey(v.id));
      const r = rule ? known.get(ruleKey(rule, v.id)) : undefined;
      const a: Answers = {
        ...(Array.isArray(k) ? { kinds: unpackKinds(KIND_NAMES, k) } : {}),
        ...(typeof r === "number" ? { rule: r } : {}),
      };
      return [v.id, hides(settings, a) ? "hide" : "ok"];
    }),
  );
}

// --- messages -----------------------------------------------------------------------------------------------------

chrome.runtime.onMessage.addListener((m, sender, reply) => {
  if (sender.id !== chrome.runtime.id || m?.target === "offscreen") return;
  if (m?.type === "progress" && sender.url?.startsWith(chrome.runtime.getURL("offscreen.html"))) {
    // from the offscreen page: shown in the popup while the model downloads (and keeps it from being closed as idle)
    void chrome.storage.session.set({
      lastUsed: Date.now(),
      model: {
        state: "loading",
        file: String(m.file),
        loaded: Number(m.loaded),
        total: Number(m.total),
        cached: !!m.cached,
      },
    });
    return;
  }
  if (m?.type === "triage" && sender.tab && sender.url?.startsWith("https://www.youtube.com/")) {
    consented()
      .then((ok) => (ok ? triage(m.videos) : Promise.reject(new Error("no-model"))))
      .then(
        (verdicts) => reply({ ok: true, verdicts }),
        (e: unknown) => reply({ ok: false, error: String(e instanceof Error ? e.message : e) }),
      );
    return true;
  }
  // everything below is for the extension's own pages (the popup, the guard window), which may have a tab too
  if (!sender.url?.startsWith(chrome.runtime.getURL(""))) return;
  if (m?.type === "download") {
    chrome.storage.local
      .set({ consent: true })
      .then(() => load(true)) // the viewer asked: try now, even right after a failure
      .then(
        () => reply({ ok: true }),
        (e: unknown) => reply({ ok: false, error: String(e) }),
      );
    return true;
  }
  if (m?.type === "guard") {
    consented()
      .then((ok) => (ok ? load() : Promise.reject(new Error("no-model"))))
      .then(() => offscreen({ type: "guard", text: String(m.text ?? "").slice(0, GUARD_CHARS) }))
      .then(
        (r) => reply({ ok: true, r }),
        (e: unknown) => reply({ ok: false, error: String(e instanceof Error ? e.message : e) }),
      );
    return true;
  }
});

// --- right-click: check selected text with the guard --------------------------------------------------------------

chrome.runtime.onInstalled.addListener((details) => {
  chrome.contextMenus.create({ id: "guard", title: "Check with OpenDecider guard", contexts: ["selection"] });
  // each new version of the extension tries WebGPU again, in case it fell back to the CPU build after a passing failure
  if (details.reason === "install" || details.reason === "update") void chrome.storage.local.remove("build");
});

chrome.contextMenus.onClicked.addListener(async (info) => {
  if (info.menuItemId !== "guard" || !info.selectionText) return;
  const id = crypto.randomUUID();
  await chrome.storage.session.set({ [`check:${id}`]: info.selectionText.slice(0, GUARD_CHARS) });
  await chrome.windows.create({ url: `check.html#${id}`, type: "popup", width: 440, height: 360 });
});

// the cache keeps the newest answers only (and on the "prune" alarm above)
chrome.runtime.onStartup.addListener(prune);
