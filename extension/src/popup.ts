/** The toolbar popup: the model (download once, with consent), the filters, and this page's counts. */
import { pickBuild } from "./build.js";
import { DEFAULTS, FOCUS_HIDES, KIND_NAMES, readSettings, type Kind, type Settings } from "./settings.js";

const LABELS: Record<Kind, string> = {
  news: "News and politics",
  howto: "How-tos and recipes",
  science_tech: "Science and tech",
  education: "Lectures and explainers",
  business: "Business and finance",
  talk: "Interviews and podcasts",
  music: "Music",
  gaming: "Gaming",
  comedy: "Comedy and pranks",
  showbiz: "Celebrity, TV and trailers",
  vlog: "Vlogs and challenges",
};
const MIB = (n: number) => `${Math.round(n / 2 ** 20)} MiB`;
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

let settings: Settings = DEFAULTS;
let typing = 0;

/** Saved at once: the popup closes as soon as the viewer clicks elsewhere, and a pending save would be lost. */
function save(change: Partial<Settings>) {
  clearTimeout(typing);
  typing = 0;
  settings = readSettings({ ...settings, ...change });
  render();
  void chrome.storage.sync.set({ settings });
}

/** The rule is saved when the viewer stops typing (each save sorts the feed again), or when the popup closes. */
function typed(rule: string) {
  clearTimeout(typing);
  typing = window.setTimeout(() => save({ rule }), 800);
}

function render() {
  $<HTMLInputElement>("on").checked = settings.on;
  $<HTMLSelectElement>("ruleMode").value = settings.ruleMode;
  if (document.activeElement !== $("rule")) $<HTMLInputElement>("rule").value = settings.rule;
  $<HTMLInputElement>("strictness").value = String(settings.strictness);
  $<HTMLInputElement>("shorts").checked = settings.shorts;
  for (const k of KIND_NAMES) $<HTMLInputElement>(`kind-${k}`).checked = settings.hide.includes(k);
}

function buildKinds() {
  const box = $("kinds");
  for (const k of KIND_NAMES) {
    const label = document.createElement("label");
    label.className = "check";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.id = `kind-${k}`;
    input.addEventListener("change", () =>
      save({ hide: input.checked ? [...settings.hide, k] : settings.hide.filter((h) => h !== k) }),
    );
    label.append(input, LABELS[k]);
    box.append(label);
  }
}

/** The model card: download (with consent), progress, ready, or an error with a retry. */
async function renderModel() {
  const box = $("model");
  const { consent } = await chrome.storage.local.get("consent");
  const { model } = (await chrome.storage.session.get("model")) as {
    model?: { state: string; loaded?: number; total?: number; cached?: boolean; device?: string; error?: string };
  };
  box.replaceChildren();
  const p = document.createElement("p");
  box.append(p);
  const size = MIB((await pickBuild()).bytes); // this computer's build: fp16 with a GPU, q8 without
  if (consent !== true || model?.state === "error") {
    p.textContent =
      model?.state === "error"
        ? `The model could not load: ${model.error ?? "unknown error"}`
        : `The filter runs on this computer: download its model once (${size}). After that it works offline, and nothing you watch is sent anywhere.`;
    if (model?.state === "error") p.className = "error";
    const b = document.createElement("button");
    b.className = "primary";
    b.textContent = model?.state === "error" ? "Try again" : `Download the model (${size})`;
    b.addEventListener("click", () => {
      b.disabled = true;
      void chrome.runtime.sendMessage({ type: "download" });
    });
    box.append(b);
    return;
  }
  if (model?.state === "loading") {
    const fromCache = model.cached !== false;
    p.textContent = fromCache
      ? "Loading the model…"
      : `Downloading the model: ${MIB(model.loaded ?? 0)} of ${MIB(model.total ?? 0)}`;
    if (!fromCache && model.total) {
      const bar = document.createElement("progress");
      bar.max = model.total;
      bar.value = model.loaded ?? 0;
      box.append(bar);
    }
    return;
  }
  p.className = "muted";
  p.textContent =
    model?.state === "ready"
      ? `Model ready, on this computer's ${model.device === "webgpu" ? "GPU (WebGPU)" : "CPU (WebAssembly)"}.`
      : "Model downloaded: it loads when you open YouTube.";
}

async function renderCounts() {
  const out = $("counts");
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab?.id === undefined) return;
  try {
    const c = await chrome.tabs.sendMessage(tab.id, { type: "counts" });
    if (c?.filtering)
      out.textContent = `On this page: ${c.shown} shown, ${c.hidden} hidden${c.waiting ? `, ${c.waiting} being sorted` : ""}.`;
  } catch {
    out.textContent = ""; // not a YouTube page
  }
}

async function main() {
  buildKinds();
  settings = readSettings((await chrome.storage.sync.get("settings")).settings);
  render();
  $<HTMLInputElement>("on").addEventListener("change", (e) => save({ on: (e.target as HTMLInputElement).checked }));
  $<HTMLSelectElement>("ruleMode").addEventListener("change", (e) =>
    save({ ruleMode: (e.target as HTMLSelectElement).value as Settings["ruleMode"] }),
  );
  const rule = $<HTMLInputElement>("rule");
  rule.addEventListener("input", () => typed(rule.value));
  rule.addEventListener("change", () => save({ rule: rule.value }));
  window.addEventListener("pagehide", () => typing && save({ rule: rule.value }));
  $<HTMLInputElement>("strictness").addEventListener("change", (e) =>
    save({ strictness: Number((e.target as HTMLInputElement).value) }),
  );
  $<HTMLInputElement>("shorts").addEventListener("change", (e) =>
    save({ shorts: (e.target as HTMLInputElement).checked }),
  );
  $("focus").addEventListener("click", () => save({ hide: [...FOCUS_HIDES] }));
  $("none").addEventListener("click", () => save({ hide: [] }));
  chrome.storage.onChanged.addListener((c, area) => {
    if ((area === "session" && c.model) || (area === "local" && c.consent)) void renderModel();
  });
  await renderModel();
  await renderCounts();
  window.setInterval(renderCounts, 1500);
}

void main();
