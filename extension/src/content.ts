/** OpenDecider Focus on YouTube (the page's isolated world). Each feed tile is marked "wait" as soon as its title exists,
 * which the stylesheet hides, until the service worker answers "ok" or "hide". Anything that goes wrong shows the
 * video: the filter fails open, never into an empty feed. */
import { MAX_VIDEOS } from "./messages.js";
import { readSettings } from "./settings.js";

const TILES = "ytd-rich-item-renderer, ytd-video-renderer, ytd-compact-video-renderer, yt-lockup-view-model";
const FILTERED = new Set(["/", "/results", "/watch"]); // home, search, and the list beside a video
const WAIT_MS = 15_000; // a tile still waiting this long is shown
const root = document.documentElement;
const verdicts = new Map<string, "wait" | "ok" | "hide">();
const queue = new Map<string, { id: string; title: string; channel: string }>();
const waitingSince = new Map<string, number>();
let on = false;
let busy = false;
let generation = 0; // new settings: answers to an earlier batch are dropped, and its videos are asked again
let gather = 0;

const text = (el: Element | null | undefined) => (el?.textContent ?? "").replace(/\s+/g, " ").trim();
const channelOf = (tile: Element) =>
  [...tile.querySelectorAll('a[href^="/@"], a[href^="/channel/"]')].map(text).find(Boolean) ?? "";

function videoId(tile: Element): string | null {
  const a = tile.querySelector<HTMLAnchorElement>('a[href*="watch?v="]');
  try {
    return a ? new URL(a.href).searchParams.get("v") : null;
  } catch {
    return null;
  }
}

function mark(id: string, v: "wait" | "ok" | "hide") {
  verdicts.set(id, v);
  for (const t of document.querySelectorAll<HTMLElement>(`[data-odf-id="${CSS.escape(id)}"]`)) t.dataset.odf = v;
}

function sweep() {
  const filtering = on && !!chrome.runtime?.id && FILTERED.has(location.pathname); // extension reloaded: show all
  root.toggleAttribute("data-odf-feed", filtering);
  if (!filtering) return;
  for (const tile of document.querySelectorAll<HTMLElement>(TILES)) {
    if (tile.parentElement?.closest(TILES)) continue; // the grid cell, not the card inside it
    const id = videoId(tile);
    if (!id) continue; // Shorts, playlists, channels: not judged (Shorts are hidden by their links)
    if (tile.dataset.odfId !== id) {
      // a new tile, or YouTube reused this one for another video
      tile.dataset.odfId = id;
      delete tile.dataset.odf;
    }
    if (verdicts.has(id)) {
      tile.dataset.odf = verdicts.get(id)!;
      continue;
    }
    const title = text(tile.querySelector("#video-title, h3"));
    if (!title) continue; // not rendered yet: shown until it is, then judged
    tile.dataset.odf = "wait";
    verdicts.set(id, "wait");
    waitingSince.set(id, Date.now());
    queue.set(id, { id, title, channel: channelOf(tile) });
  }
  if (!busy && queue.size && !gather) gather = window.setTimeout(judge, 120);
}

async function judge() {
  gather = 0;
  if (busy || !queue.size) return;
  busy = true;
  const gen = generation;
  const batch = [...queue.values()].slice(0, MAX_VIDEOS);
  for (const v of batch) queue.delete(v.id);
  let got: Record<string, string> | null = null;
  try {
    const r = await chrome.runtime.sendMessage({ type: "triage", videos: batch });
    if (r?.ok) got = r.verdicts;
    else if (r?.error === "no-model") on = false; // no model yet: show everything until the settings change
  } catch {
    // the extension was reloaded or the service worker failed: show these videos
  } finally {
    if (gen === generation)
      for (const v of batch) {
        waitingSince.delete(v.id);
        mark(v.id, got?.[v.id] === "hide" ? "hide" : "ok");
      }
    busy = false;
    if (!on) for (const id of queue.keys()) (mark(id, "ok"), queue.delete(id));
    sweep();
  }
}

// fail open: a tile that waits too long is shown
window.setInterval(() => {
  const now = Date.now();
  for (const [id, t] of waitingSince) if (now - t > WAIT_MS && verdicts.get(id) === "wait") mark(id, "ok");
  sweep(); // YouTube sometimes swaps a tile's link without adding nodes
}, 1000);

async function apply(raw: unknown) {
  const s = readSettings(raw);
  // nothing is hidden before the viewer has downloaded the model (from the popup)
  on = s.on && (await chrome.storage.local.get("consent")).consent === true;
  root.toggleAttribute("data-odf-shorts", s.on && s.shorts);
  // new settings: every verdict is computed again (from the cache, mostly)
  generation++;
  verdicts.clear();
  queue.clear();
  waitingSince.clear();
  for (const t of document.querySelectorAll<HTMLElement>("[data-odf]")) delete t.dataset.odf;
  sweep();
}

chrome.storage.sync.get("settings").then((s) => apply(s.settings));
chrome.storage.onChanged.addListener((c, area) => {
  if (area === "sync" && c.settings) void apply(c.settings.newValue);
  if (area === "local" && c.consent) void chrome.storage.sync.get("settings").then((s) => apply(s.settings));
});

// the popup asks how this page was filtered
chrome.runtime.onMessage.addListener((m, sender, reply) => {
  if (m?.type !== "counts" || sender.id !== chrome.runtime.id) return;
  let shown = 0;
  let hidden = 0;
  for (const v of verdicts.values()) v === "hide" ? hidden++ : v === "ok" && shown++;
  reply({
    shown,
    hidden,
    waiting: [...verdicts.values()].filter((v) => v === "wait").length,
    filtering: root.hasAttribute("data-odf-feed"),
  });
});

// YouTube changes the page constantly: one sweep per frame, still before the frame is painted (so nothing flashes)
let scheduled = false;
new MutationObserver(() => {
  if (scheduled) return;
  scheduled = true;
  requestAnimationFrame(() => {
    scheduled = false;
    sweep();
  });
}).observe(document, { childList: true, subtree: true });
