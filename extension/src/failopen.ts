/** When content.ts shows a tile that is still waiting for its answer: the filter fails open when the model stalls, not
 * while a long feed is still being answered a batch at a time (WebAssembly takes seconds a batch). */
export const WAIT_MS = 15_000; // no batch answered for this long: the waiting tiles are shown
export const MAX_WAIT_MS = 60_000; // and no tile waits longer than this in all (a long queue on a slow computer)

/** Whether a tile waiting since `since` is shown now, the last batch having been answered at `progress`. */
export const failOpen = (now: number, since: number, progress: number) =>
  now - Math.max(since, progress) > WAIT_MS || now - since > MAX_WAIT_MS;

/** A tile's verdict once its answer comes: one already shown (it waited too long) stays shown until the settings
 * change, so a video never appears and then vanishes. */
export const settle = (verdict: "wait" | "ok" | "hide" | undefined, answer: "ok" | "hide") =>
  verdict === "ok" ? verdict : answer;
