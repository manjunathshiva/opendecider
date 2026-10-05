/** What a YouTube page may send: checked field by field, since a page's content is never trusted. */
import { oneLine } from "./settings.js";
import type { Video } from "./questions.js";

export const MAX_VIDEOS = 16; // per message: verdicts arrive a batch at a time, so the feed fills in as it is judged
export const TITLE_CHARS = 150;
export const CHANNEL_CHARS = 60;
const VIDEO_ID = /^[\w-]{11}$/;

/** The well-formed videos of a page's list, at most MAX_VIDEOS, titles and channels shortened; anything else dropped. */
export function readVideos(list: unknown): Video[] {
  if (!Array.isArray(list)) return [];
  const out: Video[] = [];
  const seen = new Set<string>();
  for (const v of list) {
    if (out.length === MAX_VIDEOS) break;
    if (!v || typeof v !== "object") continue;
    const { id, title, channel } = v as Record<string, unknown>;
    if (typeof id !== "string" || !VIDEO_ID.test(id) || seen.has(id) || typeof title !== "string") continue;
    const t = oneLine(title).slice(0, TITLE_CHARS);
    if (!t) continue;
    seen.add(id);
    out.push({ id, title: t, channel: typeof channel === "string" ? oneLine(channel).slice(0, CHANNEL_CHARS) : "" });
  }
  return out;
}

/** The longest text the guard check takes from a selection (its windows cover all of it). */
export const GUARD_CHARS = 20_000;
