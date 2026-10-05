/** What the viewer chose in the popup (chrome.storage.sync), read defensively: storage can hold anything. */

import KINDS_JSON from "./kinds.json";

/** The kinds of video opendecider-nano tells apart, as one choice question. kinds.json is shared with the benchmark
 * (benchmarks/items.py, the youtube suite), so the published numbers are for exactly this question. */
export const KINDS = KINDS_JSON.kinds;
export type Kind = keyof typeof KINDS;
export const KIND_NAMES = Object.keys(KINDS) as Kind[];
export const KIND_QUESTION_TEXT = KINDS_JSON.question;

/** The kinds Focus hides: everything that is not learning, news or work. */
export const FOCUS_HIDES: readonly Kind[] = KIND_NAMES.filter((k) => !(KINDS_JSON.focus_keeps as string[]).includes(k));

export const RULE_CHARS = 200;

export interface Settings {
  /** Filtering at all. */
  on: boolean;
  /** Kinds to hide. */
  hide: Kind[];
  /** The viewer's own rule: topics ("crypto, celebrity gossip") or a whole yes/no question; "" for none. */
  rule: string;
  /** "hide": hide the videos the rule matches; "only": show only those. */
  ruleMode: "hide" | "only";
  /** 0.2-0.8: how sure the model must be that a video is wanted, else it is hidden. 0.5 is even odds. */
  strictness: number;
  /** Hide Shorts (by their links: no model). */
  shorts: boolean;
}

export const DEFAULTS: Settings = {
  on: true,
  hide: [...FOCUS_HIDES],
  rule: "",
  ruleMode: "hide",
  strictness: 0.5,
  shorts: true,
};

/** Settings from storage: every field checked, anything unusable replaced by its default. */
export function readSettings(raw: unknown): Settings {
  const r = (raw && typeof raw === "object" ? raw : {}) as Record<string, unknown>;
  const hide = Array.isArray(r.hide)
    ? [...new Set(r.hide.filter((k): k is Kind => typeof k === "string" && Object.hasOwn(KINDS, k)))]
    : [...DEFAULTS.hide];
  const strictness =
    typeof r.strictness === "number" && Number.isFinite(r.strictness)
      ? Math.min(0.8, Math.max(0.2, r.strictness))
      : DEFAULTS.strictness;
  return {
    on: typeof r.on === "boolean" ? r.on : DEFAULTS.on,
    hide,
    rule: typeof r.rule === "string" ? oneLine(r.rule).slice(0, RULE_CHARS) : DEFAULTS.rule,
    ruleMode: r.ruleMode === "only" ? "only" : "hide",
    strictness,
    shorts: typeof r.shorts === "boolean" ? r.shorts : DEFAULTS.shorts,
  };
}

/** Text on one line with single spaces (titles, channels, rules). */
export const oneLine = (s: string) => s.replace(/\s+/g, " ").trim();
