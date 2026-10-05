/** Answers already computed, in chrome.storage.local: a video is asked about once per question and model build. */
import type { Kind } from "./settings.js";

export const MAX_ENTRIES = 5000;

type Area = Pick<chrome.storage.StorageArea, "get" | "set" | "remove">;

interface Entry {
  /** The model build and question the answer came from: a different one is a miss. */
  m: string;
  /** The kinds' probabilities (in KIND_NAMES order) or the rule's yes probability. */
  p: number[] | number;
  /** When it was answered, in minutes (for pruning). */
  t: number;
}

export const kindKey = (id: string) => `k:${id}`;
export const ruleKey = (rule: string, id: string) => `r:${rule}:${id}`;

export class AnswerCache {
  constructor(
    private readonly area: Area,
    private readonly model: string,
    private readonly now: () => number = () => Math.floor(Date.now() / 60_000),
  ) {}

  /** The cached answers among `keys`, for this model build only. */
  async get(keys: string[]): Promise<Map<string, number[] | number>> {
    const got = keys.length ? await this.area.get(keys) : {};
    const out = new Map<string, number[] | number>();
    for (const k of keys) {
      const e = got[k] as Entry | undefined;
      if (e && e.m === this.model && (typeof e.p === "number" || Array.isArray(e.p))) out.set(k, e.p);
    }
    return out;
  }

  async set(answers: Map<string, number[] | number>): Promise<void> {
    if (!answers.size) return;
    const t = this.now();
    await this.area.set(Object.fromEntries([...answers].map(([k, p]) => [k, { m: this.model, p, t } satisfies Entry])));
  }

  /** Keep the newest MAX_ENTRIES answers. `all` is every stored item (storage.local.get(null)). */
  async prune(all: Record<string, unknown>): Promise<number> {
    const entries = Object.entries(all).filter(([k]) => k.startsWith("k:") || k.startsWith("r:"));
    if (entries.length <= MAX_ENTRIES) return 0;
    const old = entries
      .sort(([, a], [, b]) => ((a as Entry)?.t ?? 0) - ((b as Entry)?.t ?? 0))
      .slice(0, entries.length - MAX_ENTRIES)
      .map(([k]) => k);
    await this.area.remove(old);
    return old.length;
  }
}

/** The kinds' probabilities as stored (KIND_NAMES order) and back. */
export const packKinds = (names: readonly Kind[], p: Record<string, number>) => names.map((k) => p[k] ?? 0);
export const unpackKinds = (names: readonly Kind[], p: number[]) =>
  Object.fromEntries(names.map((k, i) => [k, p[i] ?? 0])) as Record<Kind, number>;
