import { describe, expect, it } from "vitest";
import { AnswerCache, MAX_ENTRIES, kindKey, packKinds, ruleKey, unpackKinds } from "../src/cache.js";
import { KIND_NAMES } from "../src/settings.js";

/** chrome.storage.local, in memory. */
function memory() {
  const data: Record<string, unknown> = {};
  return {
    data,
    async get(keys: string[]) {
      return Object.fromEntries(keys.filter((k) => k in data).map((k) => [k, data[k]]));
    },
    async set(items: Record<string, unknown>) {
      Object.assign(data, items);
    },
    async remove(keys: string[]) {
      for (const k of keys) delete data[k];
    },
  } as unknown as chrome.storage.StorageArea & { data: Record<string, unknown> };
}

describe("AnswerCache", () => {
  it("returns what was stored for the same model build only", async () => {
    const area = memory();
    const a = new AnswerCache(area, "fp16@rev1");
    await a.set(
      new Map<string, number[] | number>([
        [kindKey("v1"), [0.1, 0.9]],
        [ruleKey("h", "v1"), 0.7],
      ]),
    );
    expect(await a.get([kindKey("v1"), ruleKey("h", "v1"), kindKey("v2")])).toEqual(
      new Map<string, number[] | number>([
        [kindKey("v1"), [0.1, 0.9]],
        [ruleKey("h", "v1"), 0.7],
      ]),
    );
    expect((await new AnswerCache(area, "fp16@rev2").get([kindKey("v1")])).size).toBe(0);
  });

  it("ignores a damaged entry", async () => {
    const area = memory();
    area.data[kindKey("v1")] = { m: "fp16@rev1", p: "oops", t: 1 };
    expect((await new AnswerCache(area, "fp16@rev1").get([kindKey("v1")])).size).toBe(0);
  });

  it("prunes to the newest MAX_ENTRIES answers, and never touches other keys", async () => {
    const area = memory();
    let minute = 0;
    const a = new AnswerCache(area, "m", () => minute);
    for (let i = 0; i < MAX_ENTRIES + 10; i++) {
      minute = i;
      await a.set(new Map([[kindKey(`v${i}`), [1]]]));
    }
    area.data.consent = true;
    expect(await a.prune(area.data)).toBe(10);
    expect(kindKey("v0") in area.data).toBe(false);
    expect(kindKey(`v${MAX_ENTRIES + 9}`) in area.data).toBe(true);
    expect(area.data.consent).toBe(true);
  });

  it("packs the kinds' probabilities in a fixed order", () => {
    const p = Object.fromEntries(KIND_NAMES.map((k, i) => [k, i / 100]));
    expect(unpackKinds(KIND_NAMES, packKinds(KIND_NAMES, p))).toEqual(p);
  });
});
