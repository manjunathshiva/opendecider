import { describe, expect, it } from "vitest";
import { MAX_WAIT_MS, WAIT_MS, failOpen, settle } from "../src/failopen.js";

const S = 1000;

describe("failOpen (when a waiting tile is shown anyway)", () => {
  it("keeps a tile waiting past WAIT_MS of age while batches keep being answered", () => {
    // waiting since 0, batches answered at 9 s, 18 s, 27 s: never WAIT_MS without one
    expect(failOpen(16 * S, 0, 9 * S)).toBe(false);
    expect(failOpen(26 * S, 0, 18 * S)).toBe(false);
    expect(failOpen(33 * S, 0, 27 * S)).toBe(false);
  });

  it("shows it once no batch has been answered for WAIT_MS", () => {
    expect(failOpen(WAIT_MS, 0, 0)).toBe(false);
    expect(failOpen(WAIT_MS + 1, 0, 0)).toBe(true);
    expect(failOpen(9 * S + WAIT_MS + 1, 0, 9 * S)).toBe(true);
    // an answer long before the tile appeared does not count: its own wait does
    expect(failOpen(100 * S + WAIT_MS + 1, 100 * S, 3 * S)).toBe(true);
    expect(failOpen(100 * S + WAIT_MS, 100 * S, 3 * S)).toBe(false);
  });

  it("never lets a tile wait longer than MAX_WAIT_MS in all", () => {
    expect(failOpen(MAX_WAIT_MS, 0, MAX_WAIT_MS - S)).toBe(false);
    expect(failOpen(MAX_WAIT_MS + 1, 0, MAX_WAIT_MS)).toBe(true);
  });
});

describe("settle (a tile's verdict when its answer comes)", () => {
  it("takes the answer for a waiting tile", () => {
    expect(settle("wait", "hide")).toBe("hide");
    expect(settle("wait", "ok")).toBe("ok");
  });

  it("keeps a tile shown by failOpen shown, whatever the answer", () => {
    expect(settle("ok", "hide")).toBe("ok");
  });
});

/** content.ts's bookkeeping, on a clock: tiles waiting since 0, sent MAX_VIDEOS at a time, each batch answered `per`
 * ms after it was sent, the fail-open check once a second. Returns the verdicts, and how many tiles were shown before
 * their answer came. */
function feed(count: number, per: number, answers: (i: number) => "ok" | "hide", verdicts = new Map<number, string>()) {
  const waitingSince = new Map<number, number>();
  for (let i = 0; i < count; i++) (verdicts.set(i, "wait"), waitingSince.set(i, 0));
  let progress = 0;
  let early = 0;
  for (let b = 0; b * 16 < count; b++) {
    const at = (b + 1) * per;
    for (let now = b * per; now < at; now += S)
      for (const [i, t] of waitingSince)
        if (failOpen(now, t, progress) && verdicts.get(i) === "wait") (verdicts.set(i, "ok"), early++);
    progress = at;
    for (let i = b * 16; i < Math.min(count, (b + 1) * 16); i++) {
      waitingSince.delete(i);
      verdicts.set(i, settle(verdicts.get(i) as "wait" | "ok", answers(i)));
    }
  }
  return { early, verdicts };
}

describe("a long feed on WebAssembly (9 s a batch of 16, measured on a 16 GB M4 Mac mini)", () => {
  it("keeps every tile hidden until its own answer, though the last ones wait 27 s", () => {
    const { early, verdicts } = feed(40, 9 * S, (i) => (i % 2 ? "hide" : "ok"));
    expect(early).toBe(0);
    expect([...verdicts.values()].filter((v) => v === "hide")).toHaveLength(20);
  });

  it("shows tiles whose batch takes longer than WAIT_MS, and keeps them shown in that generation", () => {
    const { early, verdicts } = feed(16, 20 * S, () => "hide");
    expect(early).toBe(16);
    expect([...verdicts.values()].every((v) => v === "ok")).toBe(true);
    // new settings forget every verdict: the same tiles wait again, and their answers count
    verdicts.clear();
    expect([...feed(16, 9 * S, () => "hide", verdicts).verdicts.values()].every((v) => v === "hide")).toBe(true);
  });

  it("bounds a very long queue by MAX_WAIT_MS", () => {
    // 160 tiles, 9 s a batch: the 7th batch on would answer after 60 s (63 s, ..., 90 s), so they are shown at 61 s
    const { verdicts } = feed(160, 9 * S, () => "hide");
    expect([...verdicts].filter(([, v]) => v === "ok").map(([i]) => i)).toEqual(
      Array.from({ length: 64 }, (_, i) => 96 + i),
    );
  });
});
