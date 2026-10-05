import { describe, expect, it } from "vitest";
import { hash, hides, ruleQuestion, videoState } from "../src/questions.js";
import { DEFAULTS, KIND_NAMES, type Kind } from "../src/settings.js";

const kinds = (top: Kind, p = 0.9) =>
  Object.fromEntries(KIND_NAMES.map((k) => [k, k === top ? p : (1 - p) / (KIND_NAMES.length - 1)])) as Record<
    Kind,
    number
  >;

describe("hides", () => {
  it("hides a video whose kept kinds hold too little probability", () => {
    expect(hides(DEFAULTS, { kinds: kinds("music") })).toBe(true);
    expect(hides(DEFAULTS, { kinds: kinds("news") })).toBe(false);
  });

  it("follows strictness: stricter hides an unsure video, lenient shows it", () => {
    // a coin toss between a kept kind and a hidden one
    const unsure = {
      kinds: { ...Object.fromEntries(KIND_NAMES.map((k) => [k, 0])), news: 0.5, music: 0.5 } as Record<Kind, number>,
    };
    expect(hides({ ...DEFAULTS, strictness: 0.7 }, unsure)).toBe(true);
    expect(hides({ ...DEFAULTS, strictness: 0.3 }, unsure)).toBe(false);
  });

  it("never hides on a question that was not asked", () => {
    expect(hides(DEFAULTS, {})).toBe(false);
    expect(hides({ ...DEFAULTS, hide: [] }, { kinds: kinds("music") })).toBe(false);
    expect(hides({ ...DEFAULTS, hide: [], rule: "crypto" }, {})).toBe(false);
  });

  it("applies the rule both ways", () => {
    const hideCrypto = { ...DEFAULTS, hide: [], rule: "crypto", ruleMode: "hide" as const };
    expect(hides(hideCrypto, { rule: 0.9 })).toBe(true);
    expect(hides(hideCrypto, { rule: 0.1 })).toBe(false);
    const onlyCooking = { ...hideCrypto, rule: "cooking", ruleMode: "only" as const };
    expect(hides(onlyCooking, { rule: 0.9 })).toBe(false);
    expect(hides(onlyCooking, { rule: 0.1 })).toBe(true);
  });

  it("hides when either filter says so", () => {
    const both = { ...DEFAULTS, rule: "crypto" };
    expect(hides(both, { kinds: kinds("news"), rule: 0.95 })).toBe(true);
    expect(hides(both, { kinds: kinds("music"), rule: 0.05 })).toBe(true);
    expect(hides(both, { kinds: kinds("news"), rule: 0.05 })).toBe(false);
  });
});

describe("the questions", () => {
  it("turns topics into a yes/no question, and keeps a question as written", () => {
    expect(ruleQuestion("crypto, celebrity gossip").instructions).toBe("Is this video about crypto, celebrity gossip?");
    expect(ruleQuestion(" Is this a cooking video? ").instructions).toBe("Is this a cooking video?");
  });

  it("puts the video in the state", () => {
    expect(videoState({ id: "x", title: "T", channel: "" })).toBe("YouTube video. Title: T. Channel: unknown.");
  });

  it("hashes stably", () => {
    expect(hash("crypto")).toBe(hash("crypto"));
    expect(hash("crypto")).not.toBe(hash("cryptos"));
    expect(hash("")).toMatch(/^[0-9a-f]{8}$/);
  });
});
